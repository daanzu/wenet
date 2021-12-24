// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
// Author: daanzu@gmail.com (David Zurow)

#include "decoder/ctc_prefix_wfst_beam_search.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "kaldi/base/kaldi-utils.h"
#include "kaldi/fstext/fstext-utils.h"

#include "utils/log.h"
#include "utils/string.h"

namespace wenet {

using PrefixScore = CtcPrefixWfstBeamSearch::PrefixScore;
using PrefixHash = CtcPrefixWfstBeamSearch::PrefixHash;
using PrefixState = CtcPrefixWfstBeamSearch::PrefixState;
using PrefixStateHash = CtcPrefixWfstBeamSearch::PrefixStateHash;
using HypsMap = CtcPrefixWfstBeamSearch::HypsMap;
using StateId = fst::StdArc::StateId;
using Label = fst::StdArc::Label;
using Weight = fst::StdArc::Weight;

static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
constexpr float SuperNoLikelihood = -std::numeric_limits<float>::infinity();
constexpr float NoLikelihood = -std::numeric_limits<float>::max();
constexpr float FullLikelihood = 0.0f;

std::string target_string_;

CtcPrefixWfstBeamSearch::CtcPrefixWfstBeamSearch(
    std::shared_ptr<fst::StdFst> fst,
    std::shared_ptr<fst::SymbolTable> word_table,
    std::shared_ptr<fst::SymbolTable> unit_table,
    const CtcPrefixWfstBeamSearchOptions& opts)
    : opts_(opts),
      grammar_fst_(fst),
      grammar_matcher_(std::make_unique<CtcPrefixWfstBeamSearch::Matcher>(*grammar_fst_, fst::MATCH_INPUT)),
      word_table_(word_table),
      unit_table_(unit_table),
      dictation_lexiconfree_label_(word_table->Find(opts_.dictation_lexiconfree_label)),
      nonterm_end_label_(word_table->Find(opts_.nonterm_end_label)) {
  if (opts_.process_partial_word_prefixes) {
    BuildUnitDictionaryTrie();
  }
  Reset();
}

void CtcPrefixWfstBeamSearch::Reset() {
  hypotheses_.clear();
  likelihood_.clear();
  cur_hyps_.clear();
  viterbi_likelihood_.clear();
  times_.clear();
  abs_time_step_ = 0;
  PrefixScore prefix_score;
  prefix_score.s = 0.0;
  prefix_score.ns = -kFloatMax;
  prefix_score.v_s = 0.0;
  prefix_score.v_ns = 0.0;
  std::vector<int> empty;
  cur_hyps_.emplace(PrefixStateHash::make_prefix_state(empty, prefix_score), prefix_score);
}

void CtcPrefixWfstBeamSearch::ResetFst(std::shared_ptr<fst::StdFst> fst) {
  Reset();
  grammar_fst_ = fst;
  grammar_matcher_.reset(new CtcPrefixWfstBeamSearch::Matcher(*grammar_fst_, fst::MATCH_INPUT));
}

static bool PrefixScoreCompare(
    const std::pair<PrefixState, PrefixScore>& a,
    const std::pair<PrefixState, PrefixScore>& b) {
  return a.second.score() > b.second.score();
}

PrefixScore& GetNextHyp(HypsMap& next_hyps, const std::vector<int>& prefix, const PrefixScore& current_score) {
  // next_hyps is initialized empty at the beginning of each time stamp, so each of its prefix_scores are only generated for/in the current time stamp.
  // The prefix_scores inherit their scores (and other state data) exclusively from prefix_scores from the previous time stamp (no data from others in the current time stamp).
  // Obviously all prefix_scores for a given prefix represent that same prefix, but they may have taken different paths to get there.
  // Inheritance (backward-looking) of prefix_score???
  const auto& key = PrefixStateHash::make_prefix_state(prefix, current_score);
  const auto it = next_hyps.find(key);
  if (it != next_hyps.end()) {
    return it->second;
  }
  return next_hyps.emplace(key, PrefixScore::FromFstStateOnly(current_score)).first->second;
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in it.
void CtcPrefixWfstBeamSearch::Search(const torch::Tensor& logp) {
  CHECK_EQ(logp.dtype(), torch::kFloat);
  CHECK_EQ(logp.dim(), 2);
  bool verbose = true;
  for (int t = 0; t < logp.size(0); ++t, ++abs_time_step_) {
    torch::Tensor logp_t = logp[t];
    HypsMap next_hyps;
    // 1. First beam prune, only select topk candidates
    std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
    Tensor topk_score = std::get<0>(topk);
    Tensor topk_index = std::get<1>(topk);

    if (verbose) {
      for (int i = 0; i < topk_index.size(0); ++i) {
        VLOG(1) << "start t" << abs_time_step_ << ": " << unit_table_->Find(topk_index[i].item<int>()) << " " << topk_index[i].item<int>() << " " << topk_score[i].item<float>();
      }
      for (const auto& it : cur_hyps_) {
        const auto& prefix_score = it.second;
        // CHECK(prefix_score.delayed_fst_update_tokens.empty());
        CHECK(!prefix_score.delayed_fst_update);
      }
    }

    // 2. Token passing
    for (int i = 0; i < topk_index.size(0); ++i) {
      int id = topk_index[i].item<int>();
      auto prob = topk_score[i].item<float>();
      for (const auto& it : cur_hyps_) {
        const std::vector<int>& prefix = std::get<0>(it.first);
        const PrefixScore& prefix_score = it.second;

        // If prefix doesn't exist in next_hyps, next_hyps[prefix] will insert PrefixScore(-inf, -inf) by default, since the default constructor of PrefixScore will set fields s(blank ending score) and ns(none blank ending score) to -inf, respectively.
        // Cases 0 and 1: Prefix is unchanged, so don't/can't modify the fst states, but they must inherit them from the previous time step to maintain continuity.
        // Cases 2 and 3: Prefix is extended by a single unit, so do/can modify the fst states: first inherit them from the previous time step, then compute the new fst state.
        //   These cases (2 & 3) are mutually exclusive for a given prefix & unit, they can't both modify a given prefix_score. They can modify with 0 and/or 1, however.
        //   These cases (2 & 3) can inherit from another of their type (2 or 3) from the previous time step, but they likely also must sometimes inherit from the other types (0 and/or 1) of the previous time step as well.
        // The two cases 1 & 2 together are the splitting case: they both inherit from the same prefix_score from the previous time step, but each inherits from separate mutually exclusive scores from it (.s xor .ns).
        // Each case only updates either .s or .ns, never both, because it is only handling a single unit. However, multiple cases can update the same prefix_score at the same time.

        if (false) {
          std::vector<int> new_prefix(prefix);
          if (id != opts_.blank) new_prefix.emplace_back(id);
          std::string new_prefix_string = IdsToString(new_prefix);
          if (new_prefix_string == target_string_.substr(0, new_prefix_string.size())) {
            VLOG(2) << "new_prefix_string: " << new_prefix_string;
            if (!verbose && new_prefix_string.size() >= 150) {
              // VLOG(1) << "!!!!";
              verbose = true;
            }
          }
          if (new_prefix_string.substr(0, 6) == "▁BUT") {
            VLOG(2) << "new_prefix_string: " << new_prefix_string;
          }
        }

        if (id == opts_.blank) {
          // Case 0: *a + ε => *a
          // If we propose a blank, the prefix doesn't change. Only the probability of ending in blank gets updated, and the previous character could be either blank or nonblank. Progression of prefix_score: *a at time t-1 => *a at time t.
          PrefixScore& next_score = GetNextHyp(next_hyps, prefix, prefix_score);
          next_score.UpdateStamp("case0(" + IdsToString(prefix) + "|" + unit_table_->Find(id) + ")@" + std::to_string(abs_time_step_), prefix);
          next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
          next_score.v_s = prefix_score.viterbi_score() + prob;
          next_score.times_s = prefix_score.times();

        } else if (!prefix.empty() && id == prefix.back()) {
          // Case 1: *a + a => *a
          // If we propose a repeat nonblank, after a nonblank, the prefix doesn't change (the repeats are merged). Only the probability of ending in nonblank gets updated, and we know the previous character was nonblank. Progression of prefix_score: *a at time t-1 => *a at time t.
          PrefixScore& next_score1 = GetNextHyp(next_hyps, prefix, prefix_score);
          next_score1.UpdateStamp("case1(" + IdsToString(prefix) + "|" + unit_table_->Find(id) + ")@" + std::to_string(abs_time_step_), prefix);
          next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);
          if (next_score1.v_ns < prefix_score.v_ns + prob) {
            next_score1.v_ns = prefix_score.v_ns + prob;
            if (next_score1.cur_token_prob < prob) {
              next_score1.cur_token_prob = prob;
              next_score1.times_ns = prefix_score.times_ns;
              CHECK_GT(next_score1.times_ns.size(), 0);
              next_score1.times_ns.back() = abs_time_step_;
            }
          }

          // Case 2: *aε + a => *aa
          // If we propose a repeat nonblank, after a blank, the prefix does change. Only the probability of ending in nonblank gets updated, and we know the previous character was blank. Progression of prefix_score: *a at time t-1 => *aa at time t.
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score2 = GetNextHyp(next_hyps, new_prefix, prefix_score);
          next_score2.UpdateStamp("case2(" + IdsToString(prefix) + "|" + unit_table_->Find(id) + ")@" + std::to_string(abs_time_step_), new_prefix);
          next_score2.SetDelayedFstUpdate(prefix, prefix_score, id);
          next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob);
          if (next_score2.v_ns < prefix_score.v_s + prob) {
            next_score2.v_ns = prefix_score.v_s + prob;
            next_score2.cur_token_prob = prob;
            next_score2.times_ns = prefix_score.times_s;
            next_score2.times_ns.emplace_back(abs_time_step_);
          }

        } else {
          // Case 3: *a + b => *ab, *aε + b => *ab
          // If we propose a non-repeat nonblank, the prefix must change. Only the probability of ending in nonblank gets updated, and the previous character could be either blank or nonblank. Progression of prefix_score: *a at time t-1 => *ab at time t.
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score = GetNextHyp(next_hyps, new_prefix, prefix_score);
          next_score.UpdateStamp("case3(" + IdsToString(prefix) + "|" + unit_table_->Find(id) + ")@" + std::to_string(abs_time_step_), new_prefix);
          next_score.SetDelayedFstUpdate(prefix, prefix_score, id);
          next_score.ns = LogAdd(next_score.ns, prefix_score.score() + prob);
          if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
            next_score.v_ns = prefix_score.viterbi_score() + prob;
            next_score.cur_token_prob = prob;
            next_score.times_ns = prefix_score.times();
            next_score.times_ns.emplace_back(abs_time_step_);
          }
        }
      }
    }

    ProcessFstUpdates(next_hyps, false);

    // 3. Second beam prune, only keep top n best paths
    // 4. Update cur_hyps_ and get new result
    PruneAndUpdateHyps(next_hyps);
    VLOG(1) << "";
  }
  // FIXME: When utterance is complete, we should handle checking for FST Final states.
}

void CtcPrefixWfstBeamSearch::PruneAndUpdateHyps(const HypsMap& next_hyps) {
  std::vector<std::pair<PrefixState, PrefixScore>> arr(next_hyps.begin(), next_hyps.end());
  int second_beam_size =
      std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
  std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
                    PrefixScoreCompare);
  arr.resize(second_beam_size);
  std::sort(arr.begin(), arr.end(), PrefixScoreCompare);
  VLOG(1) << "PruneAndUpdateHyps: Pruned from " << next_hyps.size() << " hyps to " << arr.size();

  cur_hyps_.clear();
  hypotheses_.clear();
  likelihood_.clear();
  viterbi_likelihood_.clear();
  times_.clear();
  for (auto& item : arr) {
    const auto& prefix = std::get<0>(item.first);
    VLOG(1) << "end t" << abs_time_step_ << ": "
            << IdsToString(prefix, -1, 60) << " = " << item.second.score()
            << "    state=(" << item.second.StateString() << ")"
            // << "    grammar_fst_state=" << item.second.grammar_fst_state
            // << "    dict_fst_state=" << item.second.dictionary_fst_state
            // << "    updates=" << item.second.updates.size()
            // << "    updates=" << JoinString(" | ", item.second.updates)
            ;
    cur_hyps_.emplace(item.first, item.second);
    hypotheses_.emplace_back(prefix);
    likelihood_.emplace_back(item.second.score());
    viterbi_likelihood_.emplace_back(item.second.viterbi_score());
    times_.emplace_back(item.second.times());
  }
}

inline StateId GetPrefixScoreGrammarFstStateOrFstStart(const PrefixScore& prefix_score, const fst::StdFst& fst) {
  return prefix_score.grammar_fst_state != fst::kNoStateId ? prefix_score.grammar_fst_state : fst.Start();
}

// Called by TorchAsrDecoder::AttentionRescoring(), which is called by TorchAsrDecoder::Rescoring().
void CtcPrefixWfstBeamSearch::FinalizeSearch() {
  // Do final processing: exiting dictation, endpointing words, following epsilons.
  ProcessFstUpdates(cur_hyps_, true);

  // Add in the grammar final weights, removing any hypotheses that are not final.
  for (auto it = cur_hyps_.begin(); it != cur_hyps_.end(); /* see ++it below */) {
    auto& hyp = *it;
    const auto& prefix_state = hyp.first;
    const auto& prefix = std::get<0>(prefix_state);
    PrefixScore& prefix_score = hyp.second;
    auto grammar_fst_state = GetPrefixScoreGrammarFstStateOrFstStart(prefix_score, *grammar_fst_);
    auto final_weight = grammar_fst_->Final(grammar_fst_state);
    VLOG(2) << "FinalizeSearch: " << IdsToString(prefix) << " state=" << grammar_fst_state << " score=" << prefix_score.score() << " final_weight=" << final_weight.Value();
    if (opts_.strict && final_weight == Weight::Zero()) {
      it = cur_hyps_.erase(it);
      continue;
    }
    prefix_score.ns += -final_weight.Value();
    ++it;
  }

  PruneAndUpdateHyps(cur_hyps_);
}

void AddCombineToHypsMap(HypsMap& hyps, const PrefixState& prefix_state, const PrefixScore& prefix_score) {
  HypsMap::iterator it;
  bool inserted;
  std::tie(it, inserted) = hyps.emplace(prefix_state, prefix_score);
  if (!inserted) {
    // Combine the existing and new prefix scores, both for the same prefix_state.
    const auto& old_prefix_state = it->first;
    auto& old_prefix_score = it->second;
    VLOG(2) << "    combine state (" << WfstPrefixStateHash::prefix_state_string(prefix_state) << "):  s = " << old_prefix_score.s << " + " << prefix_score.s << "  ns = " << old_prefix_score.ns << " + " << prefix_score.ns;
    // old_prefix_score.s = LogAdd(old_prefix_score.s, prefix_score.s);
    // old_prefix_score.ns = LogAdd(old_prefix_score.ns, prefix_score.ns);
    old_prefix_score.s = std::max(old_prefix_score.s, prefix_score.s);
    old_prefix_score.ns = std::max(old_prefix_score.ns, prefix_score.ns);
    // FIXME: how best to combine/choose?
    // FIXME: other members?
  }
}

void CtcPrefixWfstBeamSearch::ProcessFstUpdates(HypsMap& next_hyps, bool final) {
  // Now that we have completed all of the standard CTC processing to compute scores, we can do FST processing, which may modify the FST states and possibly even split a single prefix_score into multiple prefix_scores (due to nondeterministic/epsilon FST transitions).
  // TODO: Is it more efficient to make a new map and copy over, or to modify the existing map (considering no rehashes when erasing but rehashes when inserting)?
  VLOG(2) << "ProcessFstUpdates(" << (final ? "FINAL" : "") << "): " << next_hyps.size() << " hyps";
  HypsMap new_next_hyps;
  for (auto& hyp : next_hyps) {
    const auto& prefix_state = hyp.first;
    const auto& new_prefix = std::get<0>(prefix_state);
    PrefixScore& next_score = hyp.second;  // Note: allowed to be non-const because it is only used once below, and then we iterate onward and eventually swap.

    // Verify that the FST state has not been modified yet, and is still equal to what it was originally copied from.
    CHECK(next_score.grammar_fst_state == std::get<1>(prefix_state) && next_score.dictionary_fst_state == std::get<2>(prefix_state) && next_score.prefix_word_id == std::get<3>(prefix_state));
    // If this hypothesis has no delayed update, we just copy it over to new_next_hyps unmodified.
    if (!next_score.delayed_fst_update && !final) {
      VLOG(2) << "ProcessFstUpdates type0 hyp: " << IdsToString(new_prefix, -1, 50) << " " << next_score.grammar_fst_state << " " << next_score.dictionary_fst_state << " " << next_score.prefix_word_id << " = " << next_score.score();
      AddCombineToHypsMap(new_next_hyps, prefix_state, next_score);
      continue;
    }
    VLOG(2) << "ProcessFstUpdates type1 hyp: " << IdsToString(new_prefix, -1, 50) << " " << next_score.grammar_fst_state << " " << next_score.dictionary_fst_state << " " << next_score.prefix_word_id << " = " << next_score.score();

    const auto& token = !final ? next_score.delayed_fst_update_token : std::make_tuple(&new_prefix, &next_score, opts_.blank);
    const auto& current_prefix = *std::get<0>(token);
    const auto& current_prefix_score = *std::get<1>(token);
    auto id = std::get<2>(token);
    if (!final) {
      CHECK(new_prefix.back() == id && std::equal(current_prefix.begin(), current_prefix.end(), new_prefix.begin()));
      next_score.delayed_fst_update = false;
    }

    auto orig_next_score = next_score;
    ComputeFstScores(current_prefix, current_prefix_score, id, next_score, final,
      [this, &new_next_hyps, &new_prefix, &current_prefix, &orig_next_score, id](PrefixScore& new_next_score, float fst_score) {
        // Warning: this takes new_next_score by reference for efficiency, and its modifications are not intended to be used outside of this function.
        // TODO: Is it more efficient to just take as parameters the new next state, and use a new constructor which takes the new state along with the fst_score to add?
        CHECK_LE(fst_score, 0);
        if (opts_.strict && fst_score <= NoLikelihood) return;  // If in strict mode, drop any scores with no likelihood in the grammar.
        VLOG(2) << "    add_new_next_prefix_score: " << IdsToString(current_prefix, id) << " : grammar_fst_state " << orig_next_score.grammar_fst_state << " -> " << new_next_score.grammar_fst_state;
        new_next_score.ns += fst_score;  // Both Case 2 & 3 only update .ns, and they are the only ones that add these tokens.
        if (fst_score < FullLikelihood) {
          VLOG(2) << "        ns " << orig_next_score.ns << " + " << fst_score << " -> " << new_next_score.ns;
        }
        AddCombineToHypsMap(new_next_hyps, PrefixStateHash::make_prefix_state(new_prefix, new_next_score), new_next_score);
      }
    );
  }
  next_hyps.swap(new_next_hyps);
  VLOG(2) << "ProcessFstUpdates(" << (final ? "FINAL" : "") << "): done, left with " << next_hyps.size() << " hyps";
}

// Follow epsilon transitions on grammar, accumulating weights, calling handler function with each grammar state reached (including the initial state). Implements BFS. Does not call handler function on initial state.
void FollowEpsilons(CtcPrefixWfstBeamSearch::Matcher& matcher, const PrefixScore& initial_prefix_score, float initial_weight, bool handle_initial, std::function<void(PrefixScore&, float)> handle_prefix_score) {
  auto initial_fst_state = GetPrefixScoreGrammarFstStateOrFstStart(initial_prefix_score, matcher.GetFst());
  std::list<std::pair<int, float>> state_queue({std::make_pair(initial_fst_state, initial_weight)});
  std::unordered_set<int> queued_states({initial_fst_state});
  while (!state_queue.empty()) {
    const auto fst_state = state_queue.front().first;
    const auto weight = state_queue.front().second;
    state_queue.pop_front();

    if (handle_initial || fst_state != initial_fst_state) {
      // Skip initial state.
      PrefixScore new_prefix_score = initial_prefix_score;
      new_prefix_score.grammar_fst_state = fst_state;
      // VLOG(2) << "FollowEpsilons: handle_prefix_score: " << initial_fst_state << " -> " << new_prefix_score.grammar_fst_state;
      handle_prefix_score(new_prefix_score, weight);
    }

    matcher.SetState(fst_state);
    for (matcher.Find(0); !matcher.Done(); matcher.Next()) {
      const auto& arc = matcher.Value();
      if (queued_states.count(arc.nextstate)) continue;
      state_queue.emplace_back(arc.nextstate, weight + arc.weight.Value());
      queued_states.insert(arc.nextstate);
    }
  }
}

// Check whether there is an epsilon-path from start_state to a state with an arc with the given ilabel.
template <class Arc>
bool CheckIfEpsilonPathToILabel(CtcPrefixWfstBeamSearch::Matcher& matcher, typename Arc::StateId start_state, typename Arc::Label ilabel) {
  auto& fst = matcher.GetFst();
  while (start_state != fst::kNoStateId) {
    matcher.SetState(start_state);
    if (matcher.Find(ilabel)) {
      return true;
    }
    if (!matcher.Find(0)) {
      return false;
    }
    start_state = matcher.Value().nextstate;
    CHECK((matcher.Next(), matcher.Done()));  // Should only have at most one epsilon to follow.
  }
  return false;
}

// DFS traversal to find a final state, following only epsilon transitions. Should be sufficient since FST should be deterministic, so only one path to follow.
template <class Arc>
bool FindFinalState(CtcPrefixWfstBeamSearch::Matcher& matcher, typename Arc::StateId state, typename Arc::StateId* final_state, typename Arc::Label* final_olabel) {
  auto& fst = matcher.GetFst();
  typename Arc::Label olabel = (final_olabel) ? *final_olabel : fst::kNoLabel;
  while (state != fst::kNoStateId) {
    if (fst.Final(state) != Arc::Weight::Zero()) {
      if (final_state) *final_state = state;
      if (final_olabel) *final_olabel = olabel;
      return true;
    }
    matcher.SetState(state);
    if (!matcher.Find(0)) {
      return false;
    }
    state = matcher.Value().nextstate;
    // CHECK(matcher.Value().olabel == 0 || olabel == fst::kNoLabel);  // This should only be set if we are guaranteed not to change word predictions.
    olabel = (matcher.Value().olabel != 0) ? matcher.Value().olabel : olabel;
    CHECK((matcher.Next(), matcher.Done()));  // Should only have at most one epsilon to follow.
  }
  return false;
}

// Computes the negative log likelihood (-infinity..0) of the given prefix + id in the FST, for the given next_prefix_score (containing the scores to use/pass on), and adds the resulting new next_prefix_score (single or multiple) with updated FST states (and inherited scores) using the given handler function.
void CtcPrefixWfstBeamSearch::ComputeFstScores(const std::vector<int>& current_prefix, const PrefixScore& current_prefix_score, int id, PrefixScore next_prefix_score, bool final, std::function<void(PrefixScore&, float)> add_new_next_prefix_score) {
  CHECK_GE(id, 0);
  // We are never called with the blank unit id or an id<0, unless we doing final processing. For final processing, we are called with the blank unit id, to indicate that we should consider any in-progress words to now be complete.
  CHECK((id != opts_.blank) != final);  // XOR

  if (final) {
    // If final, we follow any epsilon transitions, after normal processing.
    add_new_next_prefix_score = [this, add_new_next_prefix_score](PrefixScore& new_next_prefix_score, float weight) {
      FollowEpsilons(*grammar_matcher_, new_next_prefix_score, weight, true, add_new_next_prefix_score);
    };
  }

  // Check whether the completed word fits in the grammar FST, and if so, passes updated prefix_score (following the matching single arc or multiple arcs) (along with the negative log likelihood) to given handler. Only needs to handle grammar_fst, because it only considers complete words. Also handles transitioning to dictation.
  auto check_complete_word = [this, &current_prefix_score](const PrefixScore& next_prefix_score, int word_id, std::function<void(PrefixScore&, float)> add_new_next_prefix_score) {
    // static std::set<int> seen_word_ids;
    // if (seen_word_ids.find(word_id) == seen_word_ids.end()) {
    //   seen_word_ids.insert(word_id);
    // }
    VLOG(2) << "ComputeFstScores: check_complete_word: " << word_table_->Find(word_id);

    CHECK_NE(word_id, fst::kNoLabel);
    auto grammar_fst_state = GetPrefixScoreGrammarFstStateOrFstStart(current_prefix_score, *grammar_fst_);
    grammar_matcher_->SetState(grammar_fst_state);

    auto follow_arc = [&add_new_next_prefix_score](PrefixScore& new_next_prefix_score, const fst::StdArc& arc) {
      auto weight = arc.weight.Value();
      auto nextstate = arc.nextstate;
      new_next_prefix_score.grammar_fst_state = nextstate;
      add_new_next_prefix_score(new_next_prefix_score, -weight);
    };

    if (dictation_lexiconfree_label_ != fst::kNoLabel && grammar_matcher_->Find(dictation_lexiconfree_label_)) {
      VLOG(1) << "    found dictation_lexiconfree_label_";
      PrefixScore new_next_prefix_score = next_prefix_score;
      new_next_prefix_score.is_in_grammar = false;
      follow_arc(new_next_prefix_score, grammar_matcher_->Value());
      CHECK((grammar_matcher_->Next(), grammar_matcher_->Done()));  // Assume deterministic FST.
    }

    grammar_matcher_->Find(word_id);
    for (; !grammar_matcher_->Done(); grammar_matcher_->Next()) {
      PrefixScore new_next_prefix_score = next_prefix_score;
      follow_arc(new_next_prefix_score, grammar_matcher_->Value());
    }
  };  // check_complete_word()

  // Generate debugging info.
  auto current_prefix_words = IdsToString(current_prefix);
  auto all_words = IdsToString(current_prefix, id);
  if (false && id != -1) {
    std::vector<std::string> word_pieces;
    std::transform(current_prefix.begin(), current_prefix.end(), std::back_inserter(word_pieces), [this](int id) { return unit_table_->Find(id); });
    word_pieces.emplace_back(unit_table_->Find(id));
    auto word = JoinString(" | ", word_pieces);
    static std::set<std::pair<std::vector<int>, int>> seen_combinations;
    if (seen_combinations.find(std::make_pair(current_prefix, id)) == seen_combinations.end()) {
      seen_combinations.insert(std::make_pair(current_prefix, id));
      VLOG(3) << "ComputeFstScores: new: t" << abs_time_step_ << ": " << word;
    } else {
      VLOG(3) << "ComputeFstScores: old: t" << abs_time_step_ << ": " << word;
    }
  }
  if (false && all_words == target_string_.substr(0, all_words.size()) && all_words.size() >= 47) {
    VLOG(2) << "ComputeFstScores: target: " << all_words;
  }
  // dbg(all_words + "  " + std::to_string(current_prefix_score.grammar_fst_state));

  // Handle initialization.
  // if (current_prefix.empty()) {
  // }

  // Handle epsilon transitions at beginning of this-time-step processing, rather than at the end of the previous time step.
  if (true) {
    auto grammar_fst_state = GetPrefixScoreGrammarFstStateOrFstStart(current_prefix_score, *grammar_fst_);
    grammar_matcher_->SetState(grammar_fst_state);
    // Check if we have any epsilon transitions.
    if (grammar_matcher_->Find(0)) {
      // Recurse, following epsilon transitions.
      FollowEpsilons(*grammar_matcher_, current_prefix_score, 0, false,
        [&](PrefixScore& new_current_prefix_score, float weight) {
          ComputeFstScores(current_prefix, new_current_prefix_score, id, next_prefix_score, final, add_new_next_prefix_score);
        }
      );
    }
    // Fall through and continue, to handle this state.
  }

  // Handle free dictation, outside the grammar.
  if (!current_prefix_score.is_in_grammar) {
    auto grammar_fst_state = GetPrefixScoreGrammarFstStateOrFstStart(next_prefix_score, *grammar_fst_);
    grammar_matcher_->SetState(grammar_fst_state);
    CHECK(nonterm_end_label_ != fst::kNoLabel && grammar_matcher_->Find(nonterm_end_label_));  // We must have at least one way to end the dictation, and possibly multiple.
    auto weight = grammar_matcher_->Value().weight.Value();
    auto nextstate = grammar_matcher_->Value().nextstate;
    CHECK((grammar_matcher_->Next(), grammar_matcher_->Done()));  // Assume deterministic FST.
    CHECK_EQ(weight, 0);  // We don't support weights on the dictation-end arc.

    // Recurse, assuming we ended the dictation immediately before receiving this word, i.e. we followed the nonterm_end_label_ arc.
    auto new_current_prefix_score = current_prefix_score;
    new_current_prefix_score.grammar_fst_state = nextstate;
    new_current_prefix_score.is_in_grammar = true;
    auto new_next_prefix_score = next_prefix_score;
    new_next_prefix_score.grammar_fst_state = nextstate;
    new_next_prefix_score.is_in_grammar = true;
    ComputeFstScores(current_prefix, new_current_prefix_score, id, new_next_prefix_score, final, add_new_next_prefix_score);

    // Finally, just absorb the new word, without ending the dictation. But we can't end the utterance (final) while in dictation without ending it first.
    if (final) {
      next_prefix_score.grammar_fst_state = nextstate;
      next_prefix_score.is_in_grammar = true;
    }
    add_new_next_prefix_score(next_prefix_score, -opts_.dictation_wordpiece_insertion_penalty);
    return;
  }

  // Handle either: partial word prefixes, or only complete words.
  if (!opts_.process_partial_word_prefixes) {
    if (current_prefix.empty() || (!final && !IdIsStartOfWord(id))) {
      VLOG(3) << "    return: prefix empty or not start of word";
      return add_new_next_prefix_score(next_prefix_score, FullLikelihood);
    }

    // Build the previously-completed word.
    auto start_of_word_revit = std::find_if(current_prefix.rbegin(), current_prefix.rend(), [this](int id) { return IdIsStartOfWord(id); });
    auto start_of_word = start_of_word_revit == current_prefix.rend() ? current_prefix.begin() : start_of_word_revit.base() - 1;
    std::vector<std::string> word_pieces;
    std::transform(start_of_word, current_prefix.end(), std::back_inserter(word_pieces), [this](int id) { return unit_table_->Find(id); });
    auto word = JoinString("", word_pieces);
    if (WordIsStartOfWord(word)) {
      word = word.substr(3);
    }
    auto word_id = word_table_->Find(word);
    if (word_id == fst::SymbolTable::kNoSymbol) {
      VLOG(3) << "    return: not in word_table_";
      return add_new_next_prefix_score(next_prefix_score, NoLikelihood);
    }

    check_complete_word(next_prefix_score, word_id, add_new_next_prefix_score);
    return;

  } else {  // (opts_.process_partial_word_prefixes)
    float final_weight = FullLikelihood;
    auto dictionary_fst_state = current_prefix_score.dictionary_fst_state != fst::kNoStateId ? current_prefix_score.dictionary_fst_state : dictionary_trie_fst_->Start();
    auto prefix_word_id = current_prefix_score.prefix_word_id;

    // Check if we are starting a new word, thus guaranteeing that the tail of the prefix is now a complete word.
    if (!current_prefix.empty() && IdIsStartOfWord(id)) {
      // We have now completed the previous prefix word. We must have ended the dictionary in a final state. Then we check the completed word.
      CHECK_NE(dictionary_fst_state, dictionary_trie_fst_->Start());
      auto word_id = current_prefix_score.prefix_word_id;
      auto state_is_final = FindFinalState<fst::StdArc>(*dictionary_trie_matcher_, dictionary_fst_state, &dictionary_fst_state, &word_id);
      if (!state_is_final) {
        VLOG(3) << "    return: completed word not in dictionary (though it was a valid prefix)";
        return add_new_next_prefix_score(next_prefix_score, NoLikelihood);
      }
      auto state_final_weight = dictionary_trie_fst_->Final(dictionary_fst_state).Value();
      // final_weight += -state_final_weight + check_complete_word(next_prefix_score, word_id);  // FIXME!!!!!

      // Start over fresh for a new word.
      dictionary_fst_state = dictionary_trie_fst_->Start();
      prefix_word_id = fst::kNoLabel;
    }

    // Check whether the current partial word prefix could be a word in the dictionary.
    auto& matcher = *dictionary_trie_matcher_;
    matcher.SetState(dictionary_fst_state);
    if (!matcher.Find(id)) {
      VLOG(3) << "    return: partial word prefix not in dictionary";
      CHECK(!CheckIfEpsilonPathToILabel<fst::StdArc>(matcher, dictionary_fst_state, id));
      return add_new_next_prefix_score(next_prefix_score, final_weight + NoLikelihood);
    }
    auto olabel = matcher.Value().olabel;
    auto nextstate = matcher.Value().nextstate;
    // Note: we ignore the weight.
    CHECK((matcher.Next(), matcher.Done()));  // Assume deterministic FST.

    if (olabel != 0) {
      CHECK_EQ(prefix_word_id, fst::kNoLabel);  // This should only be set if we are guaranteed not to change word predictions.
      next_prefix_score.prefix_word_id = olabel;
    } else {
      next_prefix_score.prefix_word_id = prefix_word_id;
    }
    next_prefix_score.dictionary_fst_state = nextstate;
    if (opts_.prune_directly_impossible_prefixes || opts_.prune_indirectly_impossible_prefixes) {
      bool nextstate_is_final = false;
      if (opts_.prune_indirectly_impossible_prefixes) {
        nextstate_is_final = FindFinalState<fst::StdArc>(*dictionary_trie_matcher_, nextstate, &nextstate, nullptr);
      } else if (opts_.prune_directly_impossible_prefixes) {
        nextstate_is_final = (dictionary_trie_fst_->Final(nextstate).Value() != Weight::Zero());
      }
      if (!nextstate_is_final) {
        VLOG(3) << "    return: partial word prefix okay so far, but not a guaranteed-completed word yet";
        return add_new_next_prefix_score(next_prefix_score, final_weight + FullLikelihood);
      }
    }

    // Now guaranteed to be a word in the word table.
    // But adding more units after this one may not be a word, so we must wait for a space character before officially moving forward???
    CHECK_EQ(dictionary_trie_fst_->NumArcs(nextstate), 0);  // We shouldn't be able to continue after a final state???
    return add_new_next_prefix_score(next_prefix_score, final_weight + FullLikelihood);
  }

  LOG(FATAL) << "We should never reach here.";
}

bool CtcPrefixWfstBeamSearch::WordIsStartOfWord(const std::string& word) {
  return word.size() >= space_symbol_.size() && word.substr(0, 3) == space_symbol_;
}

bool CtcPrefixWfstBeamSearch::IdIsStartOfWord(int id) {
  auto word_piece = unit_table_->Find(id);
  return WordIsStartOfWord(word_piece);
}

std::string CtcPrefixWfstBeamSearch::IdsToString(const std::vector<int> ids, int extra_id, int max_length) {
  std::vector<std::string> word_pieces;
  std::transform(ids.begin(), ids.end(), std::back_inserter(word_pieces), [this](int id) { return unit_table_->Find(id); });
  if (extra_id != -1) {
    word_pieces.emplace_back(unit_table_->Find(extra_id));
  }
  auto word = JoinString("", word_pieces);
  if (max_length != -1 && word.size() > max_length) {
    word = word.substr(word.size() - max_length);
  }
  return word;
}

// Build dictionary trie of the words, in terms of characters.
std::unique_ptr<fst::StdVectorFst> CtcPrefixWfstBeamSearch::BuildCharacterDictionaryTrie(const fst::SymbolTable& word_table) {
  fst::StdVectorFst dictionary;
  auto start = dictionary.AddState();
  CHECK_EQ(start, 0);
  dictionary.SetStart(start);

  fst::SymbolTable character_table;
  character_table.AddSymbol("<eps>", 0);
  auto space_id = character_table.AddSymbol(space_symbol_, 1);
  start = dictionary.AddState();
  // Optionally accept the space character to start every word.
  dictionary.AddArc(dictionary.Start(), fst::StdArc(space_id, 0, 0, start));
  dictionary.AddArc(dictionary.Start(), fst::StdArc(0, 0, 0, start));

  for (fst::SymbolTableIterator it(word_table); !it.Done(); it.Next()) {
    const auto& word = it.Symbol();
    std::vector<std::string> chars;
    SplitUTF8StringToChars(word, &chars);
    if (!std::all_of(chars.begin(), chars.end(), CheckEnglishChar)) continue;

    auto current_state = start;
    auto word_id = it.Value();
    for (const auto& c : chars) {
      auto next_state = dictionary.AddState();
      auto character_id = character_table.Find(c);
      if (character_id == fst::SymbolTable::kNoSymbol) {
        character_id = character_table.AddSymbol(c);
      }
      dictionary.AddArc(current_state, fst::StdArc(character_id, word_id, 0, next_state));
      word_id = 0;  // Epsilon for all but the first arc.
      current_state = next_state;
    }
    dictionary.SetFinal(current_state, fst::StdArc::Weight::One());
  }

  auto final_dictionary = std::make_unique<fst::StdVectorFst>();
  // fst::RmEpsilon(&dictionary);
  // fst::Determinize(dictionary, final_dictionary.get());
  fst::ArcSort(&dictionary, fst::ILabelCompare<fst::StdArc>());
  fst::DeterminizeStar(dictionary, final_dictionary.get());  // Standard Determinize messes up output symbols.
  fst::Minimize(final_dictionary.get());
  // fst::MinimizeEncoded(final_dictionary.get());
  fst::ArcSort(final_dictionary.get(), fst::ILabelCompare<fst::StdArc>());
  final_dictionary->SetInputSymbols(&character_table);
  return final_dictionary;
}

// Build dictionary trie of the words, in terms of units.
void CtcPrefixWfstBeamSearch::BuildUnitDictionaryTrie() {
  auto character_dictionary = BuildCharacterDictionaryTrie(*word_table_);
  auto character_table = character_dictionary->InputSymbols();
  CHECK_NOTNULL(character_table);

  fst::StdVectorFst unit_to_chars_fst;
  auto start = unit_to_chars_fst.AddState();
  CHECK_EQ(start, 0);
  unit_to_chars_fst.SetStart(start);

  for (fst::SymbolTableIterator it(*unit_table_); !it.Done(); it.Next()) {
    const auto& unit = it.Symbol();
    if (unit == "<blank>") CHECK_EQ(it.Value(), 0);  // ComputeFstScore() assumes it will not receive the blank symbol, so it can use 0 for epsilon.
    if (unit.substr(0, 1) == "<") continue;
    std::vector<std::string> unit_chars;
    SplitUTF8StringToChars(unit, &unit_chars);

    auto current_state = start;
    auto unit_id = it.Value();
    auto broke = true;
    for (const auto& c : unit_chars) {
      auto character_id = character_table->Find(c);
      if (character_id == fst::SymbolTable::kNoSymbol) {
        LOG(WARNING) << "Character " << c << " not found in character table. Skipping unit " << unit;
        broke = true;
        break;
      }

      auto next_state = unit_to_chars_fst.AddState();
      unit_to_chars_fst.AddArc(current_state, fst::StdArc(unit_id, character_id, 0, next_state));
      unit_id = 0;  // Epsilon for all but the first arc.
      current_state = next_state;
      broke = false;
    }

    if (!broke) {
      unit_to_chars_fst.SetFinal(current_state, fst::StdArc::Weight::One());
      unit_to_chars_fst.AddArc(current_state, fst::StdArc(0, 0, 0, start));  // Loop back to start, to accept a series of units.
    }
  }

  // unit_to_chars_fst.Write("/tmp/dictionary_trie_raw_unit_to_chars_fst.fst");
  // character_dictionary->Write("/tmp/dictionary_trie_raw_character_dictionary.fst");
  // character_table->WriteText("/tmp/dictionary_trie_raw_character_table.syms.txt");
  fst::ArcSort(&unit_to_chars_fst, fst::OLabelCompare<fst::StdArc>());
  fst::StdVectorFst composed_fst;
  fst::Compose(unit_to_chars_fst, *character_dictionary, &composed_fst);
  // composed_fst.Write("/tmp/dictionary_trie_raw.fst");
  fst::Connect(&composed_fst);

  auto final_dictionary = std::make_unique<fst::StdVectorFst>();
  // fst::RmEpsilon(&composed_fst);
  // fst::Determinize(composed_fst, final_dictionary.get());
  fst::ArcSort(&composed_fst, fst::ILabelCompare<fst::StdArc>());
  fst::DeterminizeStar(composed_fst, final_dictionary.get());
  fst::Minimize(final_dictionary.get());
  // fst::MinimizeEncoded(final_dictionary.get());
  fst::RmEpsilon(final_dictionary.get());
  fst::ArcSort(final_dictionary.get(), fst::ILabelCompare<fst::StdArc>());
  // final_dictionary->Write("/tmp/dictionary_trie.fst");

  dictionary_trie_fst_ = std::move(final_dictionary);
  dictionary_trie_matcher_ = std::make_unique<CtcPrefixWfstBeamSearch::Matcher>(*dictionary_trie_fst_, fst::MATCH_INPUT);
}

// TODO:
// - update implementation for partial-word-prefixes to work with latest changes.
// - lexicon-bonus mode: don't restrict to only the lexicon, but give bonus to lexicon words.

}  // namespace wenet
