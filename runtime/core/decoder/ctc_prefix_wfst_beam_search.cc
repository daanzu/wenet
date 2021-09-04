// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
// Author: daanzu@gmail.com (David Zurow)

#include "decoder/ctc_prefix_wfst_beam_search.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "utils/log.h"
#include "utils/string.h"

namespace wenet {

using PrefixScore = CtcPrefixWfstBeamSearch::PrefixScore;
using PrefixHash = CtcPrefixWfstBeamSearch::PrefixHash;

CtcPrefixWfstBeamSearch::CtcPrefixWfstBeamSearch(std::shared_ptr<fst::StdFst> fst, std::shared_ptr<fst::SymbolTable> word_table, std::shared_ptr<fst::SymbolTable> unit_table, const CtcPrefixWfstBeamSearchOptions& opts)
    : opts_(opts), fst_(fst), matcher_(*fst_, fst::MATCH_INPUT), word_table_(word_table), unit_table_(unit_table) {
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
  cur_hyps_[empty] = prefix_score;
}

static bool PrefixScoreCompare(
    const std::pair<std::vector<int>, PrefixScore>& a,
    const std::pair<std::vector<int>, PrefixScore>& b) {
  return a.second.score() > b.second.score();
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in it.
void CtcPrefixWfstBeamSearch::Search(const torch::Tensor& logp) {
  CHECK_EQ(logp.dtype(), torch::kFloat);
  CHECK_EQ(logp.dim(), 2);
  for (int t = 0; t < logp.size(0); ++t, ++abs_time_step_) {
    torch::Tensor logp_t = logp[t];
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
    // 1. First beam prune, only select topk candidates
    std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
    Tensor topk_score = std::get<0>(topk);
    Tensor topk_index = std::get<1>(topk);

    // 2. Token passing
    for (int i = 0; i < topk_index.size(0); ++i) {
      int id = topk_index[i].item<int>();
      auto prob = topk_score[i].item<float>();
      for (const auto& it : cur_hyps_) {
        const std::vector<int>& prefix = it.first;
        const PrefixScore& prefix_score = it.second;

        // If prefix doesn't exist in next_hyps, next_hyps[prefix] will insert
        // PrefixScore(-inf, -inf) by default, since the default constructor
        // of PrefixScore will set fields s(blank ending score) and
        // ns(none blank ending score) to -inf, respectively.

        if (id == opts_.blank) {
          // Case 0: *a + ε => *a
          // If we propose a blank, the prefix doesn't change. Only the probability of ending in blank gets updated, and the previous character could be either blank or nonblank.
          PrefixScore& next_score = next_hyps[prefix];
          next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
          next_score.v_s = prefix_score.viterbi_score() + prob;
          next_score.times_s = prefix_score.times();

        } else if (!prefix.empty() && id == prefix.back()) {
          // Case 1: *a + a => *a
          // If we propose a repeat nonblank, after a nonblank, the prefix doesn't change. Only the probability of ending in nonblank gets updated, and we know the previous character was nonblank.
          PrefixScore& next_score1 = next_hyps[prefix];
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
          // If we propose a repeat nonblank, after a blank, the prefix does change. Only the probability of ending in nonblank gets updated, and we know the previous character was blank.
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score2 = next_hyps[new_prefix];
          auto fst_score = GetFstScore(prefix, prefix_score, id, next_score2);
          next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob + fst_score);
          if (next_score2.v_ns < prefix_score.v_s + prob) {
            next_score2.v_ns = prefix_score.v_s + prob;
            next_score2.cur_token_prob = prob;
            next_score2.times_ns = prefix_score.times_s;
            next_score2.times_ns.emplace_back(abs_time_step_);
          }

        } else {
          // Case 3: *a + b => *ab, *aε + b => *ab
          // If we propose a non-repeat nonblank, the prefix must change. Only the probability of ending in nonblank gets updated, and the previous character could be either blank or nonblank.
          std::vector<int> new_prefix(prefix);
          new_prefix.emplace_back(id);
          PrefixScore& next_score = next_hyps[new_prefix];
          auto fst_score = GetFstScore(prefix, prefix_score, id, next_score);
          next_score.ns = LogAdd(next_score.ns, prefix_score.score() + prob + fst_score);
          if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
            next_score.v_ns = prefix_score.viterbi_score() + prob;
            next_score.cur_token_prob = prob;
            next_score.times_ns = prefix_score.times();
            next_score.times_ns.emplace_back(abs_time_step_);
          }
        }
      }
    }

    // 3. Second beam prune, only keep top n best paths
    std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
                                                              next_hyps.end());
    int second_beam_size =
        std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
    std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
                     PrefixScoreCompare);
    arr.resize(second_beam_size);
    std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

    // 4. Update cur_hyps_ and get new result
    cur_hyps_.clear();
    hypotheses_.clear();
    likelihood_.clear();
    viterbi_likelihood_.clear();
    times_.clear();
    for (auto& item : arr) {
      cur_hyps_[item.first] = item.second;
      hypotheses_.emplace_back(std::move(item.first));
      likelihood_.emplace_back(item.second.score());
      viterbi_likelihood_.emplace_back(item.second.viterbi_score());
      times_.emplace_back(item.second.times());
    }
  }
}

float CtcPrefixWfstBeamSearch::GetFstScore(const std::vector<int>& current_prefix, const PrefixScore& current_prefix_score, int id, PrefixScore& next_prefix_score) {
  if (current_prefix.empty()) {
    next_prefix_score.fst_state = current_prefix_score.fst_state;
    return 0.0f;
  }
  if (!IdIsStartOfWord(id)) {
    next_prefix_score.fst_state = current_prefix_score.fst_state;
    return 0.0f;
  }

  // Build the just-completed word.
  auto start_of_word = std::find_if(current_prefix.rbegin(), current_prefix.rend(), [this](int id) { return IdIsStartOfWord(id); }).base() - 1;
  // auto word_piece_ids = std::vector<int>(start_of_word, current_prefix.end());
  std::vector<int> word_piece_ids;
  std::copy_if(start_of_word, current_prefix.end(), std::back_inserter(word_piece_ids), [this](int id) { return (id != opts_.blank); });
  std::vector<std::string> word_pieces;
  std::transform(word_piece_ids.begin(), word_piece_ids.end(), std::back_inserter(word_pieces), [this](int id) { return unit_table_->Find(id); });
  auto word = JoinString("", word_pieces);
  // CHECK_EQ(word.substr(0, 3), space_symbol_);
  if (word.substr(0, 3) == space_symbol_) {
    word = word.substr(3);
  } else {
    // word = word.substr(0);
  }
  auto word_id = word_table_->Find(word);
  if (word_id == fst::SymbolTable::kNoSymbol) {
    next_prefix_score.fst_state = current_prefix_score.fst_state;
    return kFloatMax;
    return 0.0f;
  }

  static std::set<int> seen_word_ids;
  if (seen_word_ids.find(word_id) == seen_word_ids.end()) {
    seen_word_ids.insert(word_id);
  }
  matcher_.SetState(current_prefix_score.fst_state != fst::kNoStateId ? current_prefix_score.fst_state : fst_->Start());
  auto found = matcher_.Find(word_id);
  if (!found) {
    next_prefix_score.fst_state = current_prefix_score.fst_state;
    return kFloatMax;
    return 0.0f;
  }

  int num_arcs = 1;
  auto weight = matcher_.Value().weight.Value();
  auto nextstate = matcher_.Value().nextstate;

  for (matcher_.Next(); !matcher_.Done(); matcher_.Next()) {
    ++num_arcs;
    auto weight = matcher_.Value().weight.Value();
    auto nextstate = matcher_.Value().nextstate;
    LOG(WARNING) << "GetFstScore num_arcs>1: #" << num_arcs << " weight=" << weight << " nextstate=" << nextstate;
  }
  // if (num_arcs > 1) {
  //   LOG(WARNING) << "GetFstScore num_arcs>1";
  // }

  next_prefix_score.fst_state = nextstate;
  return -weight;
}

bool CtcPrefixWfstBeamSearch::IdIsStartOfWord(int id) {
  auto word_piece = unit_table_->Find(id);
  return word_piece.size() > space_symbol_.size()
    && std::equal(space_symbol_.begin(), space_symbol_.end(), word_piece.begin());
}

}  // namespace wenet
