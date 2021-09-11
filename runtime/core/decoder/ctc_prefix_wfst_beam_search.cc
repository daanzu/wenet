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

static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
constexpr float SuperNoLikelihood = -std::numeric_limits<float>::infinity();
constexpr float NoLikelihood = -std::numeric_limits<float>::max();
constexpr float FullLikelihood = 0.0f;

CtcPrefixWfstBeamSearch::CtcPrefixWfstBeamSearch(std::shared_ptr<fst::StdFst> fst, std::shared_ptr<fst::SymbolTable> word_table, std::shared_ptr<fst::SymbolTable> unit_table, const CtcPrefixWfstBeamSearchOptions& opts)
    : opts_(opts), fst_(fst), matcher_(*fst_, fst::MATCH_INPUT), word_table_(word_table), unit_table_(unit_table),
      dictation_lexiconfree_state_(word_table->Find("#NONTERM:DICTATION_LEXICONFREE")) {
  BuildUnitDictionaryTrie();
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
  bool verbose = false;
  for (int t = 0; t < logp.size(0); ++t, ++abs_time_step_) {
    torch::Tensor logp_t = logp[t];
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
    // 1. First beam prune, only select topk candidates
    std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
    Tensor topk_score = std::get<0>(topk);
    Tensor topk_index = std::get<1>(topk);

    if (verbose) {
      for (int i = 0; i < topk_index.size(0); ++i) {
        VLOG(1) << "t" << abs_time_step_ << ": " << unit_table_->Find(topk_index[i].item<int>()) << " " << topk_index[i].item<int>() << " " << topk_score[i].item<float>();
      }
    }

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

        static std::string target_string = "▁I'D▁LIKE▁TO▁SHARE▁WITH▁YOU▁A▁DISCOVERY▁THAT▁I▁MADE▁A▁FEW▁MONTHS▁AGO▁WHILE▁WRITING▁AN▁ARTICLE▁FOR▁ITALIAN▁WIRED▁I▁ALWAYS▁KEEP▁MY▁THESAURUS▁HANDY▁WHENEVER▁I'M▁WRITING▁ANYTHING▁BUT";
        std::vector<int> new_prefix(prefix);
        if (id != opts_.blank) new_prefix.emplace_back(id);
        std::string new_prefix_string = IdsToString(new_prefix);
        if (new_prefix_string == target_string.substr(0, new_prefix_string.size())) {
          VLOG(2) << "new_prefix_string: " << new_prefix_string;
          if (!verbose && new_prefix_string.size() >= 150) {
            // VLOG(1) << "!!!!";
            verbose = true;
          }
        }

        if (id == opts_.blank) {
          // Case 0: *a + ε => *a
          // If we propose a blank, the prefix doesn't change. Only the probability of ending in blank gets updated, and the previous character could be either blank or nonblank.
          PrefixScore& next_score = GetNextHyp(next_hyps, prefix, prefix_score);
          // next_score.set_fst_state(prefix_score.fst_state);
          next_score.update_at_time(abs_time_step_);
          next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
          next_score.v_s = prefix_score.viterbi_score() + prob;
          next_score.times_s = prefix_score.times();

        } else if (!prefix.empty() && id == prefix.back()) {
          // Case 1: *a + a => *a
          // If we propose a repeat nonblank, after a nonblank, the prefix doesn't change. Only the probability of ending in nonblank gets updated, and we know the previous character was nonblank.
          PrefixScore& next_score1 = GetNextHyp(next_hyps, prefix, prefix_score);
          // next_score1.set_fst_state(prefix_score.fst_state);
          next_score1.update_at_time(abs_time_step_);
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
          PrefixScore& next_score2 = GetNextHyp(next_hyps, new_prefix, prefix_score);
          next_score2.update_at_time(abs_time_step_);
          auto fst_score = GetFstScore(prefix, prefix_score, id, next_score2);
          if (opts_.strict && fst_score == NoLikelihood) {
            next_hyps.erase(new_prefix);
            continue;
          }
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
          PrefixScore& next_score = GetNextHyp(next_hyps, new_prefix, prefix_score);
          next_score.update_at_time(abs_time_step_);
          auto fst_score = GetFstScore(prefix, prefix_score, id, next_score);
          if (opts_.strict && fst_score == NoLikelihood) {
            next_hyps.erase(new_prefix);
            continue;
          }
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
      VLOG(1) << "t" << abs_time_step_ << ": " << IdsToString(item.first, -1, 80) << " = " << item.second.score() << "    fst_state=" << item.second.fst_state;
              // << "    update_times=" << item.second.update_times.size();
      cur_hyps_[item.first] = item.second;
      hypotheses_.emplace_back(std::move(item.first));
      likelihood_.emplace_back(item.second.score());
      viterbi_likelihood_.emplace_back(item.second.viterbi_score());
      times_.emplace_back(item.second.times());
    }
    VLOG(1) << "";
  }
}

PrefixScore& CtcPrefixWfstBeamSearch::GetNextHyp(std::unordered_map<std::vector<int>, PrefixScore, PrefixHash>& next_hyps, const std::vector<int>& prefix, const PrefixScore& current_score) {
  PrefixScore& next_score = next_hyps[prefix];
  if (next_score.fst_state == fst::kNoStateId) {
    next_score.set_fst_state(current_score.fst_state);
  }
  return next_score;
}

float CtcPrefixWfstBeamSearch::GetFstScore(const std::vector<int>& current_prefix, const PrefixScore& current_prefix_score, int id, PrefixScore& next_prefix_score) {
  if (!current_prefix_score.is_in_grammar) {
    return FullLikelihood;
  }

  auto current_prefix_words = IdsToString(current_prefix);
  auto all_words = IdsToString(current_prefix, id);
  if (true) {
    std::vector<std::string> word_pieces;
    std::transform(current_prefix.begin(), current_prefix.end(), std::back_inserter(word_pieces), [this](int id) { return unit_table_->Find(id); });
    word_pieces.emplace_back(unit_table_->Find(id));
    auto word = JoinString(" | ", word_pieces);
    static std::set<std::pair<std::vector<int>, int>> seen_combinations;
    if (seen_combinations.find(std::make_pair(current_prefix, id)) == seen_combinations.end()) {
      seen_combinations.insert(std::make_pair(current_prefix, id));
      VLOG(2) << "GetFstScore: new: t" << abs_time_step_ << ": " << word;
    } else {
      VLOG(2) << "GetFstScore: old: t" << abs_time_step_ << ": " << word;
    }
  }

  int word_id = fst::kNoLabel;
  if (true) {
    if (!current_prefix.empty() && IdIsStartOfWord(id)) {
      // We have now completed the previous prefix word.
      CHECK_NE(dictionary_trie_fst_->Final(current_prefix_score.dictionary_fst_state), fst::StdArc::Weight::Zero());
      word_id = current_prefix_score.prefix_word_id;
      // Check it!!!
    }

    // Check whether the current partial word prefix could be a word in the dictionary.
    auto& matcher = *dictionary_trie_matcher_;
    auto dictionary_fst_state = current_prefix_score.dictionary_fst_state != fst::kNoStateId ? current_prefix_score.dictionary_fst_state : dictionary_trie_fst_->Start();
    matcher.SetState(dictionary_fst_state);
    if (!matcher.Find(id)) {
      VLOG(3) << "    return: partial word prefix not in dictionary";
      return NoLikelihood;
    }
    auto olabel = matcher.Value().olabel;
    auto nextstate = matcher.Value().nextstate;
    // auto nextstate_is_final = (dictionary_trie_fst_->Final(nextstate) != fst::StdArc::Weight::Zero());

    // Assume deterministic.
    matcher.Next();
    CHECK(matcher.Done());

    if (olabel != 0) {
      CHECK_EQ(current_prefix_score.prefix_word_id, fst::kNoLabel);  // This should only be set if we are guaranteed not to change word predictions.
      next_prefix_score.prefix_word_id = olabel;
    } else {
      next_prefix_score.prefix_word_id = current_prefix_score.prefix_word_id;
    }
    // CHECK(olabel == 0 || current_prefix_score.prefix_word_id == fst::kNoLabel);  // This should only be set if we are guaranteed not to change word predictions.
    // next_prefix_score.prefix_word_id = (olabel == 0) ? current_prefix_score.prefix_word_id : olabel;
    next_prefix_score.dictionary_fst_state = nextstate;
    // if (!nextstate_is_final) {
    //   VLOG(3) << "    return: partial word prefix okay so far, but not a full word yet";
    //   return FullLikelihood;
    // }
    // Now guaranteed to be a word in the word table. But adding more units after this one may not be a word, so we must wait for a space character before officially moving forward.
    VLOG(3) << "    return: partial word prefix okay so far, but not a guaranteed-completed word yet";
    return FullLikelihood;

  } else {
    if (current_prefix.empty() || !IdIsStartOfWord(id)) {
      VLOG(3) << "    return: prefix empty or not start of word";
      return FullLikelihood;
    }

    // Build the previously-completed word.
    auto start_of_word_revit = std::find_if(current_prefix.rbegin(), current_prefix.rend(), [this](int id) { return IdIsStartOfWord(id); });
    auto start_of_word = start_of_word_revit == current_prefix.rend() ? current_prefix.begin() : start_of_word_revit.base() - 1;
    // if (std::any_of(start_of_word, current_prefix.end(), [this](int id) { return (id == opts_.blank); })) {
    //   VLOG(0) << "    found blank " << (start_of_word == current_prefix.end());
    // }
    std::vector<std::string> word_pieces;
    std::transform(start_of_word, current_prefix.end(), std::back_inserter(word_pieces), [this](int id) { return unit_table_->Find(id); });
    auto word = JoinString("", word_pieces);
    if (word.substr(0, 3) == space_symbol_) {
      word = word.substr(3);
    }
    word_id = word_table_->Find(word);
    if (word_id == fst::SymbolTable::kNoSymbol) {
      VLOG(3) << "    return: not in word_table_";
      return NoLikelihood;
    }
  }

  // static std::set<int> seen_word_ids;
  // if (seen_word_ids.find(word_id) == seen_word_ids.end()) {
  //   seen_word_ids.insert(word_id);
  // }

  // Check whether the completed word fits in the grammar fst.
  CHECK_NE(word_id, fst::kNoLabel);
  auto fst_state = current_prefix_score.fst_state != fst::kNoStateId ? current_prefix_score.fst_state : fst_->Start();
  matcher_.SetState(fst_state);
  if (true && matcher_.Find(0)) {
    VLOG(0) << "    found epsilon";
  }
  if (true && matcher_.Find(dictation_lexiconfree_state_)) {
    VLOG(0) << "    found dictation_lexiconfree_state_";
    next_prefix_score.is_in_grammar = false;
  }
  else if (!matcher_.Find(word_id)) {
    VLOG(3) << "    return: not found at fst_state " << fst_state;
    return NoLikelihood;
  }

  auto weight = matcher_.Value().weight.Value();
  auto nextstate = matcher_.Value().nextstate;

  for (matcher_.Next(); !matcher_.Done(); matcher_.Next()) {
    auto weight = matcher_.Value().weight.Value();
    auto nextstate = matcher_.Value().nextstate;
    LOG(FATAL) << "GetFstScore num_arcs>1: weight=" << weight << " nextstate=" << nextstate;
  }

  VLOG(1) << "    " << IdsToString(current_prefix, id) << " : fst_state " << next_prefix_score.fst_state << " -> " << nextstate;
  next_prefix_score.set_fst_state(nextstate);
  return -weight;
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
      // dictionary.AddArc(current_state, fst::StdArc(character_id, character_id, 0, next_state));
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
    if (unit == "<blank>") CHECK_EQ(it.Value(), 0);  // GetFstScore() assumes it will not receive the blank symbol, so it can use 0 for epsilon.
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
      // unit_to_chars_fst.AddArc(current_state, fst::StdArc(character_id, character_id, 0, next_state));
      unit_to_chars_fst.AddArc(current_state, fst::StdArc(unit_id, character_id, 0, next_state));
      unit_id = 0;  // Epsilon for all but the first arc.
      current_state = next_state;
      broke = false;
    }

    if (!broke) {
      unit_to_chars_fst.SetFinal(current_state, fst::StdArc::Weight::One());
      unit_to_chars_fst.AddArc(current_state, fst::StdArc(0, 0, 0, start));  // Loop back to start, to accept a series of units.
    }
    // if (it.Value() >= 100) break;
  }

  unit_to_chars_fst.Write("/home/daz/tmp/dictionary_trie_raw_unit_to_chars_fst.fst");
  character_dictionary->Write("/home/daz/tmp/dictionary_trie_raw_character_dictionary.fst");
  character_table->WriteText("/home/daz/tmp/dictionary_trie_raw_character_table.syms.txt");
  fst::ArcSort(&unit_to_chars_fst, fst::OLabelCompare<fst::StdArc>());
  fst::StdVectorFst composed_fst;
  fst::Compose(unit_to_chars_fst, *character_dictionary, &composed_fst);
  composed_fst.Write("/home/daz/tmp/dictionary_trie_raw.fst");
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
  final_dictionary->Write("/home/daz/tmp/dictionary_trie.fst");

  dictionary_trie_fst_ = std::move(final_dictionary);
  dictionary_trie_matcher_ = std::make_unique<fst::SortedMatcher<fst::StdFst>>(*dictionary_trie_fst_, fst::MATCH_INPUT);
}

// TODO:
// - partial match, trie
// - turn on/off free ctc
// - epsilon transitions
// - multiple ambiguous arcs

}  // namespace wenet
