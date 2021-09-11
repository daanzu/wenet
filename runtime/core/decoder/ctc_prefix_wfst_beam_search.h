// Copyright 2020 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
// Author: daanzu@gmail.com (David Zurow)

#ifndef DECODER_CTC_PREFIX_WFST_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_WFST_BEAM_SEARCH_H_

#include <unordered_map>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "decoder/search_interface.h"
#include "utils/utils.h"

#include "fst/fstlib.h"

namespace wenet {

using TorchModule = torch::jit::script::Module;
using Tensor = torch::Tensor;
using StateId = fst::StdFst::StateId;

struct CtcPrefixWfstBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
  bool strict = true;
};

// Represents everything for a single Prefix, which is a sequence of regularized CTC labels, and so can only grow monotonically.
struct WfstPrefixScore {
  float s = -kFloatMax;               // blank ending score
  float ns = -kFloatMax;              // none blank ending score
  float v_s = -kFloatMax;             // viterbi blank ending score
  float v_ns = -kFloatMax;            // viterbi none blank ending score
  float cur_token_prob = -kFloatMax;  // prob of current token
  std::vector<int> times_s;           // times of viterbi blank path
  std::vector<int> times_ns;          // times of viterbi none blank path

  StateId fst_state = fst::kNoStateId;
  StateId dictionary_fst_state = fst::kNoStateId;
  bool is_in_grammar = true;

  void set_fst_state(StateId state) {
    if (fst_state != fst::kNoStateId) {
      // LOG(FATAL) << "fst_state is already set";
      // if (fst_state != state) {
      if (fst_state > state) {
        LOG(FATAL) << "fst_state is already set to " << fst_state << " not " << state;
      }
    }
    fst_state = state;
  }

  std::vector<int> update_times;
  void update_at_time(int time) {
    update_times.push_back(time);
  }

  WfstPrefixScore() = default;
  float score() const { return LogAdd(s, ns); }
  float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
  const std::vector<int>& times() const {
    return v_s > v_ns ? times_s : times_ns;
  }
};

struct WfstPrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // here we use KB&DR hash code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
    }
    return hash_code;
  }
};

class CtcPrefixWfstBeamSearch : public SearchInterface {
 public:
  explicit CtcPrefixWfstBeamSearch(std::shared_ptr<fst::StdFst> fst, std::shared_ptr<fst::SymbolTable> word_table, std::shared_ptr<fst::SymbolTable> unit_table, const CtcPrefixWfstBeamSearchOptions& opts);

  using PrefixScore = WfstPrefixScore;
  using PrefixHash = WfstPrefixHash;

  void Search(const torch::Tensor& logp) override;
  void Reset() override;
  // CtcPrefixWfstBeamSearch do nothing at FinalizeSearch
  void FinalizeSearch() override {}
  SearchType Type() const override { return SearchType::kPrefixWfstBeamSearch; }

  const std::vector<std::vector<int>>& hypotheses() const {
    return hypotheses_;
  }
  const std::vector<float>& likelihood() const { return likelihood_; }
  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }
  const std::vector<std::vector<int>>& times() const { return times_; }
  // For CTC prefix beam search, both inputs and outputs are hypotheses_
  const std::vector<std::vector<int>>& Inputs() const override {
    return hypotheses_;
  }
  const std::vector<std::vector<int>>& Outputs() const override {
    return hypotheses_;
  }
  const std::vector<float>& Likelihood() const override { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const override { return times_; }

 private:
  int abs_time_step_ = 0;

  // N-best list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;

  // Map from prefix to its score
  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;

  const CtcPrefixWfstBeamSearchOptions& opts_;

  std::shared_ptr<fst::StdFst> fst_;
  fst::ExplicitMatcher<fst::SortedMatcher<fst::StdFst>> matcher_;
  std::shared_ptr<fst::SymbolTable> word_table_;
  std::shared_ptr<fst::SymbolTable> unit_table_;
  std::unique_ptr<fst::StdFst> dictionary_trie_fst_;
  std::unique_ptr<fst::SortedMatcher<fst::StdFst>> dictionary_trie_matcher_;

  const std::string space_symbol_ = kSpaceSymbol;
  const StateId dictation_lexiconfree_state_ = fst::kNoStateId;

  PrefixScore& GetNextHyp(std::unordered_map<std::vector<int>, PrefixScore, PrefixHash>& next_hyps, const std::vector<int>& prefix, const PrefixScore& current_score);
  float GetFstScore(const std::vector<int>& current_prefix, const PrefixScore& current_prefix_score, int id, PrefixScore& next_prefix_score);
  bool WordIsStartOfWord(const std::string& word);
  bool IdIsStartOfWord(int id);
  std::string IdsToString(const std::vector<int> ids, int extra_id = -1, int max_len = -1);
  void BuildUnitDictionaryTrie();

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(CtcPrefixWfstBeamSearch);
};

}  // namespace wenet

#endif  // DECODER_CTC_PREFIX_WFST_BEAM_SEARCH_H_
