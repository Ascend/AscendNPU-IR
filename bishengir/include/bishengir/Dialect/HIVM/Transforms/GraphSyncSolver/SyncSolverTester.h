//===------------- SyncSolverTester.h ---- Graph Sync Solver --------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERTESTER_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERTESTER_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <climits>
#include <deque>
#include <optional>
#include <random>

namespace mlir::hivm::syncsolver {

// NOLINTBEGIN
constexpr int max_depth = 4;
constexpr int loop_unrolling_num = 3;
constexpr int read_write_vals_max_num = 2;
constexpr int function_blocks_max_num = 5;

constexpr int true_branch_prob_a = 1;
constexpr int true_branch_prob_b = 2;
constexpr int dead_loop_prob_a = 1;
constexpr int dead_loop_prob_b = 10;
constexpr int skip_function_block_prob_a = 1;
constexpr int skip_function_block_prob_b = 10;

constexpr int multi_function_blocks_prob_a = 1;
constexpr int multi_function_blocks_prob_b = 10;
constexpr int scope_in_prob_a = 1;
constexpr int scope_in_prob_b = 5;
constexpr int unlikely_cond_prob_a = 1;
constexpr int unlikely_cond_prob_b = 10;
constexpr int scope_for_loop_prob_a = 1;
constexpr int scope_for_loop_prob_b = 3;
constexpr int scope_while_loop_prob_a = 1;
constexpr int scope_while_loop_prob_b = 5;
constexpr int scope_cond_prob_a = 1;
constexpr int scope_cond_prob_b = 5;
constexpr int scope_out_prob_a = 1;
constexpr int scope_out_prob_b = 3;
// NOLINTEND

class SyncTester {

public:
private:
  int64_t initSeed{42};
  int numOperations{20};
  int numPointers{10};
  bool enableMultiBuffer{false};
  SyncMode syncMode{SyncMode::TEST_INTRA_CORE_MODE};
  std::unique_ptr<std::mt19937> randGenerator;

  int idx{0};
  llvm::DenseMap<CorePipeInfo,
                 std::deque<std::pair<std::pair<int, int>,
                                      std::pair<const OperationBase *, int>>>>
      pipelineQue;

public:
  SyncTester(int numOperations, int numPointers, bool enableMultiBuffer,
             bool enableCrossCoreMode, std::optional<int64_t> seed = {})
      : numOperations(numOperations), numPointers(numPointers),
        enableMultiBuffer(enableMultiBuffer),
        syncMode(enableCrossCoreMode ? SyncMode::TEST_CROSS_CORE_MODE
                                     : SyncMode::TEST_INTRA_CORE_MODE) {
    this->initSeed =
        seed.has_value()
            ? seed.value()
            : static_cast<unsigned>(
                  std::chrono::steady_clock::now().time_since_epoch().count());
    randGenerator = std::make_unique<std::mt19937>(this->initSeed);
  }

  // Generate a full solver test, run the solver and simulate the result.
  llvm::LogicalResult test();

  static size_t getOptionsNum() { return 6; }

  // Helper to toggle running as test mode (external use).
  static void runTestMode(const SmallVector<int64_t> &options);

private:
  void reset() {
    idx = 0;
    pipelineQue.clear();
  }

  int getRand() { return getRand(0, INT_MAX); }

  int getRand(int minNum, int maxNum) {
    return std::uniform_int_distribution<int>(minNum, maxNum)(*randGenerator);
  }

  std::vector<int> getNDifferentRandNums(int n, int mod) {
    assert(n <= mod);
    std::set<int> ret;
    while (static_cast<int>(ret.size()) < n) {
      int rnd = getRand();
      int val = rnd % (mod - static_cast<int>(ret.size()));
      for (auto e : ret) {
        if (e <= val) {
          val++;
        }
      }
      ret.insert(val);
    }
    return std::vector<int>(ret.begin(), ret.end());
  }

  bool isTrueWithProbability(int a, int b) {
    // p = a/b
    assert(a <= b);
    int rnd = getRand();
    return (rnd % b) < a;
  }

  // Produce a randomly generated IR (OperationBase tree) used by the tester.
  std::unique_ptr<OperationBase> getGeneratedRandomTest();

  // Recursively create a random region body (scopes/loops/cond/rw ops).
  void generateRandTest(Scope *scopeOp, const std::vector<int> &pointerOps,
                        const llvm::SmallVector<hivm::PIPE> &pipesVec,
                        int &remOpNum, int depth);

  // Walk the generated IR and place operations into pipeline queues for
  // simulation.
  void fillPipelines(const OperationBase *op, int loopCnt = 0, int loopIdx = 0);

  // Simulate execution of queued pipeline operations, checking for memory/sync
  // conflicts.
  llvm::LogicalResult runSimulation(int runId, bool debugPrint = false);
};

} // namespace mlir::hivm::syncsolver
#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERTESTER_H
