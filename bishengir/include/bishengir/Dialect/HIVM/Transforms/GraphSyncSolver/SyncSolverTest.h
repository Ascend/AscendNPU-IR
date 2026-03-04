//===------------- SyncSolverTest.h ---- Graph Sync Solver ----------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERTEST_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERTEST_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/ADT/SmallVector.h"
#include <climits>

namespace mlir::hivm::syncsolver {

class SolverTest : public Solver {
public:
private:
  llvm::DenseMap<
      std::pair<syncsolver::RWOperation *, syncsolver::RWOperation *>,
      std::vector<std::pair<hivm::PIPE, hivm::PIPE>>>
      checkTestMemoryConflictsMem;

public:
  SolverTest() = delete;

  SolverTest(std::unique_ptr<IRTranslator> irTranslator)
      : Solver(std::move(irTranslator)) {}

private:
  // Alternative processing used for tests.
  void processConflict(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                       RWOperation *rwOp2, bool isUseless) override;

  // Helpers for test-mode event id num estimation.
  int64_t getTestEventIdNum(RWOperation *rwOp1, RWOperation *rwOp2);

  int64_t getTestEventIdNum(Occurrence *occ1, Occurrence *occ2,
                            hivm::PIPE setPipe, hivm::PIPE waitPipe);
  // Check if given eventIdNum can be used without RW conflicts.
  bool
  checkEventIdNum(const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
                  const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2,
                  int lcmLen, int eventIdNum);

  std::vector<std::pair<hivm::PIPE, hivm::PIPE>>
  checkTestMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  bool checkTestRWMemoryConflicts(
      const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
      const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2);
};

} // namespace mlir::hivm::syncsolver
#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERTEST_H
