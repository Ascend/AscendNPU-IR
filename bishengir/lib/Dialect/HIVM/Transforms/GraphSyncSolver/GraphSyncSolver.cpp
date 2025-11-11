//===------------- GraphSyncSolver.cpp ---- Graph Sync Solver -----===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverTester.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include <cstring>

#define DEBUG_TYPE "hivm-graph-sync-solver"

namespace mlir {
#define GEN_PASS_DEF_GRAPHSYNCSOLVER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm::syncsolver;

namespace mlir {
struct GraphSyncSolverPass
    : public impl::GraphSyncSolverBase<GraphSyncSolverPass> {
  explicit GraphSyncSolverPass(const GraphSyncSolverOptions &options)
      : GraphSyncSolverBase(options) {}

public:
  void runOnOperation() override;
};
} // namespace mlir

// Detailed overview:
// This translation unit defines the GraphSyncSolverPass, the entry point that
// ties the in-memory sync solver to MLIR function passes. Responsibilities:
//  1) Optionally run the test-mode harness (SyncTester) and exit early.
//  2) Skip host-side functions (we only transform device kernels).
//  3) Build an in-memory representation (funcIr) of the MLIR function and
//     instantiate the Solver that will analyze memory/read-write conflicts,
//     compute synchronization (Set/Wait/Barrier) placements and optimizations.
//  4) Optionally emit debug dumps of the in-memory IR and the linearized
//     sync-IR used by the solver to reason about ordering.
//  5) Run the solver phases (conflict discovery and decisions).
//  6) Generate either debug-only func-ir results (for inspection) or actual
//     MLIR operations inserted into the function body via IRRewriter.
// Important notes:
//  - The pass preserves the original MLIR structure and only inserts
//    synchronization ops when the solver determines they are necessary.
//  - The pass supports options like enabling the unit-flag feature which
//    change the solver codegen behavior; these options are read from the pass
//    configuration and propagated into the Solver instance.
//  - Debug printing is guarded by LLVM_DEBUG and controlled by build/runtime
//    flags.

void GraphSyncSolverPass::runOnOperation() {
  // 1) Test-mode: If the environment indicates testing, run the randomized
  //    SyncTester driver which builds random IR, runs the solver and
  //    validates correctness by simulation. When running in test-mode the
  //    pass does nothing to the current MLIR function and returns early.
  if (SyncTester::runTestMode()) {
    return;
  }

  // 2) Get the function op we are running on and skip host-side functions.
  auto func = getOperation();
  if (hacc::utils::isHost(func)) {
    return;
  }

  // 3) Build the Solver instance around the MLIR function. The Solver
  //    constructs an in-memory funcIr representation and linearizes it into
  //    syncIr occurrences used by conflict analysis.
  Solver solver(func);

  // 4) Propagate pass-level options (e.g., enableUnitFlag) into the solver.
  //    These options influence pattern detection and codegen behavior.
  solver.enableUnitFlagFeature = this->enableUnitFlag;

  // 5) Debug dump: print the pre-solve funcIr for developer inspection. These
  //    dumps are expensive and only enabled in debug builds.
  LLVM_DEBUG(llvm::dbgs() << "before:\n"
                          << solver.funcIr->str(0, true) << '\n';);

  // 6) Additional debug: dump the linearized occurrence list (syncIr) which
  //    shows the internal ordering the solver will reason about.
  LLVM_DEBUG({
    for (auto &occ : solver.syncIr) {
      llvm::dbgs() << std::string(occ->depth, ' ') << occ->op->id << ' '
                   << occ->syncIrIndex << ' ' << occ->startIndex << ' '
                   << occ->endIndex << '\n';
      llvm::dbgs() << occ->op->str(occ->depth, false) << '\n';
    }
  });

  // 7) Core analysis/decision phase: run the solver which performs:
  //    - pairwise processing order scans
  //    - memory-conflict detection
  //    - graph-based feasibility checks (to avoid deadlocks/liveness issues)
  //    - event-id allocation, reuse heuristics and barrier fallbacks
  solver.solve();

  // 8) Debug-only result generation: build funcIr with inserted sync ops for
  //    textual inspection without mutating the actual MLIR function. This is
  //    useful for debugging the solver decisions (kept inside LLVM_DEBUG).
  LLVM_DEBUG({
    solver.generateFuncIrResultOps();
    llvm::dbgs() << "after:\n" << solver.funcIr->str(0, true) << '\n';
  });

  // 9) Materialize results into the MLIR function. This step emits actual
  //    MLIR SetFlag/WaitFlag/PipeBarrier ops (or updates existing ops) using
  //    IRRewriter and is the point where the transformation becomes visible
  //    to subsequent passes and to codegen.
  solver.generateResultOps();
}

std::unique_ptr<Pass>
mlir::hivm::createGraphSyncSolverPass(const GraphSyncSolverOptions &options) {
  return std::make_unique<GraphSyncSolverPass>(options);
}