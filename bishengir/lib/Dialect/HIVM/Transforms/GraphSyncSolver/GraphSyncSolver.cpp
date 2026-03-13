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
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverTester.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include <cstring>
#include <memory>

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

void GraphSyncSolverPass::runOnOperation() {
  if (this->enableTesterMode) {
    auto testerOptions = SmallVector<int64_t>(this->syncTesterOptions.begin(),
                                              this->syncTesterOptions.end());
    SyncTester::runTestMode(testerOptions);
    return;
  }

  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp)) {
    return;
  }

  bool isMemBasedArch = true;
  bool isRegBasedArch = false;
  assert(isMemBasedArch != isRegBasedArch);

  SyncSolverOptions options(SyncMode::INTRA_CORE_SYNC, isMemBasedArch,
                            isRegBasedArch);
  options.enableUnitFlagFeature = this->enableUnitFlag;

  auto irTranslator = std::make_unique<IRTranslator>(funcOp, options);

  LLVM_DEBUG({
    llvm::dbgs() << "before:\n" << irTranslator->funcIr->str(0, true) << '\n';
  });

  auto solver = std::make_unique<Solver>(std::move(irTranslator));

  DEBUG_WITH_TYPE("gss-print-unrolled-sync-ir", {
    for (auto &occ : solver->syncIr) {
      llvm::dbgs() << std::string(occ->depth, ' ') << occ->op->id << ' '
                   << occ->syncIrIndex << ' ' << occ->startIndex << ' '
                   << occ->endIndex << '\n';
      llvm::dbgs() << occ->op->str(occ->depth, false) << '\n';
    }
  });

  solver->solve();

  CodeGenerator codeGen(std::move(solver));
  codeGen.generateResultOps();

  LLVM_DEBUG({
    codeGen.generateFuncIrResultOps();
    llvm::dbgs() << "after:\n" << codeGen.funcIr->str(0, true) << '\n';
  });
}

std::unique_ptr<Pass>
mlir::hivm::createGraphSyncSolverPass(const GraphSyncSolverOptions &options) {
  return std::make_unique<GraphSyncSolverPass>(options);
}