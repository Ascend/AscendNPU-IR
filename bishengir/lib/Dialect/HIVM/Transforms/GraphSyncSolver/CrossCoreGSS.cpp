//===------------- CrossCoreGSS.cpp ---- Graph Sync Solver -----===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <cstring>

#define DEBUG_TYPE "hivm-cross-core-gss"

namespace mlir {
#define GEN_PASS_DEF_CROSSCOREGSS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm::syncsolver;

namespace mlir {
struct CrossCoreGSSPass : public impl::CrossCoreGSSBase<CrossCoreGSSPass> {
  explicit CrossCoreGSSPass(const CrossCoreGSSOptions &options)
      : CrossCoreGSSBase(options) {}

public:
  void runOnOperation() override;

private:
  std::optional<Value> getFFTSBaseAddrFromFunc(func::FuncOp funcOp) {
    auto funcArgs = funcOp.getArguments();
    auto *it = llvm::find_if(funcArgs, [&](BlockArgument arg) {
      return hacc::utils::isKernelArg(funcOp, arg.getArgNumber(),
                                      hacc::KernelArgType::kFFTSBaseAddr);
    });
    return it != funcArgs.end() ? (*it) : std::optional<Value>();
  }

  void insertSetFFTSBaseAddrOp(func::FuncOp funcOp, Value baseAddr) {
    auto *firstBlock = &funcOp.getBody().front();
    assert(firstBlock != nullptr);
    IRRewriter rewriter(funcOp->getContext());
    rewriter.setInsertionPointToStart(firstBlock);
    rewriter.create<hivm::SetFFTSBaseAddrOp>(funcOp->getLoc(), baseAddr);
  }

  LogicalResult checkWorkSpaceValidity() {
    auto funcOp = getOperation();
    for (auto [i, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (!hacc::utils::isKernelArg(funcOp, i,
                                    hacc::KernelArgType::kWorkspace)) {
        continue;
      }
      auto argUsers = arg.getUsers();
      auto noneAllocWorkSpaceUserIter =
          llvm::find_if(argUsers, [](Operation *user) {
            return !isa<bishengir::memref_ext::AllocWorkspaceOp>(user);
          });
      if (noneAllocWorkSpaceUserIter != argUsers.end()) {
        return noneAllocWorkSpaceUserIter->emitError(
            "All users of workspace arg must be AllocWorkspaceOp!");
      }
    }
    return success();
  }
};
} // namespace mlir

void CrossCoreGSSPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp)) {
    return;
  }
  auto funcCoreType = hivm::queryFuncCoreType(funcOp);
  if (!funcCoreType.has_value() ||
      (funcCoreType.value() != hivm::TFuncCoreType::MIX)) {
    return;
  }

  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  bool isMemBasedArch = hacc::utils::isMemBasedArch(moduleOp);
  bool isRegBasedArch = hacc::utils::isRegBasedArch(moduleOp);
  assert(isMemBasedArch != isRegBasedArch);

  if (this->forceIsRegBased) {
    isMemBasedArch = false;
    isRegBasedArch = true;
  }
  if (this->forceIsMemBased) {
    isMemBasedArch = true;
    isRegBasedArch = false;
  }

  if (isMemBasedArch) {
    // get && set ffts base addr
    std::optional<Value> baseAddr = getFFTSBaseAddrFromFunc(funcOp);
    assert(baseAddr.has_value() &&
           "The mix kernel parameter must have a ffts_addr value");
    insertSetFFTSBaseAddrOp(funcOp, baseAddr.value());
  }

  if (this->disableAutoInjectBlockSync) {
    return;
  }

  SyncSolverOptions options(SyncMode::CROSS_CORE_SYNC, isMemBasedArch,
                            isRegBasedArch);
  if (this->alwaysUsePipeSAsWaitingPipe) {
    options.alwaysUsePipeSAsWaitingPipe = true;
  }
  if (this->useDifferentMultiBufferFlagIds) {
    options.useDifferentMultiBufferFlagIds = true;
  }

  auto irTranslator = std::make_unique<IRTranslator>(funcOp, options);

  LLVM_DEBUG({
    llvm::dbgs() << "before:\n" << irTranslator->funcIr->str(0, true) << '\n';
  });

  auto solver = std::make_unique<Solver>(std::move(irTranslator));

  LLVM_DEBUG({
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
mlir::hivm::createCrossCoreGSSPass(const CrossCoreGSSOptions &options) {
  return std::make_unique<CrossCoreGSSPass>(options);
}