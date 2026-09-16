//===- AllOpKindAnalyzer.cpp - Implementation of AllOpKindAnalyzer --------===//
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

#include "bishengir/Dialect/Analysis/VFFusion/VFFusionAnalyzer.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vf-fusion-all-op-kind-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {

bool AllOpKindAnalyzer::isFusibleImpl(const int xIndex, const int yIndex) {
  LDBG("checking fusible");
  return true;
}

LogicalResult AllOpKindAnalyzer::fuseImpl(Block &block) {
  LDBG("AllOpKind Fusing " << block << "\n");
  initialize(block);
  for (Operation &op : block.getOperations()) {
    LDBG("Try to fuse " << op);
    const int yIndex = opToIndex.at(&op);
    for (auto opr : op.getOperands()) {
      auto *defOp = opr.getDefiningOp();
      if (!defOp)
        defOp = cast<BlockArgument>(opr).getOwner()->getParentOp();

      // defOp is outside of the block.
      LDBG("check if defOp is outside of the block " << defOp->getName());
      if (!opToIndex.contains(defOp))
        continue;

      const int xIndex = opToIndex.at(defOp);
      if (!VFFusionAnalyzerBase::isFusible(xIndex, yIndex))
        continue;

      LDBG("fusing index of " << xIndex << " with " << yIndex);
      if (!VFFusionAnalyzerBase::fuseIndexWith(xIndex, yIndex))
        continue;

      LDBG("Fused " << op.getName() << " " << defOp->getName());
    }
  }

  return success();
}

} // namespace mlir::analysis
