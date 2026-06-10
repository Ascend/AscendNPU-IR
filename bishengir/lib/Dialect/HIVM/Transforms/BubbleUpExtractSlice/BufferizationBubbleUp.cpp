//===- BufferizationBubbleUp.cpp - Bufferization bubble-up pattern --------===//
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

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BufferizationBubbleUp.h"

#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/TileUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "bufferization-bubble-up-pattern"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << X << "\n")

namespace mlir::hivm::detail {

LogicalResult BufferizationBubbleUpPattern::matchAndRewrite(
    UnrealizedConversionCastOp upPropagator,
    PatternRewriter &rewriter) const {
  if (!upPropagator->hasAttr(kBubbleUpPropagateUp))
    return failure();

  auto funcOp = upPropagator->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return failure();

  LDBG("BufferizationBubbleUpPattern propagate on " << upPropagator);

  SmallVector<Operation *> roots;
  funcOp.walk([&](UnrealizedConversionCastOp ucc) {
    if (ucc->hasAttr(kBubbleUpPropagateUp) ||
        ucc->hasAttr(kBubbleUpPropagateDown))
      roots.push_back(ucc.getOperation());
  });
  if (llvm::is_contained(roots, upPropagator.getOperation()) == false)
    roots.push_back(upPropagator.getOperation());

  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<BufferizationPropagateUpPattern>(patterns.getContext());
  patterns.add<BufferizationPropagateDownPattern>(patterns.getContext());
  GreedyRewriteConfig config;
  config.maxIterations = 50;
  config.fold = false;
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  if (failed(applyOpPatternsGreedily(roots, std::move(patterns), config)))
    return failure();

  return success();
}

} // namespace mlir::hivm::detail
