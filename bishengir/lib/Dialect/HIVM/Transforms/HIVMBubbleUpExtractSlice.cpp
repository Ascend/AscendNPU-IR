//===- HIVMBubbleUpExtractSlice.cpp - Bubble Up ExtractSliceOp on HIVM ops ===//
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
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/CSEPattern.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/HoistAffine.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Transforms/Passes.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMBUBBLEUPEXTRACTSLICE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-bubble-up-extract-slice"

namespace {

using namespace mlir::hivm::detail;
class HIVMBubbleUpExtractSlicePass
    : public impl::HIVMBubbleUpExtractSliceBase<HIVMBubbleUpExtractSlicePass> {
public:
  using Base::Base;

  static bool traceAndCheckIsGM(Value value) {
    return !traceDefOp<memref::AllocOp>(value).has_value();
  }

  LogicalResult
  verifyMarkedExtractSlicesAreBubbledUp(func::FuncOp funcOp) const {
    auto walkResult = funcOp->walk([](Operation *op) {
      if (!isa<tensor::ExtractSliceOp>(op)) {
        return WalkResult::advance();
      }
      auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);

      if (extractSliceOp->hasAttrOfType<UnitAttr>(toBeBubbleUpSlice)) {
        auto extractSrc = extractSliceOp->getOperand(0);
        if (isa<BlockArgument>(extractSrc)) {
          return WalkResult::advance();
        }
        if (failed(findContainingSubblockLoop(extractSrc.getDefiningOp()))) {
          return WalkResult::advance();
        }
        if (auto bufferizeToTensor = dyn_cast<bufferization::ToTensorOp>(
                (extractSrc.getDefiningOp()))) {
          if (!traceAndCheckIsGM(bufferizeToTensor->getOperand(0))) {
            return WalkResult::interrupt();
          } else {
            return WalkResult::advance();
          }
        }
        if (!isa<tensor::EmptyOp>(extractSrc.getDefiningOp())) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    GreedyRewriteConfig config;
    config.maxIterations = 50;
    // Apply bubble up patterns
    RewritePatternSet patterns(funcOp.getContext());
    populateHoistAffinePattern(patterns);
    populateBubbleUpExtractSliceOpPatterns(patterns);
    populateCSEPattern(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns, true);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      return signalPassFailure();
    }
    PassManager pm(funcOp->getContext());
    CanonicalizerOptions options;
    SmallVector<std::string> disabledPatterns(
        {"ReinterpretCastConstantArgumentFolder"});
    options.disabledPatterns = disabledPatterns;
    pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
    pm.addPass(createCSEPass());
    if (failed(pm.run(funcOp))) {
      return signalPassFailure();
    }
    // Apply bubble up once more, because canonicalize might bring more
    // opportunity.
    RewritePatternSet patterns2(funcOp.getContext());
    populateHoistAffinePattern(patterns2);
    populateBubbleUpExtractSliceOpPatterns(patterns2);
    populateCSEPattern(patterns2);
    tensor::populateFoldTensorEmptyPatterns(patterns2, true);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns2), config))) {
      return signalPassFailure();
    }
    if (failed(verifyMarkedExtractSlicesAreBubbledUp(funcOp))) {
      return signalPassFailure();
    }
  }

private:
  void
  populateBubbleUpExtractSliceOpPatterns(RewritePatternSet &patterns) const {
    auto *context = patterns.getContext();

    // Create strategies
    SmallVector<std::shared_ptr<BubbleUpStrategy>> strategies;
    strategies.push_back(std::make_shared<BroadcastBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ReduceBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ExpandBubbleUpStrategy>());
    strategies.push_back(std::make_shared<CollapseBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ElementwiseBubbleUpStrategy>());
    strategies.push_back(std::make_shared<LoopBubbleUpStrategy>());
    strategies.push_back(std::make_shared<LoopArgsBubbleUpStrategy>());
    strategies.push_back(std::make_shared<ExtractSliceBubbleUpStrategy>());
    strategies.push_back(std::make_shared<InsertSliceBubbleUpStrategy>());
    strategies.push_back(std::make_shared<BitcastBubbleUpStrategy>());

    // Add pattern with strategies
    patterns.add<BubbleUpPattern>(context, std::move(strategies));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createHIVMBubbleUpExtractSlicePass() {
  return std::make_unique<HIVMBubbleUpExtractSlicePass>();
}
