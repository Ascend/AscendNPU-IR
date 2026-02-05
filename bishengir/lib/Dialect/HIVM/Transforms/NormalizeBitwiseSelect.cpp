//===- NormalizeBitwiseSelect.cpp - normalize hivm bitwise select op.------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEBITWISESELECT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-normalize-bitwise-select"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

struct NormalizeBitwiseSelectPattern
    : public OpRewritePattern<annotation::MarkOp> {
public:
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;

  // TODO: the annotation should be added on the vselOp condition but not its
  // result
  LogicalResult matchAndRewrite(annotation::MarkOp op,
                                PatternRewriter &rewriter) const override {
    // Backtrace pattern:
    // %0 = ...
    // %1 = vcast ins(%0: tensor<Nxi8>) -> tensor<Nxf16>
    // %2 = vcmp ins(%1, %zero: tensor<Nxf16>, tensor<Nxf16>) -> tensor<Nxi1>
    // %3 = vnot ins(%2: tensor<Nxi1>) -> tensor<Nxi1>
    // %4 = vsel ins(%3, %true, %false)
    // annotation.mark %4 {bitwise_mask}
    if (!op->getAttrDictionary().contains("bitwise_mask"))
      return failure();
    auto selOp = dyn_cast_or_null<hivm::VSelOp>(op.getSrc().getDefiningOp());
    if (!selOp)
      return failure();
    auto notOp =
        dyn_cast_or_null<hivm::VNotOp>(selOp.getSrc()[0].getDefiningOp());
    if (!notOp)
      return failure();
    auto cmpOp =
        dyn_cast_or_null<hivm::VCmpOp>(notOp.getSrc()[0].getDefiningOp());
    if (!cmpOp)
      return failure();
    auto castOp =
        dyn_cast_or_null<hivm::VCastOp>(cmpOp.getSrc()[0].getDefiningOp());
    if (!castOp)
      return failure();
    while (auto nextOp = dyn_cast_or_null<hivm::VCastOp>(
               castOp.getSrc()[0].getDefiningOp()))
      castOp = nextOp;
    Value initMask = castOp.getSrc()[0];
    auto maskType = dyn_cast<ShapedType>(initMask.getType());
    if (!maskType || !maskType.getElementType().isInteger(8))
      return failure();
    // Generate pattern:
    // %0 = ...
    // vsel ins(%0, %false, %true)
    auto newSelOp = rewriter.create<hivm::VSelOp>(
        op.getLoc(), TypeRange({selOp.getDst()[0].getType()}),
        ValueRange({initMask, selOp.getSrc()[2], selOp.getSrc()[1]}),
        selOp.getDst(), Value());
    rewriter.eraseOp(op);
    selOp.replaceAllUsesWith(newSelOp.getResults());
    return success();
  }
};

class NormalizeBitwiseSelectPass
    : public impl::NormalizeBitwiseSelectBase<NormalizeBitwiseSelectPass> {
public:
  using NormalizeBitwiseSelectBase<
      NormalizeBitwiseSelectPass>::NormalizeBitwiseSelectBase;
  void runOnOperation() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<NormalizeBitwiseSelectPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::hivm::createNormalizeBitwiseSelectPass() {
  return std::make_unique<NormalizeBitwiseSelectPass>();
}
