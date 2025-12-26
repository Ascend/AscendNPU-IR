//===- InlineBrc.cpp - Inline Broadcast-like Ops For HFusion Ops ----------===//
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

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-inline-brc"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_HFUSIONINLINEBRC
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

// Currently enumerate all possible Ops
// TODO: generalize inlinable Op.
// An Op is consider inlinable if its operand shape can be easily
// inferred by init
static bool isInlinableOp(Operation *op) {
  return isa<linalg::ElemwiseBinaryOp>(op) ||
         isa<linalg::ElemwiseUnaryOp>(op) ||
         isa<hfusion::ElemwiseBinaryOp>(op) ||
         isa<hfusion::ElemwiseUnaryOp>(op) ||
         isa<hfusion::SelectOp>(op) ||
         isa<linalg::SelectOp>(op);
}

// TODO : add platform information
// check whether brc op can be inlined to current brc user op
static bool canInlineBrc(Operation *useOp, OpOperand *oper,
                         bool isScalar = false) {
  // only support inline brc to elemwise binary op currently
  if (!isInlinableOp(useOp)) {
    return false;
  }

  bool isSameAsInit = false;
  auto dstStyleOp = cast<DestinationStyleOpInterface>(useOp);
  for (auto initOper : dstStyleOp.getDpsInits()) {
    if (initOper == oper->get()) {
      isSameAsInit = true;
    }
  }
  return !isSameAsInit || !isScalar;
}

static void findUsersToInline(Value src, DenseSet<OpOperand *> &usesToInline,
                              bool isInlineScalar, int level = 0) {
  if (level >= 100) {
    llvm::report_fatal_error("findUsersToInline in infinite recursion");
  }
  assert(src && "src must not be nullptr");
  for (OpOperand &use : src.getUses()) {
    Operation *user = use.getOwner();
    if (canInlineBrc(user, &use, isInlineScalar)) {
      usesToInline.insert(&use);
      continue;
    }
    if (auto hfusionCast = dyn_cast<hfusion::CastOp>(user)) {
      Value castResult = hfusionCast.getResult(0);
      findUsersToInline(castResult, usesToInline, isInlineScalar, level + 1);
    } else if (auto tensorCast = dyn_cast<tensor::CastOp>(user)) {
      Value castResult = tensorCast.getResult();
      findUsersToInline(castResult, usesToInline, isInlineScalar, level + 1);
    }
  }
}

// replace brc result in current user op with brc scalar input
static LogicalResult replaceBrcWithInput(Operation *brcOp, Value brcResult,
                                         Value input, Location loc,
                                         PatternRewriter &rewriter) {
  if (!utils::isScalarLike(input)) {
    return rewriter.notifyMatchFailure(brcOp, "input is not scalar like.");
  }

  DenseSet<OpOperand *> usesToInline;
  bool isScalar = input.getType().isIntOrFloat();
  findUsersToInline(brcResult, usesToInline, isScalar);
  if (usesToInline.empty()) {
    return rewriter.notifyMatchFailure(brcOp, "cannot find users to inline.");
  }

  auto scalarMaybe = utils::extractScalarValue(rewriter, loc, input);
  if (!scalarMaybe.has_value()) {
    return rewriter.notifyMatchFailure(brcOp, "failed to get scalar value.");
  }

  Value scalar = scalarMaybe.value();
  Type srcElemType = getElementTypeOrSelf(scalar.getType());

  for (OpOperand *use : usesToInline) {
    Type dstElemType = getElementTypeOrSelf(use->get().getType());
    Value replacement = scalar;
    if (srcElemType != dstElemType) {
      hfusion::RoundMode roundMode =
          mlir::utils::selectRoundMode<hfusion::RoundMode>(srcElemType,
                                                           dstElemType);
      replacement = hfusion::castTo(rewriter, scalar, dstElemType, roundMode);
    }
    rewriter.modifyOpInPlace(use->getOwner(), [&]() { use->set(replacement); });
  }
  return success();
}

struct InlineBroadcastOpWithScalarInput
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp brcOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Got BroadcastOp: " << brcOp);
    Value input = brcOp.getInput();
    return replaceBrcWithInput(brcOp, brcOp->getResult(0), input,
                               brcOp->getLoc(), rewriter);
  }
};

struct InlineFillOpWithScalarInput : public OpRewritePattern<linalg::FillOp> {
public:
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Got FillOp: " << fillOp);
    if (fillOp->getNumResults() != 1) {
      return failure();
    }
    ValueRange inputs = fillOp.getInputs();
    if (inputs.size() != 1) {
      return failure();
    }

    Value input = inputs.front();
    return replaceBrcWithInput(fillOp, fillOp->getResult(0), input,
                               fillOp->getLoc(), rewriter);
  }
};

namespace {
struct HFusionInlineBrcPass
    : public impl::HFusionInlineBrcBase<HFusionInlineBrcPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineBroadcastOpWithScalarInput>(patterns.getContext());
    patterns.add<InlineFillOpWithScalarInput>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionInlineBrcPass() {
  return std::make_unique<HFusionInlineBrcPass>();
}
