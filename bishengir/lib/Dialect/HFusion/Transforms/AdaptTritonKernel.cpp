//===- AdaptTritonKernel.cpp - Adapt triton kernel                       -===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Transform.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
#define GEN_PASS_DEF_ADAPTTRITONKERNEL
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "adapt-triton-kernel"

using namespace mlir;

namespace {

/// Workaround to preserve Triton RTZ floating-point cast semantics when
/// lowering to VCVTFF.
///
/// A narrowing cast with round_mode<trunc> requests rounding toward zero (RTZ).
/// For example:
///
/// ```mlir
/// %y = hfusion.cast {round_mode = #hfusion.round_mode<trunc>}
///     ins(%x : tensor<16xf32>) outs(%init : tensor<16xf16>)
///     -> tensor<16xf16>
/// ```
///
/// This eventually lowers to VCVTFF after all. Its non-saturating RTZ behavior
/// does not implement the required finite-overflow semantics. Enabling
/// saturation gives the required finite-overflow behavior, but may also clamp
/// infinities and does not by itself preserve NaNs. Therefore, use the
/// saturating RTZ cast only for finite inputs and obtain non-finite results
/// from a separate, non-saturating RINT cast:
///
/// ```mlir
/// %finite = hfusion.isfinite %x : tensor<16xf32> -> tensor<16xi1>
/// %special = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
///     ins(%x : tensor<16xf32>) outs(%tmp : tensor<16xf16>)
///     -> tensor<16xf16>
/// %normal = hfusion.cast {enable_saturate = true, hfusion.rtz_processed,
///                        round_mode = #hfusion.round_mode<trunc>}
///     ins(%x : tensor<16xf32>) outs(%init : tensor<16xf16>)
///     -> tensor<16xf16>
/// %y = hfusion.select
///     ins(%finite, %normal, %special
///         : tensor<16xi1>, tensor<16xf16>, tensor<16xf16>)
///     outs(%init : tensor<16xf16>) -> tensor<16xf16>
/// ```
///
/// Here %tmp is a fresh tensor.empty and the auxiliary cast has saturation
/// disabled. Its finite results are discarded; only NaN and signed infinities
/// are selected. Rounding is irrelevant for these special values, and using
/// RINT avoids recursively applying this workaround to the auxiliary cast.
/// The processed marker makes the transformation idempotent. Saturation is
/// updated in both the named op and its scalar region so that generalization
/// preserves the same semantics.
///
/// Note that hfusion.isfinite / hfusion.isnan accept only f16, bf16 and f32.
/// For FP8 sources, maskInput is a non-saturating widening cast to f32 used
/// solely for classification; both result-producing casts still consume the
/// original input. This is an elementwise selection, not a masked gather or a
/// change of tensor shape.
///
/// The f8E4M3FN type has no infinities. For this destination, retain saturating
/// conversion for all non-NaN inputs and select a broadcast destination NaN for
/// NaN inputs instead of generating the auxiliary result cast.
struct WorkaroundRTZCastPattern : public OpRewritePattern<hfusion::CastOp> {
  using OpRewritePattern<hfusion::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::CastOp op,
                                PatternRewriter &rewriter) const override {
    constexpr StringLiteral processedAttr = "hfusion.rtz_processed";
    if (op->hasAttr(processedAttr) ||
        op.getRoundMode() != hfusion::RoundMode::TRUNC ||
        !op.hasPureTensorSemantics()) {
      return failure();
    }

    Value input = op.getDpsInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inputType || !outputType) {
      return failure();
    }

    auto srcType = dyn_cast<FloatType>(inputType.getElementType());
    auto dstType = dyn_cast<FloatType>(outputType.getElementType());
    if (!srcType || !dstType || dstType.getWidth() > srcType.getWidth()) {
      return failure();
    }

    Location loc = op.getLoc();

    rewriter.setInsertionPoint(op);

    // hfusion.isfinite accept f16, bf16 and f32, so widen FP8 first
    Value maskInput = input;
    if (isa<Float8E4M3FNType, Float8E5M2Type>(srcType)) {
      maskInput = hfusion::castTo(
          rewriter, input, rewriter.getF32Type(), hfusion::RoundMode::RINT,
          std::nullopt, op.getEnableOverflow(), false,
          hfusion::TypeFn::cast_signed, hfusion::UnsignedMode::SI2SI);
    }

    auto maskType = inputType.clone(rewriter.getI1Type());
    bool finiteOnlyDestination = isa<Float8E4M3FNType>(dstType);
    Value mask;
    Value special;
    if (finiteOnlyDestination) {
      // The E4M3FN type cannot represent infinity.
      // So, retain its existing saturation policy.
      mask = rewriter.create<hfusion::IsNanOp>(loc, maskType, maskInput);
      special = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(
                   dstType, APFloat::getNaN(dstType.getFloatSemantics())));
    } else {
      mask = rewriter.create<hfusion::IsFiniteOp>(loc, maskType, maskInput);
      // Only non-finite results are used.
      // RINT preserves them without recursively triggering this RTZ workaround.
      special = hfusion::castTo(
          rewriter, input, dstType, hfusion::RoundMode::RINT, std::nullopt,
          op.getEnableOverflow(), false, hfusion::TypeFn::cast_signed,
          hfusion::UnsignedMode::SI2SI);
    }

    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr(processedAttr, rewriter.getUnitAttr());
      op->setAttr("enable_saturate", rewriter.getBoolAttr(true));

      // Named structured ops already contain a scalar region. Updating only
      // the outer attribute would leave generalization using stale semantics.
      op->getRegion(0).walk([&](Operation *scalarOp) {
        if (scalarOp->hasAttr("round_mode"))
          scalarOp->setAttr("enable_saturate", rewriter.getBoolAttr(true));
      });
    });

    rewriter.setInsertionPointAfter(op);
    auto select = rewriter.create<hfusion::SelectOp>(
        loc, TypeRange{outputType},
        finiteOnlyDestination ? ValueRange{mask, special, op.getResult(0)}
                              : ValueRange{mask, op.getResult(0), special},
        ValueRange{op.getDpsInits()[0]});
    rewriter.replaceAllUsesExcept(op.getResult(0), select.getResult(0), select);
    return success();
  }
};

/// This pass labels the triton entry kernel
struct AdaptTritonKernelPass
    : public impl::AdaptTritonKernelBase<AdaptTritonKernelPass> {
  using AdaptTritonKernelBase<AdaptTritonKernelPass>::AdaptTritonKernelBase;

public:
  void runOnOperation() override;
};

void markWorkspaceArgument(func::FuncOp funcOp) {
  constexpr StringRef workspaceArgName = "WorkspaceArgIdx";
  if (!funcOp->hasAttr(workspaceArgName))
    return;
  int64_t workspaceArgIdx =
      funcOp->getAttrOfType<IntegerAttr>(workspaceArgName).getInt();
  assert(funcOp.getNumArguments() > workspaceArgIdx);

  funcOp.setArgAttrs(
      workspaceArgIdx,
      SmallVector<NamedAttribute>{hacc::createHACCKernelArgAttr(
          funcOp.getContext(), hacc::KernelArgType::kWorkspace)});
}

void markSyncBlockLockArgument(func::FuncOp funcOp) {
  constexpr StringRef lockArgName = "SyncBlockLockArgIdx";
  if (!funcOp->hasAttr(lockArgName))
    return;
  int64_t syncBlockLockArgIdx =
      funcOp->getAttrOfType<IntegerAttr>(lockArgName).getInt();
  assert(funcOp.getNumArguments() > syncBlockLockArgIdx);

  funcOp.setArgAttrs(
      syncBlockLockArgIdx,
      SmallVector<NamedAttribute>{hacc::createHACCKernelArgAttr(
          funcOp.getContext(), hacc::KernelArgType::kSyncBlockLock)});
}

/// triton print op is converted triton_print func call in triton adaptor,
/// therefore this pattern match func call of which func name starts with
/// "triton_print".
struct TritonPrintToHFusionPrintPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef printFuncName = "triton_print";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(printFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp, funcName + " does not starts with the prefix triton_print");
    }
    auto printOpNameStr = hfusion::PrintOp::getOperationName();
    auto printOpName =
        mlir::OperationName(printOpNameStr, rewriter.getContext());
    auto prefixAttrName = hfusion::PrintOp::getPrefixAttrName(printOpName);
    bool hasPrefixAttr = funcOp->hasAttr(prefixAttrName);
    if (!hasPrefixAttr) {
      return rewriter.notifyMatchFailure(
          funcOp, funcName + " has no attribute of prefix");
    }
    auto hexAttrName = hfusion::PrintOp::getHexAttrName(printOpName);
    bool hasHexAttr = funcOp->hasAttr(hexAttrName);
    if (!hasHexAttr) {
      return rewriter.notifyMatchFailure(funcOp,
                                         funcName + " has no attribute of hex");
    }

    auto prefixAttr = funcOp->getAttrOfType<StringAttr>(prefixAttrName);
    auto hexAttr = funcOp->getAttrOfType<BoolAttr>(hexAttrName);
    for (Value operand : callOp.getArgOperands()) {
      rewriter.create<hfusion::PrintOp>(callOp.getLoc(), prefixAttr, hexAttr,
                                        operand);
    }
    rewriter.eraseOp(callOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

/// triton assert op is converted triton_assert func call in triton adaptor,
/// therefore this pattern match func call of which func name starts with
/// "triton_assert".
struct TritonAssertToHFusionAssertPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef assertFuncName = "triton_assert";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(assertFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp, funcName + " does not starts with the prefix triton_assert");
    }
    auto assertOpNameStr = hfusion::AssertOp::getOperationName();
    auto assertOpName =
        mlir::OperationName(assertOpNameStr, rewriter.getContext());
    auto msgAttrName = hfusion::AssertOp::getMsgAttrName(assertOpName);
    bool hasMsgAttr = funcOp->hasAttr(msgAttrName);
    if (!hasMsgAttr) {
      return rewriter.notifyMatchFailure(funcOp,
                                         funcName + " has no attribute of msg");
    }
    auto msgAttr = funcOp->getAttrOfType<StringAttr>(msgAttrName);
    if (callOp.getArgOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          callOp,
          "calling " + funcName + " with wrong number of args (expecting 1)");
    }
    auto originArg = callOp.getArgOperands()[0];
    auto ty = originArg.getType();
    if (isa<RankedTensorType>(ty)) {
      auto i1TensorTy = cast<RankedTensorType>(ty);
      SmallVector<int64_t> shape(i1TensorTy.getShape());
      RankedTensorType i8TensorTy =
          RankedTensorType::get(shape, rewriter.getI8Type());
      auto arg = rewriter.create<arith::ExtUIOp>(callOp.getLoc(), i8TensorTy,
                                                 originArg);
      rewriter.create<hfusion::AssertOp>(callOp.getLoc(), msgAttr, arg);
    } else {
      rewriter.create<hfusion::AssertOp>(callOp.getLoc(), msgAttr, originArg);
    }
    rewriter.eraseOp(callOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

/// triton gather op is converted triton_gather func call in triton adaptor,
/// therefore this pattern match func call of which func name starts with
/// "triton_gather".
struct TritonGatherToHFusionGatherPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton_gather";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value index = callOp.getOperand(1);
    Value axisVal = callOp.getOperand(2);
    auto axis = mlir::utils::getArithConstantOpValue<int64_t>(axisVal);
    if (failed(axis)) {
      return callOp->emitError("Failed to extract the value of arith.constant "
                               "defining the gather axis.");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto elemTy = srcTy.getElementType();
    auto indexTy = cast<RankedTensorType>(index.getType());
    auto resShape = indexTy.getShape();
    auto init = rewriter.create<tensor::EmptyOp>(loc, resShape, elemTy);
    auto gatherOp =
        rewriter.create<hfusion::GatherOp>(loc, src, index, init, *axis);
    rewriter.replaceOp(callOp, gatherOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

// SUFFIX SPLIT for reverse extraction in triton_cum{sum,prod,max,min} pattern.
//   _membase (strict):     FailureOr<bool> via getArithConstantOpValue<bool>
//                          + emitError on parse failure (preserves master behavior).
//   _regbase (permissive): dyn_cast default false, no diagnostic
//                          (preserves A5 behavior).
static FailureOr<bool> getReverseValue_membase(func::CallOp callOp) {
  Value reverseVals = callOp.getOperand(2);
  auto reverse = mlir::utils::getArithConstantOpValue<bool>(reverseVals);
  if (failed(reverse))
    callOp->emitError("Failed to extract the value of arith.constant "
                      "defining the reverse.");
  return reverse;
}

static bool getReverseValue_regbase(func::CallOp callOp) {
  Value reverseAttr = callOp.getOperand(2);
  bool reverse{false};
  if (auto constOp =
          dyn_cast<arith::ConstantOp>(reverseAttr.getDefiningOp())) {
    if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue()))
      reverse = boolAttr.getValue();
  }
  return reverse;
}

/// triton cumsum/cumprod op is converted triton_cumsum func call in triton
/// adaptor, therefore this pattern match func call of which func name starts
/// with "triton_cumsum/triton_cumprod".
struct TritonCumToHFusionCumPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef cumsumFuncName = "triton_cumsum";
  static constexpr StringRef cumprodFuncName = "triton_cumprod";
  static constexpr StringRef cummaxFuncName = "triton_cummax";
  static constexpr StringRef cumminFuncName = "triton_cummin";
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }

    auto isRegBasedArch = hacc::utils::isRegBasedArch(callOp->getParentOfType<ModuleOp>());

    auto funcName = funcOp.getSymName();
    mlir::hfusion::CumOpType cumOpType = mlir::hfusion::CumOpType::UNDEFINED;
    if (funcName.starts_with(cumsumFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMSUM;
    } else if (funcName.starts_with(cumprodFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMPROD;
    } else if (isRegBasedArch && funcName.starts_with(cummaxFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMMAX;
    } else if (isRegBasedArch && funcName.starts_with(cumminFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMMIN;
    } else {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + cumsumFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value dimVals = callOp.getOperand(1);
    bool reverse{false};
    if (isRegBasedArch) {
      reverse = getReverseValue_regbase(callOp);
    } else {
      auto reverseOr = getReverseValue_membase(callOp);
      if (failed(reverseOr))
        return failure();
      reverse = *reverseOr;
    }
    // cummax/cummin carry an optional trailing propagateNan flag (operand 3),
    // selecting torch-like NaN propagation (Maximum/Minimum) vs ignore-NaN
    // (MaxNum/MinNum). Default to propagation when the operand is absent.
    bool propagateNan{true};
    if (isRegBasedArch && callOp.getNumOperands() > 3) {
      if (auto constOp =
              dyn_cast<arith::ConstantOp>(callOp.getOperand(3).getDefiningOp())) {
        if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
          propagateNan = boolAttr.getValue();
        }
      }
    }
    auto cumDim = mlir::utils::getArithConstantOpValue<int64_t>(dimVals);
    if (failed(cumDim)) {
      return callOp->emitError("Failed to extract the value of arith.constant "
                               "defining the cum dims.");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    llvm::SmallVector<int64_t> cumDims{*cumDim};
    if (cumOpType == mlir::hfusion::CumOpType::CUMSUM) {
      rewriter.replaceOp(callOp, rewriter.create<hfusion::CumsumOp>(
                                     loc, srcTy, src, cumDims, reverse));
    } else if (cumOpType == mlir::hfusion::CumOpType::CUMPROD) {
      rewriter.replaceOp(callOp, rewriter.create<hfusion::CumprodOp>(
                                     loc, srcTy, src, cumDims, reverse));
    } else if (cumOpType == mlir::hfusion::CumOpType::CUMMAX) {
      assert(isRegBasedArch);
      auto cumOp =
          rewriter.create<hfusion::CummaxOp>(loc, srcTy, src, cumDims, reverse);
      cumOp.setPropagateNan(propagateNan);
      rewriter.replaceOp(callOp, cumOp);
    } else if (cumOpType == mlir::hfusion::CumOpType::CUMMIN) {
      assert(isRegBasedArch);
      auto cumOp =
          rewriter.create<hfusion::CumminOp>(loc, srcTy, src, cumDims, reverse);
      cumOp.setPropagateNan(propagateNan);
      rewriter.replaceOp(callOp, cumOp);
    } else {
      llvm::report_fatal_error("unsupport cumulative function");
    }
    rewriter.eraseOp(funcOp);

    return success();
  }
};

/// triton indirect_load op is converted triton_indirect_load func call in
/// triton adaptor, therefore this pattern match func call of which func name
/// starts with "triton_indirect_load".
/// TODO: Based on CustomOp
struct TritonIndirectLoadToHFusionIndirectLoadPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton_indirect_load";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value offsets = callOp.getOperand(1);

    Value mask = nullptr;
    Value other = nullptr;

    unsigned numOperands = callOp.getNumOperands();
    if (numOperands > 2) {
      mask = callOp.getOperand(2);
    }
    if (numOperands > 3) {
      other = callOp.getOperand(3);
    }

    if (mask) {
      auto maskType = mlir::cast<RankedTensorType>(mask.getType());
      if (!maskType)
        return failure();
      auto elemType = maskType.getElementType();

      Value actualMask = mask;
      if (elemType.isInteger(1)) {

        Type i8Type = rewriter.getIntegerType(8);
        TensorType resultType = maskType.clone(i8Type);

        actualMask = rewriter.create<arith::ExtUIOp>(loc, resultType, mask);
      }
      mask = actualMask;
    } else {
      auto offsetsType = mlir::cast<RankedTensorType>(offsets.getType());
      auto maskType =
          RankedTensorType::get(offsetsType.getShape(), rewriter.getI8Type());
      auto trueAttr = rewriter.getI8IntegerAttr(1);
      auto trueScalar = rewriter.create<arith::ConstantOp>(loc, trueAttr);
      auto maskOp = rewriter.create<tensor::SplatOp>(loc, maskType,
                                                     trueScalar->getResults());
      mask = maskOp.getResult();
    }

    if (!other) {
      auto offsetType = mlir::cast<RankedTensorType>(offsets.getType());
      auto srcType = mlir::cast<MemRefType>(src.getType());
      auto elementType = srcType.getElementType();
      auto otherType =
          RankedTensorType::get(offsetType.getShape(), elementType);

      assert(mlir::isa<IntegerType>(elementType) ||
             mlir::isa<FloatType>(elementType));
      Value zeroValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(elementType));
      other = rewriter.create<tensor::SplatOp>(loc, otherType, zeroValue)
                  .getResult();
    }

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto srcArgIdx = utils::getArgumentIndex(src);
    argIndices.push_back(srcArgIdx);
    SmallVector<Attribute> attrList;
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    auto resultType = cast<RankedTensorType>(callOp.getResult(0).getType());
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 resultType.getElementType());

    hfusion::IndirectLoadOp indirectLoadOp =
        rewriter.create<hfusion::IndirectLoadOp>(loc, resultType, src, offsets,
                                                 init, mask, other);
    bool isVolatile = true;
    if (auto attr = callOp->getAttrOfType<BoolAttr>("isVolatile")) {
      isVolatile = attr.getValue();
    }
    indirectLoadOp->setAttr("isVolatile", rewriter.getBoolAttr(isVolatile));
    rewriter.replaceOp(callOp, indirectLoadOp);
    return success();
  }
};

/// triton stride_load op is converted to a triton_stride_load func call in
/// triton adaptor, therefore this pattern matches func calls whose names start
/// with "triton_stride_load".
struct TritonStrideLoadToHFusionStrideLoadPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef strideLoadFuncName = "triton_stride_load";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(strideLoadFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp, funcName + " does not starts with the prefix " +
                      strideLoadFuncName);
    }

    if (callOp.getNumResults() != 1)
      return failure();

    auto loc = callOp.getLoc();
    auto resultType = cast<RankedTensorType>(callOp.getResult(0).getType());
    int64_t rank = resultType.getRank();
    if (callOp.getNumOperands() != static_cast<unsigned>(3 + 2 * rank))
      return failure();

    Value src = callOp.getOperand(0);
    Value offset = callOp.getOperand(1);
    Value other = callOp.getOperand(2);
    ValueRange strides = callOp.getOperands().slice(3, rank);
    ValueRange numels = callOp.getOperands().slice(3 + rank, rank);

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto srcArgIdx = utils::getArgumentIndex(src);
    argIndices.push_back(srcArgIdx);
    SmallVector<Attribute> attrList;
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 resultType.getElementType());
    auto strideLoadOp = rewriter.create<hfusion::StrideLoadOp>(
        loc, resultType, src, init, offset, other, strides, numels);
    rewriter.replaceOp(callOp, strideLoadOp);
    return success();
  }
};

/// triton stride_store op is converted to a triton_stride_store func call in
/// triton adaptor, therefore this pattern matches func calls whose names start
/// with "triton_stride_store".
struct TritonStrideStoreToHFusionStrideStorePattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef strideStoreFuncName = "triton_stride_store";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(strideStoreFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp, funcName + " does not starts with the prefix " +
                      strideStoreFuncName);
    }

    if (callOp.getNumResults() != 0)
      return failure();
    if (callOp.getNumOperands() < 5)
      return failure();

    auto loc = callOp.getLoc();
    Value dst = callOp.getOperand(0);
    Value src = callOp.getOperand(1);
    auto srcType = cast<RankedTensorType>(src.getType());
    int64_t rank = srcType.getRank();
    if (callOp.getNumOperands() != static_cast<unsigned>(3 + 2 * rank))
      return failure();

    Value offset = callOp.getOperand(2);
    ValueRange strides = callOp.getOperands().slice(3, rank);
    ValueRange numels = callOp.getOperands().slice(3 + rank, rank);

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto dstArgIdx = utils::getArgumentIndex(dst);
    argIndices.push_back(dstArgIdx);
    SmallVector<Attribute> attrList;
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    rewriter.create<hfusion::StrideStoreOp>(loc, dst, src, offset, strides,
                                            numels);
    rewriter.eraseOp(callOp);
    return success();
  }
};

/// triton gather op is converted triton__gather_out_to_ub func call in triton
/// adaptor, therefore this pattern match func call of which func name starts
/// with "triton__gather_out_to_ub".
/// TODO: Based on CustomOp
struct TritonGatherTToHFusionGatherTPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton__gather_out_to_ub";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value index = callOp.getOperand(1);
    Value bound = callOp.getOperand(2);
    Value dim = callOp.getOperand(3);

    SmallVector<Value> src_stride;
    SmallVector<Value> index_shape;
    SmallVector<Value> offsets;

    auto idxTy = cast<RankedTensorType>(index.getType());
    auto idxShape = idxTy.getShape();
    auto idxRank = idxShape.size();
    unsigned pushBackPos = 4;
    unsigned numOperands = callOp.getNumOperands();

    for (size_t i = 0; i < idxRank; i++) {
      src_stride.push_back(callOp.getOperand(pushBackPos++));
    }
    for (size_t i = 0; i < idxRank; i++) {
      index_shape.push_back(callOp.getOperand(pushBackPos++));
    }
    for (size_t i = 0; i < idxRank; i++) {
      offsets.push_back(callOp.getOperand(pushBackPos++));
    }

    Value other;

    auto srcType = mlir::cast<MemRefType>(src.getType());
    auto elementType = srcType.getElementType();

    if (numOperands > 4 + 3 * idxRank) {
      other = callOp.getOperand(pushBackPos++);
    } else {
      if (mlir::isa<IntegerType>(elementType)) {
        other = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(elementType));
      } else {
        other = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elementType, 0.0));
      }
    }

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto srcArgIdx = utils::getArgumentIndex(src);
    argIndices.push_back(srcArgIdx);
    SmallVector<Attribute> attrList;
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    auto resultType = cast<RankedTensorType>(callOp.getResult(0).getType());

    auto memrefType = MemRefType::get(idxShape, elementType);
    Value init = rewriter.create<mlir::memref::AllocOp>(loc, memrefType);
    rewriter.create<mlir::linalg::FillOp>(loc, other, init);
    // TODO: Based on CustomOp and remove bufferization
    Value tensorResult = rewriter.create<mlir::bufferization::ToTensorOp>(
        loc, init, /*restrict=*/true);

    auto gatherOp = rewriter.create<hfusion::GatherTOp>(
        loc, resultType, src, index, tensorResult, bound, dim, src_stride,
        index_shape, offsets);
    rewriter.replaceOp(callOp, gatherOp);

    return success();
  }
};

/// triton index_put op is converted triton_index_put func call in triton
/// adaptor, therefore this pattern match func call of which func name starts
/// with "triton_index_put".
/// TODO: Based on CustomOp
struct TritonIndexPutToHFusionIndexPutPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton_index_put";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value dst = callOp.getOperand(0);
    Value index = callOp.getOperand(1);
    Value value = callOp.getOperand(2);
    Value scatter_dim = callOp.getOperand(3);
    Value bound = callOp.getOperand(4);
    unsigned pushBackPos = 5;

    SmallVector<Value> end_offset;
    SmallVector<Value> start_offset;
    SmallVector<Value> dst_stride;

    auto valueTy = cast<RankedTensorType>(value.getType());
    auto valueShape = valueTy.getShape();
    auto valueRank = valueShape.size();

    for (size_t i = 0; i < valueRank; i++) {
      end_offset.push_back(callOp.getOperand(pushBackPos++));
    }
    for (size_t i = 0; i < valueRank; i++) {
      start_offset.push_back(callOp.getOperand(pushBackPos++));
    }
    for (size_t i = 0; i < valueRank; i++) {
      dst_stride.push_back(callOp.getOperand(pushBackPos++));
    }

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto dstArgIdx = utils::getArgumentIndex(dst);
    argIndices.push_back(dstArgIdx);
    SmallVector<Attribute> attrList;
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    hfusion::IndexPutOp indexPutOp = rewriter.create<hfusion::IndexPutOp>(
        loc, dst, index, value, scatter_dim, bound, end_offset, start_offset,
        dst_stride);
    rewriter.replaceOp(callOp, indexPutOp);

    return success();
  }
};

struct TritonFlipToHFusionFlipPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef flipFuncName = "triton_flip";
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp)
      return callOp->emitError(
          "Unable to resolve called function for flip pattern.");
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(flipFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp, funcName + " does not start with the prefix " + flipFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value flipAxisVals = callOp.getOperand(1);
    auto flipAxis = mlir::utils::getArithConstantOpValue<int64_t>(flipAxisVals);
    if (failed(flipAxis)) {
      return callOp->emitError("Failed to extract the value of arith.constant"
                               "defining the Flip axis.");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto flipOp = rewriter.create<hfusion::FlipOp>(loc, srcTy, src, *flipAxis);
    rewriter.replaceOp(callOp, flipOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

struct TritonSortToHFusionSortPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef sortFuncName = "triton_sort";
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {

    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(sortFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + sortFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value sortAxisVals = callOp.getOperand(1);
    auto sortAxis = mlir::utils::getArithConstantOpValue<int64_t>(sortAxisVals);
    if (llvm::failed(sortAxis)) {
      return callOp->emitError("Failed to extract the value of arith.constant "
                               "defining the sort axis");
    }

    Value descendingVals = callOp.getOperand(2);
    auto descending =
        mlir::utils::getArithConstantOpValue<bool>(descendingVals);
    if (llvm::failed(descending)) {
      return callOp->emitError("Failed to extract the value of arith.constant "
                               "defining the descending");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto sortOp = rewriter.create<hfusion::SortOp>(loc, srcTy, src, *descending,
                                                   *sortAxis);
    rewriter.replaceOp(callOp, sortOp);
    rewriter.eraseOp(funcOp);

    return llvm::success();
  }
};

/// TODO: Based on CustomOp
struct TritonScatterTOpToHFusionScatterTOpPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton_scatter_ub_to_out";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value dst = callOp.getOperand(0);
    Value value = callOp.getOperand(1);
    Value index_tile = callOp.getOperand(2);
    Value index_boundary = callOp.getOperand(3);
    Value dim = callOp.getOperand(4);

    SmallVector<Value> dst_stride;
    SmallVector<Value> index_shape;
    SmallVector<Value> offsets;

    auto idxTy = cast<RankedTensorType>(index_tile.getType());
    auto idxShape = idxTy.getShape();
    auto idxRank = idxShape.size();

    for (size_t i = 0; i < idxRank; i++) {
      dst_stride.push_back(callOp.getOperand(5 + i));
    }

    for (size_t i = 0; i < idxRank; i++) {
      index_shape.push_back(callOp.getOperand(5 + i + idxRank));
    }

    for (size_t i = 0; i < idxRank; i++) {
      offsets.push_back(callOp.getOperand(5 + i + idxRank + idxRank));
    }

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto dstArgIdx = utils::getArgumentIndex(dst);
    argIndices.push_back(dstArgIdx);
    SmallVector<Attribute> attrList;
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    hfusion::ScatterTOp scatterTOp = rewriter.create<hfusion::ScatterTOp>(
        loc, dst, value, index_tile, index_boundary, dim, dst_stride,
        index_shape, offsets);
    rewriter.replaceOp(callOp, scatterTOp);
    return success();
  }
};

struct TritonBindSubBlockAttrToHFusionPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  static constexpr llvm::StringLiteral kSplitVectorAttrName = "bind_sub_block";

  static bool hasBindSubBlockAttr(PatternRewriter &rewriter, scf::ForOp op) {
    if (!op->hasAttr(kSplitVectorAttrName))
      return false;
    return op->getAttr(kSplitVectorAttrName) == rewriter.getBoolAttr(true);
  }

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasBindSubBlockAttr(rewriter, op))
      return failure();

    // The reason for normalizing is that for changing into parallel blocks,
    // only index type is acceptable
    auto newForOp = bishengir::scf::normalizeToIndex(rewriter, op);
    rewriter.modifyOpInPlace(
        newForOp, [&]() { newForOp->removeAttr(kSplitVectorAttrName); });
    rewriter.modifyOpInPlace(newForOp, [&]() {
      newForOp->setAttr(hfusion::BindSubBlockAttr::name,
                        UnitAttr::get(newForOp->getContext()));
    });
    return success();
  }
};

struct TritonEmbeddingGatherToHFusionEmbeddingGatherPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton_embedding_gather";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not start with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value idx = callOp.getOperand(1);
    Value bound = callOp.getOperand(2);
    Value blksiz = callOp.getOperand(3);
    SmallVector<Value> offsets;
    SmallVector<Value> numels;
    auto idxTy = cast<RankedTensorType>(idx.getType());
    auto idxShape = idxTy.getShape();
    auto idxRank = idxShape.size();
    for (size_t i = 0; i <= idxRank; i++) {
      offsets.push_back(callOp.getOperand(4 + i));
    }
    for (size_t i = 0; i <= idxRank; i++) {
      numels.push_back(callOp.getOperand(4 + idxRank + 1 + i));
    }

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto srcArgIdx = utils::getArgumentIndex(src);
    argIndices.push_back(srcArgIdx);
    SmallVector<Attribute> attrList;
    for (auto _idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), _idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    auto srcTy = cast<MemRefType>(src.getType());
    auto elemTy = srcTy.getElementType();
    std::vector<int64_t> retShape(idxShape.begin(), idxShape.end());
    auto blksiz_int = blksiz.getDefiningOp<arith::ConstantIntOp>().value();
    retShape.push_back(blksiz_int);
    auto retTy = RankedTensorType::get(retShape, elemTy);
    auto init = rewriter.create<tensor::EmptyOp>(loc, retShape, elemTy);
    auto gatherOp = rewriter.create<hfusion::EmbeddingGatherOp>(
        loc, retTy, src, idx, init, bound, offsets, numels);
    rewriter.replaceOp(callOp, gatherOp);

    return success();
  }
};

/// triton indirect_store op is converted triton_indirect_store func call in
/// triton adaptor, therefore this pattern match func call of which func name
/// starts with "triton_indirect_store".
struct TritonIndirectStoreToHFusionIndirectStorePattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton_indirect_store";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    if (!funcOp) {
      return rewriter.notifyMatchFailure(
          callOp, "Called funcOp is null");
    }
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(2);
    Value offsets = callOp.getOperand(1);
    Value dst = callOp->getOperand(0);

    Value mask = nullptr;

    unsigned numOperands = callOp.getNumOperands();
    if (numOperands > 3) {
      mask = callOp.getOperand(3);
    }

    if (mask) {
      auto maskType = mlir::cast<RankedTensorType>(mask.getType());
      if (!maskType)
        return failure();
      auto elemType = maskType.getElementType();

      Value actualMask = mask;
      if (elemType.isInteger(1)) {

        Type i8Type = rewriter.getIntegerType(8);
        TensorType resultType = maskType.clone(i8Type);

        actualMask = rewriter.create<arith::ExtUIOp>(loc, resultType, mask);
      }
      mask = actualMask;
    }

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    auto parFuncOp = callOp->getParentOfType<func::FuncOp>();
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            parFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }
    auto srcArgIdx = utils::getArgumentIndex(dst);
    argIndices.push_back(srcArgIdx);
    SmallVector<Attribute> attrList;
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
    }
    parFuncOp->setAttr(directlyUsedGMArgListName,
                       ArrayAttr::get(rewriter.getContext(), attrList));

    hfusion::IndirectStoreOp indirectStoreOp =
        rewriter.create<hfusion::IndirectStoreOp>(loc, dst, offsets, src, mask);
    rewriter.replaceOp(callOp, indirectStoreOp);
    return success();
  }
};

static constexpr StringLiteral kGMAddrArgIndicesAttrName = "gm_addr_args_indices";

template <typename CustomOpT>
void setCustomOpBuiltinGMArgAttrs(Operation *op) {
  if (!isa<CustomOpT>(op))
    return;

  CustomOpT customOp = cast<CustomOpT>(op);
  auto funcOp = customOp->template getParentOfType<func::FuncOp>();
  assert(funcOp);
  if (const auto indicesAttr =
          customOp->template getAttrOfType<DenseI32ArrayAttr>(
              kGMAddrArgIndicesAttrName)) {
    for (int32_t i : indicesAttr.asArrayRef()) {
      Value operand = customOp.getOperand(static_cast<size_t>(i));
      // A GM pointer reaches the op either directly as a kernel argument or as
      // a view of one: folding an offset into it, as in `k_ptr + group *
      // stride`, makes it a memref.reinterpret_cast of the argument. Both name
      // the same argument, so resolve the view chain before asking which
      // argument it is -- getArgumentIndex itself only answers for a value that
      // is already a block argument.
      if (isa<OpResult>(operand) && isa<BaseMemRefType>(operand.getType()))
        operand = utils::tracebackMemRef(operand);
      const int64_t funcArgIdx = utils::getArgumentIndex(operand);
      // Fails only for an operand not rooted in a kernel argument at all, which
      // no GM address should be; leave those unmarked rather than indexing the
      // argument attribute list out of bounds.
      if (funcArgIdx < 0 ||
          funcArgIdx >= static_cast<int64_t>(funcOp.getNumArguments())) {
        customOp->emitWarning("gm_addr_args_indices names operand ")
            << i << ", which does not resolve to a kernel argument";
        continue;
      }
      funcOp.setArgAttr(static_cast<unsigned>(funcArgIdx),
                        hacc::KernelArgTypeAttr::name,
                        hacc::KernelArgTypeAttr::get(
                            op->getContext(), hacc::KernelArgType::kGMAddr));
    }
  }
}

} // end anonymous namespace

void AdaptTritonKernelPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ModuleOp module = getOperation();
  patterns.add<WorkaroundRTZCastPattern>(context);

  if (hacc::utils::isRegBasedArch(module)) {
    patterns
        .add<TritonPrintToHFusionPrintPattern, TritonAssertToHFusionAssertPattern,
            TritonGatherToHFusionGatherPattern, TritonCumToHFusionCumPattern,
            TritonBindSubBlockAttrToHFusionPattern,
            TritonEmbeddingGatherToHFusionEmbeddingGatherPattern,
            TritonIndirectLoadToHFusionIndirectLoadPattern,
            TritonStrideLoadToHFusionStrideLoadPattern,
            TritonStrideStoreToHFusionStrideStorePattern,
            TritonIndirectStoreToHFusionIndirectStorePattern,
            TritonGatherTToHFusionGatherTPattern,
            TritonIndexPutToHFusionIndexPutPattern,
            TritonFlipToHFusionFlipPattern, TritonSortToHFusionSortPattern,
            TritonScatterTOpToHFusionScatterTOpPattern>(patterns.getContext());
  } else {
    patterns
        .add<TritonPrintToHFusionPrintPattern, TritonAssertToHFusionAssertPattern,
            TritonGatherToHFusionGatherPattern, TritonCumToHFusionCumPattern,
            TritonBindSubBlockAttrToHFusionPattern,
            TritonSortToHFusionSortPattern, TritonFlipToHFusionFlipPattern>(
            patterns.getContext());
  }
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  if (hacc::utils::isRegBasedArch(module)) {
    module.walk([&](Operation *op) {
      setCustomOpBuiltinGMArgAttrs<hivm::CustomOp>(op);
      setCustomOpBuiltinGMArgAttrs<hivm::CustomMacroOp>(op);
    });
  }

  module.walk([&](func::FuncOp funcOp) {
    std::string globalKernelStr{"global_kernel"};
    auto attr = funcOp->getAttr(globalKernelStr);
    if (!attr) {
      return;
    }
    hacc::utils::setDeviceEntry(funcOp);
    funcOp->removeAttr(globalKernelStr);

    // Extract workspace and sync-block-lock argument info if it exists and then
    // remark HACC attribute
    markWorkspaceArgument(funcOp);
    markSyncBlockLockArgument(funcOp);
  });

  // membase-only
  if (hacc::utils::isMemBasedArch(module)) {
    module->setAttr(utils::kMemrefAsPtr, UnitAttr::get(context));
  }
}

std::unique_ptr<Pass> mlir::hfusion::createAdaptTritonKernelPass() {
  return std::make_unique<AdaptTritonKernelPass>();
}
