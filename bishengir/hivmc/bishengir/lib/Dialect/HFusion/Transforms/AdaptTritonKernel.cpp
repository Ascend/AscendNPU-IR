//===- AdaptTritonKernel.cpp - Adapt triton kernel                       -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Transform.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
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

/// triton cumsum/cumprod op is converted triton_cumsum func call in triton
/// adaptor, therefore this pattern match func call of which func name starts
/// with "triton_cumsum/triton_cumprod".
struct TritonCumToHFusionCumPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef cumsumFuncName = "triton_cumsum";
  static constexpr StringRef cumprodFuncName = "triton_cumprod";
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    auto funcName = funcOp.getSymName();
    mlir::hfusion::CumOpType cumOpType = mlir::hfusion::CumOpType::UNDEFINED;
    if (funcName.starts_with(cumsumFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMSUM;
    } else if (funcName.starts_with(cumprodFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMPROD;
    } else {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + cumsumFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value dimVals = callOp.getOperand(1);
    Value reverseAttr = callOp.getOperand(2);
    bool reverse{false};
    if (auto constOp =
            dyn_cast<arith::ConstantOp>(reverseAttr.getDefiningOp())) {
      if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
        reverse = boolAttr.getValue();
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
    } else {
      llvm_unreachable("unsupport cumulative function");
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
    rewriter.replaceOp(callOp, indirectLoadOp);
    rewriter.eraseOp(funcOp);
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
    rewriter.eraseOp(funcOp);

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
    rewriter.eraseOp(funcOp);

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
    rewriter.eraseOp(funcOp);
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
    for (auto idx : argIndices) {
      attrList.push_back(IntegerAttr::get(rewriter.getI64Type(), idx));
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
    rewriter.eraseOp(funcOp);

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
    rewriter.eraseOp(funcOp);
    return success();
  }
};

template <typename CustomOpT>
void setCustomOpBuiltinGMArgAttrs(Operation *op) {
  if (!isa<CustomOpT>(op))
    return;

  CustomOpT customOp = cast<CustomOpT>(op);
  auto funcOp = customOp->template getParentOfType<func::FuncOp>();
  assert(funcOp);
  if (const auto maybeGMAddrArgsIndices = customOp.getGMAddrArgsIndices()) {
    const auto &gmAddrArgsIndices = *maybeGMAddrArgsIndices;
    for (const size_t i : gmAddrArgsIndices) {
      const size_t funcArgIdx = static_cast<size_t>
                                (utils::getArgumentIndex(customOp.getOperand(i)));
      funcOp.setArgAttr(funcArgIdx, hacc::KernelArgTypeAttr::name,
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
  patterns
      .add<TritonPrintToHFusionPrintPattern, TritonAssertToHFusionAssertPattern,
           TritonGatherToHFusionGatherPattern, TritonCumToHFusionCumPattern,
           TritonBindSubBlockAttrToHFusionPattern,
           TritonEmbeddingGatherToHFusionEmbeddingGatherPattern,
           TritonIndirectLoadToHFusionIndirectLoadPattern,
           TritonIndirectStoreToHFusionIndirectStorePattern,
           TritonGatherTToHFusionGatherTPattern,
           TritonIndexPutToHFusionIndexPutPattern,
           TritonFlipToHFusionFlipPattern, TritonSortToHFusionSortPattern,
           TritonScatterTOpToHFusionScatterTOpPattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  module.walk([&](Operation *op) {
    setCustomOpBuiltinGMArgAttrs<hivm::CustomOp>(op);
    setCustomOpBuiltinGMArgAttrs<hivm::CustomMacroOp>(op);
  });

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
}

std::unique_ptr<Pass> mlir::hfusion::createAdaptTritonKernelPass() {
  return std::make_unique<AdaptTritonKernelPass>();
}
