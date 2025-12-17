//===------------- HIVMDecomposeOp.cpp - hivm op decompose-----------------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/RWMutex.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMDECOMPOSEOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-decompose-op"

using namespace mlir;
using namespace mlir::hivm;
using namespace utils;

namespace {

//===----------------------------------------------------------------------===//
// VCastOp Decompose
//===----------------------------------------------------------------------===//

/// Cast src to dst convert to cast src to f16 to dst, where tmp buffer or
/// tensor of f16 type is used.
static LogicalResult castSrcToF16ToDst(hivm::VCastOp &op,
                                       PatternRewriter &rewriter) {
  // One cast op converts to two cast ops.
  // 1. cast src to f16
  auto tmpVCastOp =
      castTo(rewriter, op.getLoc(), op.getSingleSrc(), op.getRoundModeAttr(),
             Float16Type::get(op.getContext()));

  // 2. cast f16 to dst
  // Note that args transpose and broadcast are used in the second cast op.
  TypeRange dstTypeRange;
  if (op.hasPureTensorSemantics()) {
    dstTypeRange = TypeRange(op.getResult());
  }

  auto transpose = op.getTransposeAttr();
  auto broadcast = op.getBroadcastAttr();
  Value srcF16 = op.hasPureTensorSemantics() ? tmpVCastOp->getResult(0)
                                             : tmpVCastOp.getSingleDst();

  hivm::VCastOp castF16ToDst = rewriter.create<hivm::VCastOp>(
      op.getLoc(), dstTypeRange, srcF16, op.getSingleDst(),
      op.getRoundModeAttr(), op.getCastAttr(), transpose, broadcast);

  rewriter.replaceOp(op, castF16ToDst);
  return success();
}

static LogicalResult castSrcToTargetTypeAndCmpiToDst(hivm::VCastOp &op,
                                                     PatternRewriter &rewriter,
                                                     Type targetElemType) {
  // 1. cast src to targetelemtype
  auto tmpVCastOp = castTo(rewriter, op.getLoc(), op.getSingleSrc(),
                           op.getRoundModeAttr(), targetElemType);

  // 2. brc targetElemType scalar zeros into tensor
  Value tmpZeros = createTmpBufferOrTensorWithTargetType(
      rewriter, op.getLoc(), op.getSingleSrc(), targetElemType);

  auto floatZero = rewriter.getFloatAttr(targetElemType, 0.0);
  auto tmpVBrcOp = brcScalar(rewriter, op.getLoc(), floatZero, tmpZeros);
  // 3. cmp f16 to dst
  TypeRange dstTypeRange =
      op.hasPureTensorSemantics() ? TypeRange(op.getResult()) : TypeRange();
  Value srctargetElemType = op.hasPureTensorSemantics()
                                ? tmpVCastOp->getResult(0)
                                : tmpVCastOp.getSingleDst();
  Value srcFZero = op.hasPureTensorSemantics() ? tmpVBrcOp->getResult(0)
                                               : tmpVBrcOp.getDst();
  llvm::SmallVector<Value> inputs{srctargetElemType, srcFZero};
  auto compareAttr =
      rewriter.getAttr<hivm::CompareModeAttr>(hivm::CompareMode::NE);
  hivm::VCmpOp cmpToDstOp = rewriter.create<hivm::VCmpOp>(
      op.getLoc(), dstTypeRange, ValueRange(inputs), op.getDst(), compareAttr,
      op.getTransposeAttr(), op.getBroadcastAttr());
  rewriter.replaceOp(op, cmpToDstOp);
  return success();
}

/// Match cast f32 to i8, rewrite to f32 to f16 to i8.
/// Match cast i4 to i8, rewrite to i4 to f16 to i8.
/// Match cast bool to i8, rewrite to bool to f16 to i8.
struct VCastLowering : public OpRewritePattern<hivm::VCastOp> {
  using OpRewritePattern<hivm::VCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics()) {
      return op.emitOpError(
          "VCastOp should have pure buffer or tensor Semantics!");
    }

    auto srcShapedType = cast<ShapedType>(op.getSingleSrc().getType());
    auto dstShapedType = cast<ShapedType>(op.getSingleDst().getType());
    auto srcElemType = srcShapedType.getElementType();
    auto dstElemType = dstShapedType.getElementType();
    const bool isF32ToI8 = srcElemType.isF32() && dstElemType.isInteger(8);
    const bool isI4ToI8 = srcElemType.isInteger(4) && dstElemType.isInteger(8);
    const bool isI8orI16ToI1 =
        (srcElemType.isInteger(8) || srcElemType.isInteger(16)) &&
        dstElemType.isInteger(1);
    const bool isI32ToI1 =
        srcElemType.isInteger(32) && dstElemType.isInteger(1);
    const bool isBoolToI8 =
        srcElemType.isInteger(1) && dstElemType.isInteger(8);
    if (isF32ToI8 || isI4ToI8 || isBoolToI8) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << srcElemType << " to "
                 << dstElemType << ", and rewrite to cast (from " << srcElemType
                 << " to f16) and cast(from f16 to " << dstElemType << ")\n ");
      return castSrcToF16ToDst(op, rewriter);
    } else if (isI8orI16ToI1) {
      return castSrcToTargetTypeAndCmpiToDst(op, rewriter,
                                             Float16Type::get(op.getContext()));
    } else if (isI32ToI1) {
      return castSrcToTargetTypeAndCmpiToDst(op, rewriter,
                                             Float32Type::get(op.getContext()));
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// VBrcOp Decompose
//===----------------------------------------------------------------------===//
static LogicalResult decomposeMultiAxesVBrcOp(hivm::VBrcOp op,
                                              PatternRewriter &rewriter) {
  llvm::ArrayRef<int64_t> brcDims = op.getBroadcastDims();
  const int brcDimSize = static_cast<int>(brcDims.size());
  auto dst = op.getDst();
  auto dstShapes = cast<ShapedType>(dst.getType()).getShape();
  Value curSrc = op.getSrc();
  hivm::VBrcOp tmpBrcOp;
  for (int i = brcDimSize - 1; i >= 0; --i) {
    // init curDstShape.
    auto srcShapes = cast<ShapedType>(curSrc.getType()).getShape();
    SmallVector<int64_t> curDstShapes(dstShapes.size(), 0);
    for (size_t shape = 0; shape < dstShapes.size(); shape++) {
      curDstShapes[shape] = srcShapes[shape];
    }
    curDstShapes[brcDims[i]] = dstShapes[brcDims[i]];
    // create curDst. Last brc use origin dst.
    Value curDst;
    if (i > 0) {
      curDst = createTmpBufferOrTensorWithTargetType(
          rewriter, op.getLoc(), dst, getElementTypeOrSelf(dst), curDstShapes);
    } else {
      curDst = op.getDst();
    }
    // create brcop.
    auto singleBrcDim =
        rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(brcDims[i])});
    auto curDstType = curDst.getType();
    TypeRange resTypeRange =
        op.hasPureTensorSemantics() ? TypeRange(curDstType) : TypeRange();
    tmpBrcOp = rewriter.create<hivm::VBrcOp>(op.getLoc(), resTypeRange, curSrc,
                                             curDst, singleBrcDim);
    // Update curSrc for next use in loop
    curSrc = op.hasPureTensorSemantics() ? tmpBrcOp->getResult(0)
                                         : tmpBrcOp.getDst();
  }
  rewriter.replaceOp(op, tmpBrcOp);
  return success();
}

/// Decompose pattern for VBrcOp.
///
/// Decompose VBrcOp with multiple broadcast axes to multiple VBrcOp with a
/// single broadcast axis. broadcast from the inner to the outer axis.
/// e.g.
///   %src = memref.alloc() : memref<1x1x10x1xi16>
///   %dst = memref.alloc() : memref<16x8x10x32xi16>
///   hivm.hir.vbrc ins(%src : memref<1x1x10x1xi16>) outs(%dst :
///   memref<16x8x10x32xi16>) broadcast_dims = [0, 1, 3]
/// converts to
///   %src = memref.alloc() : memref<1x1x10x1xi16>
///   %dst = memref.alloc() : memref<16x8x10x32xi16>
///   %tmp0 = memref.alloc() : memref<1x1x10x32xi16>
///   hivm.hir.vbrc ins(%src : memref<1x1x10x1xi16>)
///     outs(%tmp0 : memref<1x1x10x32xi16>) broadcast_dims = [3]
///   %tmp1 = memref.alloc() : memref<1x8x10x32xi16>
///   hivm.hir.vbrc ins(%tmp0 : memref<1x1x10x32xi16>)
///     outs(%tmp1 : memref<1x8x10x32xi16>) broadcast_dims = [1]
///   hivm.hir.vbrc ins(%tmp1 : memref<1x8x10x32xi16>)
///     outs(%dst : memref<16x8x10x32xi16>) broadcast_dims = [0]
struct MultiAxesVBrcLowering : public OpRewritePattern<hivm::VBrcOp> {
  using OpRewritePattern<hivm::VBrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VBrcOp op,
                                PatternRewriter &rewriter) const override {
    const int brcDimSize = static_cast<int>(op.getBroadcastDims().size());
    const int minToDecompose = 2;
    if (brcDimSize < minToDecompose) {
      return failure();
    }

    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics()) {
      return op.emitOpError(
          "hivm::VBrcOp should have pure buffer or tensor Semantics!");
    }

    return decomposeMultiAxesVBrcOp(op, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// VRecOp High Precision Lowering
//===----------------------------------------------------------------------===//

/// VRecOp has a relatively low precision. Decompose it to VBrcOp + VDivOp.
struct VRecOpHighPrecisionLowering : public OpRewritePattern<hivm::VRecOp> {
  using OpRewritePattern<hivm::VRecOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VRecOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return op.emitOpError("VRecOp should have pure buffer semantics!");
    }

    auto *init = op.getDpsInitOperand(0);
    auto *input = op.getDpsInputOperand(0);
    auto shapedType = cast<ShapedType>(input->get().getType());
    assert(shapedType);

    auto elementType = shapedType.getElementType();
    auto constOneValue = getConstOneValue(rewriter, elementType, op->getLoc());

    auto brcDst = init->get();
    if (input->get() == init->get()) {
      // if input and init are same variable, brcOp need define an empty alloc
      // as dst buffer to store temp result
      auto allocMaybe = createTempAllocOpForBrc(rewriter, brcDst, op->getLoc());
      if (!allocMaybe.has_value()) {
        return failure();
      }
      brcDst = allocMaybe.value();
    }
    // Use vrec op's dest buffer to store vbrc op's result
    auto brcOp = rewriter.create<hivm::VBrcOp>(
        op.getLoc(), /*result=*/op.getResultTypes(),
        /*src=*/constOneValue,
        /*dst=*/brcDst,
        /*broadcast_dims=*/
        rewriter.getDenseI64ArrayAttr(
            {})); // broadcast dims should be empty for scalar src

    rewriter.replaceOpWithNewOp<hivm::VDivOp>(
        op,
        /*result=*/op.getResultTypes(),
        /*src=*/ValueRange{brcOp.getDst(), input->get()},
        /*dst=*/ValueRange{init->get()});
    return success();
  }

private:
  Value getConstOneValue(PatternRewriter &rewriter, Type elementType,
                         Location loc) const {
    return llvm::TypeSwitch<Type, Value>(elementType)
        .Case([&](IntegerType intType) {
          return rewriter.create<arith::ConstantIntOp>(loc, 1, intType);
        })
        .Case([&](FloatType floatType) {
          return rewriter.create<arith::ConstantOp>(
              loc, rewriter.getFloatAttr(floatType, 1.0));
        })
        .Default([](Type) {
          llvm_unreachable("Unsupported type!");
          return Value{};
        });
  }

  std::optional<Value> createTempAllocOpForBrc(PatternRewriter &rewriter,
                                               Value brcDst,
                                               Location loc) const {
    auto maybeAlloc = utils::tracebackMemRefToAlloc(brcDst);
    if (!maybeAlloc.has_value()) {
      return std::nullopt;
    }

    memref::AllocOp allocOp = *maybeAlloc;
    auto memRefType = cast<MemRefType>(brcDst.getType());
    if (memRefType.hasStaticShape()) {
      return rewriter
          .create<memref::AllocOp>(loc, memRefType, allocOp.getAlignmentAttr())
          .getMemref();
    }
    auto allocType = cast<MemRefType>(allocOp.getResult().getType());
    std::optional<int64_t> totalStaticSize =
        utils::getStaticTotalSize(allocType.getShape());
    if (!totalStaticSize.has_value()) {
      return std::nullopt;
    }

    // Required tmpRefType.
    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, loc, brcDst);
    SmallVector<int64_t> staticShape;
    SmallVector<Value> dynamicSizes;
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticShape);
    MemRefType tmpRefType =
        MemRefType::get(staticShape, memRefType.getElementType(),
                        memRefType.getLayout(), memRefType.getMemorySpace());

    // Required intermediate allocOp
    Value tmpAlloc =
        rewriter.create<memref::AllocOp>(loc, tmpRefType, dynamicSizes);
    memref::ViewOp viewOp = mlir::utils::createAllocWithSettingBufferSize(
        tmpAlloc.getDefiningOp(), totalStaticSize.value(), rewriter);
    return viewOp.getResult();
  }
};

//===----------------------------------------------------------------------===//
// SynBlockOpLowering
//===----------------------------------------------------------------------===//
void appendBlockSyncOperations(PatternRewriter &rewriter, Location loc,
                               TCoreTypeAttr tCoreTypeAttr,
                               hivm::SyncBlockInstrModeAttr modeAttr,
                               PipeAttr tpipe, PipeAttr pipe,
                               IntegerAttr flagID, Value fftsBaseAddr) {
  rewriter.create<SyncBlockSetOp>(loc, tCoreTypeAttr, tpipe, pipe, flagID,
                                  fftsBaseAddr, modeAttr);
  rewriter.create<SyncBlockWaitOp>(loc, tCoreTypeAttr, tpipe, pipe, flagID);
}

struct SyncBlockOpLowering : public OpRewritePattern<SyncBlockOp> {
  using OpRewritePattern<SyncBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SyncBlockOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op->getContext();
    Value fftsBaseAddr = op.getFftsBaseAddr();
    auto syncBlockMode = op.getSyncBlockModeAttr().getSyncMode();
    IntegerAttr flagID = op.getFlagIdAttr();
    if (syncBlockMode == SyncBlockMode::BARRIER_CUBE ||
        syncBlockMode == SyncBlockMode::BARRIER_VECTOR) {
      rewriter.create<PipeBarrierOp>(loc, PipeAttr::get(ctx, PIPE::PIPE_ALL));
    } else if (syncBlockMode == SyncBlockMode::ALL_CUBE) {
      auto pipe = op.getTcubePipeAttr();
      appendBlockSyncOperations(
          rewriter, loc, TCoreTypeAttr::get(ctx, TCoreType::CUBE),
          SyncBlockInstrModeAttr::get(
              ctx, SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION),
          pipe, pipe, flagID, fftsBaseAddr);
    } else if (syncBlockMode == SyncBlockMode::ALL_VECTOR) {
      auto pipe = op.getTvectorPipeAttr();
      appendBlockSyncOperations(
          rewriter, loc, TCoreTypeAttr::get(ctx, TCoreType::VECTOR),
          SyncBlockInstrModeAttr::get(
              ctx, SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION),
          pipe, pipe, flagID, fftsBaseAddr);
    } else if (syncBlockMode == SyncBlockMode::ALL) {
      auto tcubePipe = op.getTcubePipeAttr();
      auto tvectorPipe = op.getTvectorPipeAttr();
      appendBlockSyncOperations(
          rewriter, loc, TCoreTypeAttr::get(ctx, TCoreType::CUBE),
          SyncBlockInstrModeAttr::get(
              ctx, SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION),
          tcubePipe, tcubePipe, flagID, fftsBaseAddr);
      appendBlockSyncOperations(
          rewriter, loc, TCoreTypeAttr::get(ctx, TCoreType::CUBE_OR_VECTOR),
          SyncBlockInstrModeAttr::get(
              ctx, SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION),
          tcubePipe, tvectorPipe, flagID, fftsBaseAddr);
      appendBlockSyncOperations(
          rewriter, loc, TCoreTypeAttr::get(ctx, TCoreType::VECTOR),
          SyncBlockInstrModeAttr::get(
              ctx, SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION),
          tvectorPipe, tvectorPipe, flagID, fftsBaseAddr);
    } else {
      llvm_unreachable("unsupported sync mode");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct VCmpOpLowering : public OpRewritePattern<VCmpOp> {
  using OpRewritePattern<VCmpOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(VCmpOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          op, "VCmpOp should have pure buffer Semantics!");
    }

    Value src = op->getOperand(0);
    Type srcType = src.getType();
    if (!isa<MemRefType>(srcType) && !isa<TensorType>(srcType)) {
      return rewriter.notifyMatchFailure(
          op, "VCmpOp first operand should be memref or tensor type!");
    }

    if (!getElementTypeOrSelf(srcType).isInteger()) {
      return failure();
    }

    // TODO: replace with a unified interface to decide whether to decompose
    CompareMode cmpMode = op.getCompareMode();
    if (getElementTypeOrSelf(srcType).isInteger(32) &&
        (cmpMode == CompareMode::NE || cmpMode == CompareMode::EQ)) {
      return failure();
    }

    // output type need to be i1
    Value dst = op.getDst()[0];
    auto dstElemType = getElementTypeOrSelf(dst);
    if (!dstElemType.isInteger(1)) {
      return failure();
    }

    // Since the store does not support i1 handling during the
    // decomposeVecToScalar process, it can only be moved through i8 and then
    // cast back to i1.
    // step 1: Add required intermediate allocOp
    // and create annotation mark if it is dynamic size.
    auto tmpAlloc = createTmpBufferOrTensorWithTargetType(
        rewriter, op.getLoc(), src, rewriter.getIntegerType(8));

    // step 2: cast i8 to i1
    rewriter.setInsertionPointAfter(op);
    hivm::RoundMode rounding = mlir::utils::selectRoundMode<hivm::RoundMode>(
        IntegerType::get(op.getContext(), 8),
        IntegerType::get(op.getContext(), 1));
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
    rewriter.create<hivm::VCastOp>(op.getLoc(), TypeRange(op.getODSResults(0)),
                                   tmpAlloc, op.getDst(), roundingAttr,
                                   hivm::TypeFnAttr{});

    // step 3: VCmpOp output memref i1 to i8
    rewriter.modifyOpInPlace(
        op, [&]() { op.getDpsInitsMutable()[0].assign(tmpAlloc); });
    return success();
  }
};

template <typename ExtOp>
struct DecomposeI32ScalarExtOp : public OpRewritePattern<ExtOp> {
  using OpRewritePattern<ExtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtOp op,
                                PatternRewriter &rewriter) const final {
    // The type of operand is i32(scalar)
    Value oper = op->getOperand(0);
    if (!oper.getType().isInteger(32)) {
      return failure();
    }
    if constexpr (std::is_same<arith::MulUIExtendedOp, ExtOp>::value) {
      // cast i32 inputs to i64
      auto lhsExtSIOp = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getI64Type(), op.getLhs());
      auto rhsExtSIOp = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getI64Type(), op.getRhs());
      // mul i64
      auto mulI64Res =
          rewriter.create<arith::MulIOp>(op.getLoc(), lhsExtSIOp, rhsExtSIOp);
      // get low 32 bits of a 64-bit number
      auto constThirtyTwo = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), 32, rewriter.getI64Type());
      auto shLIOp = rewriter.create<arith::ShLIOp>(op.getLoc(), mulI64Res,
                                                   constThirtyTwo);
      auto shRSIOp1 = rewriter.create<arith::ShRSIOp>(
          op.getLoc(), shLIOp.getResult(), constThirtyTwo);
      auto resLow32Bits =
          rewriter
              .create<arith::TruncIOp>(op.getLoc(), rewriter.getI32Type(),
                                       shRSIOp1.getResult())
              .getResult();
      // get high 32 bits of a 64-bit number
      auto shRSIOp = rewriter.create<arith::ShRSIOp>(op.getLoc(), mulI64Res,
                                                     constThirtyTwo);
      auto resHigh32Bits =
          rewriter
              .create<arith::TruncIOp>(op.getLoc(), rewriter.getI32Type(),
                                       shRSIOp.getResult())
              .getResult();
      rewriter.replaceOp(op, {resLow32Bits, resHigh32Bits});
    }
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// VReduceOp Any Lowering
// ===----------------------------------------------------------------------===//

struct VReduceAnyLowering : public OpRewritePattern<hivm::VReduceOp> {
  using OpRewritePattern<hivm::VReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto reduceOpArith = op.getArithAttr();
    auto reduceOpAttr = reduceOpArith.getReduceOp();
    if (reduceOpAttr != hivm::ReduceOperation::any) {
      return failure();
    }

    // step 1: cast i1 to f16
    hivm::RoundMode rounding = hivm::RoundMode::RINT;
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);

    auto tmpVCastI1ToF16Op =
        castTo(rewriter, op.getLoc(), op.getSrc(), roundingAttr,
               Float16Type::get(op.getContext()));

    // step 2: reduce_max -> 0/1 (fp16)
    auto reduceMaxOpAttr =
        hivm::ReduceOpAttr::get(op.getContext(), hivm::ReduceOperation::max);
    Value reduceMaxInit = createTmpBufferOrTensorWithTargetType(
        rewriter, op.getLoc(), op.getDstValue(),
        Float16Type::get(op.getContext()));

    auto reduceMaxInitType = reduceMaxInit.getType();
    TypeRange resTypeRange = op.hasPureTensorSemantics()
                                 ? TypeRange(reduceMaxInitType)
                                 : TypeRange();
    auto tmpVCastI1ToF16OpSrc = op.hasPureTensorSemantics()
                                    ? tmpVCastI1ToF16Op->getResult(0)
                                    : tmpVCastI1ToF16Op.getSingleDst();

    auto tmpReduceMaxOp = rewriter.create<hivm::VReduceOp>(
        op.getLoc(), resTypeRange, tmpVCastI1ToF16OpSrc,
        ValueRange{reduceMaxInit}, reduceMaxOpAttr, op.getReduceDimsAttr());

    // step 3: cast f16 to i1
    // TODO: after add f16 to i1 hivm decompose, rewrite here to cast to i1
    // directly

    auto tmpVCastF16ToI8OpSrc = op.hasPureTensorSemantics()
                                    ? tmpReduceMaxOp->getResult(0)
                                    : tmpReduceMaxOp.getDstValue();

    auto tmpVCastF16ToI8Op =
        castTo(rewriter, op.getLoc(), tmpVCastF16ToI8OpSrc, roundingAttr,
               IntegerType::get(op.getContext(), 8));

    auto tmpVCastI8ToI1Opsrc = op.hasPureTensorSemantics()
                                   ? tmpVCastF16ToI8Op->getResult(0)
                                   : tmpVCastF16ToI8Op.getSingleDst();

    TypeRange dstTypeRange = op.hasPureTensorSemantics()
                                 ? TypeRange(op.getODSResults(0))
                                 : TypeRange();

    auto tmpVCastI8ToI1Op = rewriter.create<hivm::VCastOp>(
        op.getLoc(), dstTypeRange, tmpVCastI8ToI1Opsrc, op.getDst(),
        roundingAttr, hivm::TypeFnAttr{});

    rewriter.replaceOp(op, tmpVCastI8ToI1Op);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// VReduceOp ALL Lowering
// ===----------------------------------------------------------------------===//

struct VReduceAllLowering : public OpRewritePattern<hivm::VReduceOp> {
  using OpRewritePattern<hivm::VReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto reduceOpArith = op.getArithAttr();
    auto reduceOpAttr = reduceOpArith.getReduceOp();
    if (reduceOpAttr != hivm::ReduceOperation::all) {
      return failure();
    }

    // step 1: cast i1 to f16
    hivm::RoundMode rounding = hivm::RoundMode::RINT;
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);

    auto tmpVCastI1ToF16Op =
        castTo(rewriter, op.getLoc(), op.getSrc(), roundingAttr,
               Float16Type::get(op.getContext()));

    // step 2: reduce_min -> 0/1 (fp16)
    auto reduceMinOpAttr =
        hivm::ReduceOpAttr::get(op.getContext(), hivm::ReduceOperation::min);
    Value reduceMinInit = createTmpBufferOrTensorWithTargetType(
        rewriter, op.getLoc(), op.getDstValue(),
        Float16Type::get(op.getContext()));

    auto reduceMinInitType = reduceMinInit.getType();
    TypeRange resTypeRange = op.hasPureTensorSemantics()
                                 ? TypeRange(reduceMinInitType)
                                 : TypeRange();
    auto tmpVCastI1ToF16OpSrc = op.hasPureTensorSemantics()
                                    ? tmpVCastI1ToF16Op->getResult(0)
                                    : tmpVCastI1ToF16Op.getSingleDst();

    auto tmpReduceMinOp = rewriter.create<hivm::VReduceOp>(
        op.getLoc(), resTypeRange, tmpVCastI1ToF16OpSrc,
        ValueRange{reduceMinInit}, reduceMinOpAttr, op.getReduceDimsAttr());

    // step 3: cast f16 to i1
    // TODO: after add f16 to i1 hivm decompose, rewrite here to cast to i1
    // directly

    auto tmpVCastF16ToI8OpSrc = op.hasPureTensorSemantics()
                                    ? tmpReduceMinOp->getResult(0)
                                    : tmpReduceMinOp.getDstValue();

    auto tmpVCastF16ToI8Op =
        castTo(rewriter, op.getLoc(), tmpVCastF16ToI8OpSrc, roundingAttr,
               IntegerType::get(op.getContext(), 8));

    auto tmpVCastI8ToI1Opsrc = op.hasPureTensorSemantics()
                                   ? tmpVCastF16ToI8Op->getResult(0)
                                   : tmpVCastF16ToI8Op.getSingleDst();

    TypeRange dstTypeRange = op.hasPureTensorSemantics()
                                 ? TypeRange(op.getODSResults(0))
                                 : TypeRange();

    auto tmpVCastI8ToI1Op = rewriter.create<hivm::VCastOp>(
        op.getLoc(), dstTypeRange, tmpVCastI8ToI1Opsrc, op.getDst(),
        roundingAttr, hivm::TypeFnAttr{});

    rewriter.replaceOp(op, tmpVCastI8ToI1Op);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// VReduceInitInitializing
// ===----------------------------------------------------------------------===//

struct VReduceInitInitializing : public OpRewritePattern<hivm::VReduceOp> {
  using OpRewritePattern<hivm::VReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return failure();
    }

    if (!op.shouldLowerToScalarLoops()) {
      return failure();
    }

    static constexpr llvm::StringLiteral kAlreadyInitalizeInit =
        "already_initialize_init";
    if (op->hasAttr(kAlreadyInitalizeInit)) {
      return failure();
    }

    // initialize reduce init operand
    auto dstType = getElementTypeOrSelf(op.getDstValue().getType());
    TypedAttr initScalr;
    if (dstType.isInteger()) {
      initScalr = dyn_cast<IntegerAttr>(op.getInit());
    } else {
      initScalr = dyn_cast<FloatAttr>(op.getInit());
    }

    if (initScalr) {
      brcScalar(rewriter, op.getLoc(), initScalr, op.getDpsInits()[0]);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op->setAttr(kAlreadyInitalizeInit, UnitAttr::get(op->getContext()));
    });

    return success();
  }
};

// TODO : add platform information
static bool isHWSupportedAbs(hivm::VAbsOp op) {
  Value src = op.getSrc()[0];
  Type elemType = getElementTypeOrSelf(src.getType());
  return elemType.isInteger(16) || elemType.isInteger(32);
}

struct VAbsIntegerLowering : public OpRewritePattern<hivm::VAbsOp> {
  using OpRewritePattern<hivm::VAbsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VAbsOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return failure();
    }

    if (!isHWSupportedAbs(op)) {
      return failure();
    }

    Value src = op.getSrc()[0];
    Value dst = op.getDst()[0];
    if (!isa<MemRefType>(dst.getType())) {
      return failure();
    }

    // decompose abs to: vmax(vadd(vnot(src), one), src)
    Type elemType = getElementTypeOrSelf(src.getType());
    auto one = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, elemType);
    auto vnotInit = createTmpBufferOrTensorWithTargetType(
        rewriter, op->getLoc(), src, elemType);

    auto vnot = rewriter.create<hivm::VNotOp>(op->getLoc(),
                                              /*result=*/TypeRange{},
                                              /*src=*/ValueRange({src}),
                                              /*dst=*/ValueRange({vnotInit}));
    auto vaddInit = createTmpBufferOrTensorWithTargetType(
        rewriter, op->getLoc(), src, elemType);
    auto vadd = rewriter.create<hivm::VAddOp>(
        op->getLoc(), /*result=*/TypeRange{},
        /*src=*/ValueRange{vnot.getDst()[0], one->getResult(0)},
        /*dst=*/ValueRange{vaddInit});
    auto vmax =
        rewriter.create<hivm::VMaxOp>(op->getLoc(), /*result=*/TypeRange{},
                                      /*src=*/ValueRange{src, vadd.getDst()[0]},
                                      /*dst=*/ValueRange{dst});
    rewriter.replaceOp(op, vmax);
    return success();
  }
};

static bool operationOnScalar(Operation *op) {
  return llvm::all_of(op->getOperandTypes(), [](Type type) {
    return type.isSignlessIntOrIndexOrFloat();
  });
}

template <typename CastOp>
struct DecomposeCastScalarToVecOp : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    if (!operationOnScalar(op)) {
      return failure();
    }

    // create tensor (shape is 1), and store scalar into the tensor
    Value scalarSrc = op.getOperand();
    const Type srcType = scalarSrc.getType();
    const Type dstType = op.getType();
    if (isHWSupportedCast(srcType, dstType)) {
      return failure();
    }

    auto loc = op.getLoc();
    Value src = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, srcType));
    createSinglePointStore(rewriter, loc, scalarSrc, src);

    // cast src tensor to target tensor
    auto resType = op.getType();
    Value castInit = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, resType));
    hivm::VCastOp castOp;

    // TODO: only do scalar to vec and put hivm cast vec op decomposition into
    // createHIVMAggregatedDecomposeOpPass pass.
    if ((srcType.isInteger(32) || srcType.isInteger(64)) && dstType.isBF16()) {
      auto roundingAttr =
          rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);
      castOp = castTo(rewriter, loc, src, roundingAttr, rewriter.getF32Type());
      castOp = castTo(rewriter, loc, castOp.getSingleDst(), roundingAttr,
                      getElementTypeOrSelf(resType));
    } else {
      auto roundingAttr =
          rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);
      castOp = castTo(rewriter, loc, src, roundingAttr,
                      getElementTypeOrSelf(resType));
    }

    // load target tensor to scalar
    auto loadOp = createSinglePointLoad(rewriter, loc, castOp.getSingleDst());
    rewriter.replaceOp(op, loadOp);
    return success();
  }

private:
  bool isHWSupportedCast(Type srcType, Type dstType) const {
    if ((srcType.isInteger(32) && dstType.isF32()) ||
        (srcType.isF32() && dstType.isInteger(32)) ||
        (srcType.isF16() && dstType.isF32()) ||
        (srcType.isF32() && dstType.isF16())) {
      return true;
    }

    // decompose to vector when dstType or srcType is BF16
    return !dstType.isBF16() && !srcType.isBF16();
  }
};

template <>
struct DecomposeCastScalarToVecOp<arith::TruncFOp>
    : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    if (!operationOnScalar(op)) {
      return failure();
    }

    // create tensor (shape is 1), and store scalar into the tensor
    Value scalarSrc = op.getOperand();
    if (!isa<BFloat16Type>(getElementTypeOrSelf(scalarSrc.getType())) &&
        !isa<BFloat16Type>(getElementTypeOrSelf(op.getType()))) {
      return failure();
    }
    auto loc = op.getLoc();
    Value src = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, scalarSrc.getType()));
    createSinglePointStore(rewriter, loc, scalarSrc, src);

    // cast src tensor to target tensor
    auto resType = op.getType();
    auto roundingAttr =
        rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::ROUND);
    hivm::VCastOp castOp =
        castTo(rewriter, loc, src, roundingAttr, getElementTypeOrSelf(resType));

    // load target tensor to scalar
    auto loadOp = createSinglePointLoad(rewriter, loc, castOp.getSingleDst());
    rewriter.replaceOp(op, loadOp);
    return success();
  }
};

template <typename SCALAROP, typename VECOP>
struct DecomposeScalarOpToVecOp : public OpRewritePattern<SCALAROP> {
  using OpRewritePattern<SCALAROP>::OpRewritePattern;
  LogicalResult matchAndRewrite(SCALAROP op,
                                PatternRewriter &rewriter) const override {
    if (!operationOnScalar(op)) {
      return failure();
    }

    // create tensor (shape is 1), and store scalar into the tensor
    Value scalarSrc = op.getOperand();
    auto loc = op.getLoc();
    Value src = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, scalarSrc.getType()));
    createSinglePointStore(rewriter, loc, scalarSrc, src);

    // create vector op
    Value dst = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, op.getType()));
    auto vecOp = rewriter.create<VECOP>(op->getLoc(),
                                        /*result=*/TypeRange{},
                                        /*src=*/ValueRange({src}),
                                        /*dst=*/ValueRange({dst}));

    // load target tensor to scalar
    auto loadOp = createSinglePointLoad(rewriter, loc, vecOp.getDst()[0]);
    rewriter.replaceOp(op, loadOp);
    return success();
  }
};

struct DecomposeVSubScalarOp : public OpRewritePattern<hivm::VSubOp> {
  using OpRewritePattern<hivm::VSubOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VSubOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics())
      return failure();

    Value scalarSrc = op.getSrc()[1];
    Type scalarType = scalarSrc.getType();
    if (!scalarType.isIntOrFloat())
      return failure();

    Location loc = op.getLoc();
    auto zeroAttr = rewriter.getZeroAttr(scalarType);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto newSrc =
        llvm::TypeSwitch<Type, Value>(scalarType)
            .Case([&](IntegerType) {
              return rewriter.create<arith::SubIOp>(loc, zero, scalarSrc);
            })
            .Case([&](FloatType) {
              return rewriter.create<arith::SubFOp>(loc, zero, scalarSrc);
            })
            .Default([](Type) {
              llvm_unreachable("Unsupported type!");
              return Value{};
            });

    auto vadd = rewriter.create<hivm::VAddOp>(
        loc, /*result=*/TypeRange{},
        /*src=*/ValueRange{op.getSrc()[0], newSrc},
        /*dst=*/ValueRange{op.getDst()[0]});
    rewriter.replaceOp(op, vadd);
    return success();
  }
};

class DecomposeVDeinterleaveOp
    : public OpRewritePattern<hivm::VDeinterleaveOp> {
  using OpRewritePattern<hivm::VDeinterleaveOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VDeinterleaveOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return failure();
    }

    if (op.getIndexMode() != DeinterleaveMode::ALL_CHANNELS)
      return failure();

    auto dst = op.getDst();
    int64_t channelNum = op.getDeInterLeaveChannelNum();
    assert(dst.size() == channelNum);

    for (int i = 0; i < channelNum; ++i) {
      Value curDst = dst[i];
      rewriter.create<hivm::VDeinterleaveOp>(
          op.getLoc(), TypeRange{}, op.getSrc(), curDst, channelNum,
          symbolizeDeinterleaveMode(i).value());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class AtomicStoreOpLowering : public OpRewritePattern<hivm::StoreOp> {
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType = getElementTypeOrSelf(op.getDstOperandType());
    auto atomicKind = op.getAtomicKind();
    if (op.isAtomic()) {
      if (elemType.isInteger(64))
        return decomposeEltwiseAtomic(op, rewriter, loc);
      if (*atomicKind == hivm::AtomicKind::UMAX ||
          *atomicKind == hivm::AtomicKind::UMIN) {
        assert(elemType.getIntOrFloatBitWidth() == 8);
        return decomposeEltwiseAtomic(op, rewriter, loc, /*isUnsigned=*/true);
      }
    }
    if (!op.isSWAtomic()) {
      return failure();
    }
    switch (atomicKind.value()) {
    case hivm::AtomicKind::AND:
    case hivm::AtomicKind::OR:
    case hivm::AtomicKind::XOR:
      return decomposeEltwiseAtomic(op, rewriter, loc);
    default:
      return failure();
    }
  }

private:
  /// implement atomic by software way
  /// e.g.store ins(% res_ub) outs(% res_gm) with atomic XOR is converted to
  /// % lock_var = create_sync_lock()
  /// sync_block_lock(% lock_var)
  ///
  /// % tmp0_ub = load % res_gm % tmp0_ub =
  /// % tmp0_ub xor % res_ub
  /// store ins(% tmp0_ub) outs(% res_gm)
  ///
  /// sync_block_unlock(% lock_var)
  LogicalResult decomposeEltwiseAtomic(hivm::StoreOp op,
                                       PatternRewriter &rewriter, Location loc,
                                       bool isUnsigned = false) const {
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());

    // 1. insert sync_block_lock
    rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);

    // 2. create tmp memref alloc and load dst to tmp
    auto src = op.getSrc();
    auto tmpUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src);

    auto dst = op.getDst();
    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, dst, tmpUB);

    if (isUnsigned) {
      hivm::RoundMode rounding = mlir::utils::selectRoundMode<hivm::RoundMode>(
          getElementTypeOrSelf(dst), rewriter.getF16Type());
      auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
      hivm::TypeFn typeFn = hivm::TypeFn::cast_unsigned;
      auto typeFnAttr = rewriter.getAttr<hivm::TypeFnAttr>(typeFn);
      src = castTo(rewriter, src.getLoc(), src, roundingAttr,
                   rewriter.getF16Type(), typeFnAttr)
                .getSingleDst();
      tmpUB = castTo(rewriter, src.getLoc(), tmpUB, roundingAttr,
                     rewriter.getF16Type(), typeFnAttr)
                  .getSingleDst();
    }

    // 3. do eltwise vv between src and tmp(and/or/xor)
    auto resUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src);
    auto eltwiseOp = createEltwiseOpByAtomicKind(
        rewriter, loc, TypeRange{}, ValueRange{src, tmpUB}, ValueRange{resUB},
        op.getAtomicKind().value());
    if (!eltwiseOp.has_value()) {
      return op.emitError("not support block-sync atomic kind!!");
    }

    if (isUnsigned) {
      hivm::RoundMode rounding = mlir::utils::selectRoundMode<hivm::RoundMode>(
          rewriter.getF16Type(), getElementTypeOrSelf(dst));
      auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
      hivm::TypeFn typeFn = hivm::TypeFn::cast_unsigned;
      auto typeFnAttr = rewriter.getAttr<hivm::TypeFnAttr>(typeFn);
      resUB = castTo(rewriter, src.getLoc(), resUB, roundingAttr,
                     getElementTypeOrSelf(dst), typeFnAttr)
                  .getSingleDst();
    }

    // 4. store tmp to dst
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, resUB, dst);

    // 5. insert sync_block_unlock
    rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);

    rewriter.eraseOp(op);
    return success();
  }
};

/// implement atomic cas in software way
/// e.g. hivm.hir.atomic_cas ins(%src0_ub, src1_ub) outs(%dst_gm) is converted
/// to
/// 1. %lock_var = create_sync_lock()
/// 2. sync_block_lock(%lock_var)
/// 3. %tmp0_ub = load(%dst_gm)
/// 4. %cond = vcmp(tmp0_ub, src0_ub)
/// 5. %tmp0_ub = vsel(%cond, src1_ub, tmp0_ub)
/// 6. %dst_gm = store(%tmp0_ub)
/// 7. sync_block_unlock(%lock_var)
class AtomicCasOpLowering : public OpRewritePattern<hivm::AtomicCasOp> {
  using OpRewritePattern<hivm::AtomicCasOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::AtomicCasOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());

    // insert sync_block_lock
    rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);

    // step1: load old val in gm to ub
    // create memref.alloc op
    auto src0 = op.getSrc()[0];
    auto tmpUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src0);

    auto dst = op.getDst();
    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, dst, tmpUB);

    // step2: condition = vcmp(dst, expected_val)
    //        dst = vsel(condition, new_val, dst)
    // create condition alloc
    auto condUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src0,
                                                        rewriter.getI1Type());
    auto compareAttr =
        rewriter.getAttr<hivm::CompareModeAttr>(hivm::CompareMode::EQ);
    auto elemType = getElementTypeOrSelf(src0);
    auto src1 = op.getSrc()[1];
    hivm::RoundMode rounding = hivm::RoundMode::RINT;
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
    if (elemType.isInteger(8)) {
      src0 = castTo(rewriter, src0.getLoc(), src0, roundingAttr,
                       rewriter.getF16Type()).getSingleDst();
      src1 = castTo(rewriter, src1.getLoc(), src1, roundingAttr,
                       rewriter.getF16Type()).getSingleDst();
      tmpUB = castTo(rewriter, tmpUB.getLoc(), tmpUB, roundingAttr,
                        rewriter.getF16Type()).getSingleDst();
    } else if (elemType.isBF16()) {
      src0 = castTo(rewriter, src0.getLoc(), src0, roundingAttr,
                       rewriter.getF32Type()).getSingleDst();
      src1 = castTo(rewriter, src1.getLoc(), src1, roundingAttr,
                       rewriter.getF32Type()).getSingleDst();
      tmpUB = castTo(rewriter, tmpUB.getLoc(), tmpUB, roundingAttr,
                        rewriter.getF32Type()).getSingleDst();
    }
    rewriter.create<hivm::VCmpOp>(op.getLoc(), TypeRange(),
                                  ValueRange({tmpUB, src0}),
                                  Value(condUB), compareAttr);

    auto resUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src0);
    rewriter.create<hivm::VSelOp>(op.getLoc(), TypeRange(),
                                  ValueRange({condUB, src1, tmpUB}),
                                  ValueRange({resUB}), Value());
    if (elemType.isInteger(8)) {
      resUB = castTo(rewriter, resUB.getLoc(), resUB, roundingAttr,
                     rewriter.getI8Type()).getSingleDst();
    } else if (elemType.isBF16()) {
      resUB = castTo(rewriter, resUB.getLoc(), resUB, roundingAttr,
                     rewriter.getBF16Type()).getSingleDst();
    }

    // step3: store res_ub to dst
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, resUB, dst);

    rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    rewriter.eraseOp(op);
    return success();
  }
};

/// implement atomic xchg in software way
/// e.g. hivm.hir.atomic_xchg ins(%src_ub) outs(%dst_gm) is converted to
/// 1. %lock_var = create_sync_lock()
/// 2. sync_block_lock(%lock_var)
/// 3. %tmp0_ub = load(%dst_gm)
/// 4. %dst_gm = store(%src_ub)
/// 5. %src_ub = copy(%tmp0_ub)
/// 7. sync_block_unlock(%lock_var)
class AtomicXchgOpLowering : public OpRewritePattern<hivm::AtomicXchgOp> {
  using OpRewritePattern<hivm::AtomicXchgOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::AtomicXchgOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());

    // insert sync_block_lock
    rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);

    // step1: load old val in dst gm to ub
    auto src = op.getSrc()[0];
    auto tmpUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src);

    auto dst = op.getDst();
    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, dst, tmpUB);

    // step2: store new val to dst gm
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, src, dst);

    // step3: copy old val to src ub
    rewriter.create<hivm::CopyOp>(loc, TypeRange{}, tmpUB, src);

    rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    rewriter.eraseOp(op);
    return success();
  }
};

struct HIVMDecomposeOpPass
    : public impl::HIVMDecomposeOpBase<HIVMDecomposeOpPass> {
  void runOnOperation() override;
};
} // namespace

void HIVMDecomposeOpPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns
      .add<MultiAxesVBrcLowering, VCastLowering, VAbsIntegerLowering,
           VRecOpHighPrecisionLowering, VReduceAnyLowering, VReduceAllLowering,
           VReduceInitInitializing, SyncBlockOpLowering, VCmpOpLowering,
           DecomposeCastScalarToVecOp<arith::ExtFOp>,
           DecomposeCastScalarToVecOp<arith::ExtSIOp>,
           DecomposeCastScalarToVecOp<arith::ExtUIOp>,
           DecomposeCastScalarToVecOp<arith::FPToSIOp>,
           DecomposeCastScalarToVecOp<arith::FPToUIOp>,
           DecomposeCastScalarToVecOp<arith::SIToFPOp>,
           DecomposeCastScalarToVecOp<arith::TruncFOp>,
           DecomposeScalarOpToVecOp<math::LogOp, hivm::VLnOp>,
           DecomposeI32ScalarExtOp<arith::MulUIExtendedOp>,
           DecomposeVSubScalarOp, DecomposeVDeinterleaveOp,
           AtomicStoreOpLowering, AtomicCasOpLowering, AtomicXchgOpLowering>(
          &getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createHIVMDecomposeOpPass() {
  return std::make_unique<HIVMDecomposeOpPass>();
}
