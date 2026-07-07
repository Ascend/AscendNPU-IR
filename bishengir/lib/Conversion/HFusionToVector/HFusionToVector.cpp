//===----------- HFusionToVector.cpp - convert hfusion to vector //--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HFusionToVector/HFusionToVector.h"
#include "bishengir/Conversion/HFusionToHIVM/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include <optional>

#define DEBUG_TYPE "convert-hfusion-to-vector"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_CONVERTHFUSIONTOVECTOR
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;
using namespace mlir::hivm::util;

namespace {
struct HFusionToVectorPass
    : public impl::ConvertHFusionToVectorBase<HFusionToVectorPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

namespace {
/// map element bitwidth to fixed size
static std::optional<int64_t> getFixedVecSizeForElementType(Type eltTy) {
  if (auto intTy = mlir::dyn_cast<IntegerType>(eltTy)) {
    switch (intTy.getWidth()) {
    case 8:
      return 256;
    case 16:
      return 128;
    case 32:
      return 64;
    case 64:
      return 64;
    default:
      return std::nullopt;
    }
  }
  if (auto fpTy = mlir::dyn_cast<FloatType>(eltTy)) {
    switch (fpTy.getWidth()) {
    case 8:
      return 256;
    case 16:
      return 128;
    case 32:
      return 64;
    case 64:
      return 64;
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

/// Pattern that lowers hfusion.interleave -> masked transfer_read -> Vector VF
/// -> masked transfer_write without creating redundant extract/insert
/// operations.
struct InterleaveToVectorPattern
    : public OpRewritePattern<hfusion::InterleaveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp interleaveOp,
                                PatternRewriter &rewriter) const override {
    Location loc = interleaveOp.getLoc();
    Value src0, src1;
    RankedTensorType src0Type, src1Type, resType;
    Type eltTy;

    if (failed(validateInputs(interleaveOp, src0, src1, src0Type, src1Type,
                              resType, eltTy)))
      return failure();

    VectorType vecTy, maskTy;
    int64_t vecSize;
    if (failed(getVectorTypesAndSize(eltTy, rewriter, vecTy, maskTy, vecSize)))
      return failure();

    // allow inputs that come either from tensor.extract_slice *or* from ops
    // that produce tensors (e.g. vector.transfer_write / vector.mask). Do not
    // fail just because the defining op isn't tensor::ExtractSliceOp.
    Value activeElements = getActiveElements(rewriter, loc, src0);
    if (!activeElements)
      return failure();

    Value readMask0 = createReadMask(rewriter, loc, maskTy, activeElements);
    Value readMask1 = createReadMask(rewriter, loc, maskTy, activeElements);
    Value padScalar = createPadScalar(rewriter, loc, eltTy);

    auto permMap = createPermutationMap(rewriter, src0Type.getRank());
    SmallVector<Value, 4> zeroIndices =
        createZeroIndices(rewriter, loc, src0Type.getRank());

    auto [vec0, vec1] =
        performMaskedReads(rewriter, loc, vecTy, src0, src1, zeroIndices,
                           padScalar, permMap, readMask0, readMask1);
    auto vec2Ty = VectorType::get({vecSize * 2}, eltTy);
    auto interleavedVec = rewriter.create<vector::InterleaveOp>(
        loc, vec2Ty, ValueRange{vec0, vec1});

    Value outVec = interleavedVec.getResult();
    auto ucc = rewriter.create<UnrealizedConversionCastOp>(loc, vecTy, outVec);
    Value outVec0 = ucc.getResult(0);
    Value twiceActive = rewriter.create<arith::MulIOp>(
        loc, activeElements, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    // create 1D write mask (kept for the direct-insert case)
    Value writeMask = createWriteMask(rewriter, loc, maskTy, twiceActive);

    if (succeeded(handleDirectInsertCase(interleaveOp, rewriter, loc, outVec0,
                                         writeMask, permMap))) {
      return success();
    }

    // reshape vector -> n-D and create n-D mask, then transfer_write (so other
    // pass can fold this away)
    handleGeneralCase(interleaveOp, rewriter, loc, outVec0, writeMask, permMap,
                      twiceActive);
    return success();
  }

private:
  LogicalResult validateInputs(hfusion::InterleaveOp interleaveOp, Value &src0,
                               Value &src1, RankedTensorType &src0Type,
                               RankedTensorType &src1Type,
                               RankedTensorType &resType, Type &eltTy) const {
    // expect exactly two operands
    if (interleaveOp.getNumOperands() != 2)
      return failure();

    src0 = interleaveOp.getOperand(0);
    src1 = interleaveOp.getOperand(1);

    src0Type = mlir::dyn_cast<RankedTensorType>(src0.getType());
    src1Type = mlir::dyn_cast<RankedTensorType>(src1.getType());
    resType =
        mlir::dyn_cast<RankedTensorType>(interleaveOp.getResult().getType());

    if (!src0Type || !src1Type || !resType)
      return failure();

    eltTy = src0Type.getElementType();
    if (eltTy != src1Type.getElementType() || eltTy != resType.getElementType())
      return failure();

    return success();
  }

  LogicalResult getVectorTypesAndSize(Type eltTy, PatternRewriter &rewriter,
                                      VectorType &vecTy, VectorType &maskTy,
                                      int64_t &vecSize) const {
    // get the correct vector size to load
    auto maybeVecSize = getFixedVecSizeForElementType(eltTy);
    if (!maybeVecSize)
      return failure();

    vecSize = *maybeVecSize;

    // create 1D vector type with vecSize elements
    vecTy = VectorType::get({vecSize}, eltTy);
    maskTy = VectorType::get({vecSize}, rewriter.getI1Type());
    return success();
  }

  // accept any Value that represents the (possibly sliced) tensor
  // and return the "active elements" (size for the last dimension).
  Value getActiveElements(PatternRewriter &rewriter, Location loc,
                          Value tensorVal) const {
    auto tt = mlir::dyn_cast<RankedTensorType>(tensorVal.getType());
    if (!tt)
      return nullptr;

    int64_t lastDim = tt.getShape().back();
    if (lastDim != ShapedType::kDynamic) {
      return rewriter.create<arith::ConstantIndexOp>(loc, lastDim);
    }

    Operation *defOp = tensorVal.getDefiningOp();
    if (!defOp) {
      // If no defining op, try tensor.dim on the value itself (block arg case).
      Value dimIdx =
          rewriter.create<arith::ConstantIndexOp>(loc, tt.getRank() - 1);
      return rewriter.create<tensor::DimOp>(loc, tensorVal, dimIdx);
    }

    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      auto sizeOfr = extractOp.getMixedSizes().back();
      if (auto val = mlir::dyn_cast<Value>(sizeOfr))
        return val;
      if (auto attr = mlir::dyn_cast<Attribute>(sizeOfr)) {
        int64_t staticSize = cast<IntegerAttr>(attr).getInt();
        return rewriter.create<arith::ConstantIndexOp>(loc, staticSize);
      }
      return nullptr;
    }

    // If the value is produced by vector.transfer_write / vector.mask /
    // other tensor-producing ops, fall back to emitting a tensor.dim on the
    // value to obtain the runtime size of the last dimension.
    Value dimIdx =
        rewriter.create<arith::ConstantIndexOp>(loc, tt.getRank() - 1);
    return rewriter.create<tensor::DimOp>(loc, tensorVal, dimIdx);
  }

  Value createReadMask(PatternRewriter &rewriter, Location loc,
                       VectorType maskTy, Value activeElements) const {
    return rewriter.create<vector::CreateMaskOp>(loc, maskTy,
                                                 ValueRange{activeElements});
  }

  Value createPadScalar(PatternRewriter &rewriter, Location loc,
                        Type eltTy) const {
    return rewriter.create<arith::ConstantOp>(loc, eltTy,
                                              rewriter.getZeroAttr(eltTy));
  }

  //  permutation helper 1-D if op returns is explicitly written into
  AffineMap createPermutationMap(PatternRewriter &rewriter,
                                 unsigned rank) const {
    SmallVector<AffineExpr, 1> exprs{rewriter.getAffineDimExpr(rank - 1)};
    return AffineMap::get(rank, 0, exprs, rewriter.getContext());
  }

  // Overload: this is for n-D permutation need for folding when other ops needs
  // it
  AffineMap createPermutationMap(PatternRewriter &rewriter, unsigned memrefRank,
                                 unsigned resultCount) const {
    SmallVector<AffineExpr, 4> exprs;
    if (resultCount == 0)
      return AffineMap::get(memrefRank, 0, exprs, rewriter.getContext());
    if (resultCount <= memrefRank) {
      unsigned start = memrefRank - resultCount;
      for (unsigned i = 0; i < resultCount; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(start + i));
    } else {
      // resultCount > memrefRank: repeat the last dim to fill results
      for (unsigned i = 0; i < memrefRank; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      for (unsigned i = memrefRank; i < resultCount; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(memrefRank - 1));
    }
    return AffineMap::get(memrefRank, 0, exprs, rewriter.getContext());
  }

  SmallVector<Value, 4> createZeroIndices(PatternRewriter &rewriter,
                                          Location loc, unsigned rank) const {
    SmallVector<Value, 4> indices(rank);
    for (unsigned i = 0; i < rank; ++i)
      indices[i] = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    return indices;
  }

  std::pair<Value, Value>
  performMaskedReads(PatternRewriter &rewriter, Location loc, VectorType vecTy,
                     Value src0, Value src1,
                     const SmallVector<Value, 4> &indices, Value padScalar,
                     AffineMap permMap, Value mask0, Value mask1) const {
    auto tr0 = rewriter.create<vector::TransferReadOp>(loc, vecTy, src0,
                                                       indices, padScalar);
    tr0.setPermutationMapAttr(AffineMapAttr::get(permMap));
    Operation *mTr0 =
        vector::maskOperation(rewriter, tr0.getOperation(), mask0);
    Value vec0 = mTr0->getResult(0);

    auto tr1 = rewriter.create<vector::TransferReadOp>(loc, vecTy, src1,
                                                       indices, padScalar);
    tr1.setPermutationMapAttr(AffineMapAttr::get(permMap));
    Operation *mTr1 =
        vector::maskOperation(rewriter, tr1.getOperation(), mask1);
    Value vec1 = mTr1->getResult(0);

    return {vec0, vec1};
  }

  Value createWriteMask(PatternRewriter &rewriter, Location loc,
                        VectorType maskTy, Value twiceActive) const {
    return rewriter.create<vector::CreateMaskOp>(loc, maskTy,
                                                 ValueRange{twiceActive});
  }

  LogicalResult handleDirectInsertCase(hfusion::InterleaveOp interleaveOp,
                                       PatternRewriter &rewriter, Location loc,
                                       Value outVec0, Value writeMask,
                                       AffineMap permMap) const {
    if (interleaveOp.getResult().hasOneUse()) {
      Operation *user = *interleaveOp.getResult().user_begin();
      if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(user)) {
        // case 1: direct insert
        Value destTensor = insertOp.getDest();
        SmallVector<OpFoldResult, 4> origOffsets = insertOp.getMixedOffsets();
        SmallVector<Value, 4> offsetsVal;
        for (auto ofr : origOffsets) {
          if (auto v = mlir::dyn_cast<Value>(ofr))
            offsetsVal.push_back(v);
          else if (auto attr = mlir::dyn_cast<Attribute>(ofr))
            offsetsVal.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, mlir::cast<IntegerAttr>(attr).getInt()));
          else
            return failure();
        }
        auto tw = rewriter.create<vector::TransferWriteOp>(
            loc, outVec0, destTensor, offsetsVal);
        tw.setPermutationMapAttr(AffineMapAttr::get(permMap));
        Operation *mTw =
            vector::maskOperation(rewriter, tw.getOperation(), writeMask);
        rewriter.replaceOp(insertOp, mTw->getResult(0));
        rewriter.eraseOp(interleaveOp);
        return success();
      }
    }
    return failure();
  }

  // compute an ND mask
  void handleGeneralCase(hfusion::InterleaveOp interleaveOp,
                         PatternRewriter &rewriter, Location loc, Value outVec0,
                         Value writeMask, AffineMap /*origPermMap*/,
                         Value twiceActive) const {
    auto resTensorType =
        mlir::dyn_cast<RankedTensorType>(interleaveOp.getResult().getType());
    if (!resTensorType)
      return;

    unsigned destRank = static_cast<unsigned>(resTensorType.getRank());

    // reshape 1-D vector result to n-D vector with leading unit dims
    Value resultVecND = reshapeVectorToND(rewriter, loc, outVec0, destRank);

    // create a write permutation map matching dest tensor rank and result
    // vector rank.
    auto resultVecTy = mlir::dyn_cast<VectorType>(resultVecND.getType());
    unsigned resultVecRank =
        resultVecTy ? static_cast<unsigned>(resultVecTy.getRank()) : 1;
    AffineMap writePermMap =
        createPermutationMap(rewriter, resTensorType.getRank(), resultVecRank);

    // build mixed sizes for tensor.empty: static dims as IndexAttr, replace the
    // dynamic last-dimension with `twiceActive` (we know they match).
    SmallVector<OpFoldResult, 4> mixedSizes;
    ArrayRef<int64_t> shape = resTensorType.getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      if (!ShapedType::isDynamic(shape[i])) {
        mixedSizes.push_back(rewriter.getIndexAttr(shape[i]));
      } else if (i == shape.size() - 1) {
        // last dim is dynamic and equals twiceActive
        mixedSizes.push_back(twiceActive);
      } else {
        return;
      }
    }

    // create an empty tensor of the same type as the original result
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, mixedSizes, resTensorType.getElementType());

    // create an ND mask: derive base mask type from the 1-D writeMask type
    VectorType baseMaskTy = mlir::dyn_cast<VectorType>(writeMask.getType());
    VectorType maskTyND = makeNDMaskType(rewriter, baseMaskTy, destRank);

    // build lengths vector for CreateMaskOp: leading ones and last =
    // twiceActive
    SmallVector<Value, 4> lens;
    for (unsigned i = 0; i < destRank; ++i) {
      if (i + 1 == destRank) {
        lens.push_back(twiceActive);
      } else {
        lens.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }
    }
    Value writeMaskND =
        rewriter.create<vector::CreateMaskOp>(loc, maskTyND, ValueRange(lens));

    // write the ND vector into the new tensor with mask
    SmallVector<Value, 4> zeroIdx2(
        resTensorType.getRank(),
        rewriter.create<arith::ConstantIndexOp>(loc, 0));
    auto tw = rewriter.create<vector::TransferWriteOp>(loc, resultVecND,
                                                       emptyTensor, zeroIdx2);
    tw.setPermutationMapAttr(AffineMapAttr::get(writePermMap));
    Operation *mTw =
        vector::maskOperation(rewriter, tw.getOperation(), writeMaskND);
    Value finalTensor = mTw->getResult(0);

    rewriter.replaceOp(interleaveOp, finalTensor);
  }

  // helpers to create n-D vector/mask types and to shape-cast a 1-D vector to
  // an n-D vector where the first n-1 dims are 1 and last dim is the base len.
  VectorType makeNDVectorType(PatternRewriter &rewriter, VectorType base,
                              unsigned targetRank) const {
    if (targetRank == 1)
      return base;
    SmallVector<int64_t, 4> shape;
    for (unsigned i = 0; i < targetRank - 1; ++i)
      shape.push_back(1);
    ArrayRef<int64_t> baseShape = base.getShape();
    shape.push_back(baseShape.back());
    return VectorType::get(shape, base.getElementType());
  }

  VectorType makeNDMaskType(PatternRewriter &rewriter, VectorType baseMaskTy,
                            unsigned targetRank) const {
    if (targetRank == 1)
      return baseMaskTy;
    SmallVector<int64_t, 4> maskShape;
    for (unsigned i = 0; i < targetRank - 1; ++i)
      maskShape.push_back(1);
    ArrayRef<int64_t> baseShape = baseMaskTy.getShape();
    maskShape.push_back(baseShape.back());
    return VectorType::get(maskShape, rewriter.getI1Type());
  }

  Value reshapeVectorToND(PatternRewriter &rewriter, Location loc, Value vec,
                          unsigned targetRank) const {
    auto vecTy = mlir::dyn_cast<VectorType>(vec.getType());
    if (!vecTy)
      return vec;
    VectorType ndTy = makeNDVectorType(rewriter, vecTy, targetRank);
    if (ndTy == vecTy)
      return vec;
    return rewriter.create<vector::ShapeCastOp>(loc, ndTy, vec);
  }
};

/// Pattern that lowers hfusion.deinterleave -> masked transfer_read -> Vector
/// VF
/// -> masked transfer_write without creating redundant extract/insert
/// operations.
struct DeInterleaveToVectorPattern
    : public OpRewritePattern<hfusion::DeinterleaveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::DeinterleaveOp deinterleaveOp,
                                PatternRewriter &rewriter) const override {
    Location loc = deinterleaveOp.getLoc();

    Value src;
    RankedTensorType srcType, resType;
    int64_t channel;
    if (failed(validateInputs(deinterleaveOp, src, srcType, resType, channel)))
      return failure();

    Type eltTy = srcType.getElementType();
    auto [vecTy, maskTy, vecSize] = getVectorTypesAndSize(eltTy, rewriter);
    if (vecSize == 0)
      return failure();

    // allow inputs that come either from tensor.extract_slice or from ops that
    // produce tensors (e.g. vector.transfer_write / vector.mask). Do not fail
    // just because the defining op isn't tensor::ExtractSliceOp.
    Value activeElements = getActiveElements(rewriter, loc, src);
    if (!activeElements)
      return failure();

    Value padScalar = createPadScalar(rewriter, loc, eltTy);

    // read and write can have different vector ranks: create separate
    // permutation maps
    AffineMap readPermMap =
        createPermutationMap(rewriter, srcType.getRank(), vecTy.getRank());

    SmallVector<Value, 4> inputOffsetValues =
        getInputOffsetValues(rewriter, loc, src);
    SmallVector<Value, 4> zeroIndices =
        createZeroIndices(rewriter, loc, srcType.getRank());

    Value vecSizeValue = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);
    auto [vec0, vec1] = performMaskedReads(rewriter, loc, vecTy, maskTy, src,
                                           zeroIndices, padScalar, readPermMap,
                                           activeElements, vecSizeValue);
    // vector.deinterleave operates on src of 2xN vector and returns two vectors
    // of length N. ave.deinterleave operates on two vectors of length N and
    // returns two vectors of length N. Thus we directly use UCC to bridge this
    // gap. UCC is a placeholder. VectorToAVE would materialize this bridge.
    auto vec2Ty = VectorType::get({vecSize * 2}, eltTy);
    auto ucc = rewriter.create<UnrealizedConversionCastOp>(
        loc, vec2Ty, ValueRange{vec0, vec1});
    Value vec01 = ucc.getResult(0);
    auto deinterleavedVecs =
        rewriter.create<vector::DeinterleaveOp>(loc, vecTy, vecTy, vec01);

    Value resultVec1D = channel == 0 ? deinterleavedVecs.getResult(0)
                                     : deinterleavedVecs.getResult(1);

    unsigned destRank = static_cast<unsigned>(resType.getRank());
    Value resultVec = reshapeVectorToND(rewriter, loc, resultVec1D, destRank);

    // create a write permutation map matching dest tensor rank and result
    // vector rank
    auto resultVecTy = mlir::dyn_cast<VectorType>(resultVec.getType());
    unsigned resultVecRank = resultVecTy
                                 ? static_cast<unsigned>(resultVecTy.getRank())
                                 : static_cast<unsigned>(vecTy.getRank());
    AffineMap writePermMap =
        createPermutationMap(rewriter, resType.getRank(), resultVecRank);

    Value halfActive = rewriter.create<arith::DivSIOp>(
        loc, activeElements, rewriter.create<arith::ConstantIndexOp>(loc, 2));

    VectorType maskTyND = makeNDMaskType(rewriter, maskTy, destRank);
    Value writeMask = createWriteMask(rewriter, loc, maskTyND, halfActive);

    if (succeeded(handleDirectInsertCase(deinterleaveOp, rewriter, loc,
                                         resultVec, writeMask, writePermMap))) {
      return success();
    }

    handleGeneralCase(deinterleaveOp, rewriter, loc, resultVec, writeMask,
                      writePermMap, halfActive);
    return success();
  }

private:
  LogicalResult validateInputs(hfusion::DeinterleaveOp deinterleaveOp,
                               Value &src, RankedTensorType &srcType,
                               RankedTensorType &resType,
                               int64_t &channel) const {
    // get the channel attribute
    channel = static_cast<int64_t>(deinterleaveOp.getChannelIndex());
    if (channel != 0 && channel != 1) {
      return deinterleaveOp.emitError("Only channels 0 and 1 are supported");
    }
    src = deinterleaveOp.getOperand();
    srcType = dyn_cast<RankedTensorType>(src.getType());
    resType =
        mlir::dyn_cast<RankedTensorType>(deinterleaveOp.getResult(0).getType());

    if (!srcType || !resType)
      return failure();

    Type eltTy = srcType.getElementType();
    if (eltTy != resType.getElementType())
      return failure();

    return success();
  }

  std::tuple<VectorType, VectorType, int64_t>
  getVectorTypesAndSize(Type eltTy, PatternRewriter &rewriter) const {
    auto maybeVecSize = getFixedVecSizeForElementType(eltTy);
    if (!maybeVecSize)
      return {VectorType(), VectorType(), 0};

    int64_t vecSize = *maybeVecSize;
    VectorType vecTy = VectorType::get({vecSize}, eltTy);
    VectorType maskTy = VectorType::get({vecSize}, rewriter.getI1Type());
    return {vecTy, maskTy, vecSize};
  }

  // accept any Value that represents the (possibly sliced) tensor
  // and return the "active elements" (size for the last dimension).
  Value getActiveElements(PatternRewriter &rewriter, Location loc,
                          Value tensorVal) const {
    auto tt = mlir::dyn_cast<RankedTensorType>(tensorVal.getType());
    if (!tt)
      return nullptr;

    int64_t lastDim = tt.getShape().back();
    if (lastDim != ShapedType::kDynamic) {
      return rewriter.create<arith::ConstantIndexOp>(loc, lastDim);
    }

    Operation *defOp = tensorVal.getDefiningOp();
    if (!defOp) {
      // If no defining op (e.g. block arg), use tensor.dim
      Value dimIdx =
          rewriter.create<arith::ConstantIndexOp>(loc, tt.getRank() - 1);
      return rewriter.create<tensor::DimOp>(loc, tensorVal, dimIdx);
    }

    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      auto sizeOfr = extractOp.getMixedSizes().back();
      if (auto val = mlir::dyn_cast<Value>(sizeOfr))
        return val;
      if (auto attr = mlir::dyn_cast<Attribute>(sizeOfr)) {
        int64_t staticSize = cast<IntegerAttr>(attr).getInt();
        return rewriter.create<arith::ConstantIndexOp>(loc, staticSize);
      }
      return nullptr;
    }
    Value dimIdx =
        rewriter.create<arith::ConstantIndexOp>(loc, tt.getRank() - 1);
    return rewriter.create<tensor::DimOp>(loc, tensorVal, dimIdx);
  }

  Value createPadScalar(PatternRewriter &rewriter, Location loc,
                        Type eltTy) const {
    return rewriter.create<arith::ConstantOp>(loc, eltTy,
                                              rewriter.getZeroAttr(eltTy));
  }

  // n-d permutation map
  // memrefRank: number of input dims (e.g. tensor rank)
  // resultCount: desired number of results (vector rank)
  AffineMap createPermutationMap(PatternRewriter &rewriter, unsigned memrefRank,
                                 unsigned resultCount) const {
    SmallVector<AffineExpr, 4> exprs;
    if (resultCount == 0)
      return AffineMap::get(memrefRank, 0, exprs, rewriter.getContext());
    if (resultCount <= memrefRank) {
      unsigned start = memrefRank - resultCount;
      for (unsigned i = 0; i < resultCount; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(start + i));
    } else {
      // resultCount > memrefRank: repeat the last dim to fill results
      for (unsigned i = 0; i < memrefRank; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      for (unsigned i = memrefRank; i < resultCount; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(memrefRank - 1));
    }
    return AffineMap::get(memrefRank, 0, exprs, rewriter.getContext());
  }

  // accept either an ExtractSliceOp (and return its mixed offsets) or return
  // zero offsets when the source is produced by other ops (e.g.
  // transfer_write).
  SmallVector<Value, 4> getInputOffsetValues(PatternRewriter &rewriter,
                                             Location loc,
                                             Value tensorVal) const {
    SmallVector<Value, 4> offsetValues;

    auto tt = mlir::dyn_cast<RankedTensorType>(tensorVal.getType());
    if (!tt)
      return {};

    Operation *defOp = tensorVal.getDefiningOp();
    if (auto extractOp = dyn_cast_or_null<tensor::ExtractSliceOp>(defOp)) {
      SmallVector<OpFoldResult> inputOffsets = extractOp.getMixedOffsets();
      for (auto ofr : inputOffsets) {
        if (auto v = mlir::dyn_cast<Value>(ofr)) {
          offsetValues.push_back(v);
        } else if (auto attr = mlir::dyn_cast<Attribute>(ofr)) {
          int64_t staticOffset = cast<IntegerAttr>(attr).getInt();
          offsetValues.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, staticOffset));
        } else {
          return {};
        }
      }
      return offsetValues;
    }
    for (unsigned i = 0; i < tt.getRank(); ++i)
      offsetValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    return offsetValues;
  }

  SmallVector<Value, 4> createZeroIndices(PatternRewriter &rewriter,
                                          Location loc, unsigned rank) const {
    SmallVector<Value, 4> indices(rank);
    for (unsigned i = 0; i < rank; ++i)
      indices[i] = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    return indices;
  }

  std::pair<Value, Value>
  performMaskedReads(PatternRewriter &rewriter, Location loc, VectorType vecTy,
                     VectorType maskTy, Value src,
                     const SmallVector<Value, 4> &zeroIndices, Value padScalar,
                     AffineMap permMap, Value activeElements,
                     Value vecSizeValue) const {

    Value firstMaskLen =
        rewriter.create<arith::MinSIOp>(loc, activeElements, vecSizeValue);
    Value firstMask = rewriter.create<vector::CreateMaskOp>(
        loc, maskTy, ValueRange{firstMaskLen});

    auto firstRead = rewriter.create<vector::TransferReadOp>(
        loc, vecTy, src, zeroIndices, padScalar);
    firstRead.setPermutationMapAttr(AffineMapAttr::get(permMap));
    Operation *maskedFirstRead =
        vector::maskOperation(rewriter, firstRead.getOperation(), firstMask);
    Value vec0 = maskedFirstRead->getResult(0);

    SmallVector<Value, 4> secondOffsetValues = zeroIndices;
    Value remainingElements =
        rewriter.create<arith::SubIOp>(loc, activeElements, vecSizeValue);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value secondMaskLen =
        rewriter.create<arith::MaxSIOp>(loc, zeroIdx, remainingElements);
    Value secondMask = rewriter.create<vector::CreateMaskOp>(
        loc, maskTy, ValueRange{secondMaskLen});

    auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                              secondMaskLen, zeroIdx);
    Value lastDimOffsetIfAny =
        rewriter.create<arith::SelectOp>(loc, cmp, vecSizeValue, zeroIdx);
    secondOffsetValues[zeroIndices.size() - 1] = lastDimOffsetIfAny;

    auto secondRead = rewriter.create<vector::TransferReadOp>(
        loc, vecTy, src, secondOffsetValues, padScalar);
    secondRead.setPermutationMapAttr(AffineMapAttr::get(permMap));
    Operation *maskedSecondRead =
        vector::maskOperation(rewriter, secondRead.getOperation(), secondMask);
    Value vec1 = maskedSecondRead->getResult(0);

    return {vec0, vec1};
  }

  Value createWriteMask(PatternRewriter &rewriter, Location loc,
                        VectorType maskTy, Value halfActive) const {
    // createMaskOp uses the provided length for the last dimension; leading
    // dims are implicitly full (ones) because their sizes are 1.
    unsigned rank = static_cast<unsigned>(maskTy.getRank());
    SmallVector<Value, 4> lens;
    for (unsigned i = 0; i < rank; ++i) {
      if (i + 1 == rank) {
        lens.push_back(halfActive);
      } else {
        lens.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }
    }
    return rewriter.create<vector::CreateMaskOp>(loc, maskTy, ValueRange{lens});
  }

  LogicalResult handleDirectInsertCase(hfusion::DeinterleaveOp deinterleaveOp,
                                       PatternRewriter &rewriter, Location loc,
                                       Value resultVec, Value writeMask,
                                       AffineMap permMap) const {
    if (deinterleaveOp.getResult(0).hasOneUse()) {
      Operation *user = *deinterleaveOp.getResult(0).user_begin();
      if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(user)) {
        // case 1: directly insert
        Value destTensor = insertOp.getDest();
        SmallVector<OpFoldResult, 4> origOffsets = insertOp.getMixedOffsets();
        SmallVector<Value, 4> offsetsVal;
        for (auto ofr : origOffsets) {
          if (auto v = mlir::dyn_cast<Value>(ofr))
            offsetsVal.push_back(v);
          else if (auto attr = mlir::dyn_cast<Attribute>(ofr))
            offsetsVal.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, mlir::cast<IntegerAttr>(attr).getInt()));
          else
            return failure();
        }
        auto tw = rewriter.create<vector::TransferWriteOp>(
            loc, resultVec, destTensor, offsetsVal);
        tw.setPermutationMapAttr(AffineMapAttr::get(permMap));
        Operation *mTw =
            vector::maskOperation(rewriter, tw.getOperation(), writeMask);
        rewriter.replaceOp(insertOp, mTw->getResult(0));
        rewriter.eraseOp(deinterleaveOp);
        return success();
      }
    }
    return failure();
  }

  void handleGeneralCase(hfusion::DeinterleaveOp deinterleaveOp,
                         PatternRewriter &rewriter, Location loc,
                         Value resultVec, Value writeMask, AffineMap permMap,
                         Value halfActive) const {
    auto resTypeEmpty =
        mlir::dyn_cast<RankedTensorType>(deinterleaveOp.getResult(0).getType());
    if (!resTypeEmpty)
      return;

    // build mixed sizes for tensor.empty: static dims as IndexAttr, replace the
    // dynamic last-dimension with `halfActive` (we know they match).
    SmallVector<OpFoldResult, 4> mixedSizes;
    ArrayRef<int64_t> shape = resTypeEmpty.getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      if (!ShapedType::isDynamic(shape[i])) {
        mixedSizes.push_back(rewriter.getIndexAttr(shape[i]));
      } else if (i == shape.size() - 1) {
        // last dim is dynamic and equals halfActive
        mixedSizes.push_back(halfActive);
      } else {
        // unexpected dynamic dimension we don't know how to bound
        return;
      }
    }

    // create an empty tensor of the same type as the original result
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, mixedSizes, resTypeEmpty.getElementType());

    // write the vector into the new tensor with mask
    SmallVector<Value, 4> zeroIdx2(
        resTypeEmpty.getRank(),
        rewriter.create<arith::ConstantIndexOp>(loc, 0));
    auto tw = rewriter.create<vector::TransferWriteOp>(loc, resultVec,
                                                       emptyTensor, zeroIdx2);
    tw.setPermutationMapAttr(AffineMapAttr::get(permMap));
    Operation *mTw =
        vector::maskOperation(rewriter, tw.getOperation(), writeMask);
    Value finalTensor = mTw->getResult(0);

    // replace all uses of deinterleaveOp result with the new tensor
    rewriter.replaceOp(deinterleaveOp, finalTensor);
  }

  VectorType makeNDVectorType(PatternRewriter &rewriter, VectorType base,
                              unsigned targetRank) const {
    if (targetRank == 1)
      return base;
    SmallVector<int64_t, 4> shape;
    for (unsigned i = 0; i < targetRank - 1; ++i)
      shape.push_back(1);
    ArrayRef<int64_t> baseShape = base.getShape();
    shape.push_back(baseShape.back());
    return VectorType::get(shape, base.getElementType());
  }

  VectorType makeNDMaskType(PatternRewriter &rewriter, VectorType baseMaskTy,
                            unsigned targetRank) const {
    if (targetRank == 1)
      return baseMaskTy;
    SmallVector<int64_t, 4> maskShape;
    for (unsigned i = 0; i < targetRank - 1; ++i)
      maskShape.push_back(1);
    ArrayRef<int64_t> baseShape = baseMaskTy.getShape();
    maskShape.push_back(baseShape.back());
    return VectorType::get(maskShape, rewriter.getI1Type());
  }

  Value reshapeVectorToND(PatternRewriter &rewriter, Location loc, Value vec,
                          unsigned targetRank) const {
    auto vecTy = mlir::dyn_cast<VectorType>(vec.getType());
    if (!vecTy)
      return vec;
    VectorType ndTy = makeNDVectorType(rewriter, vecTy, targetRank);
    if (ndTy == vecTy)
      return vec;
    return rewriter.create<vector::ShapeCastOp>(loc, ndTy, vec);
  }
};
struct FlipToAVEPattern : public OpRewritePattern<hfusion::FlipOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::FlipOp flipOp,
                                PatternRewriter &rewriter) const override {
    Location loc = flipOp.getLoc();
    Value input = flipOp.getInput();

    // Get the input type and shape
    auto inputType = mlir::cast<ShapedType>(input.getType());
    if (!inputType.hasRank()) {
      return failure();
    }

    ArrayRef<int64_t> shape = inputType.getShape();
    int64_t rank = static_cast<int64_t>(shape.size());

    if (rank == 0) {
      return failure();
    }

    // Create output tensor (same shape as input)
    Value output = rewriter.create<tensor::EmptyOp>(loc, shape,
                                                    inputType.getElementType());

    // Build nested loops for all dimensions
    SmallVector<Value> lowerBounds(rank);
    SmallVector<Value> upperBounds(rank);
    SmallVector<Value> steps(rank);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    for (int64_t i = 0; i < rank; ++i) {
      lowerBounds[i] = zero;
      upperBounds[i] = rewriter.create<arith::ConstantIndexOp>(loc, shape[i]);
      steps[i] = one;
    }

    // Create the nested scf.for loops
    auto nestedLoop = rewriter.create<scf::ForOp>(
        loc, lowerBounds[0], upperBounds[0], steps[0], ValueRange{output},
        [&](OpBuilder &b, Location l, Value iv0, ValueRange iterArgs) {
          Value currentOutput = iterArgs[0];

          // Build remaining nested loops recursively
          currentOutput = buildNestedLoops(b, l, input, currentOutput, shape,
                                           rank, 1, SmallVector<Value>{iv0},
                                           lowerBounds, upperBounds, steps);

          b.create<scf::YieldOp>(l, currentOutput);
        });

    rewriter.replaceOp(flipOp, nestedLoop.getResults());
    return success();
  }

private:
  Value buildNestedLoops(OpBuilder &builder, Location loc, Value input,
                         Value output, ArrayRef<int64_t> shape, int64_t rank,
                         int64_t currentDim, SmallVector<Value> indices,
                         ArrayRef<Value> lowerBounds,
                         ArrayRef<Value> upperBounds,
                         ArrayRef<Value> steps) const {
    if (currentDim == rank) {
      // Base case: perform the actual flip operation
      // For the last dimension, reverse the index
      SmallVector<Value> readIndices = indices;
      SmallVector<Value> writeIndices = indices;

      // Flip the last dimension: flipped_idx = size - 1 - idx
      Value lastDimSize =
          builder.create<arith::ConstantIndexOp>(loc, shape[rank - 1]);
      Value lastIdx = indices.back();
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value sizeMinusOne = builder.create<arith::SubIOp>(loc, lastDimSize, one);
      Value flippedIdx =
          builder.create<arith::SubIOp>(loc, sizeMinusOne, lastIdx);

      readIndices.back() = flippedIdx;

      // Extract element from input with flipped last dimension
      Value element =
          builder.create<tensor::ExtractOp>(loc, input, readIndices);

      // Insert element into output at original position
      Value newOutput =
          builder.create<tensor::InsertOp>(loc, element, output, writeIndices);

      return newOutput;
    }

    // Recursive case: create another loop
    auto forOp = builder.create<scf::ForOp>(
        loc, lowerBounds[currentDim], upperBounds[currentDim],
        steps[currentDim], ValueRange{output},
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          SmallVector<Value> newIndices = indices;
          newIndices.push_back(iv);

          Value result = buildNestedLoops(b, l, input, iterArgs[0], shape, rank,
                                          currentDim + 1, newIndices,
                                          lowerBounds, upperBounds, steps);

          b.create<scf::YieldOp>(l, result);
        });

    return forOp.getResult(0);
  }
};

} // namespace

static void
populateHFusionToVectorConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<InterleaveToVectorPattern>(patterns.getContext());
  patterns.add<DeInterleaveToVectorPattern>(patterns.getContext());
}

void HFusionToVectorPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = &getContext();

  RewritePatternSet conversionPatterns(ctx);
  populateHFusionToVectorConversionPatterns(conversionPatterns);

  if (failed(applyPatternsGreedily(op, std::move(conversionPatterns))))
    signalPassFailure();
}

// Pass factory
std::unique_ptr<Pass> mlir::createHFusionToVectorConversionPass() {
  return std::make_unique<HFusionToVectorPass>();
}
