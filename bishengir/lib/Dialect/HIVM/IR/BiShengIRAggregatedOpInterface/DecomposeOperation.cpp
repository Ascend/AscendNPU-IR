//===- DecomposeOperation.cpp - DecomposeOperation implementations --------===//
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
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "decompose-operation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

inline RoundModeAttr getRoundAttr(mlir::OpBuilder &b, Type srcType,
                                  Type dstType) {
  return hivm::RoundModeAttr::get(
      b.getContext(),
      mlir::utils::selectRoundMode<hivm::RoundMode>(srcType, dstType));
}

} // namespace mlir::hivm

//===----------------------------------------------------------------------===//
// VBrcOp
//===----------------------------------------------------------------------===//

/// Decomposes a VBrcOp (vector broadcast operation) for I1 and I8 element
/// types.
///
/// This decomposition is necessary because the hardware doesn't support direct
/// broadcasting of I1/I8 types. Instead, we convert to F16, perform the
/// broadcast, then convert back to the original type.
///
/// Decomposition strategy:
/// - For I1: I1 -> F16 -> broadcast -> F16 -> I1 (via comparison with 0.0)
/// - For I8: I8 -> F16 -> broadcast -> F16 -> I8 (via cast)
mlir::FailureOr<llvm::SmallVector<mlir::Value>>
VBrcOp::decomposeOperation(mlir::OpBuilder &b) {
  const Type srcType = getSrc().getType();
  const Type srcElemType = getElementTypeOrSelf(srcType);
  const Type dstElemType = b.getF16Type();
  bool isI8 = srcElemType.isInteger(8);
  bool isI1 = srcElemType.isInteger(1);
  if (!isa<MemRefType>(srcType) || (!isI8 && !isI1))
    return llvm::failure();
  auto srcRoundAttr = getRoundAttr(b, srcElemType, dstElemType);
  auto dstRoundAttr = getRoundAttr(b, dstElemType, srcElemType);

  // Convert I1/I8 -> F16 by VCast
  hivm::VCastOp srcCast =
      castTo(b, getLoc(), getSrc(), srcRoundAttr, dstElemType);

  Value castedDst = mlir::utils::createTmpBufferOrTensorWithTargetType(
      b, getLoc(), getDst(), dstElemType);
  b.create<hivm::VBrcOp>(getLoc(), TypeRange(), srcCast.getSingleDst(),
                         castedDst, getBroadcastDimsAttr());
  if (isI1) {
    // Convert F16 -> I1 by VCompare F16 != 0
    Value floatZero =
        b.create<arith::ConstantOp>(getLoc(), b.getFloatAttr(dstElemType, 0.0));
    b.create<hivm::VCmpOp>(
        getLoc(), TypeRange(), ValueRange({castedDst, floatZero}), getDst(),
        b.getAttr<hivm::CompareModeAttr>(hivm::CompareMode::NE));
  } else if (isI8) {
    // Convert F16 -> I8 by VCast
    b.create<hivm::VCastOp>(getLoc(), TypeRange(), castedDst, getDst(),
                            dstRoundAttr, hivm::TypeFnAttr{});
  } else {
    return failure();
  }
  return SmallVector<Value>();
}

//===----------------------------------------------------------------------===//
// VConcatOp
//===----------------------------------------------------------------------===//

std::optional<Value> traceSource(Value value) {
  if (isa<BlockArgument>(value) || utils::isAllocLikeOp(value)) {
    return value;
  }
  auto *defOp = value.getDefiningOp();
  if (auto subviewOp = llvm::dyn_cast<memref::SubViewOp>(defOp)) {
    return traceSource(subviewOp.getViewSource());
  }
  return std::nullopt;
}

std::optional<Value> traceSameSource(ArrayRef<Value> values) {
  DenseSet<Value> sources;
  for (Value v : values) {
    auto srcMaybe = traceSource(v);
    if (!srcMaybe.has_value()) {
      return std::nullopt;
    }
    sources.insert(srcMaybe.value());
  }
  if (sources.size() != 1) {
    return std::nullopt;
  }
  return *sources.begin();
}

SmallVector<Value> filterValues(ArrayRef<Value> values, int64_t excludeIndex) {
  SmallVector<Value> result;
  int64_t size = static_cast<int64_t>(values.size());
  for (int64_t i = 0; i < size; ++i) {
    if (i == excludeIndex) {
      continue;
    }
    result.push_back(values[i]);
  }
  return result;
}

LogicalResult decomposeInsertSliceConcat(VConcatOp concatOp, OpBuilder &b) {
  IntegerAttr attr = concatOp->getAttrOfType<IntegerAttr>(
      hivm::InsertSliceSourceIndexAttr::name);
  if (!attr) {
    return failure();
  }
  int64_t srcIndex = attr.getInt();
  SmallVector<Value> inputs = concatOp.getSrc();
  if (inputs.size() != 3 || srcIndex != 1) {
    // only handle concat op with three inputs, where the insert_slice src is
    // the second input
    return failure();
  }

  SmallVector<Value> insertDst =
      filterValues(inputs, /*excludeIndex=*/srcIndex);
  auto dstSourceMaybe = traceSameSource(insertDst);
  if (!dstSourceMaybe.has_value()) {
    return failure();
  }
  Value dstSource = dstSourceMaybe.value();
  LDBG("traced dstSource for concat with insert_slice pattern: " << dstSource);

  Value copySrc = inputs[srcIndex];
  MemRefType srcType = llvm::cast<MemRefType>(copySrc.getType());
  int64_t rank = srcType.getRank();
  Location loc = concatOp.getLoc();
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = memref::getMixedSizes(b, loc, copySrc);
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

  if (inputs.size() == 3) {
    assert(srcIndex == 1 && "insert_slice src must be the middle operand");
    SmallVector<OpFoldResult> firstSizes =
        memref::getMixedSizes(b, loc, inputs[0]);
    auto dim = static_cast<int64_t>(concatOp.getDim());
    offsets[dim] = firstSizes[dim];
  }

  Value copyDst =
      b.create<memref::SubViewOp>(loc, dstSource, offsets, sizes, strides);
  (void)b.create<hivm::CopyOp>(loc, concatOp->getResultTypes(), copySrc,
                               copyDst);
  mlir::IRRewriter rewriter(b);
  rewriter.replaceAllUsesWith(concatOp.getDst(), dstSource);
  return success();
}

/// Here we specify VConcat's decompose behavior
/// VConcat will get erased and become Copy ops
/// from inputs of concat to subviews of it's output
FailureOr<SmallVector<Value>> VConcatOp::decomposeOperation(OpBuilder &b) {
  if (hasPureTensorSemantics()) {
    return failure();
  }
  auto srcNums = getODSOperands(0).size();
  auto dim = getDim();
  auto dst = getDst();
  ValueRange inputs = getODSOperands(0);
  SmallVector<OpFoldResult> concatSizes;
  for (auto input : inputs) {
    auto concatSize = memref::getMixedSize(b, this->getLoc(), input, dim);
    concatSizes.push_back(concatSize);
  }
  OpFoldResult totalSize = concatSizes[0];
  SmallVector<OpFoldResult> offsets;
  offsets.push_back(b.getIndexAttr(0));
  offsets.push_back(concatSizes[0]);

  for (size_t i = 1; i < concatSizes.size() - 1; ++i) {
    AffineExpr sumExpr = b.getAffineSymbolExpr(0) + b.getAffineSymbolExpr(1);
    totalSize = affine::makeComposedFoldedAffineApply(
        b, this->getLoc(), sumExpr, {totalSize, concatSizes[i]});
    offsets.push_back(totalSize);
  }

  if (succeeded(decomposeInsertSliceConcat(*this, b))) {
    return SmallVector<Value>{};
  }

  for (size_t i = 0; i < srcNums; ++i) {
    auto src = getODSOperands(0)[i];
    SmallVector<OpFoldResult> sliceSizes =
        memref::getMixedSizes(b, this->getLoc(), src);

    // Prepare offset, sizes and strides for SubViewOp
    SmallVector<OpFoldResult> vecOffsets;
    auto srcShapedType = cast<ShapedType>(src.getType());
    auto srcShapes = srcShapedType.getShape();
    const SmallVector<OpFoldResult> vecStrides(srcShapes.size(),
                                               b.getIndexAttr(1));

    for (uint32_t dim0 = 0; dim0 < srcShapes.size(); dim0++) {
      if (dim0 == dim) {
        vecOffsets.push_back(offsets[i]);
      } else {
        vecOffsets.push_back(b.getIndexAttr(0));
      }
    }

    auto subviewOp = b.create<memref::SubViewOp>(
        this->getLoc(), dst, vecOffsets, sliceSizes, vecStrides);

    (void)b.create<hivm::CopyOp>(this->getLoc(), getResultTypes(),
                                 getODSOperands(0)[i], subviewOp);
  }
  return SmallVector<Value>{};
}

//===----------------------------------------------------------------------===//
// VDeinterleaveOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {

inline FailureOr<llvm::SmallVector<mlir::Value>>
decomposeTensorDeinterleave(VDeinterleaveOp &op, mlir::OpBuilder &b) {
  assert(op.getResult().size() == op.getDst().size());
  assert(getElementTypeOrSelf(op.getSrc().getType()).isInteger(8));

  const Location loc = op->getLoc();
  auto fstRound = getRoundAttr(b, b.getI8Type(), b.getF16Type());
  auto bwdRound = getRoundAttr(b, b.getF16Type(), b.getI8Type());
  // create first cast op to convert i8 src to f16 src
  hivm::VCastOp fstCast =
      castTo(b, loc, /*src = i8 src*/ op.getSrc(), fstRound, b.getF16Type());
  assert(fstCast.getDst().size() == 1);
  assert(fstCast.getResult().size() == 1);

  // create f16 buffer for dst of new deinterleave
  unsigned int resultNum = op.getResult().size();
  SmallVector<Type> newResultTypes(resultNum);
  SmallVector<Value> newDestRange(op.getDst().size());
  for (const auto &[idx, dst] : llvm::enumerate(op.getDst())) {
    assert(getElementTypeOrSelf(dst.getType()).isInteger(8));

    newDestRange[idx] = mlir::utils::createTmpBufferOrTensorWithTargetType(
        b, loc, dst, b.getF16Type());
    newResultTypes[idx] = newDestRange[idx].getType();
  }

  // create the new deinterleave op
  auto newOp = b.create<hivm::VDeinterleaveOp>(
      loc, /*resultType = f16 resultType*/ newResultTypes,
      /*src = f16 casted*/ fstCast.getResult()[0],
      /*dst = f16 temp*/ newDestRange, op.getChannelNumAttr(),
      op.getIndexModeAttr());

  // create second casts to convert hivm.deinterleave results back to i8.
  // * For tensor operands, the result is the result of new op.
  // * the old op's dst is in shaped type of i8, which fits in the dst of second
  //   cast op.
  // * cast result types are the result types of old op.

  SmallVector<Value> sndCastResults(resultNum);
  for (const auto &[idx, newOpResult] : llvm::enumerate(newOp.getResult())) {
    hivm::VCastOp sndCast = b.create<hivm::VCastOp>(
        loc, /*resultType = i8 resultType*/ op.getResult()[idx].getType(),
        /*src = f16 temp*/ newOpResult,
        /*dst = i8 dst*/ op.getDst()[idx], bwdRound, hivm::TypeFnAttr{});
    sndCastResults[idx] = sndCast.getResult()[0];
  }

  return sndCastResults;
}

FailureOr<llvm::SmallVector<mlir::Value>>
decomposeMemRefDeinterleave(VDeinterleaveOp &op, mlir::OpBuilder &b) {
  assert(op.getResult().empty());
  assert(getElementTypeOrSelf(op.getSrc().getType()).isInteger(8));

  const Location loc = op->getLoc();

  // create first cast op to convert i8 src to f16 src
  auto fstRound = getRoundAttr(b, b.getI8Type(), b.getF16Type());
  auto bwdRound = getRoundAttr(b, b.getF16Type(), b.getI8Type());
  hivm::VCastOp fstCast =
      castTo(b, loc, /*src = i8 src*/ op.getSrc(), fstRound, b.getF16Type());
  assert(fstCast.getDst().size() == 1);
  assert(fstCast.getResult().empty());

  // create f16 buffer for dst of new deinterleave
  SmallVector<Value> newDestRange(op.getDst().size());
  for (const auto &[idx, dst] : llvm::enumerate(op.getDst())) {
    assert(getElementTypeOrSelf(dst.getType()).isInteger(8));

    newDestRange[idx] = mlir::utils::createTmpBufferOrTensorWithTargetType(
        b, loc, dst, b.getF16Type());
  }

  // create the new deinterleave op
  b.create<hivm::VDeinterleaveOp>(loc, TypeRange({}),
                                  /*src = f16 casted*/ fstCast.getDst()[0],
                                  /*dst = f16 temp*/ newDestRange,
                                  op.getChannelNumAttr(),
                                  op.getIndexModeAttr());

  // create second casts to convert hivm.deinterleave results back to i8.
  // * for memref operands, the result is stored in new dst of op
  // * the old op's dst is in shaped type of i8, which fits in the dst of second
  //   cast op.
  // * cast result types are the result types of old op.
  for (const auto &[idx, newOpResult] : llvm::enumerate(newDestRange)) {
    b.create<hivm::VCastOp>(loc, TypeRange({}),
                            /*src = f16 temp*/ newOpResult,
                            /*dst = i8 dst*/ op.getDst()[idx], bwdRound,
                            hivm::TypeFnAttr{});
  }

  return SmallVector<Value>();
}

} // namespace mlir::hivm

mlir::FailureOr<llvm::SmallVector<mlir::Value>>
VDeinterleaveOp::decomposeOperation(mlir::OpBuilder &b) {
  // only apply pattern for hivm.deinterleave on shaped type of i8
  const Type srcType = getSrc().getType();
  if (!getElementTypeOrSelf(srcType).isInteger(8))
    return llvm::failure();
  assert(isa<ShapedType>(srcType));

  if (isa<TensorType>(srcType))
    return decomposeTensorDeinterleave(*this, b);
  return decomposeMemRefDeinterleave(*this, b);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> LoadOp::decomposeOperation(OpBuilder &b) {
  if (!hasPureBufferSemantics())
    return failure();
  if (!getInitOutBuffer())
    return failure();

  MemRefType dstMemRefTy = cast<MemRefType>(getDst().getType());
  auto toAddrSpace = cast<hivm::AddressSpaceAttr>(dstMemRefTy.getMemorySpace());
  if (toAddrSpace.getAddressSpace() != hivm::AddressSpace::UB)
    return failure();

  auto maybeAlloc = traceDefOp<memref::AllocOp>(getDst());
  assert(maybeAlloc.has_value());
  auto padMemref = cast<memref::AllocOp>(maybeAlloc.value());
  auto loc = getLoc();
  if (getInitCondition()) {
    scf::IfOp ifOp =
        b.create<scf::IfOp>(getLoc(), TypeRange(), getInitCondition(), false);
    OpBuilder::InsertionGuard insertionGuard(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    b.create<hivm::VBrcOp>(loc, TypeRange(), getPadValue(), padMemref,
                           b.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
  } else {
    b.create<hivm::VBrcOp>(loc, TypeRange(), getPadValue(), padMemref,
                           b.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
  }
  b.create<hivm::LoadOp>(loc, TypeRange{}, getSrc(), getDst(), getPadModeAttr(),
                         getPadValue(), getLeftPaddingNum(), false,
                         getMayImplicitTransposeWithLastAxis());
  return SmallVector<Value>{};
}

//===----------------------------------------------------------------------===//
// ND2NZOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> ND2NZOp::decomposeOperation(OpBuilder &b) {
  if (!hasPureBufferSemantics())
    return failure();
  if (!getInitOutBuffer())
    return failure();
  auto maybeAlloc = traceDefOp<memref::AllocOp>(getDst());
  assert(maybeAlloc.has_value());
  auto padMemref = cast<memref::AllocOp>(maybeAlloc.value());
  auto loc = getLoc();
  if (getInitCondition()) {
    scf::IfOp ifOp =
        b.create<scf::IfOp>(getLoc(), TypeRange(), getInitCondition(), false);
    OpBuilder::InsertionGuard insertionGuard(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    b.create<hivm::VBrcOp>(loc, TypeRange(), getPadValue(), padMemref,
                           b.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
  } else {
    b.create<hivm::VBrcOp>(loc, TypeRange(), getPadValue(), padMemref,
                           b.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
  }
  b.create<hivm::ND2NZOp>(loc, TypeRange{}, getSrc(), getDst(),
                          b.getUnitAttr());
  return SmallVector<Value>{};
}

//===----------------------------------------------------------------------===//
// VPadOp
//===----------------------------------------------------------------------===//

/// hivm.hir.vpad ins(%src) outs(%dst) low[%low] high[%high] pad_value %cst
/// The result tensor dimensions are low[i] + dim[i] + high[i] for each
/// dimension
///
/// for positive low & high:
///
///                  ------------------------------
/// src:             |0102030405060708091011121314|
///                  ------------------------------
///                   <-------- srcSize --------->
///                              ||
///                              \/
///      -------------------------------------------------------
/// dst: |PPPPPPPPPPPP0102030405060708091011121314PPPPPPPPPPPPP|
///      -------------------------------------------------------
///       <---low---><--------- srcSize ---------><-- high --->
///
/// For negative low, positive high, and |low| < srcSize:
///
///                  ------------------------------
/// src:             |0102030405060708091011121314|
///                  ------------------------------
///                   <--------- srcSize -------->
///                              ||
///                              \/
///                             --------------------------------
/// dst:              <-|low|-->|60708091011121314PPPPPPPPPPPPP|
///                             --------------------------------
///                   <--------- srcSize --------><-- high --->
///
/// For negative low, positive high, and |low| > srcSize:
///
///                  ------------------------------
/// src:             |0102030405060708091011121314|
///                  ------------------------------
///                   <-------- srcSize --------->
///                              ||
///                              \/
///                                                   ----------
/// dst:              <----------|low|--------------->|PPPPPPPP|
///                                                   ----------
///                   <--------- srcSize --------><-- high --->
///
/// Note that we assume the result tensor dimensions are all non-negative, i.e.
/// dst = low + dim + high >= 0. In this case the pad op is equivalent to a
/// broadcast op that fills result with padValue
///
/// The case of positive low, negative high is similar to the case above
///
/// For negative low, negative high
///
///                  ------------------------------
/// src:             |0102030405060708091011121314|
///                  ------------------------------
///                   <--------- srcSize -------->
///                              ||
///                              \/
///                            ---------
/// dst:              <-|low|->|0607080|<-|high|->
///                            ---------
///                   <--------- srcSize -------->
/// Note we assume that the result tensor dimensions are all non-negative, i.e.
/// dst = low + dim + high >= 0. In this case the pad op is equivalent to
/// slicing the src
namespace {
/// This class wraps the logic for decomposing vpad op that has non-zero low or
/// high on one single dimension. It will generate a broadcast op to fill the
/// low (beginning of dst tensor) with padValue, a broadcast op to fill the high
/// (end of dst tensor) with padValue, and a copy op to copy mid (middle of dst
/// tensor that corresponds to original src tensor) from src to dst.
class VPadOpDecomposer {
public:
  VPadOpDecomposer(const VPadOpDecomposer &) = delete;
  VPadOpDecomposer &operator=(const VPadOpDecomposer &) = delete;
  VPadOpDecomposer(VPadOpDecomposer &&) = delete;
  VPadOpDecomposer &operator=(VPadOpDecomposer &&) = delete;

  static inline FailureOr<SmallVector<Value>> run(VPadOp op,
                                                  OpBuilder &builder) {
    if (op.hasPureTensorSemantics()) {
      return failure();
    }
    ArrayRef<int64_t> staticLowPad = op.getStaticLow();
    ArrayRef<int64_t> staticHighPad = op.getStaticHigh();
    std::optional<unsigned> optPadDim = std::nullopt;
    for (const auto &[idx, low, high] :
         llvm::enumerate(staticLowPad, staticHighPad)) {
      // If low or high is non zero or dynamic (equals to ShapedType::kDynamic =
      // -2^63, which is also non zero)
      if (low != 0 || high != 0) {
        if (optPadDim.has_value())
          // not support decomposing multi-dim padding for now
          return failure();
        optPadDim = idx;
      }
    }
    if (!optPadDim.has_value()) {
      // not padding on any dim: equivalent to copy op
      builder.create<hivm::CopyOp>(op->getLoc(), TypeRange(), op.getSrc(),
                                   op.getDst());
    } else {
      unsigned padDim = optPadDim.value();
      VPadOpDecomposer decomposer(op, builder);
      if (staticLowPad[padDim] != 0) // non zero or dynamic
        decomposer.broadcastLow(padDim);
      if (staticHighPad[padDim] != 0) // non zero or dynamic
        decomposer.broadcastHigh(padDim);
      decomposer.copySrcToDst(padDim);
    }
    return SmallVector<Value>{};
  }

private:
  OpBuilder &builder;
  VPadOp op;
  // All the fields below are shorthands of the fields in VPadOp
  const SmallVector<OpFoldResult> mixedLowPad, mixedHighPad, srcSizes, dstSizes;

  const SmallVector<OpFoldResult> allOneVecStrides;

  VPadOpDecomposer(VPadOp op, OpBuilder &builder)
      : builder(builder), op(op), mixedLowPad(op.getMixedLowPad()),
        mixedHighPad(op.getMixedHighPad()),
        srcSizes(memref::getMixedSizes(builder, op->getLoc(), op.getSrc())),
        dstSizes(memref::getMixedSizes(builder, op->getLoc(), op.getDst())),
        allOneVecStrides(srcSizes.size(), builder.getIndexAttr(1)) {}

  /// Saturates an affine expression to the range [0, maxValue].
  /// This implements: result = min(max(0, expr), maxValue)
  /// Where expr is a formula that we are going to compute
  ///
  /// Cases handled:
  /// - If expr < 0: returns 0 (clamps negative values to zero)
  /// - If 0 <= expr <= maxValue: returns expr (value is already in valid range)
  /// - If expr > maxValue: returns maxValue (clamps oversized values to
  /// maximum)
  ///
  /// This prevents negative indices and out-of-bounds access, ensures iteration
  /// stays within valid tensor dimensions, guarantees addresses stay within
  /// allocated regions
  ///
  /// \param expr The affine expression to saturate (e.g., computed index)
  /// \param operands The operand values for evaluating the expression
  /// \param maxValue The upper bound (e.g., array size - 1, tensor dimension)
  /// \return Saturated value guaranteed to be in range [0, maxValue]
  inline OpFoldResult getSaturatedIndex(AffineExpr expr,
                                        ArrayRef<OpFoldResult> operands,
                                        OpFoldResult maxValue) {
    const AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineMap saturatingMap =
        AffineMap::get(0, operands.size(), {expr, zero}, builder.getContext());
    OpFoldResult positiveIdx = affine::makeComposedFoldedAffineMax(
        builder, op->getLoc(), saturatingMap, operands);
    return affine::makeComposedFoldedAffineMin(
        builder, op->getLoc(),
        AffineMap::getMultiDimIdentityMap(2, builder.getContext()),
        {positiveIdx, maxValue});
  }

  /// Creates a subview on the destination tensor corresponding to the "low"
  /// padding region
  /// and broadcasts it with the pad value.
  ///
  /// This handles the beginning portion of the padded tensor where we need to
  /// fill with pad values before the actual source data starts.
  ///
  /// Cases handled:
  /// - If low < 0: broadcasts an empty subview--subview with dimSize 0 at its
  /// padDim (no low padding needed)
  /// - If low > dstSize: the entire destination is low padding (source data
  /// doesn't fit)
  /// - If 0 <= low <= dstSize: broadcasts exactly 'low' elements at the
  /// beginning
  ///
  /// The subview always starts at offset 0 and extends for min(max(0, low),
  /// dstSize) elements in the padding dimension, while maintaining source sizes
  /// and zero offsets in other dimensions.
  ///
  /// \param padDim The dimension along which padding is being applied
  inline void broadcastLow(const unsigned padDim) {
    // offsets of subview: all zero (start from beginning of destination)
    const SmallVector<OpFoldResult> lowOffsets(srcSizes.size(),
                                               builder.getIndexAttr(0));
    SmallVector<OpFoldResult> lowPadSizes = srcSizes;
    lowPadSizes[padDim] =
        getSaturatedIndex(getIdExpr(), {mixedLowPad[padDim]},
                          /*maxValue = dstSize*/ dstSizes[padDim]);
    auto subviewLow = builder.create<memref::SubViewOp>(
        op->getLoc(), /*source=*/op.getDst(), /*offsets=*/lowOffsets,
        /*sizes=*/lowPadSizes, /*strides=*/allOneVecStrides);
    builder.create<hivm::VBrcOp>(op->getLoc(), TypeRange(), op.getPadValue(),
                                 subviewLow);
  }

  /// Creates a subview on the destination tensor corresponding to the "high"
  /// padding region and broadcasts it with the pad value.
  ///
  /// This handles the end portion of the padded tensor where we need to fill
  /// with pad values after the actual source data ends.
  ///
  /// Cases handled:
  /// - If high < 0: broadcasts an empty subview (no high padding needed)
  ///   subview size at padDim dimension is 0
  /// Let's define remainingSize as in (lowPad + srcSize)
  /// - If remainingSize >= destSize: no room for high padding (offset
  ///   at/beyond end)
  /// - If high > remaining space: clamps high padding to available space
  /// - Normal case: broadcasts 'high' elements starting after source data
  ///
  /// The subview starts at offset (low + srcSize) and extends for the remaining
  /// space in the padding dimension, while maintaining source sizes and zero
  /// offsets in other dimensions.
  ///
  /// @param padDim The dimension along which padding is being applied
  inline void broadcastHigh(const unsigned padDim) {
    // offsets of subview: On padDim if satisfies 0 <= srcSize+low <=
    // low+srcSize+high; On other dims trivially, zero
    SmallVector<OpFoldResult> rightOffsets(srcSizes.size(),
                                           builder.getIndexAttr(0));
    rightOffsets[padDim] = getSaturatedIndex(
        getBinAddExpr(), {mixedLowPad[padDim], srcSizes[padDim]},
        /*maxValue = dstSize*/ dstSizes[padDim]);
    SmallVector<OpFoldResult> rightPadSizes = srcSizes;
    rightPadSizes[padDim] =
        getSaturatedIndex(getIdExpr(), {mixedHighPad[padDim]},
                          /*maxValue = dstSize*/ dstSizes[padDim]);
    auto subviewRight = builder.create<memref::SubViewOp>(
        op->getLoc(), /*source=*/op.getDst(), /*offsets=*/rightOffsets,
        /*sizes=*/rightPadSizes, /*strides=*/allOneVecStrides);
    builder.create<hivm::VBrcOp>(op->getLoc(), TypeRange(), op.getPadValue(),
                                 subviewRight);
  }

  /// Creates a subview on the destination tensor corresponding to the "mid"
  /// region and copied it with the values from source tensor.
  ///
  /// Cases handled:
  /// - Normal case: copy 'srcSize' elements from source to destination
  /// - If high + srcSize < 0 or low + srcSize < 0: copies an empty subview (no
  /// copy needed). Subview size at padDim dimension is 0
  /// - Otherwise,
  ///   If high < 0, low >= 0: copies source tensor from 0 to srcSize - |high|
  ///   If low < 0, high >= 0: copies source tensor from |low| to srcSize
  ///   If low < 0, high < 0: copies source tensor from |low| to srcSize -
  ///   |high|
  ///
  /// In the fomula below, we use " a[[ b ]]c " to represent the result of b
  /// after saturated into range [a,c], i.e. "a [[ b ]] c" = min(max(a,b), c)
  inline void copySrcToDst(const unsigned padDim) {
    // Gettting the subview on SRC:

    // offsets of the subview on SRC: On padDim, offset = 0[[ -low ]]srcSize;
    // On other dims, all zero. Note that:
    // *       If low < 0, then src is truncated from |low|.
    // *       If srcSize < |low|, meaning that src won't contribute to dst. The
    // subview from SRC is empty.
    SmallVector<OpFoldResult> midOffsetsOnSrc(srcSizes.size(),
                                              builder.getIndexAttr(0));
    midOffsetsOnSrc[padDim] = getSaturatedIndex(
        getNegExpr(), {mixedLowPad[padDim]}, /*srcSize*/ srcSizes[padDim]);

    // Sizes of the subview on SRC: On padDim, size = rIdx - offset, where
    // rIdx = 0[[srcSize + high]]srcSize; on other dims, equal to SRC sizes.

    // Here we prove that the subview is a valid subview on SRC. Note that
    // offset + size = rIdx - offset + offset = rIdx < srcSize, so it is
    // sufficient to prove by showing (*) size >= 0.
    // *      If low > 0, high > 0, meaning that src is fully copied. Here
    // offset = 0, size = rIdx = 0[[srcSize + high]]srcSize = srcSize  (*)
    // *      If low > 0, high < 0, meaning that src is truncated from high.
    // Here offset = 0, so size = rIdx  (*)
    // *      If low < 0, high > 0, meaning that src is truncated from low. Here
    // rIdx = srcSize, size = srcSize - offset. Since 0 <= offset <= srcSize,
    // srcSize >= size >= 0 (*).
    // *      If low < 0, high < 0, meaning that src is truncated from both low
    // and high. However, the resulting dst is still non-neg (assume no
    // dimension in tensor can be negative), therefore the dst is purely
    // produced by copying the remaining parts of src. Then 0 <= srcSize + low +
    // high <= srcSize + high <= srcSize, and -low < srcSize. Therefore offset =
    // -low, rIdx = srcSize+high, hence size = rIdx - offset = srcSize+high+low
    // >= 0 (*), which is consistant with dst
    SmallVector<OpFoldResult> midSizesOnSrc = srcSizes;
    OpFoldResult rIdx = getSaturatedIndex(
        getBinAddExpr(), {srcSizes[padDim], mixedHighPad[padDim]},
        /*srcSize*/ srcSizes[padDim]);
    midSizesOnSrc[padDim] = affine::makeComposedFoldedAffineApply(
        builder, op->getLoc(), getBinSubExpr(),
        {rIdx, /*offset*/ midOffsetsOnSrc[padDim]});

    // using all 1 stride to create subview of SRC
    auto subviewMidOnSrc = builder.create<memref::SubViewOp>(
        op->getLoc(), /*source=*/op.getSrc(), /*offsets=*/midOffsetsOnSrc,
        /*sizes=*/midSizesOnSrc, /*strides=*/allOneVecStrides);

    // Gettting the subview on DST:

    // offsets on Dst: On padDim, offset' = " 0[[ low ]]low+srcSize+high "; on
    // other dims, all zero
    SmallVector<OpFoldResult> midOffsetsOnDst(srcSizes.size(),
                                              builder.getIndexAttr(0));
    midOffsetsOnDst[padDim] =
        getSaturatedIndex(getIdExpr(), {mixedLowPad[padDim]},
                          /*low+srcSize+high = dst*/ dstSizes[padDim]);

    // Sizes on Dst: equals to size on Src

    // Here we prove that the subview is a valid subview on DST. Since 0 <=
    // offset' <= low+srcSize+high, it is sufficient to prove by showing the
    // rightmost index rIdx' is valid in DST, i.e. rIdx' = offset' + size = 0[[
    // low ]]low+srcSize+high + 0[[srcSize + high]]srcSize - 0[[ -low ]]srcSize
    // <= dst = srcSize + low + high (*)
    // *      if low > 0, high > 0, rIdx' = low + srcSize < low + srcSize + high
    // (*)
    // *      if low < 0, high > 0, rIdx' = 0 + srcSize - min(srcSize, -low) =
    // max(0, srcSize + low) < srcSize + low + high (*)
    // *      if low > 0, high < 0, rIdx' = min(low, low+srcSize+high) +
    // max(srcSize+high, 0) = low + min(0, srcSize+high) + max(srcSize+high, 0)
    // = low + srcSize + high (*)
    // *      if low < 0, high < 0, rIdx' = 0 + max(srcSize+high, 0) - min(-low,
    // srcSize) = max(srcSize + high, 0) + max(low, -srcSize) = max(srcSize +
    // low + high, high, low, -srcSize) = srcSize + low + high (*) Therefore,
    // the DST subview is a valid subview

    // using all 1 stride to create subview of DST
    auto subviewMidOnDst = builder.create<memref::SubViewOp>(
        op->getLoc(), /*source=*/op.getDst(), /*offsets=*/midOffsetsOnDst,
        /*sizes=*/midSizesOnSrc, /*strides=*/allOneVecStrides);

    // Copy mid Src to Dst
    builder.create<hivm::CopyOp>(op->getLoc(), TypeRange(), subviewMidOnSrc,
                                 subviewMidOnDst);
  }

  // util functions
  inline AffineExpr getBinAddExpr() {
    return builder.getAffineSymbolExpr(0) + builder.getAffineSymbolExpr(1);
  }
  inline AffineExpr getBinSubExpr() {
    return builder.getAffineSymbolExpr(0) - builder.getAffineSymbolExpr(1);
  }
  inline AffineExpr getIdExpr() { return builder.getAffineSymbolExpr(0); }
  inline AffineExpr getNegExpr() { return -builder.getAffineSymbolExpr(0); }
};
} // namespace

FailureOr<SmallVector<Value>> VPadOp::decomposeOperation(OpBuilder &b) {
  return VPadOpDecomposer::run(*this, b);
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {
FailureOr<SmallVector<Value>> decomposeMultiAxesVReduceOp(hivm::VReduceOp op,
                                                          OpBuilder &builder) {
  // Create tmp, which is same as src
  Value tmpOdd = mlir::utils::createTmpBufferOrTensorWithTargetType(
      builder, op.getLoc(), op.getSrc());
  Value tmpEven = mlir::utils::createTmpBufferOrTensorWithTargetType(
      builder, op.getLoc(), op.getSrc());

  auto src = op.getSrc();
  auto srcShapedType = cast<ShapedType>(src.getType());
  auto srcShapes = srcShapedType.getShape();
  // Prepare offset, sizes and strides for SubViewOp
  const SmallVector<OpFoldResult> vecOffsets(srcShapes.size(),
                                             builder.getIndexAttr(0));
  const SmallVector<OpFoldResult> vecStrides(srcShapes.size(),
                                             builder.getIndexAttr(1));
  const bool hasPureTensor = op.hasPureTensorSemantics();
  // Init sliceSizes using src
  SmallVector<OpFoldResult> sliceSizes =
      hasPureTensor ? tensor::getMixedSizes(builder, op.getLoc(), src)
                    : memref::getMixedSizes(builder, op.getLoc(), src);

  Value curSrc = src;
  auto dst = op.getDstValue();
  hivm::VReduceOp tmpReduceOp;
  const auto reduceDims = op.getReduceDims();
  const int reduceDimSize = static_cast<int>(reduceDims.size());
  // Loop from outer to inner axis.
  // The count of created VReduceOp would be reduceDimSize.
  // e.g.
  // reduceDims is [0, 2, 4], reduceDimSize = 3,
  // then loop i from 0 to 1 to 2,
  // loop i=0: src         to tmp_even_subview, reduce axis is 0,
  // loop i=1: tmp_even_subview to tmp_odd_subview, reduce axis is 2,
  // loop i=2: tmp_odd_subview to dst, reduce axis is 4.
  // Note that the final step loop2 must be tmp to dst.
  for (int i = 0; i < reduceDimSize; ++i) {
    // From the example above, dst = [tmp_even_subview, tmp_odd_subview, dst],
    // curFullDst is determined by the odd or even value of i.
    Value curFullDst = (reduceDimSize - 1 - i) % 2 == 0 ? tmpEven : tmpOdd;
    Value curDst;
    if (i == reduceDimSize - 1) {
      // No need to get subview
      curDst = dst;
    } else {
      // sliceSizes need to set the value of reduce idx to 1
      sliceSizes[reduceDims[i]] = builder.getIndexAttr(1);
      curDst = utils::getSlice(builder, op.getLoc(), curFullDst, vecOffsets,
                               sliceSizes, vecStrides);
    }

    auto singleReduceDim =
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(reduceDims[i])});
    auto curDstType = curDst.getType();
    TypeRange resTypeRange =
        hasPureTensor ? TypeRange(curDstType) : TypeRange();
    tmpReduceOp =
        builder.create<hivm::VReduceOp>(op.getLoc(), resTypeRange, curSrc,
                                        curDst, op.getArith(), singleReduceDim);

    // Update curSrc for next use in loop
    curSrc =
        hasPureTensor ? tmpReduceOp->getResult(0) : tmpReduceOp.getDstValue();
  }
  SmallVector<Value> res = {};
  if (hasPureTensor)
    res.push_back(tmpReduceOp->getResult(0));
  return res;
}
} // namespace mlir::hivm

FailureOr<SmallVector<Value>> VReduceOp::decomposeOperation(OpBuilder &b) {
  const int reduceDimSize = static_cast<int>(getReduceDims().size());
  if (reduceDimSize < 2) {
    return failure();
  }

  if (!hasPureBufferSemantics() && !hasPureTensorSemantics()) {
    return emitOpError(
        "hivm::VReduceOp should have pure buffer or tensor Semantics!");
  }

  return decomposeMultiAxesVReduceOp(*this, b);
}

//===----------------------------------------------------------------------===//
// VTransposeOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {
static Value getReshapedValue(OpBuilder &builder, Location loc, Value v,
                              llvm::ArrayRef<int64_t> newShape) {
  auto type = v.getType();
  auto elemType = cast<ShapedType>(type).getElementType();

  MemRefType newType = mlir::MemRefType::get(newShape, elemType);

  auto prevMemSpace = cast<MemRefType>(v.getType()).getMemorySpace();
  if (prevMemSpace) {
    newType = cast<MemRefType>(getBaseMemRefTypeWithNewScope(
        newType, cast<AddressSpaceAttr>(prevMemSpace)));
  }

  MemRefType idxsType = mlir::MemRefType::get(
      ArrayRef<int64_t>(newShape.size()), builder.getI64Type());

  auto gmSpaceAttr =
      AddressSpaceAttr::get(builder.getContext(), hivm::AddressSpace::UB);
  idxsType =
      cast<MemRefType>(getBaseMemRefTypeWithNewScope(idxsType, gmSpaceAttr));

  auto memrefIdxs = builder.create<memref::AllocOp>(loc, idxsType);
  for (auto [idx, dim] : llvm::enumerate(newShape)) {
    auto idxValue = builder.create<arith::ConstantIndexOp>(loc, idx);
    auto dimValue =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(dim));

    builder.create<memref::StoreOp>(loc, dimValue, memrefIdxs,
                                    ValueRange{idxValue});
  }

  Value reshapedValue =
      builder.create<memref::ReshapeOp>(loc, newType, v, memrefIdxs);

  return reshapedValue;
}

static Value reshapeAndTranspose2DStep(OpBuilder &builder, Location loc,
                                       Value v, int64_t reshapeCoef) {
  auto shapedType = cast<ShapedType>(v.getType());
  auto elemType = shapedType.getElementType();

  auto inputShape = shapedType.getShape();
  auto rank = inputShape.size();

  // 1. (Y*C, X) reshape to (Y, C*X)
  llvm::SmallVector<int64_t> newShape(inputShape.begin(), inputShape.end());
  newShape[rank - 2] /= reshapeCoef;
  newShape[rank - 1] *= reshapeCoef;

  auto reshapedValue = getReshapedValue(builder, loc, v, newShape);

  // 2. (Y, C*X) transpsoe to (C*X, Y)
  std::swap(newShape[rank - 2], newShape[rank - 1]);
  auto transposedBuffer = utils::createTmpBufferOrTensorWithTargetType(
      builder, loc, reshapedValue, elemType, newShape);

  auto permArray = llvm::to_vector(llvm::iota_range<int64_t>(0, rank, false));
  std::swap(permArray[rank - 2], permArray[rank - 1]);

  auto defaultPerm = builder.getDenseI64ArrayAttr(permArray);
  auto transposeOp = builder.create<hivm::VTransposeOp>(
      loc, TypeRange(), reshapedValue, transposedBuffer, defaultPerm);

  return transposeOp.getDst();
}

static FailureOr<SmallVector<Value>>
decomposeUnalignTransposeOp(VTransposeOp op, OpBuilder &builder) {
  if (!op.getDisableAlign()) {
    return failure();
  }

  if (!op.hasPureBufferSemantics()) {
    return failure();
  }

  auto src = op.getSrc();
  auto srcMemrefType = cast<MemRefType>(src.getType());
  auto srcSpace = srcMemrefType.getMemorySpace();
  if (!srcSpace) {
    return failure();
  }

  std::vector<std::unique_ptr<OperAlignInfo>> alignList;
  if (getUnAlignSizeInfo(op, &alignList).failed()) {
    return failure();
  }

  auto inputType = cast<ShapedType>(op.getSrc().getType());

  auto inputShape = inputType.getShape();

  auto elemTypeBytes =
      inputType.getElementTypeBitWidth() / mlir::utils::INTR_BITS_PER_BYTE;

  SmallVector<int64_t> transposeLoopDims;
  op.getTransposeLoopDims(transposeLoopDims);

  auto lastAlign = alignList[0]->alignBytes[0] / static_cast<int>(elemTypeBytes);
  auto lastDim = inputShape[transposeLoopDims[1]];

  // (K * lastAlign, lastDim) -> (K, lastAlign * lastDim) -> (lastAlign *
  // lastDim, K)
  auto firstStepResult =
      reshapeAndTranspose2DStep(builder, op.getLoc(), op.getSrc(), lastAlign);

  // (lastAlign * lastDim, K) -> (lastAlign, lastDim * K) -> (lastDim * K,
  // lastAlign)
  auto secondStepResult = reshapeAndTranspose2DStep(builder, op.getLoc(),
                                                    firstStepResult, lastDim);

  // (lastDim * K, lastAlign) -> (lastDim, K * lastAlign)
  auto rank = inputShape.size();
  auto transposedInputShape = llvm::to_vector(inputShape);
  std::swap(transposedInputShape[rank - 2], transposedInputShape[rank - 1]);

  auto thirdStepResult = getReshapedValue(
      builder, op.getLoc(), secondStepResult, transposedInputShape);

  mlir::IRRewriter rewriter(builder);
  rewriter.replaceAllUsesWith(op.getDst(), thirdStepResult);

  return SmallVector<Value>();
}

} // namespace mlir::hivm

FailureOr<SmallVector<Value>> VTransposeOp::decomposeOperation(OpBuilder &b) {
  return decomposeUnalignTransposeOp(*this, b);
}
