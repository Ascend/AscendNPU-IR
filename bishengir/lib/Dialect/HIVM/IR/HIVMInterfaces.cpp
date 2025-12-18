//===- HIVMInterfaces.cpp - HIVM interfaces implementation ----------------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.cpp.inc"
#include "bishengir/Interfaces/AggregatedOpInterface.cpp.inc"

using namespace mlir;
using namespace mlir::hivm;

namespace mlir {
namespace hivm {

bool hasHWUnsupportedScalarOperandImpl(HIVMStructuredOp op) {
  auto operands = op.getHIVMOperands(/*includeExtraBuffer=*/false);

  for (unsigned i = 0; i < operands.size(); ++i) {
    Value val = operands[i]->get();
    if (!isa<ShapedType>(val.getType()))
      continue;

    // Check for the trait corresponding to this operand index.
    if ((i == 0 && op->hasTrait<OpTrait::ScalarOnlyHWTrait<0>::Impl>()) ||
        (i == 1 && op->hasTrait<OpTrait::ScalarOnlyHWTrait<1>::Impl>())) {
      return true;
    }
  }

  return false;
}

AlignKind deduceAlignmentForMemRefType(MemRefType vecType) {
  Type eleType = vecType.getElementType();
  int eleSize = static_cast<int>(eleType.getIntOrFloatBitWidth() / 8);

  AlignKind alignKind{AlignKind::UNKNOWN};
  int64_t toCheck{0};

  StridedLayoutAttr dstLayout =
      dyn_cast<StridedLayoutAttr>(vecType.getLayout());
  if (dstLayout) {
    ArrayRef<int64_t> strides = dstLayout.getStrides();
    if (strides.size() <
        2) { // if strides is less than 2, alignment is impossible
      return AlignKind::UNKNOWN;
    }

    toCheck = strides[strides.size() - 2]; // get the 2nd last strides
  } else {
    int rank = vecType.getRank();
    if (rank == 0) {
      return AlignKind::UNKNOWN;
    }

    toCheck = vecType.getDimSize(rank - 1);
  }

  if (toCheck != ShapedType::kDynamic) {
    auto isAlignedToBlock = [](int eleNum, int eleSize) {
      return eleNum * eleSize % util::BL == 0;
    };
    if (isAlignedToBlock(toCheck, eleSize)) {
      alignKind = AlignKind::ALIGN;
    } else {
      alignKind = AlignKind::UNALIGNED;
    }
  } else {
    alignKind = AlignKind::UNKNOWN;
  }

  return alignKind;
}

/* LCOV_EXCL_START */
AlignKind deduceAlignmentForDPSInitOperand(OpOperand &operand) {
  Value operandValue = operand.get();
  MemRefType maybeMemRefType = dyn_cast<MemRefType>(operandValue.getType());
  if (maybeMemRefType)
    return deduceAlignmentForMemRefType(maybeMemRefType);

  // Try deduce alignment kind for tensor.
  AlignKind alignKind{AlignKind::UNKNOWN};
  auto owner = dyn_cast<DestinationStyleOpInterface>(operand.getOwner());
  if (!owner)
    return alignKind;

  // If tied result is tagged with alignment info, return it as it is.
  Value tiedResult = owner.getTiedOpResult(&operand);
  auto markOpsWithAlignmentInfo =
      llvm::make_filter_range(tiedResult.getUsers(), [](Operation *user) {
        return isa<annotation::MarkOp>(user) &&
               user->hasAttrOfType<AlignKindAttr>(AlignKindAttr::name);
      });
  if (markOpsWithAlignmentInfo.empty())
    return alignKind;

  auto alignmentInfo =
      llvm::map_to_vector<1>(markOpsWithAlignmentInfo, [](Operation *markOp) {
        return markOp->getAttrOfType<AlignKindAttr>(AlignKindAttr::name)
            .getValue();
      });
  if (!llvm::all_equal(alignmentInfo)) {
    return AlignKind::UNKNOWN;
  }
  return alignmentInfo.front();
}
/* LCOV_EXCL_STOP */

namespace detail {

std::optional<TCoreType> queryCoreTypeHelper(Operation *op) {
  bool isCube = op->hasTrait<OpTrait::CoreTypeTrait<TCoreType::CUBE>::Impl>();
  bool isVec = op->hasTrait<OpTrait::CoreTypeTrait<TCoreType::VECTOR>::Impl>();
  bool isCubeVec =
      (isCube && isVec) ||
      op->hasTrait<OpTrait::CoreTypeTrait<TCoreType::CUBE_OR_VECTOR>::Impl>();
  if (isCubeVec) {
    return TCoreType::CUBE_OR_VECTOR;
  }
  if (isCube) {
    return TCoreType::CUBE;
  }
  if (isVec) {
    return TCoreType::VECTOR;
  }

  if (auto infer = dyn_cast<InferCoreTypeInterface>(op)) {
    return infer.inferCoreType();
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// HIVMStructuredOpInterface related methods.
//===----------------------------------------------------------------------===//

void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    HIVMStructuredOp hivmOp) {
  for (auto *operand : hivmOp.getDpsInputOperands()) {
    if (!llvm::isa<MemRefType>(operand->get().getType()))
      continue;

    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : hivmOp.getDpsInitsMutable()) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;

    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

SmallVector<OpOperand *> getHIVMOperandsImpl(Operation *op,
                                             bool includeExtraBuffer) {
  assert(op);
  if (includeExtraBuffer || !isa<ExtraBufferOpInterface>(op)) {
    // If get temp buffer but op doesn't have temp buffer, return all operands.
    // If user requested to include extra buffer, return all operands.
    return llvm::to_vector(llvm::map_range(
        op->getOpOperands(), [](OpOperand &operand) { return &operand; }));
  }
  // Exclude temp buffer
  auto extraBufferInterfaceOp = cast<ExtraBufferOpInterface>(op);
  auto bufferRange = extraBufferInterfaceOp.getExtraBuffers();
  DenseSet<Value> tempBufferList(bufferRange.begin(), bufferRange.end());
  return llvm::to_vector(llvm::map_range(
      llvm::make_filter_range(op->getOpOperands(),
                              [&tempBufferList](const OpOperand &operand) {
                                return !tempBufferList.count(operand.get());
                              }),
      [](OpOperand &operand) { return &operand; }));
}

SmallVector<Type> getHIVMOperandTypesImpl(Operation *op,
                                          bool includeExtraBuffer) {
  return llvm::to_vector(llvm::map_range(
      detail::getHIVMOperandsImpl(op, includeExtraBuffer),
      [](OpOperand *operand) { return operand->get().getType(); }));
}

SmallVector<OpOperand *> getHIVMInputOperandsImpl(Operation *op,
                                                  bool includeExtraBuffer) {
  auto hivmStructuredOp = dyn_cast<hivm::HIVMStructuredOp>(op);
  assert(hivmStructuredOp);
  if (includeExtraBuffer || !isa<ExtraBufferOpInterface>(op)) {
    // If get temp buffer but op doesn't have temp buffer, return all operands.
    // If user requested to include extra buffer, return all operands.
    return llvm::to_vector(
        llvm::map_range(hivmStructuredOp.getDpsInputOperands(),
                        [](OpOperand *operand) { return operand; }));
  }
  // Exclude temp buffer
  auto extraBufferInterfaceOp = cast<ExtraBufferOpInterface>(op);
  auto bufferRange = extraBufferInterfaceOp.getExtraBuffers();
  DenseSet<Value> tempBufferList(bufferRange.begin(), bufferRange.end());
  auto inputOperands = hivmStructuredOp.getDpsInputOperands();
  return llvm::to_vector(llvm::map_range(
      llvm::make_filter_range(inputOperands,
                              [&tempBufferList](OpOperand *operand) {
                                return !tempBufferList.count(operand->get());
                              }),
      [](OpOperand *operand) { return operand; }));
}

SmallVector<Value> getTargetSpaceOperandsImpl(Operation *op,
                                              hivm::AddressSpace hivmSpace,
                                              bool includeExtraBuffer) {
  SmallVector<Value> results;
  assert(isa<HIVMStructuredOp>(op));
  auto operands = getHIVMOperandsImpl(op, includeExtraBuffer);

  for (const auto &oper : operands) {
    auto type = oper->get().getType();
    if (!isa<MemRefType>(type)) {
      continue;
    }

    auto memrefType = cast<MemRefType>(type);
    auto addressSpaceAttr =
        dyn_cast<AddressSpaceAttr>(memrefType.getMemorySpace());
    if (hivmSpace == addressSpaceAttr.getAddressSpace()) {
      results.push_back(oper->get());
    }
  }
  return results;
}

/* LCOV_EXCL_START */
bool isVectorOnlyOperandImpl(Operation *op, size_t idx) {
  assert(isa<HIVMStructuredOp>(op));
  auto hivmOp = cast<HIVMStructuredOp>(op);
  // TODO: support op with more than 3 input operands
  switch (idx) {
  case 3:
    return hivmOp->hasTrait<mlir::OpTrait::VectorOnlyTrait<3>::Impl>();
  case 2:
    return hivmOp->hasTrait<mlir::OpTrait::VectorOnlyTrait<2>::Impl>();
  case 1:
    return hivmOp->hasTrait<mlir::OpTrait::VectorOnlyTrait<1>::Impl>();
  case 0:
    return hivmOp->hasTrait<mlir::OpTrait::VectorOnlyTrait<0>::Impl>();
  default:
    llvm_unreachable("index not supported yet");
    break;
  }
  return false;
}
/* LCOV_EXCL_STOP */

BitVector getContiguousAxesImpl(ArrayRef<Type> shapedTypes) {
  BitVector ret;
  // Presume all same rank
  for (auto type : shapedTypes) {
    auto shapedType = dyn_cast<MemRefType>(type);
    if (!shapedType)
      continue;
    int rank = shapedType.getRank();
    if (ret.empty())
      ret.resize(shapedType.getRank(), true);

    auto stridedLayout = dyn_cast<StridedLayoutAttr>(shapedType.getLayout());
    if (!stridedLayout)
      continue;
    if (stridedLayout.isIdentity())
      continue;

    auto strides = stridedLayout.getStrides();
    auto shape = shapedType.getShape();
    for (int64_t axis = 1; axis < rank; axis++) {
      // If it's dynamic its undeterminable
      if (ShapedType::isDynamic(strides[axis]) ||
          ShapedType::isDynamic(strides[axis - 1]) ||
          ShapedType::isDynamic(shape[axis])) {
        ret[axis] = false;
        continue;
      }
      // Check if its contiguous
      if (strides[axis] * shape[axis] != strides[axis - 1])
        ret[axis] = false;
    }
  }
  // First dimension is always contiguous
  if (!ret.empty())
    ret[0] = true;
  return ret;
}

/* LCOV_EXCL_START */
/// This function will return the mask of the contiguous axes
BitVector getContiguousAxesImpl(Operation *op) {
  auto shapedTypes = getHIVMOperandTypesImpl(op);
  return getContiguousAxesImpl(shapedTypes);
}
/* LCOV_EXCL_STOP */

BitVector getUnitAxesMaskImpl(MemRefType type) {
  auto shape = type.getShape();
  BitVector ret(type.getRank());
  for (int i = 0; i < type.getRank(); ++i)
    ret[i] = (shape[i] == 1);
  return ret;
}

BitVector getUnitAxesMaskImpl(ArrayRef<Type> types) {
  BitVector ret;
  for (auto type : types) {
    if (auto shapedType = dyn_cast<MemRefType>(type)) {
      if (ret.empty())
        ret.resize(shapedType.getRank(), true);
      ret &= getUnitAxesMaskImpl(shapedType);
    }
  }
  return ret;
}

/* LCOV_EXCL_START */
/// This function will return the mask of the unit axes, true if all of the
/// shape in the current axis is 1.
/// Mask doesn't guarantee it's permutation safe
/// [A, B = 1, C, D = 1]
/// [A, D = 1, C, B = 1]
/// would still return true on both ret[1] and ret[3]
BitVector getUnitAxesMaskImpl(Operation *op) {
  auto shapedTypes = getHIVMOperandTypesImpl(op);
  return getUnitAxesMaskImpl(shapedTypes);
}
/* LCOV_EXCL_STOP */

/* LCOV_EXCL_START */
BitVector getPermutedAxesMaskImpl(Operation *op) {
  auto permutationArray = getPermutationArray(op);
  auto shapedTypes = getHIVMOperandTypesImpl(op);
  BitVector ret(permutationArray.size());
  for (const auto &[idx, val] : llvm::enumerate(permutationArray)) {
    ret[idx] = static_cast<int64_t>(idx) != val;
  }
  return ret;
}
/* LCOV_EXCL_STOP */

LogicalResult verifyStructuredOpInterface(Operation *op) {
  auto hivmStructuredOp = cast<HIVMStructuredOp>(op);
  // TODO: Support expressing transpose and broadcast behavior at the same time.
  if (hivmStructuredOp.existInlineBroadcastLoopDims() &&
      hivmStructuredOp.existInlineTransposeLoopDims())
    return op->emitError() << "Broadcast OTF and Transpose OTF cannot be "
                              "enabled at the same time";

  return success();
}

/* LCOV_EXCL_START */
Value getIsFirstIterationValue(scf::ForOp forOp, Location loc,
                               PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value lowerBound = forOp.getLowerBound();
  Value currentInd = forOp.getInductionVar();
  Value isFirstIter = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lowerBound, currentInd);
  return isFirstIter;
}
/* LCOV_EXCL_STOP */

/* LCOV_EXCL_START */
Value getIsLastIterationValue(scf::ForOp forOp, Location loc,
                              PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value currentInd = forOp.getInductionVar();
  Value nextInd = rewriter.create<arith::AddIOp>(loc, currentInd, step);
  Value isLastIter = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, nextInd, upperBound);
  return isLastIter;
}
/* LCOV_EXCL_STOP */

/* LCOV_EXCL_START */
Value getSelectedUnitFlagMode(Value isEnabled, PatternRewriter &rewriter,
                              std::optional<UNIT_FLAG> isEnabledMode = {}) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(isEnabled.getDefiningOp());
  Value disabledVal = rewriter.create<arith::ConstantOp>(
      isEnabled.getLoc(), rewriter.getI8Type(),
      rewriter.getI8IntegerAttr(static_cast<uint8_t>(UNIT_FLAG::DISABLED)));
  Value enabledVal = rewriter.create<arith::ConstantOp>(
      isEnabled.getLoc(), rewriter.getI8Type(),
      rewriter.getI8IntegerAttr(static_cast<uint8_t>(
          isEnabledMode.has_value() ? isEnabledMode.value()
                                    : UNIT_FLAG::ENABLED_WITH_UPDATE)));
  return rewriter.create<arith::SelectOp>(isEnabled.getLoc(),
                                          rewriter.getI8Type(), isEnabled,
                                          enabledVal, disabledVal);
}
/* LCOV_EXCL_STOP */

/* LCOV_EXCL_START */
Value getUnitFlagModeLibValueImpl(HIVMUnitFlagEnabled op,
                                  PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  UNIT_FLAG unitFlagMode =
      op.getUnitFlagModeValue()
          .value_or(UnitFlagAttr::get(op->getContext(), UNIT_FLAG::DISABLED))
          .getUnitFlag();
  if (unitFlagMode == UNIT_FLAG::DISABLED ||
      unitFlagMode == UNIT_FLAG::ENABLED_WITH_UPDATE ||
      unitFlagMode == UNIT_FLAG::ENABLED_WITHOUT_UPDATE) {
    rewriter.setInsertionPoint(op);
    if (op.getUnitFlagModeCondition().has_value() &&
        op.getUnitFlagModeCondition().value()) {
      return getSelectedUnitFlagMode(op.getUnitFlagModeCondition().value(),
                                     rewriter, unitFlagMode);
    } else {
      return rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getI8Type(),
          rewriter.getI8IntegerAttr(static_cast<uint8_t>(unitFlagMode)));
    }
  } else if (unitFlagMode == UNIT_FLAG::ENABLED_ONLY_FIRST_ITER) {
    auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
    assert(forOp);
    Value isFirstIter = getIsFirstIterationValue(forOp, op->getLoc(), rewriter);
    return getSelectedUnitFlagMode(isFirstIter, rewriter);
  } else if (unitFlagMode == UNIT_FLAG::ENABLED_ONLY_LAST_ITER) {
    auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
    assert(forOp);
    Value isLastIter = getIsLastIterationValue(forOp, op->getLoc(), rewriter);
    return getSelectedUnitFlagMode(isLastIter, rewriter);
  } else if (unitFlagMode == UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS) {
    auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
    assert(forOp);
    Value isFirstIter = getIsFirstIterationValue(forOp, op->getLoc(), rewriter);
    Value isLastIter = getIsLastIterationValue(forOp, op->getLoc(), rewriter);
    rewriter.setInsertionPoint(op);
    Value isFirstOrLastIter =
        rewriter.create<arith::OrIOp>(op->getLoc(), isFirstIter, isLastIter);
    return getSelectedUnitFlagMode(isFirstOrLastIter, rewriter);
  } else {
    llvm_unreachable("unsupported unit-flag mode to be lowered to std");
  }
}
/* LCOV_EXCL_STOP */

} // namespace detail

SmallVector<OpFoldResult>
HIVMStructuredOp::createFlatListOfOperandDims(OpBuilder &b, Location loc) {
  SmallVector<OpFoldResult> res;
  for (OpOperand &opOperand : getOperation()->getOpOperands()) {
    for (int64_t i = 0, e = getRank(&opOperand); i < e; ++i)
      res.push_back(linalg::createFoldedDimOp(b, loc, opOperand.get(), i));
  }
  return res;
}

/* LCOV_EXCL_START */
SmallVector<int64_t, 4> HIVMStructuredOp::createFlatListOfOperandStaticDims() {
  SmallVector<int64_t, 4> res;
  assert(!hasDynamicShape() && "expected operands to have static shapes");
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    llvm::append_range(res, getShape(&opOperand));
  return res;
}
/* LCOV_EXCL_STOP */

/* LCOV_EXCL_START */
SmallVector<Range, 4> HIVMStructuredOp::createLoopRanges(OpBuilder &b,
                                                         Location loc) {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims();
  unsigned numRes = map.getNumResults();
  auto viewSizes = createFlatListOfOperandDims(b, loc);
  SmallVector<Range, 4> res(numDims);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = dyn_cast<AffineDimExpr>(result)) {
      if (res[d.getPosition()].offset)
        continue;
      res[d.getPosition()] =
          Range{b.getIndexAttr(0), viewSizes[idx], b.getIndexAttr(1)};
    }
  }
  return res;
}
/* LCOV_EXCL_STOP */

/* LCOV_EXCL_START */
SmallVector<int64_t, 4> HIVMStructuredOp::computeStaticLoopSizes() {
  AffineMap map = getLoopsToShapesMap();
  unsigned numDims = map.getNumDims();
  unsigned numRes = map.getNumResults();
  SmallVector<int64_t, 4> allShapeSizes = createFlatListOfOperandStaticDims();
  SmallVector<int64_t, 4> res(numDims, 0);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = dyn_cast<AffineDimExpr>(result))
      res[d.getPosition()] = allShapeSizes[idx];
  }
  return res;
}
/* LCOV_EXCL_STOP */

ArrayAttr detail::getIndexingMapsImpl(HIVMStructuredOp op) {
  MLIRContext *context = op.getContext();
  AffineMap scalarMap = AffineMap::get(op.getNumParallelLoops(), 0, context);
  AffineMap tensorMap =
      AffineMap::getMultiDimIdentityMap(op.getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : op->getOpOperands())
    indexingMaps.push_back(op.getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(op.getContext()).getAffineMapArrayAttr(indexingMaps);
}

/// The indexing maps based on operands
/// 1. Scalar operands (rank 0) or regular: Maps to an identity affine map
/// 2. Tensor operands with broadcast axes: Creates maps that broadcast along
///    specified dimensions by mapping them to constant 0
///
/// @param op The HIVM structured operation for which to generate indexing maps
/// @return ArrayAttr containing AffineMap attributes for each operand. Each map
///         describes how to index into the corresponding operand during the
///         elementwise operation.
ArrayAttr detail::getIndexingMapsElementwiseImpl(HIVMStructuredOp op) {
  MLIRContext *context = op.getContext();
  AffineMap scalarMap = AffineMap::get(op.getNumParallelLoops(), 0, context);
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(op.getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  auto hivmOp = cast<HIVMStructuredOp>(op.getOperation());
  for (OpOperand &opOperand : hivmOp->getOpOperands()) {
    auto operandRank = op.getRank(&opOperand);
    if (operandRank == 0) {
      indexingMaps.push_back(scalarMap);
      continue;
    }
    SmallVector<int64_t> inlineBrcAxes =
        hivmOp.getInlinedBroadcastableAxes(&opOperand);
    if (inlineBrcAxes.empty() && hivmOp.getBroadcastArray().empty()) {
      indexingMaps.push_back(identityMap);
      continue;
    }
    SmallVector<AffineExpr> outputExprs;
    DenseSet<int64_t> brcDims(inlineBrcAxes.begin(), inlineBrcAxes.end());
    for (int64_t i = 0; i < operandRank; ++i) {
      if (brcDims.contains(i)) {
        outputExprs.push_back(mlir::getAffineConstantExpr(0, context));
      } else {
        outputExprs.push_back(mlir::getAffineDimExpr(i, context));
      }
    }
    AffineMap outputMap = AffineMap::get(operandRank, 0, outputExprs, context);
    indexingMaps.push_back(outputMap);
  }

  return Builder(op.getContext()).getAffineMapArrayAttr(indexingMaps);
}

} // namespace hivm
} // namespace mlir
