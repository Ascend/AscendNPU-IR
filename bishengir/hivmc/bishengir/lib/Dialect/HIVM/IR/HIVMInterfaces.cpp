//===- HIVMInterfaces.cpp - HIVM interfaces implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.cpp.inc"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.cpp.inc"
#include "bishengir/Interfaces/AggregatedOpInterface.cpp.inc"

using namespace mlir;
using namespace mlir::hivm;

namespace mlir {
namespace hivm {
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

/// This function will return the mask of the contiguous axes
BitVector getContiguousAxesImpl(Operation *op) {
  auto shapedTypes = getHIVMOperandTypesImpl(op);
  return getContiguousAxesImpl(shapedTypes);
}

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

BitVector getPermutedAxesMaskImpl(Operation *op) {
  auto permutationArray = getPermutationArray(op);
  auto shapedTypes = getHIVMOperandTypesImpl(op);
  BitVector ret(permutationArray.size());
  for (const auto &[idx, val] : llvm::enumerate(permutationArray)) {
    ret[idx] = static_cast<int64_t>(idx) != val;
  }
  return ret;
}

LogicalResult verifyStructuredOpInterface(Operation *op) {
  auto hivmStructuredOp = cast<HIVMStructuredOp>(op);
  // TODO: Support expressing transpose and broadcast behavior at the same time.
  if (hivmStructuredOp.existInlineBroadcastLoopDims() &&
      hivmStructuredOp.existInlineTransposeLoopDims())
    return op->emitError() << "Broadcast OTF and Transpose OTF cannot be "
                              "enabled at the same time";

  return success();
}

LogicalResult verifyUnitFlagEnabledInterface(UnitFlagEnabledInterface op) {
  if (auto unitFlagModesOpt = op.getUnitFlagModes()) {
    auto unitFlagModes = unitFlagModesOpt.value();
    if (auto unitFlagCondsOpt = op.getUnitFlagConditions()) {
      auto unitFlagConds = unitFlagCondsOpt.value();
      if (unitFlagModes.size() != unitFlagConds.size()) {
        return op.emitError() << "Number of unit-flag conditions should be "
                                 "equal to the number of unit-flag modes.";
      }
    } else if (unitFlagModes.size() > 1) {
      return op.emitError() << "Cannot have multiple unit-flag modes without "
                               "having equal number of unit-flag conditions.";
    }
  } else if (auto unitFlagCondsOpt = op.getUnitFlagConditions()) {
    return op.emitError()
           << "Cannot have unit-flag conditions without unit-flag modes.";
  }
  return llvm::success();
}

Value getUnitFlagModeLibValueImpl(UnitFlagEnabledInterface op,
                                  PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto getUnitFlagValue = [&op, &rewriter](UNIT_FLAG unitFlagMode) {
    return rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getI8Type(),
        rewriter.getI8IntegerAttr(static_cast<uint8_t>(unitFlagMode)));
  };

  auto unitFlagModesOpt = op.getUnitFlagModes();
  if (!unitFlagModesOpt.has_value() || unitFlagModesOpt->empty()) {
    return getUnitFlagValue(UNIT_FLAG::DISABLED);
  }
  auto unitFlagModes = unitFlagModesOpt.value();

  auto unitFlagConditionsOpt = op.getUnitFlagConditions();
  if (!unitFlagConditionsOpt.has_value() ||
      unitFlagConditionsOpt.value().empty()) {
    assert(unitFlagModes.size() == 1);
    return getUnitFlagValue(unitFlagModes.front().getUnitFlag());
  }
  auto unitFlagConditions = unitFlagConditionsOpt.value();

  Value retValue = getUnitFlagValue(UNIT_FLAG::DISABLED);
  assert(unitFlagModes.size() == unitFlagConditions.size());
  for (auto [unitFlagMode, condValue] :
       llvm::reverse(llvm::zip(unitFlagModes, unitFlagConditions))) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value selectOp = rewriter.create<arith::SelectOp>(
        op->getLoc(), condValue, getUnitFlagValue(unitFlagMode.getUnitFlag()),
        retValue);
    retValue = selectOp;
  }
  return retValue;
}

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

SmallVector<int64_t, 4> HIVMStructuredOp::createFlatListOfOperandStaticDims() {
  SmallVector<int64_t, 4> res;
  assert(!hasDynamicShape() && "expected operands to have static shapes");
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    llvm::append_range(res, getShape(&opOperand));
  return res;
}

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
