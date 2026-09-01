//===- AveLoopAnalysis.cpp - AVE loop analysis utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/Utils/AveLoopAnalysis.h"

#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace mlir;
using namespace mlir::hivmave;

namespace mlir {
namespace hivmave {

AveLoopAnalysis::AveLoopAnalysis(scf::ForOp loop)
    : loopOp(loop.getOperation()), loopBody(loop.getBody()),
      inductionVar(loop.getInductionVar()), step(loop.getStep()) {}

std::optional<int64_t> AveLoopAnalysis::getAffineExprCoefficient(
    AffineExpr expr, ArrayRef<std::optional<int64_t>> dimCoefficients,
    ArrayRef<std::optional<int64_t>> symbolCoefficients) const {
  if (isa<AffineConstantExpr>(expr))
    return 0;
  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return dimCoefficients[dim.getPosition()];
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr))
    return symbolCoefficients[symbol.getPosition()];

  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary)
    return std::nullopt;
  std::optional<int64_t> lhs = getAffineExprCoefficient(
      binary.getLHS(), dimCoefficients, symbolCoefficients);
  std::optional<int64_t> rhs = getAffineExprCoefficient(
      binary.getRHS(), dimCoefficients, symbolCoefficients);
  if (!lhs || !rhs)
    return std::nullopt;

  switch (expr.getKind()) {
  case AffineExprKind::Add:
    return *lhs + *rhs;
  case AffineExprKind::Mul:
    if (auto lhsConstant = dyn_cast<AffineConstantExpr>(binary.getLHS()))
      return lhsConstant.getValue() * *rhs;
    if (auto rhsConstant = dyn_cast<AffineConstantExpr>(binary.getRHS()))
      return rhsConstant.getValue() * *lhs;
    return std::nullopt;
  default:
    // Floor/mod/ceil of an IV-dependent expression is not globally linear.
    return *lhs == 0 && *rhs == 0 ? std::optional<int64_t>(0) : std::nullopt;
  }
}

std::optional<int64_t>
AveLoopAnalysis::getInductionVarCoefficient(Value value) const {
  if (value == inductionVar)
    return 1;
  if (getConstantIntValue(value))
    return 0;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == loopBody)
      return std::nullopt;
    return 0;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !loopOp->isAncestor(defOp))
    return 0;

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return getInductionVarCoefficient(castOp.getIn());
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = getInductionVarCoefficient(addOp.getLhs());
    auto rhs = getInductionVarCoefficient(addOp.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs + *rhs;
  }
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto lhs = getInductionVarCoefficient(subOp.getLhs());
    auto rhs = getInductionVarCoefficient(subOp.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs - *rhs;
  }
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    if (auto lhsConstant = getConstantIntValue(mulOp.getLhs())) {
      auto rhs = getInductionVarCoefficient(mulOp.getRhs());
      return rhs ? std::optional<int64_t>(*lhsConstant * *rhs) : std::nullopt;
    }
    if (auto rhsConstant = getConstantIntValue(mulOp.getRhs())) {
      auto lhs = getInductionVarCoefficient(mulOp.getLhs());
      return lhs ? std::optional<int64_t>(*rhsConstant * *lhs) : std::nullopt;
    }
    return std::nullopt;
  }
  if (auto applyOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
    AffineMap map = applyOp.getAffineMap();
    if (map.getNumResults() != 1)
      return std::nullopt;

    SmallVector<std::optional<int64_t>> coefficients;
    coefficients.reserve(applyOp.getMapOperands().size());
    for (Value operand : applyOp.getMapOperands())
      coefficients.push_back(getInductionVarCoefficient(operand));
    ArrayRef<std::optional<int64_t>> coefficientRef(coefficients);
    return getAffineExprCoefficient(
        map.getResult(0), coefficientRef.take_front(map.getNumDims()),
        coefficientRef.drop_front(map.getNumDims()));
  }
  return std::nullopt;
}

std::optional<int64_t>
AveLoopAnalysis::getLinearizedAccessStride(Value base, ValueRange indices,
                                           MemRefType memrefType) const {
  int64_t linearCoefficient = 0;

  auto accumulateOffsets =
      [this, &linearCoefficient](MemRefType type,
                                 ArrayRef<OpFoldResult> offsets) -> bool {
    SmallVector<int64_t> strides;
    int64_t baseOffset = 0;
    if (failed(getStridesAndOffset(type, strides, baseOffset)) ||
        strides.size() != offsets.size())
      return false;
    for (auto [offset, stride] : llvm::zip(offsets, strides)) {
      if (ShapedType::isDynamic(stride))
        return false;
      auto offsetValue = dyn_cast<Value>(offset);
      if (!offsetValue)
        continue;
      auto coefficient = getInductionVarCoefficient(offsetValue);
      if (!coefficient)
        return false;
      linearCoefficient += *coefficient * stride;
    }
    return true;
  };

  SmallVector<OpFoldResult> accessIndices;
  accessIndices.reserve(indices.size());
  for (Value index : indices)
    accessIndices.push_back(index);
  if (!accumulateOffsets(memrefType, accessIndices))
    return std::nullopt;

  Value currentMemref = base;
  while (auto subview = currentMemref.getDefiningOp<memref::SubViewOp>()) {
    auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
    if (!sourceType ||
        !accumulateOffsets(sourceType, subview.getMixedOffsets()))
      return std::nullopt;
    currentMemref = subview.getSource();
  }

  if (auto blockArg = dyn_cast<BlockArgument>(currentMemref)) {
    if (blockArg.getOwner() == loopBody)
      return std::nullopt;
  } else if (Operation *defOp = currentMemref.getDefiningOp();
             defOp && loopOp->isAncestor(defOp)) {
    return std::nullopt;
  }
  return linearCoefficient;
}

LoopAccessContinuity
AveLoopAnalysis::analyzeAccessContinuity(Value base, ValueRange indices,
                                         MemRefType memrefType,
                                         VectorType vectorType) const {
  std::optional<int64_t> ivStride =
      getLinearizedAccessStride(base, indices, memrefType);
  std::optional<int64_t> loopStep = getConstantIntValue(step);
  if (!ivStride || !loopStep)
    return LoopAccessContinuity::Unknown;

  // For index = IV, step = 64 and vector<64xT>, this proves 1 * 64 == 64.
  int64_t addressDelta = *ivStride * *loopStep;
  return addressDelta == vectorType.getNumElements()
             ? LoopAccessContinuity::Contiguous
             : LoopAccessContinuity::NonContiguous;
}

} // namespace hivmave
} // namespace mlir
