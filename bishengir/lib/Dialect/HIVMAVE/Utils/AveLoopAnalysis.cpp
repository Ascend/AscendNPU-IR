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
#include <algorithm>
#include <cmath>
#include <optional>

using namespace mlir;
using namespace mlir::hivmave;

namespace {

constexpr float kDefaultInstructionCost = 1.0f;
// Keep the same relative scale as VFFusion's normalized execution intervals:
// common vector instructions cost one unit and f32 long-latency math costs
// four.
constexpr float kExpensiveInstructionCost = 4.0f;
constexpr float kCostEpsilon = 1.0e-4f;

static std::optional<int64_t>
getAffineExprCoefficient(AffineExpr expr,
                         ArrayRef<std::optional<int64_t>> dimCoefficients,
                         ArrayRef<std::optional<int64_t>> symbolCoefficients) {
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

static std::optional<int64_t> getValueCoefficient(Value value, Value iv,
                                                  scf::ForOp loop) {
  if (value == iv)
    return 1;
  if (getConstantIntValue(value))
    return 0;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == loop.getBody())
      return std::nullopt;
    return 0;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !loop->isAncestor(defOp))
    return 0;

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return getValueCoefficient(castOp.getIn(), iv, loop);
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = getValueCoefficient(addOp.getLhs(), iv, loop);
    auto rhs = getValueCoefficient(addOp.getRhs(), iv, loop);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs + *rhs;
  }
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto lhs = getValueCoefficient(subOp.getLhs(), iv, loop);
    auto rhs = getValueCoefficient(subOp.getRhs(), iv, loop);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs - *rhs;
  }
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    if (auto lhsConstant = getConstantIntValue(mulOp.getLhs())) {
      auto rhs = getValueCoefficient(mulOp.getRhs(), iv, loop);
      return rhs ? std::optional<int64_t>(*lhsConstant * *rhs) : std::nullopt;
    }
    if (auto rhsConstant = getConstantIntValue(mulOp.getRhs())) {
      auto lhs = getValueCoefficient(mulOp.getLhs(), iv, loop);
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
      coefficients.push_back(getValueCoefficient(operand, iv, loop));
    ArrayRef<std::optional<int64_t>> coefficientRef(coefficients);
    return getAffineExprCoefficient(
        map.getResult(0), coefficientRef.take_front(map.getNumDims()),
        coefficientRef.drop_front(map.getNumDims()));
  }
  return std::nullopt;
}

static std::optional<int64_t> getLinearizedIvStride(Value base,
                                                    ValueRange indices,
                                                    MemRefType memrefType,
                                                    scf::ForOp loop) {
  Value iv = loop.getInductionVar();
  int64_t linearCoefficient = 0;

  auto accumulateOffsets = [&linearCoefficient, iv,
                            loop](MemRefType type,
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
      auto coefficient = getValueCoefficient(offsetValue, iv, loop);
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
    if (blockArg.getOwner() == loop.getBody())
      return std::nullopt;
  } else if (Operation *defOp = currentMemref.getDefiningOp();
             defOp && loop->isAncestor(defOp)) {
    return std::nullopt;
  }
  return linearCoefficient;
}

static float getExecuteCost(const Operation &op) {
  if (isa<VFDivOp, VFDivfOp, VFDivFHPOp, VFExpOp, VFSqrtOp, VFRsqrtOp, VFLnOp>(
          &op))
    return kExpensiveInstructionCost;
  return kDefaultInstructionCost;
}

} // namespace

LoopAccessContinuity
hivmave::analyzeLoopAccessContinuity(scf::ForOp loop, Value base,
                                     ValueRange indices, MemRefType memrefType,
                                     VectorType vectorType) {
  std::optional<int64_t> ivStride =
      getLinearizedIvStride(base, indices, memrefType, loop);
  std::optional<int64_t> loopStep = getConstantIntValue(loop.getStep());
  if (!ivStride || !loopStep)
    return LoopAccessContinuity::Unknown;

  // For index = IV, step = 64 and vector<64xT>, this proves 1 * 64 == 64.
  int64_t addressDelta = *ivStride * *loopStep;
  return addressDelta == vectorType.getNumElements()
             ? LoopAccessContinuity::Contiguous
             : LoopAccessContinuity::NonContiguous;
}

AVEPipelineCost &AVEPipelineCost::operator+=(const AVEPipelineCost &other) {
  load += other.load;
  execute += other.execute;
  store += other.store;
  return *this;
}

AVEPipelineCost AVEPipelineCost::scaled(float factor) const {
  AVEPipelineCost result = *this;
  result.load *= factor;
  result.execute *= factor;
  result.store *= factor;
  return result;
}

float AVEPipelineCost::bottleneck() const {
  return std::max({load, execute, store});
}

AVEPipelineCost hivmave::estimateAVEPipelineCost(const Operation &op) {
  AVEPipelineCost cost;
  if (isa<VFLoadOp>(&op)) {
    cost.load = kDefaultInstructionCost;
    return cost;
  }
  if (isa<VFMaskedStoreOp, VFStoreWithStrideOp, VFUnalignedMaskedStoreOp>(
          &op)) {
    cost.store = kDefaultInstructionCost;
    return cost;
  }
  if (isa<VFPgeOp>(&op))
    return cost;
  if (isa<VFInterleaveOp, VFDeInterleaveOp>(&op) ||
      isa<AVEElementwiseOp>(&op) || isa<AVEOpWithLibraryFunction>(&op)) {
    cost.execute = getExecuteCost(op);
    return cost;
  }
  // Keep the profitability check backward compatible as the cost table grows:
  // an unclassified op contributes no cost, while known operations are still
  // compared before and after the candidate rewrite.
  return cost;
}

AVEPipelineCost hivmave::estimateLoopPipelineCost(scf::ForOp loop) {
  AVEPipelineCost result;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (op.getNumRegions() != 0)
      continue;
    Dialect *dialect = op.getDialect();
    if (!dialect || dialect->getNamespace() != "ave")
      continue;
    result += estimateAVEPipelineCost(op);
  }
  return result;
}

AVEPipelineBound
hivmave::classifyAVEPipelineBound(const AVEPipelineCost &cost) {
  float maximum = cost.bottleneck();
  unsigned maximumCount = 0;
  maximumCount += std::abs(cost.load - maximum) <= kCostEpsilon;
  maximumCount += std::abs(cost.execute - maximum) <= kCostEpsilon;
  maximumCount += std::abs(cost.store - maximum) <= kCostEpsilon;
  if (maximumCount != 1)
    return AVEPipelineBound::Balanced;
  if (std::abs(cost.load - maximum) <= kCostEpsilon)
    return AVEPipelineBound::Load;
  if (std::abs(cost.execute - maximum) <= kCostEpsilon)
    return AVEPipelineBound::Execute;
  return AVEPipelineBound::Store;
}

AVEPipelineCost hivmave::estimateLoadMergeDelta(unsigned factor) {
  AVEPipelineCost delta;
  if (factor <= 1)
    return delta;
  float eliminated = static_cast<float>(factor - 1);
  delta.load = -eliminated * kDefaultInstructionCost;
  delta.execute = eliminated * kDefaultInstructionCost;
  return delta;
}

AVEPipelineCost hivmave::estimateNarrowChainMergeDelta(
    unsigned factor, float elementwiseChainCost, unsigned packTreeCount) {
  AVEPipelineCost delta;
  if (factor <= 1)
    return delta;
  float eliminated = static_cast<float>(factor - 1);
  delta.execute = eliminated *
                  (static_cast<float>(packTreeCount) * kDefaultInstructionCost -
                   elementwiseChainCost);
  delta.store = -eliminated * kDefaultInstructionCost;
  return delta;
}

bool hivmave::isAVEPipelinePlanProfitable(const AVEPipelineCost &before,
                                          const AVEPipelineCost &delta,
                                          float minimumGain) {
  AVEPipelineCost after = before;
  after += delta;
  float gain = before.bottleneck() - after.bottleneck();
  return gain + kCostEpsilon >= minimumGain;
}
