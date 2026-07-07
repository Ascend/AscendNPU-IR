//===-------- NormalizeTypeConversion.cpp -----------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Transforms/Normalize/NormalizeTypeConversionTemplate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "bishengir/Dialect/Utils/Util.h"

namespace mlir {

namespace {

template <typename ElemType, typename OpType>
struct HIVMNormalizeToTargetTypeTraits;

/// Materializes a stable `SmallVector<Value>` from an MLIR value range.
/// Some HIVM DPS accessors return lightweight ranges whose iterator/sentinel
/// pair should not be fed directly into the `SmallVector(begin, end)`
/// constructor.
static SmallVector<Value> collectValues(ValueRange values) {
  SmallVector<Value> collected;
  collected.reserve(values.size());
  for (Value value : values)
    collected.push_back(value);
  return collected;
}

template <typename ElemType, typename OpType>
struct HIVMDpsNormalizeToTargetTypeTraitsBase
    : public hivm::NormalizeTraitsBase {
  static CastSignKind getCastSignKind(OpType) { return CastSignKind::Signed; }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location, OpType op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    /// Clones a DPS-style HIVM op after remapping only its semantic inputs and
    /// init operands to the widened values.
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    IRMapping mapper;
    mapper.map(dpsOp.getDpsInputs(), newInputs);
    mapper.map(dpsOp.getDpsInits(), newOutputs);
    Operation *newOp = rewriter.cloneWithoutRegions(*op, mapper);
    for (const auto &[idx, result] : llvm::enumerate(newOp->getResults())) {
      if (idx < newOutputs.size())
        result.setType(newOutputs[idx].getType());
    }
    return newOp;
  }

  static void replaceResults(PatternRewriter &rewriter, OpType op,
                             ValueRange newResults, CastSignKind intKind) {
    SmallVector<Value> castBackResults;
    castBackResults.reserve(newResults.size());
    auto outputs = getOutputs(op);
    Location loc = op.getLoc();
    for (const auto &[idx, result] : llvm::enumerate(newResults)) {
      Type targetElemType = getElementTypeOrSelf(outputs[idx].getType());
      if (!isNormalizedElemType(targetElemType)) {
        castBackResults.push_back(result);
        continue;
      }
      castBackResults.push_back(castBackHIVMNormalizedResult(
          rewriter, loc, result, targetElemType, intKind));
    }
    rewriter.replaceOp(op, castBackResults);
  }

protected:
  /// Returns true when any semantic value still uses the source element family
  /// handled by the current HIVM trait (`i1` or `i8`).
  static bool hasAnyNormalizedElemType(ValueRange values) {
    return llvm::any_of(values, [](Value value) {
      return isNormalizedElemType(getElementTypeOrSelf(value.getType()));
    });
  }

  static SmallVector<Value> getInputs(OpType op) {
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    return collectValues(dpsOp.getDpsInputs());
  }

  static SmallVector<Value> getOutputs(OpType op) {
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    return collectValues(dpsOp.getDpsInits());
  }

  static Value castBackHIVMNormalizedResult(PatternRewriter &rewriter,
                                             Location loc, Value result,
                                             Type targetElemType,
                                             CastSignKind intKind) {
    return hivm::NormalizeTraitsBase::createCastOp(
        rewriter, loc, result, targetElemType, CastRoundKind::Default, Value(),
        intKind);
  }

private:
  static bool isNormalizedElemType(Type elemType) {
    if constexpr (std::is_same_v<ElemType, bool>)
      return elemType.isInteger(1);
    if constexpr (std::is_same_v<ElemType, int8_t>)
      return elemType.isInteger(8);
    return false;
  }
};

/// Returns whether this HIVM op form carries transpose/broadcast decorations
/// that block the simple type-conversion rewrite.
template <typename OpType>
static bool hasBlockedVectorLayout(OpType op) {
  if constexpr (OpType::template hasTrait<OpTrait::BroadcastableOTF>()) {
    if (!op.getBroadcast().empty())
      return true;
  }
  if constexpr (OpType::template hasTrait<OpTrait::TransposableOTF>()) {
    if (!op.getTranspose().empty())
      return true;
  }
  return false;
}

/// HIVM models both plain reduce and reduce-with-index through `hivm::VReduce`,
/// so local traits must filter out the with-index variants explicitly.
static bool isReduceWithIndex(hivm::VReduceOp op) {
  return utils::isReduceWithIndex(op.getArith().getReduceOp());
}

struct HIVMNormalizeReduceBoolToLogicalTraits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(hivm::VReduceOp op) {
    if (!op.hasPureTensorSemantics() || isReduceWithIndex(op))
      return false;

    auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = dyn_cast<RankedTensorType>(op.getDstValue().getType());
    if (!srcType || !dstType || !srcType.getElementType().isInteger(1) ||
        !dstType.getElementType().isInteger(1))
      return false;
    return true;
  }

  static std::optional<mlir::ReduceBoolLogicalKind>
  getLogicalReduceKind(hivm::VReduceOp op) {
    switch (op.getArith().getReduceOp()) {
    case hivm::ReduceOperation::sum:
    case hivm::ReduceOperation::max:
      return mlir::ReduceBoolLogicalKind::Or;
    case hivm::ReduceOperation::prod:
    case hivm::ReduceOperation::min:
      return mlir::ReduceBoolLogicalKind::And;
    default:
      return std::nullopt;
    }
  }

  static LogicalResult rewriteToLogicalReduce(PatternRewriter &rewriter,
                                              hivm::VReduceOp op,
                                              mlir::ReduceBoolLogicalKind kind) {
    auto newOp = rewriter.create<hivm::VReduceOp>(
        op.getLoc(), TypeRange(op->getResultTypes()), op.getSrc(),
        ValueRange(op.getDpsInits()),
        hivm::ReduceOpAttr::get(
            rewriter.getContext(),
            kind == mlir::ReduceBoolLogicalKind::Or
                ? hivm::ReduceOperation::any
                : hivm::ReduceOperation::all),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getReduceDimsAttr());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct HIVMReduceI1AddToSelectMaxCompareTraits
    : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(hivm::VReduceOp op) {
    if (!hacc::utils::isAscend950(op->getParentOfType<ModuleOp>()) ||
        !op.hasPureTensorSemantics() || isReduceWithIndex(op))
      return false;

    auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = dyn_cast<RankedTensorType>(op.getDstValue().getType());
    if (!srcType || !dstType || !srcType.getElementType().isInteger(1) ||
        !dstType.getElementType().isInteger(1))
      return false;
    if (op.getArith().getReduceOp() != hivm::ReduceOperation::sum)
      return false;
    auto reduceDims = op.getReduceDims();
    if (reduceDims.size() != 1 || reduceDims[0] != 0)
      return false;
    if (srcType.getRank() != 1 || dstType.getRank() != 1 ||
        dstType.getDimSize(0) != 1)
      return false;
    return true;
  }

  static Value getInput(hivm::VReduceOp op) { return op.getSrc(); }

  static Value createPredicateTensor(PatternRewriter &rewriter, Location loc,
                                     Value likeValue, Type elemType,
                                     int64_t fillValue) {
    Value empty = utils::createEmptyOpWithTargetElemType(rewriter, loc, likeValue,
                                                         elemType);
    Value scalar = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getIntegerAttr(elemType, fillValue));
    return rewriter.create<linalg::FillOp>(loc, scalar, empty).getResult(0);
  }

  static Value createReduceInit(PatternRewriter &rewriter, Location loc,
                                hivm::VReduceOp op, Type elemType) {
    Value empty =
        utils::createEmptyOpWithTargetElemType(rewriter, loc, op.getDstValue(),
                                               elemType);
    Value scalar = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getIntegerAttr(elemType, 0));
    return rewriter.create<linalg::FillOp>(loc, scalar, empty).getResult(0);
  }

  static Value createMaxReduce(PatternRewriter &rewriter, Location loc,
                               hivm::VReduceOp op, Value input, Value init) {
    auto newOp = rewriter.create<hivm::VReduceOp>(
        loc, TypeRange{init.getType()}, input, ValueRange{init},
        hivm::ReduceOpAttr::get(rewriter.getContext(), hivm::ReduceOperation::max),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getReduceDimsAttr());
    return newOp.getResult().front();
  }

  static Value createCmpNeZero(PatternRewriter &rewriter, Location loc,
                               hivm::VReduceOp, Value reduced, Type elemType) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getIntegerAttr(elemType, 0));
    return hivm::NormalizeTraitsBase::createCmpOp(rewriter, loc, reduced, zero,
                                                  CompareKind::NE);
  }

  static void replaceResults(PatternRewriter &rewriter, hivm::VReduceOp op,
                             Value result) {
    rewriter.replaceOp(op, result);
  }
};

struct HIVMReduceI1AndOrToI16Traits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(hivm::VReduceOp op) {
    if (!hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>()) ||
        !op.hasPureTensorSemantics() || isReduceWithIndex(op))
      return false;

    auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = dyn_cast<RankedTensorType>(op.getDstValue().getType());
    if (!srcType || !dstType || !srcType.getElementType().isInteger(1) ||
        !dstType.getElementType().isInteger(1))
      return false;
    return op.getArith().getReduceOp() == hivm::ReduceOperation::all ||
           op.getArith().getReduceOp() == hivm::ReduceOperation::any;
  }

  static Value getInput(hivm::VReduceOp op) { return op.getSrc(); }

  static Value castInputToI16(PatternRewriter &rewriter, Location loc,
                              Value input, Type elemType) {
    return hivm::NormalizeTraitsBase::createCastOp(rewriter, loc, input, elemType,
                                                   CastRoundKind::Default);
  }

  static bool isAndReduce(hivm::VReduceOp op) {
    return op.getArith().getReduceOp() == hivm::ReduceOperation::all;
  }

  static Value createReduceInit(PatternRewriter &rewriter, Location loc,
                                hivm::VReduceOp op, Type elemType,
                                int64_t initValue) {
    Value empty =
        utils::createEmptyOpWithTargetElemType(rewriter, loc, op.getDstValue(),
                                               elemType);
    Value scalar = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getIntegerAttr(elemType, initValue));
    return rewriter.create<linalg::FillOp>(loc, scalar, empty).getResult(0);
  }

  static Value createReducedOp(PatternRewriter &rewriter, Location loc,
                               hivm::VReduceOp op, Value input, Value init,
                               bool isAndReduce) {
    auto newOp = rewriter.create<hivm::VReduceOp>(
        loc, TypeRange{init.getType()}, input, ValueRange{init},
        hivm::ReduceOpAttr::get(rewriter.getContext(),
                                isAndReduce ? hivm::ReduceOperation::max
                                            : hivm::ReduceOperation::min),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getReduceDimsAttr());
    return newOp.getResult().front();
  }

  static Value castResultToI1(PatternRewriter &rewriter, Location loc,
                              Value result) {
    return hivm::NormalizeTraitsBase::createCastOp(rewriter, loc, result,
                                                   rewriter.getI1Type(),
                                                   CastRoundKind::Default);
  }

  static void replaceResults(PatternRewriter &rewriter, hivm::VReduceOp op,
                             Value result) {
    rewriter.replaceOp(op, result);
  }
};

struct HIVMReduceNormalize910_95Traits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(hivm::VReduceOp op) {
    if (!hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>()) ||
        !hacc::utils::isAscend950(op->getParentOfType<ModuleOp>()) ||
        !op.hasPureTensorSemantics() || isReduceWithIndex(op))
      return false;

    auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = dyn_cast<RankedTensorType>(op.getDstValue().getType());
    if (!srcType || !dstType || !srcType.getElementType().isInteger(8) ||
        !dstType.getElementType().isInteger(8))
      return false;
    if (op.getArith().getReduceOp() == hivm::ReduceOperation::xori)
      return false;
    return true;
  }

  static CastSignKind getCastSignKind(hivm::VReduceOp op) {
    return op.getUnsignedSrc() ? CastSignKind::Unsigned : CastSignKind::Signed;
  }

  static SmallVector<Value> getInputs(hivm::VReduceOp op) {
    return {op.getSrc()};
  }

  static SmallVector<Value> getOutputs(hivm::VReduceOp op) {
    return {op.getDstValue()};
  }

  static Value createI8ToI16Cast(PatternRewriter &rewriter, Location loc,
                                 Value value, CastSignKind intKind) {
    return hivm::NormalizeTraitsBase::createCastOp(
        rewriter, loc, value, rewriter.getI16Type(), CastRoundKind::Default,
        Value(), intKind);
  }

  static Operation *rebuildOpInI16(PatternRewriter &rewriter, Location loc,
                                   hivm::VReduceOp op,
                                   SmallVector<Value> &newInputs,
                                   SmallVector<Value> &newOutputs) {
    return rewriter.create<hivm::VReduceOp>(
        loc, TypeRange{newOutputs[0].getType()}, newInputs[0],
        ValueRange{newOutputs[0]}, op.getArithAttr(), op.getUnsignedSrcAttr(),
        op.getTieBreakLeftAttr(), op.getReduceDimsAttr());
  }

  static void replaceResults(PatternRewriter &rewriter, hivm::VReduceOp op,
                             ValueRange newResults, CastSignKind intKind) {
    Value castBack = hivm::NormalizeTraitsBase::createCastOp(
        rewriter, op.getLoc(), newResults.front(),
        getElementTypeOrSelf(op.getDstValue().getType()), CastRoundKind::Default,
        Value(), intKind);
    rewriter.replaceOp(op, castBack);
  }
};

template <typename OpType>
struct HIVMNormalizeF16ToF32Traits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(OpType op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty()) {
      return false;
    }

    if constexpr (std::is_same_v<OpType, hivm::VLnOp>) {
      return true;
    }
    if constexpr (std::is_same_v<OpType, hivm::VPowOp>) {
      return true;
    }
    if constexpr (std::is_same_v<OpType, hivm::VRsqrtOp>) {
      return true;
    }
    return false;
  }

  static Value rebuildOpInF32(PatternRewriter &rewriter, Location loc, OpType op,
                              ArrayRef<Value> newInputs,
                              ArrayRef<Value> newInits) {
    auto newOp = rewriter.create<OpType>(loc, TypeRange{newInits},
                                         ValueRange{newInputs},
                                         ValueRange{newInits});
    return newOp->getResult(0);
  }
};

template <>
struct HIVMNormalizeToTargetTypeTraits<bool, hivm::VSelOp>
    : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(hivm::VSelOp op) {
    if (!op.hasPureTensorSemantics() || hasBlockedVectorLayout(op))
      return false;
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    return llvm::all_of(dpsOp.getDpsInputs(), [](Value value) {
             return getElementTypeOrSelf(value.getType()).isInteger(1);
           }) ||
           llvm::any_of(dpsOp.getDpsInits(), [](Value v) {
             return getElementTypeOrSelf(v.getType()).isInteger(1);
           });
  }

  static SmallVector<Value> getInputs(hivm::VSelOp op) {
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    return collectValues(dpsOp.getDpsInputs());
  }

  static SmallVector<Value> getOutputs(hivm::VSelOp op) {
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    return collectValues(dpsOp.getDpsInits());
  }
};

template <typename OpType>
struct HIVMNormalizeToTargetTypeTraits<bool, OpType>
    : public HIVMDpsNormalizeToTargetTypeTraitsBase<bool, OpType> {
  static bool shouldNormalize(OpType op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if (hasBlockedVectorLayout(op))
      return false;
    return std::is_same_v<OpType, hivm::VBrcOp> ||
           std::is_same_v<OpType, hivm::VTransposeOp> ||
           std::is_same_v<OpType, hivm::VInterleaveOp> ||
           std::is_same_v<OpType, hivm::VConcatOp>;
  }

  static Type getTargetType(PatternRewriter &rewriter, OpType) {
    return rewriter.getF16Type();
  }
};

template <typename OpType>
struct HIVMNormalizeToTargetTypeTraits<int8_t, OpType>
    : public HIVMDpsNormalizeToTargetTypeTraitsBase<int8_t, OpType> {
  static bool shouldNormalize(OpType op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if (hasBlockedVectorLayout(op))
      return false;

    // On Ascend950, VAdd/VSub/VMax/VMin support native i8 execution and should
    // not be widened, matching HFusion's behavior.
    if constexpr (std::is_same_v<OpType, hivm::VAddOp> ||
                  std::is_same_v<OpType, hivm::VSubOp> ||
                  std::is_same_v<OpType, hivm::VMaxOp> ||
                  std::is_same_v<OpType, hivm::VMinOp>) {
      if (hivm::archisAscend950)
        return false;
    }

    if constexpr (std::is_same_v<OpType, hivm::VBrcOp>) {
      auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
      auto inputs = dpsOp.getDpsInputs();
      if (!inputs.empty()) {
        Type srcElemType = getElementTypeOrSelf(inputs[0].getType());
        if (!srcElemType.isInteger(8))
          return false;
      }
    }

    return std::is_same_v<OpType, hivm::VAddOp> ||
           std::is_same_v<OpType, hivm::VSubOp> ||
           std::is_same_v<OpType, hivm::VMulOp> ||
           std::is_same_v<OpType, hivm::VMaxOp> ||
           std::is_same_v<OpType, hivm::VMinOp> ||
           std::is_same_v<OpType, hivm::VAbsOp> ||
           std::is_same_v<OpType, hivm::VModOp> ||
           std::is_same_v<OpType, hivm::VModUIOp> ||
           std::is_same_v<OpType, hivm::VInterleaveOp> ||
           std::is_same_v<OpType, hivm::VDeinterleaveOp> ||
           std::is_same_v<OpType, hivm::VGatherOp> ||
           std::is_same_v<OpType, hivm::VBrcOp>;
  }

  static Type getTargetType(PatternRewriter &rewriter, OpType) {
    if constexpr (std::is_same_v<OpType, hivm::VMulOp> ||
                  std::is_same_v<OpType, hivm::VModOp>)
      return rewriter.getF32Type();
    return rewriter.getF16Type();
  }

  static Value createCastOp(PatternRewriter &rewriter, Location loc,
                             Value input, Type targetElemType,
                             CastRoundKind kind, Value output,
                             CastSignKind signKind) {
    if constexpr (std::is_same_v<OpType, hivm::VBrcOp>) {
      return hivm::NormalizeTraitsBase::createCastOp(rewriter, loc, input,
                                                     targetElemType,
                                                     CastRoundKind::Trunc,
                                                     output, signKind);
    }
    return hivm::NormalizeTraitsBase::createCastOp(rewriter, loc, input,
                                                   targetElemType, kind,
                                                   output, signKind);
  }
};

template <>
struct HIVMNormalizeToTargetTypeTraits<bool, hivm::VReduceOp>
    : public HIVMDpsNormalizeToTargetTypeTraitsBase<bool, hivm::VReduceOp> {
  static bool shouldNormalize(hivm::VReduceOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if (!isReduceWithIndex(op))
      return false;
    SmallVector<Value> inputs = getInputs(op);
    SmallVector<Value> outputs = getOutputs(op);
    return hasAnyNormalizedElemType(inputs) ||
           hasAnyNormalizedElemType(outputs);
  }

  static Type getTargetType(PatternRewriter &rewriter, hivm::VReduceOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(hivm::VReduceOp op) {
    return op.getUnsignedSrc() ? CastSignKind::Unsigned : CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc, hivm::VReduceOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return rewriter.create<hivm::VReduceOp>(
        loc, TypeRange(newOutputs), newInputs[0], ValueRange(newOutputs),
        op.getArithAttr(), op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getReduceDimsAttr());
  }

  static void replaceResults(PatternRewriter &rewriter, hivm::VReduceOp op,
                             ValueRange newResults, CastSignKind intKind) {
    Value castBack = hivm::NormalizeTraitsBase::createCastOp(
        rewriter, op.getLoc(), newResults.front(),
        getElementTypeOrSelf(getOutputs(op)[0].getType()),
        CastRoundKind::Default, Value(), intKind);
    SmallVector<Value> replacements{castBack};
    replacements.append(newResults.begin() + 1, newResults.end());
    rewriter.replaceOp(op, replacements);
  }
};

template <>
struct HIVMNormalizeToTargetTypeTraits<int8_t, hivm::VReduceOp>
    : public HIVMDpsNormalizeToTargetTypeTraitsBase<int8_t, hivm::VReduceOp> {
  static bool shouldNormalize(hivm::VReduceOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if (isIntegerBitwiseReduce(op))
      return false;
    SmallVector<Value> inputs = getInputs(op);
    SmallVector<Value> outputs = getOutputs(op);
    return hasAnyNormalizedElemType(inputs) ||
           hasAnyNormalizedElemType(outputs);
  }

  static Type getTargetType(PatternRewriter &rewriter, hivm::VReduceOp op) {
    return isAddLikeReduce(op) ? static_cast<Type>(rewriter.getF32Type())
                               : static_cast<Type>(rewriter.getF16Type());
  }

  static CastSignKind getCastSignKind(hivm::VReduceOp op) {
    return op.getUnsignedSrc() ? CastSignKind::Unsigned : CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc, hivm::VReduceOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return rewriter.create<hivm::VReduceOp>(
        loc, TypeRange(newOutputs), newInputs[0], ValueRange(newOutputs),
        op.getArithAttr(), op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getReduceDimsAttr());
  }

  static void replaceResults(PatternRewriter &rewriter, hivm::VReduceOp op,
                             ValueRange newResults, CastSignKind intKind) {
    SmallVector<Value> castBackResults;
    castBackResults.reserve(newResults.size());
    auto outputs = getOutputs(op);
    Location loc = op.getLoc();
    for (const auto &[idx, result] : llvm::enumerate(newResults)) {
      if (idx > 0 && isReduceWithIndex(op)) {
        castBackResults.push_back(result);
        continue;
      }
      Type targetElemType = getElementTypeOrSelf(outputs[idx].getType());
      castBackResults.push_back(hivm::NormalizeTraitsBase::createCastOp(
          rewriter, loc, result, targetElemType, CastRoundKind::Default, Value(),
          intKind));
    }
    rewriter.replaceOp(op, castBackResults);
  }

private:
  static bool isIntegerBitwiseReduce(hivm::VReduceOp op) {
    auto reduceKind = op.getArith().getReduceOp();
    return reduceKind == hivm::ReduceOperation::xori ||
           reduceKind == hivm::ReduceOperation::ori ||
           reduceKind == hivm::ReduceOperation::andi ||
           reduceKind == hivm::ReduceOperation::any ||
           reduceKind == hivm::ReduceOperation::all;
  }

  static bool isAddLikeReduce(hivm::VReduceOp op) {
    return op.getArith().getReduceOp() == hivm::ReduceOperation::sum;
  }
};

template <>
struct HIVMNormalizeToTargetTypeTraits<bool, hivm::VCmpOp>
    : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(hivm::VCmpOp op) {
    return op.hasPureTensorSemantics() && !hasBlockedVectorLayout(op);
  }

  static SmallVector<Value> getInputs(hivm::VCmpOp op) {
    auto dpsOp = cast<DestinationStyleOpInterface>(op.getOperation());
    return collectValues(dpsOp.getDpsInputs());
  }

  static SmallVector<Value> getOutputs(hivm::VCmpOp op) { return {}; }

  static Type getTargetType(PatternRewriter &rewriter, hivm::VCmpOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(hivm::VCmpOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc, hivm::VCmpOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &) {
    return rewriter.create<hivm::VCmpOp>(
        loc, TypeRange(op->getResultTypes()), ValueRange(newInputs),
        ValueRange(op.getDpsInits()), op.getIsSigned(), op.getCompareMode(),
        op.getTransposeAttr(), op.getBroadcastAttr());
  }

  static void replaceResults(PatternRewriter &rewriter, hivm::VCmpOp op,
                             ValueRange newResults, CastSignKind) {
    rewriter.replaceOp(op, newResults);
  }
};

template <>
struct HIVMNormalizeToTargetTypeTraits<int8_t, hivm::VSelOp>
    : public HIVMDpsNormalizeToTargetTypeTraitsBase<int8_t, hivm::VSelOp> {
  // `VSel` stores the condition in the first DPS input, but this type
  // conversion only widens the true/false data lanes. The shared template must
  // skip the condition lane and keep it in the original element type.
  static bool shouldNormalize(hivm::VSelOp op) {
    return op.hasPureTensorSemantics() && !hasBlockedVectorLayout(op);
  }

  static Type getTargetType(PatternRewriter &rewriter, hivm::VSelOp) {
    return rewriter.getF16Type();
  }

  static SmallVector<Value> getInputs(hivm::VSelOp op) {
    // Skip the condition lane: only the true/false data lanes participate in
    // type widening, while the condition must stay in its original type.
    SmallVector<Value> inputs =
        HIVMDpsNormalizeToTargetTypeTraitsBase<int8_t,
                                               hivm::VSelOp>::getInputs(op);
    return SmallVector<Value>{inputs[1], inputs[2]};
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc, hivm::VSelOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    SmallVector<Value> originalInputs =
        HIVMDpsNormalizeToTargetTypeTraitsBase<int8_t,
                                               hivm::VSelOp>::getInputs(op);
    Value newCond = originalInputs[0];
    return rewriter.create<hivm::VSelOp>(
        loc, TypeRange{newOutputs[0].getType()},
        ValueRange{newCond, newInputs[0], newInputs[1]}, ValueRange(newOutputs),
        op.getTempBuffer(), op.getTransposeAttr(), op.getBroadcastAttr());
  }
};

template <typename CumOpType>
struct HIVMNormalizeCumOpTraitsBase : public hivm::NormalizeTraitsBase {
  static bool shouldNormalize(CumOpType op) {
    return op.hasPureTensorSemantics() &&
           (std::is_same_v<CumOpType, hivm::VCumsumOp> ||
            std::is_same_v<CumOpType, hivm::VCumprodOp> ||
            std::is_same_v<CumOpType, hivm::VCummaxOp> ||
            std::is_same_v<CumOpType, hivm::VCumminOp>);
  }

  static Value rebuildOpInF32(PatternRewriter &rewriter, Location loc,
                              CumOpType op, Value newInput, Value newOutput) {
    auto newOp = rewriter.create<CumOpType>(loc, TypeRange{newOutput}, newInput,
                                            newOutput, op.getCumDims(),
                                            op.getReverse());
    // Preserve the NaN-propagation flag across normalization for cummax/cummin
    // (cumsum/cumprod have no such attribute).
    if constexpr (std::is_same_v<CumOpType, hivm::VCummaxOp> ||
                  std::is_same_v<CumOpType, hivm::VCumminOp>) {
      newOp.setPropagateNan(op.getPropagateNan());
    }
    return newOp->getResult(0);
  }

protected:
  static Value getInput(CumOpType op) { return op.getSrc(); }

  static Value getOutput(CumOpType op) { return op.getDst(); }
};

struct HIVMNormalizeGatherIndexToI32Traits : public mlir::hivm::NormalizeTraitsBase {
  static bool shouldNormalize(mlir::hivm::VGatherOp) { return true; }

  static mlir::Value getIndex(mlir::hivm::VGatherOp op) { return op.getIndices(); }

  static mlir::Value castIndexToI32(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value index) {
    return mlir::hivm::NormalizeTraitsBase::createCastOp(
        rewriter, loc, index, rewriter.getI32Type(),
        mlir::CastRoundKind::Default);
  }

  static void rebuildOp(mlir::PatternRewriter &rewriter,
                        mlir::hivm::VGatherOp op, mlir::Value newIndex) {
    auto newOp = rewriter.create<mlir::hivm::VGatherOp>(
        op.getLoc(), op.getResultTypes(), op.getSrc(), newIndex,
        op.getDpsInits()[0]);
    rewriter.replaceOp(op, newOp->getResults());
  }
};

template <typename CumOpType>
struct HIVMNormalizeCumOpF16ToF32Traits
    : public HIVMNormalizeCumOpTraitsBase<CumOpType> {};

template <typename CumOpType>
struct HIVMNormalizeCumOpI8ToTargetTraits
    : public HIVMNormalizeCumOpTraitsBase<CumOpType> {};

template <typename... Patterns>
void addTypeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<Patterns...>(patterns.getContext());
}

template <typename ElemType, typename... Ops>
void addNormalizeToTargetPatterns(RewritePatternSet &patterns) {
  addTypeConversionPatterns<
      mlir::NormalizeToTargetTemplate<
          ElemType, Ops, HIVMNormalizeToTargetTypeTraits<ElemType, Ops>>...>(
      patterns);
}

} // namespace

} // namespace mlir

void mlir::hivm::populateNormalizeI1ToTargetPatterns(
    RewritePatternSet &patterns) {
  if (archisAscend950)
    addTypeConversionPatterns<mlir::ReduceI1AddToSelectMaxCompareTemplate<
        hivm::VReduceOp, HIVMReduceI1AddToSelectMaxCompareTraits>>(patterns);
  addNormalizeToTargetPatterns<bool, hivm::VInterleaveOp,
                               hivm::VBrcOp>(patterns);
  addTypeConversionPatterns<mlir::NormalizeReduceBoolToLogicalTemplate<
      hivm::VReduceOp, HIVMNormalizeReduceBoolToLogicalTraits>>(patterns);
  if (archIsRegbased) {
    addTypeConversionPatterns<
        mlir::NormalizeToTargetTypeI1SelectTemplate<
            hivm::VSelOp, HIVMNormalizeToTargetTypeTraits<bool, hivm::VSelOp>>,
        mlir::ReduceI1AndOrToI16Template<hivm::VReduceOp,
                                         HIVMReduceI1AndOrToI16Traits>>(
        patterns);
  }
  addNormalizeToTargetPatterns<bool, hivm::VCmpOp, hivm::VTransposeOp,
                               hivm::VConcatOp, hivm::VReduceOp>(patterns);
}

void mlir::hivm::populateNormalizeF16ToF32Patterns(
    RewritePatternSet &patterns) {
  addTypeConversionPatterns<
      mlir::NormalizeF16ToF32Type<hivm::VLnOp, HIVMNormalizeF16ToF32Traits<hivm::VLnOp>>,
      mlir::NormalizeF16ToF32Type<hivm::VPowOp, HIVMNormalizeF16ToF32Traits<hivm::VPowOp>>,
      mlir::NormalizeF16ToF32Type<hivm::VRsqrtOp, HIVMNormalizeF16ToF32Traits<hivm::VRsqrtOp>>>(patterns);
  addTypeConversionPatterns<mlir::NormalizeCumOpF16ToF32Type<
      hivm::VCumsumOp, HIVMNormalizeCumOpF16ToF32Traits<hivm::VCumsumOp>>,
      mlir::NormalizeCumOpF16ToF32Type<
          hivm::VCumprodOp, HIVMNormalizeCumOpF16ToF32Traits<hivm::VCumprodOp>>,
      mlir::NormalizeCumOpF16ToF32Type<
          hivm::VCummaxOp, HIVMNormalizeCumOpF16ToF32Traits<hivm::VCummaxOp>>,
      mlir::NormalizeCumOpF16ToF32Type<
          hivm::VCumminOp, HIVMNormalizeCumOpF16ToF32Traits<hivm::VCumminOp>>>(patterns);
}

void mlir::hivm::populateNormalizeI8ToTargetPatterns(
    RewritePatternSet &patterns) {
  if (archIsRegbased)
    addTypeConversionPatterns<mlir::ReduceNormalize910_95Template<
        hivm::VReduceOp, HIVMReduceNormalize910_95Traits>>(patterns);
  if (!archIsRegbased) {
    addNormalizeToTargetPatterns<int8_t, hivm::VAddOp, hivm::VSubOp,
                                 hivm::VMulOp, hivm::VMaxOp, hivm::VMinOp,
                                 hivm::VModOp, hivm::VModUIOp,
                                 hivm::VAbsOp, hivm::VSelOp,
                                 hivm::VReduceOp, hivm::VInterleaveOp,
                                 hivm::VDeinterleaveOp, hivm::VGatherOp,
                                 hivm::VBrcOp>(patterns);
    addTypeConversionPatterns<mlir::NormalizeCumOpI8ToTargetType<
        hivm::VCumsumOp, HIVMNormalizeCumOpI8ToTargetTraits<hivm::VCumsumOp>>,
        mlir::NormalizeCumOpI8ToTargetType<
            hivm::VCumprodOp, HIVMNormalizeCumOpI8ToTargetTraits<hivm::VCumprodOp>>,
        mlir::NormalizeCumOpI8ToTargetType<
            hivm::VCummaxOp, HIVMNormalizeCumOpI8ToTargetTraits<hivm::VCummaxOp>>,
        mlir::NormalizeCumOpI8ToTargetType<
            hivm::VCumminOp, HIVMNormalizeCumOpI8ToTargetTraits<hivm::VCumminOp>>>(patterns);
  }
  if (archisAscend950) {
    addNormalizeToTargetPatterns<int8_t, hivm::VAddOp, hivm::VSubOp,
                                 hivm::VMulOp, hivm::VMaxOp, hivm::VMinOp>(patterns);
  }
  addNormalizeToTargetPatterns<int8_t, hivm::VReduceOp>(patterns);
}

void mlir::hivm::populateNormalizeGatherIndexPatterns(
    RewritePatternSet &patterns) {
  patterns.add<mlir::NormalizeGatherIndexToI32Template<
      mlir::hivm::VGatherOp, HIVMNormalizeGatherIndexToI32Traits>>(
      patterns.getContext());
}
