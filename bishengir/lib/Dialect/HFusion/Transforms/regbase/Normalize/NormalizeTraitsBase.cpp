//===-------- NormalizeTraitsBase.cpp ----------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Transforms/regbase/Normalize/Utils/CastingTemplateHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
namespace mlir::hfusion {

using TernaryOpFn =
    Value (*)(PatternRewriter &, Location, Value, Value, Value, Value);

bool mlir::hfusion::NormalizeTraitsBase::archIsRegbased() {
  return hfusion::archIsRegbased;
}

bool mlir::hfusion::NormalizeTraitsBase::archIsAscend950() {
  return hfusion::archisAscend950;
}

Value mlir::hfusion::NormalizeTraitsBase::createIsNanOp(
    PatternRewriter &rewriter, Location loc, Value src) {
  auto isNanResultTensorType =
      utils::getTensorTypeWithSameShape(src.getType(), rewriter.getI1Type());
  return rewriter.create<IsNanOp>(loc, isNanResultTensorType, src)->getResult(0);
}

Value mlir::hfusion::NormalizeTraitsBase::createCastValueFromSourceOp(
    PatternRewriter &rewriter, Location loc, CastOp op, Value input,
    Type targetElemType, CastRoundKind executionKind, CastSignKind signKind,
    bool enableSaturate, CastUnsignedModeKind unsignedModeKind) {
  hfusion::TypeFn typeFn = mapCastSignKind(signKind, op.getCast());
  hfusion::UnsignedMode unsignedMode =
      mapCastUnsignedModeKind(unsignedModeKind, hfusion::UnsignedMode::SI2SI);
  hfusion::RoundMode defaultRoundMode =
      selectRoundMode<hfusion::RoundMode>(
          getElementTypeOrSelf(input.getType()), targetElemType);
  hfusion::RoundMode roundMode =
      mapCastRoundKind(executionKind, defaultRoundMode);
  const bool enableOverflow = executionKind == CastRoundKind::Default
                                  ? op.getEnableOverflow()
                                  : executionKind ==
                                        CastRoundKind::TruncEnableOverflow;
  return hfusion::castTo(rewriter, input, targetElemType, roundMode,
                         std::nullopt,
                         enableOverflow, enableSaturate, typeFn, unsignedMode);
}

Value mlir::hfusion::NormalizeTraitsBase::castScalarThroughTensor(
    PatternRewriter &rewriter, Location loc, Value scalar, Type dstType) {
  auto tensorType = RankedTensorType::get({1}, scalar.getType());
  Value fromElementsOp =
      rewriter.create<tensor::FromElementsOp>(loc, tensorType, scalar);
  Value castValue = hfusion::castTo(rewriter, fromElementsOp, dstType);
  auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  return rewriter.create<tensor::ExtractOp>(loc, castValue, ValueRange{c0});
}

template <typename Kind, typename Fn>
static std::optional<Fn>
lookupMappedFn(const llvm::DenseMap<Kind, Fn> &kindToFn, Kind kind) {
  auto it = kindToFn.find(kind);
  if (it == kindToFn.end()) {
    return std::nullopt;
  }

  return it->second;
}

template <typename MapT, typename KeyT>
static typename MapT::mapped_type lookupOrUnreachable(const MapT &map, KeyT key,
                                                      const char *errorMessage) {
  auto it = map.find(key);
  if (it == map.end())
    llvm_unreachable(errorMessage);
  return it->second;
}

static std::optional<hfusion::UnaryFn>
mapUnaryKindToHFusionUnaryFn(UnaryKind kind) {
  static const llvm::DenseMap<UnaryKind, hfusion::UnaryFn> kindToFn = {
      {UnaryKind::Rec, hfusion::UnaryFn::rec},
      {UnaryKind::Sqrt, hfusion::UnaryFn::sqrt},
      {UnaryKind::Not, hfusion::UnaryFn::vnot},
      {UnaryKind::Log2, hfusion::UnaryFn::log2}
  };

  return lookupMappedFn(kindToFn, kind);
}

static std::optional<linalg::UnaryFn>
mapUnaryKindToLinalgUnaryFn(UnaryKind kind) {
  static const llvm::DenseMap<UnaryKind, linalg::UnaryFn> kindToFn = {
      {UnaryKind::Abs, linalg::UnaryFn::abs},
      {UnaryKind::Exp, linalg::UnaryFn::exp},
      {UnaryKind::Ln, linalg::UnaryFn::log},
      {UnaryKind::Floor, linalg::UnaryFn::floor},
  };

  return lookupMappedFn(kindToFn, kind);
}

static std::optional<linalg::BinaryFn>
mapBinaryKindToLinalgBinaryFn(BinaryKind kind) {
  static const llvm::DenseMap<BinaryKind, linalg::BinaryFn> kindToFn = {
      {BinaryKind::Add, linalg::BinaryFn::add},
      {BinaryKind::Mul, linalg::BinaryFn::mul},
      {BinaryKind::Div, linalg::BinaryFn::div},
      {BinaryKind::MinSigned, linalg::BinaryFn::min_signed},
      {BinaryKind::MaxSigned, linalg::BinaryFn::max_signed},
      {BinaryKind::MinUnsigned, linalg::BinaryFn::min_unsigned},
      {BinaryKind::MaxUnsigned, linalg::BinaryFn::max_unsigned},
      {BinaryKind::Sub, linalg::BinaryFn::sub},
  };

  return lookupMappedFn(kindToFn, kind);
}

static std::optional<hfusion::BinaryFn>
mapBinaryKindToHFusionBinaryFn(BinaryKind kind) {
  static const llvm::DenseMap<BinaryKind, hfusion::BinaryFn> kindToFn = {
      {BinaryKind::Mod, hfusion::BinaryFn::mod},
      {BinaryKind::Min, hfusion::BinaryFn::minf},
      {BinaryKind::Max, hfusion::BinaryFn::maxf},
      {BinaryKind::ModUnsigned, hfusion::BinaryFn::modui},
      {BinaryKind::And, hfusion::BinaryFn::vand},
      {BinaryKind::Xor, hfusion::BinaryFn::vxor},
      {BinaryKind::Or, hfusion::BinaryFn::vor},
      {BinaryKind::Powf, hfusion::BinaryFn::powf},
  };

  return lookupMappedFn(kindToFn, kind);
}

static std::optional<hfusion::BinaryFn>
mapShiftKindToBinaryFn(ShiftKind kind) {
  static const llvm::DenseMap<ShiftKind, hfusion::BinaryFn> kindToFn = {
      {ShiftKind::Left, hfusion::BinaryFn::shli},
      {ShiftKind::RightSigned, hfusion::BinaryFn::shrsi},
      {ShiftKind::RightUnsigned, hfusion::BinaryFn::shrui},
  };

  return lookupMappedFn(kindToFn, kind);
}

static Value createHFusionSelectOp(PatternRewriter &rewriter, Location loc,
                                   Value lhs, Value mid, Value rhs,
                                   Value dst) {
  return rewriter
      .create<hfusion::SelectOp>(loc, TypeRange{dst.getType()},
                                 ValueRange{lhs, mid, rhs}, ValueRange{dst})
      .getResult(0);
}

static std::optional<TernaryOpFn> mapTernaryKindToCreator(TernaryKind kind) {
  static const llvm::DenseMap<TernaryKind, TernaryOpFn> kindToFn = {
      {TernaryKind::Select, createHFusionSelectOp},
  };
  return lookupMappedFn(kindToFn, kind);
}

static bool matchUnaryOp(Operation *op, UnaryKind kind) {
  if (auto linalgFn = mapUnaryKindToLinalgUnaryFn(kind)) {
    auto unaryOp = dyn_cast_or_null<linalg::ElemwiseUnaryOp>(op);
    return unaryOp && unaryOp.hasPureTensorSemantics() &&
           unaryOp.getFun() == *linalgFn;
  }
  if (auto hfusionFn = mapUnaryKindToHFusionUnaryFn(kind)) {
    auto unaryOp = dyn_cast_or_null<hfusion::ElemwiseUnaryOp>(op);
    return unaryOp && unaryOp.hasPureTensorSemantics() &&
           unaryOp.getFun() == *hfusionFn;
  }

  llvm_unreachable("unsupported unary kind");
}

static bool matchBinaryOp(Operation *op, BinaryKind kind) {
  if (auto linalgFn = mapBinaryKindToLinalgBinaryFn(kind)) {
    auto binaryOp = dyn_cast_or_null<linalg::ElemwiseBinaryOp>(op);
    return binaryOp && binaryOp.hasPureTensorSemantics() &&
           binaryOp.getFun() == *linalgFn;
  }
  if (auto hfusionFn = mapBinaryKindToHFusionBinaryFn(kind)) {
    auto binaryOp = dyn_cast_or_null<hfusion::ElemwiseBinaryOp>(op);
    return binaryOp && binaryOp.hasPureTensorSemantics() &&
           binaryOp.getFun() == *hfusionFn;
  }

  llvm_unreachable("unsupported binary kind");
}

bool mlir::hfusion::NormalizeTraitsBase::matchOp(Operation *op,
                                                 UnaryKind kind) {
  return matchUnaryOp(op, kind);
}

bool mlir::hfusion::NormalizeTraitsBase::matchOp(Operation *op,
                                                 BinaryKind kind) {
  return matchBinaryOp(op, kind);
}

static CompareFn mapCompareKindToCompareFn(CompareKind kind) {
  static const llvm::DenseMap<CompareKind, hfusion::CompareFn> kindToFn = {
      {CompareKind::EQ, hfusion::CompareFn::veq},
      {CompareKind::NE, hfusion::CompareFn::vne},
      {CompareKind::LT, hfusion::CompareFn::vlt},
      {CompareKind::GT, hfusion::CompareFn::vgt},
      {CompareKind::GE, hfusion::CompareFn::vge},
      {CompareKind::LE, hfusion::CompareFn::vle}
  };

  return lookupOrUnreachable(kindToFn, kind, "unsupported compare kind");
}

static std::optional<hfusion::RoundMode>
mapCastRoundKindToRoundMode(CastRoundKind kind) {
  static const llvm::DenseMap<CastRoundKind, hfusion::RoundMode>
      kindToMode = {
          {CastRoundKind::Round, hfusion::RoundMode::ROUND},
          {CastRoundKind::Floor, hfusion::RoundMode::FLOOR},
          {CastRoundKind::Ceil, hfusion::RoundMode::CEIL},
          {CastRoundKind::RInt, hfusion::RoundMode::RINT},
          {CastRoundKind::Trunc, hfusion::RoundMode::TRUNC},
          {CastRoundKind::TruncEnableOverflow, hfusion::RoundMode::TRUNC},
          {CastRoundKind::TruncWithOverflow,
           hfusion::RoundMode::TRUNCWITHOVERFLOW},
      };
  return lookupMappedFn(kindToMode, kind);
}

static std::optional<hfusion::UnsignedMode>
mapCastUnsignedModeKindToUnsignedMode(CastUnsignedModeKind kind) {
  static const llvm::DenseMap<CastUnsignedModeKind, hfusion::UnsignedMode>
      kindToMode = {
          {CastUnsignedModeKind::SignedToSigned, hfusion::UnsignedMode::SI2SI},
          {CastUnsignedModeKind::SignedToUnsigned, hfusion::UnsignedMode::SI2UI},
          {CastUnsignedModeKind::UnsignedToSigned, hfusion::UnsignedMode::UI2SI},
          {CastUnsignedModeKind::UnsignedToUnsigned,
           hfusion::UnsignedMode::UI2UI},
      };

  return lookupMappedFn(kindToMode, kind);
}

static std::optional<hfusion::TypeFn> mapCastSignKindToTypeFn(
    CastSignKind kind) {
  static const llvm::DenseMap<CastSignKind, hfusion::TypeFn> kindToTypeFn = {
      {CastSignKind::Signed, hfusion::TypeFn::cast_signed},
      {CastSignKind::Unsigned, hfusion::TypeFn::cast_unsigned},
  };

  return lookupMappedFn(kindToTypeFn, kind);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createCmpOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    CompareKind kind) {
  CompareFn cmpFn = mapCompareKindToCompareFn(kind);
  Operation *cmpOp = hfusion::createCmpOp(rewriter, loc, input, dst, cmpFn);
  return cmpOp->getResult(0);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createCmpOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs,
    CompareOp sourceOp) {
  Operation *cmpOp =
      hfusion::createCmpOp(rewriter, loc, lhs, rhs, sourceOp.getCompareFn());
  return cmpOp->getResult(0);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createUnaryOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    UnaryKind kind) {
  if (auto linalgFn = mapUnaryKindToLinalgUnaryFn(kind)) {
    auto *op = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                      linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, loc, *linalgFn, ValueRange{input}, ValueRange{dst});
    return op->getResult(0);
  }
  if (auto hfusionFn = mapUnaryKindToHFusionUnaryFn(kind)) {
    auto *op = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp,
                                      hfusion::UnaryFn, hfusion::UnaryFnAttr>(
        rewriter, loc, *hfusionFn, ValueRange{input}, ValueRange{dst});
    return op->getResult(0);
  }

  llvm_unreachable("unsupported unary kind");
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createFloorOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst) {
  return createUnaryOp(rewriter, loc, input, dst, UnaryKind::Floor);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createBinaryOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value dst,
    BinaryKind kind) {
  if (auto linalgFn = mapBinaryKindToLinalgBinaryFn(kind)) {
    auto *op =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, *linalgFn, ValueRange{lhs, rhs}, ValueRange{dst});
    return op->getResult(0);
  }
  if (auto hfusionFn = mapBinaryKindToHFusionBinaryFn(kind)) {
    auto *op =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, *hfusionFn, ValueRange{lhs, rhs}, ValueRange{dst});
    return op->getResult(0);
  }

  llvm_unreachable("unsupported binary kind");
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createCastOp(
    PatternRewriter &rewriter, Location loc, Value input, Type targetElemType,
    std::optional<RoundMode> roundMode) {
  if (roundMode)
    return hfusion::castTo(rewriter, input, targetElemType, *roundMode);
  return hfusion::castTo(rewriter, input, targetElemType);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createShiftOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value dst,
    ShiftKind kind, Operation *sourceOp) {
  std::optional<hfusion::BinaryFn> binaryFn;
  if (sourceOp)
    binaryFn = cast<hfusion::ElemwiseBinaryOp>(sourceOp).getFun();
  else
    binaryFn = mapShiftKindToBinaryFn(kind);
  if (!binaryFn)
    llvm_unreachable("unsupported shift kind");
  auto *op =
      hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                              hfusion::BinaryFnAttr>(
          rewriter, loc, *binaryFn, mlir::ValueRange{lhs, rhs},
          mlir::ValueRange{dst});
  return op->getResult(0);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createTernaryOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value mid, Value rhs,
    Value dst, TernaryKind kind) {
  auto fn = mapTernaryKindToCreator(kind);
  if (!fn)
    llvm_unreachable("unsupported ternary kind");
  return (*fn)(rewriter, loc, lhs, mid, rhs, dst);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createFillOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst) {
  auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{input}, dst);
  return fillOp->getResult(0);
}

bool mlir::hfusion::NormalizeTraitsBase::matchFillOp(Operation *op) {
  return isa<linalg::FillOp>(op);
}

Value mlir::hfusion::NormalizeTraitsBase::getFillInput(Operation *op) {
  return cast<linalg::FillOp>(op).getInputs()[0];
}

bool mlir::hfusion::NormalizeTraitsBase::matchBroadcastOp(Operation *op) {
  return isa<linalg::BroadcastOp>(op);
}

Value mlir::hfusion::NormalizeTraitsBase::getBroadcastInput(Operation *op) {
  return cast<linalg::BroadcastOp>(op).getInput();
}

SmallVector<int64_t>
mlir::hfusion::NormalizeTraitsBase::getBroadcastDims(Operation *op) {
  return llvm::to_vector(cast<linalg::BroadcastOp>(op).getDimensions());
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createBroadcastOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    ArrayRef<int64_t> dims) {
  return rewriter.create<linalg::BroadcastOp>(loc, input, dst, dims)
      ->getResult(0);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createBitcastOp(
    PatternRewriter &rewriter, Location loc, Type resultType, Value source) {
  Value init = utils::createEmptyOpWithTargetElemType(
      rewriter, loc, source, getElementTypeOrSelf(resultType));
  return rewriter
      .create<hfusion::BitcastOp>(loc, TypeRange{resultType},
                                  ValueRange{source}, ValueRange{init})
      .getResult(0);
}

Operation *mlir::hfusion::NormalizeTraitsBase::createReduceWithIndexOp(
    PatternRewriter &rewriter, Location loc, ReduceWithIndexOp op,
    ArrayRef<Value> newInputs, ArrayRef<Value> newInits) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(newInits.size());
  for (Value init : newInits)
    resultTypes.push_back(init.getType());

  return rewriter.create<hfusion::ReduceWithIndexOp>(
      loc, resultTypes, newInputs, newInits, op.getReduceKindAttr(),
      op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
      op.getDimensionsAttr());
}

Value mlir::hfusion::NormalizeTraitsBase::castReduceIndexTensor(
    PatternRewriter &rewriter, Location, Value value, IntegerType targetElemType,
    Value) {
  return hfusion::castTo(rewriter, value, targetElemType);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createSelectOp(
    PatternRewriter &rewriter, Location loc, Value cond, Value a, Value b,
    Value dst) {
  return rewriter
      .create<hfusion::SelectOp>(loc, mlir::TypeRange{dst.getType()},
                                 mlir::ValueRange{cond, a, b},
                                 mlir::ValueRange{dst})
      .getResults()[0];
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createGather1DOp(
    PatternRewriter &rewriter, Location loc, Value source, Value indices) {
  Type sourceElemType = getElementTypeOrSelf(source.getType());
  Value init = utils::createEmptyOpWithTargetElemType(rewriter, loc, indices,
                                                      sourceElemType);
  return rewriter.create<hfusion::GatherOp>(loc, source, indices, init,
                                            /*axis=*/0)
      ->getResult(0);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createIsInfOp(
    PatternRewriter &rewriter, Location loc, Value y) {
  Type yType = y.getType();
  Type boolType = rewriter.getI1Type();
  if (auto shapedType = dyn_cast<ShapedType>(yType))
    boolType = shapedType.clone(rewriter.getI1Type());
  return rewriter.create<hfusion::IsInfOp>(loc, boolType, y)->getResults()[0];
}

bool mlir::hfusion::NormalizeTraitsBase::matchCastRoundMode(
    hfusion::CastOp op, CastRoundKind kind) {
  auto roundMode = mapCastRoundKindToRoundMode(kind);
  return roundMode && op.getRoundMode() == *roundMode;
}

bool mlir::hfusion::NormalizeTraitsBase::matchCastUnsignedMode(
    hfusion::CastOp op, CastUnsignedModeKind kind) {
  auto unsignedMode = mapCastUnsignedModeKindToUnsignedMode(kind);
  return unsignedMode && op.getUnsignedMode() == *unsignedMode;
}

hfusion::TypeFn mlir::hfusion::NormalizeTraitsBase::mapCastSignKind(
    CastSignKind kind, hfusion::TypeFn preserveTypeFn) {
  auto typeFn = mapCastSignKindToTypeFn(kind);
  return typeFn.value_or(preserveTypeFn);
}

hfusion::RoundMode mlir::hfusion::NormalizeTraitsBase::mapCastRoundKind(
    CastRoundKind kind, hfusion::RoundMode defaultRoundMode) {
  auto roundMode = mapCastRoundKindToRoundMode(kind);
  return roundMode.value_or(defaultRoundMode);
}

hfusion::UnsignedMode
mlir::hfusion::NormalizeTraitsBase::mapCastUnsignedModeKind(
    CastUnsignedModeKind kind, hfusion::UnsignedMode preserveMode) {
  auto unsignedMode = mapCastUnsignedModeKindToUnsignedMode(kind);
  return unsignedMode.value_or(preserveMode);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createCastOp(
    PatternRewriter &rewriter, Location loc, Value input, Type targetElemType,
    CastRoundKind kind, Value output, CastSignKind signKind) {
  if (signKind == CastSignKind::Preserve)
    llvm_unreachable("createCastOp does not support CastSignKind::Preserve");
  Type srcElemType = getElementTypeOrSelf(input.getType());
  auto roundMode =
      kind == CastRoundKind::Default
          ? selectRoundMode<hfusion::RoundMode>(srcElemType,
                                                             targetElemType)
          : mapCastRoundKindToRoundMode(kind).value();
  auto castIntType = mapCastSignKind(signKind, hfusion::TypeFn::cast_signed);
  std::optional<Value> dst =
      output ? std::optional<Value>(output) : std::nullopt;
  return hfusion::castTo(rewriter, input, targetElemType, roundMode, dst,
                         /*enableOverflow=*/true, /*enableSaturate=*/false,
                         castIntType, hfusion::UnsignedMode::SI2SI);
}

} // namespace mlir::hfusion
