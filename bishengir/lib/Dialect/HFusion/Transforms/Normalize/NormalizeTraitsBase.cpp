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

#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Transforms/Normalize/Utils/CastingTemplateHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
namespace mlir::hfusion {

bool mlir::hfusion::NormalizeTraitsBase::archIsRegbased() {
  return hfusion::archIsRegbased;
}

Value mlir::hfusion::NormalizeTraitsBase::castValue(
    PatternRewriter &rewriter, Location loc, CastOp op, Value input,
    Type targetElemType, CastExecutionKind executionKind, CastSignKind signKind,
    bool enableSaturate, CastUnsignedModeKind unsignedModeKind) {
  hfusion::TypeFn typeFn = mapCastSignKind(signKind, op.getCast());
  hfusion::UnsignedMode unsignedMode =
      mapCastUnsignedModeKind(unsignedModeKind, hfusion::UnsignedMode::SI2SI);
  hfusion::RoundMode defaultRoundMode =
      utils::selectRoundMode<hfusion::RoundMode>(
          getElementTypeOrSelf(input.getType()), targetElemType);
  hfusion::RoundMode roundMode =
      mapCastExecutionKind(executionKind, defaultRoundMode);

  const bool enableOverflow = executionKind == CastExecutionKind::Default
                                  ? op.getEnableOverflow()
                                  : executionKind ==
                                        CastExecutionKind::TruncEnableOverflow;
  return hfusion::castTo(rewriter, input, targetElemType, roundMode,
                         std::nullopt, enableOverflow, enableSaturate, typeFn,
                         unsignedMode);
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
      {BinaryKind::And, hfusion::BinaryFn::vand},
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

  auto it = kindToFn.find(kind);
  if (it == kindToFn.end()) {
    llvm_unreachable("unsupported compare kind");
  }

  return it->second;
}

static hfusion::RoundMode mapCastRoundKindToRoundMode(CastRoundKind kind) {
  static const llvm::DenseMap<CastRoundKind, hfusion::RoundMode> kindToFn = {
      {CastRoundKind::Round, hfusion::RoundMode::ROUND},
      {CastRoundKind::Floor, hfusion::RoundMode::FLOOR},
  };

  auto it = kindToFn.find(kind);
  if (it == kindToFn.end()) {
    llvm_unreachable("unsupported cast round kind");
  }

  return it->second;
}

static std::optional<hfusion::RoundMode>
mapCastExecutionKindToRoundMode(CastExecutionKind kind) {
  static const llvm::DenseMap<CastExecutionKind, hfusion::RoundMode> kindToMode = {
      {CastExecutionKind::RInt, hfusion::RoundMode::RINT},
      {CastExecutionKind::Trunc, hfusion::RoundMode::TRUNC},
      {CastExecutionKind::TruncEnableOverflow, hfusion::RoundMode::TRUNC},
      {CastExecutionKind::TruncWithOverflow,
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
    CastRoundKind kind) {
  hfusion::RoundMode roundMode = mapCastRoundKindToRoundMode(kind);
  return hfusion::castTo(rewriter, input, targetElemType, roundMode);
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createFillOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst) {
  auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{input}, dst);
  return fillOp->getResult(0);
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

bool mlir::hfusion::NormalizeTraitsBase::matchCastRoundMode(
    hfusion::CastOp op, CastExecutionKind kind) {
  auto roundMode = mapCastExecutionKindToRoundMode(kind);
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

hfusion::RoundMode mlir::hfusion::NormalizeTraitsBase::mapCastExecutionKind(
    CastExecutionKind kind, hfusion::RoundMode defaultRoundMode) {
  auto roundMode = mapCastExecutionKindToRoundMode(kind);
  return roundMode.value_or(defaultRoundMode);
}

hfusion::UnsignedMode
mlir::hfusion::NormalizeTraitsBase::mapCastUnsignedModeKind(
    CastUnsignedModeKind kind, hfusion::UnsignedMode preserveMode) {
  auto unsignedMode = mapCastUnsignedModeKindToUnsignedMode(kind);
  return unsignedMode.value_or(preserveMode);
}
} // namespace mlir::hfusion
