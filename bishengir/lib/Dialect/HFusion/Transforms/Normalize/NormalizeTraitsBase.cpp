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
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/TypeUtilities.h"

#include <optional>

using namespace mlir;
namespace mlir::hfusion {

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
      {UnaryKind::Not, hfusion::UnaryFn::vnot}
  };

  return lookupMappedFn(kindToFn, kind);
}

static std::optional<linalg::UnaryFn>
mapUnaryKindToLinalgUnaryFn(UnaryKind kind) {
  static const llvm::DenseMap<UnaryKind, linalg::UnaryFn> kindToFn = {
      {UnaryKind::Abs, linalg::UnaryFn::abs},
      {UnaryKind::Exp, linalg::UnaryFn::exp},
  };

  return lookupMappedFn(kindToFn, kind);
}

static std::optional<linalg::BinaryFn>
mapBinaryKindToLinalgBinaryFn(BinaryKind kind) {
  static const llvm::DenseMap<BinaryKind, linalg::BinaryFn> kindToFn = {
      {BinaryKind::Add, linalg::BinaryFn::add},
      {BinaryKind::Sub, linalg::BinaryFn::sub},
      {BinaryKind::Mul, linalg::BinaryFn::mul},
      {BinaryKind::Div, linalg::BinaryFn::div},
  };

  return lookupMappedFn(kindToFn, kind);
}

static std::optional<hfusion::BinaryFn>
mapBinaryKindToHFusionBinaryFn(BinaryKind kind) {
  static const llvm::DenseMap<BinaryKind, hfusion::BinaryFn> kindToFn = {
      {BinaryKind::Min, hfusion::BinaryFn::minf},
      {BinaryKind::Max, hfusion::BinaryFn::maxf},
  };

  return lookupMappedFn(kindToFn, kind);
}

static bool matchUnaryOp(Operation *op, UnaryKind kind) {
  if (auto unaryFn = mapUnaryKindToLinalgUnaryFn(kind)) {
    auto unaryOp = dyn_cast_or_null<linalg::ElemwiseUnaryOp>(op);
    return unaryOp && unaryOp.hasPureTensorSemantics() &&
           unaryOp.getFun() == *unaryFn;
  }
  if (auto unaryFn = mapUnaryKindToHFusionUnaryFn(kind)) {
    auto unaryOp = dyn_cast_or_null<hfusion::ElemwiseUnaryOp>(op);
    return unaryOp && unaryOp.hasPureTensorSemantics() &&
           unaryOp.getFun() == *unaryFn;
  }

  llvm_unreachable("unsupported unary kind");
}

static bool matchBinaryOp(Operation *op, BinaryKind kind) {
  if (auto binaryFn = mapBinaryKindToLinalgBinaryFn(kind)) {
    auto binaryOp = dyn_cast_or_null<linalg::ElemwiseBinaryOp>(op);
    return binaryOp && binaryOp.hasPureTensorSemantics() &&
           binaryOp.getFun() == *binaryFn;
  }
  if (auto binaryFn = mapBinaryKindToHFusionBinaryFn(kind)) {
    auto binaryOp = dyn_cast_or_null<hfusion::ElemwiseBinaryOp>(op);
    return binaryOp && binaryOp.hasPureTensorSemantics() &&
           binaryOp.getFun() == *binaryFn;
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
  if (auto unaryFn = mapUnaryKindToLinalgUnaryFn(kind)) {
    auto *op = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                      linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, loc, *unaryFn, ValueRange{input}, ValueRange{dst});
    return op->getResult(0);
  }
  if (auto unaryFn = mapUnaryKindToHFusionUnaryFn(kind)) {
    auto *op = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp,
                                      hfusion::UnaryFn, hfusion::UnaryFnAttr>(
        rewriter, loc, *unaryFn, ValueRange{input}, ValueRange{dst});
    return op->getResult(0);
  }

  llvm_unreachable("unsupported unary kind");
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::createBinaryOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value dst,
    BinaryKind kind) {
  if (auto binaryFn = mapBinaryKindToLinalgBinaryFn(kind)) {
    auto *op =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, *binaryFn, ValueRange{lhs, rhs}, ValueRange{dst});
    return op->getResult(0);
  }
  if (auto binaryFn = mapBinaryKindToHFusionBinaryFn(kind)) {
    auto *op =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, *binaryFn, ValueRange{lhs, rhs}, ValueRange{dst});
    return op->getResult(0);
  }

  llvm_unreachable("unsupported binary kind");
}

mlir::Value mlir::hfusion::NormalizeTraitsBase::castTo(
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
} // namespace mlir::hfusion
