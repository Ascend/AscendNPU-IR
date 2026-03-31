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

using namespace mlir;
namespace mlir::hfusion {

static UnaryFn mapUnaryKindToUnaryFn(UnaryKind kind) {
  static const llvm::DenseMap<UnaryKind, hfusion::UnaryFn> kindToFn = {
      {UnaryKind::Rec, hfusion::UnaryFn::rec},
      {UnaryKind::Sqrt, hfusion::UnaryFn::sqrt},
      {UnaryKind::Not, hfusion::UnaryFn::vnot}
  };

  auto it = kindToFn.find(kind);
  if (it == kindToFn.end()) {
    llvm_unreachable("unsupported unary kind");
  }

  return it->second;
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
  UnaryFn unaryFn = mapUnaryKindToUnaryFn(kind);
  auto *op = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                                    hfusion::UnaryFnAttr>(
      rewriter, loc, unaryFn, mlir::ValueRange{input}, mlir::ValueRange{dst});
  return op->getResult(0);
}
} // namespace mlir::hfusion
