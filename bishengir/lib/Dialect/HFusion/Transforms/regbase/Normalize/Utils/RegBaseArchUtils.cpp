//===- RegBaseArchUtils.cpp --------------------------------------*- C++ -*-===//
// TODO-A5: Remove this file after appropriate porting of dependenciesЧ
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

#include "bishengir/Dialect/HFusion/Transforms/regbase/RegBaseArchUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <unordered_set>

namespace mlir::hfusion {

static Value createEmptyLikeAnyShaped(OpBuilder &b, Location loc, Value a,
                                      Value bval) {
  auto aTy = a.getType();
  if (isa<ShapedType>(aTy))
    return ::mlir::utils::createEmptyOp(b, loc, a);
  auto bTy = bval.getType();
  if (isa<ShapedType>(bTy))
    return ::mlir::utils::createEmptyOp(b, loc, bval);
  llvm::report_fatal_error(
      "createEmptyLikeAnyShaped: both operands are scalar; "
      "elemwise tensor op requires at least one shaped operand.");
}

Operation *createVandOp(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  auto andEmptyOp = createEmptyLikeAnyShaped(b, loc, lhs, rhs);
  return hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                 hfusion::BinaryFnAttr>(
      b, loc, hfusion::BinaryFn::vand, ValueRange({lhs, rhs}),
      ValueRange{andEmptyOp});
}

Operation *createLinalgElemwiseBinaryOp(OpBuilder &b, Location loc,
                                        linalg::BinaryFn fn, Value lhs,
                                        Value rhs) {
  Value emptyOp = createEmptyLikeAnyShaped(b, loc, lhs, rhs);
  return hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                 linalg::BinaryFnAttr>(
      b, loc, fn, ValueRange{lhs, rhs}, ValueRange(emptyOp));
}

Operation *createHFusionElemwiseBinaryOp(OpBuilder &b, Location loc,
                                         hfusion::BinaryFn fn, Value lhs,
                                         Value rhs) {
  Value emptyOp = ::mlir::utils::createEmptyOp(b, loc, lhs);
  return hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                 hfusion::BinaryFnAttr>(
      b, loc, fn, ValueRange{lhs, rhs}, ValueRange(emptyOp));
}

Operation *createCmpOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                       CompareFn cmpFn) {
  auto emptyOp = ::mlir::utils::createEmptyOpWithTargetElemType(b, loc, lhs,
                                                                b.getI1Type());
  auto cmpPredicateAttr = b.getAttr<hfusion::CompareFnAttr>(cmpFn);
  auto cmpModeAttr = b.getNamedAttr(
      hfusion::CompareFnAttr::getMnemonic(), cmpPredicateAttr);
  return b.create<hfusion::CompareOp>(
      loc, TypeRange(emptyOp), ValueRange{lhs, rhs}, ValueRange(emptyOp),
      ArrayRef{cmpModeAttr});
}

template <typename T> T selectRoundMode(Type inType, Type outType) {
  if (inType.isFloat8E5M2() || inType.isFloat8E4M3FN() ||
    outType.isFloat8E5M2() || outType.isFloat8E4M3FN())
    return T::RINT;
  if (inType.isF32()) {
    if (outType.isF16() || outType.isBF16() || outType.isF32() ||
        outType.isFloat8E4M3FN() || outType.isFloat8E5M2()) {
      return T::RINT;
        }
  }

  if (outType.isF32()) {
    if (inType.isF16() || inType.isBF16() || inType.isFloat8E4M3FN() ||
        inType.isFloat8E5M2()) {
      return T::RINT;
        }
  }

  if (inType.isInteger(8) && // for bit width of 8 and 16 use RINT mode
      outType.isF16()) {
    return T::RINT;
      }

  if (inType.isInteger(16) && outType.isInteger(8)) {
    return T::TRUNCWITHOVERFLOW;
  }

  if (isa<mlir::FloatType>(inType) && outType.isInteger()) {
    return T::TRUNC;
  }

  if (inType.isInteger() && isa<mlir::FloatType>(outType)) {
    return T::TRUNC;
  }

  if (inType.isInteger() && outType.isInteger()) {
    return T::RINT;
  }
  llvm_unreachable("unsupported type cast.");
}

template hfusion::RoundMode selectRoundMode(Type inType, Type outType);

} // namespace mlir::hfusion
