//===- ValueBoundsOpInterfaceImpl.cpp - Arith value bounds models ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Arith/Transforms/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace {
static std::optional<int64_t> computeConstantBound(Value value,
                                                   presburger::BoundType type) {
  auto bound = ValueBoundsConstraintSet::computeConstantBound(
      type, ValueBoundsConstraintSet::Variable(value),
      /*stopCondition=*/nullptr,
      /*closedUB=*/type == presburger::BoundType::UB);
  if (succeeded(bound)) {
    return *bound;
  }
  return std::nullopt;
}

struct MaxSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MaxSIOpInterface,
                                                   arith::MaxSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto maxOp = cast<arith::MaxSIOp>(op);
    assert(value == maxOp.getResult() && "invalid value");

    cstr.bound(value) >= maxOp.getLhs();
    cstr.bound(value) >= maxOp.getRhs();

    Value lhs = maxOp.getLhs();
    Value rhs = maxOp.getRhs();
    std::optional<int64_t> lhsUpper =
        computeConstantBound(lhs, presburger::BoundType::UB);
    std::optional<int64_t> rhsUpper =
        computeConstantBound(rhs, presburger::BoundType::UB);
    if (lhsUpper && rhsUpper) {
      cstr.bound(value) <= std::max(*lhsUpper, *rhsUpper);
    }

    cstr.populateConstraints(lhs, std::nullopt);
    cstr.populateConstraints(rhs, std::nullopt);
    if (cstr.populateAndCompare(
            {lhs}, ValueBoundsConstraintSet::ComparisonOperator::LE, {rhs})) {
      cstr.bound(value) <= rhs;
    }
    if (cstr.populateAndCompare(
            {rhs}, ValueBoundsConstraintSet::ComparisonOperator::LE, {lhs})) {
      cstr.bound(value) <= lhs;
    }
  }
};

struct MinSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MinSIOpInterface,
                                                   arith::MinSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto minOp = cast<arith::MinSIOp>(op);
    assert(value == minOp.getResult() && "invalid value");

    cstr.bound(value) <= minOp.getLhs();
    cstr.bound(value) <= minOp.getRhs();

    Value lhs = minOp.getLhs();
    Value rhs = minOp.getRhs();
    std::optional<int64_t> lhsLower =
        computeConstantBound(lhs, presburger::BoundType::LB);
    std::optional<int64_t> rhsLower =
        computeConstantBound(rhs, presburger::BoundType::LB);
    if (lhsLower && rhsLower) {
      cstr.bound(value) >= std::min(*lhsLower, *rhsLower);
    }

    cstr.populateConstraints(lhs, std::nullopt);
    cstr.populateConstraints(rhs, std::nullopt);
    if (cstr.populateAndCompare(
            {lhs}, ValueBoundsConstraintSet::ComparisonOperator::LE, {rhs})) {
      cstr.bound(value) >= lhs;
    }
    if (cstr.populateAndCompare(
            {rhs}, ValueBoundsConstraintSet::ComparisonOperator::LE, {lhs})) {
      cstr.bound(value) >= rhs;
    }
  }
};
} // namespace

void arith::registerBiShengIRValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    auto maxInfo = RegisteredOperationName::lookup(
        arith::MaxSIOp::getOperationName(), ctx);
    if (maxInfo && !maxInfo->hasInterface<ValueBoundsOpInterface>()) {
      // only attach the interface if it hasn't been attached by upstream mlir
      maxInfo->attachInterface<MaxSIOpInterface>();
    }

    auto minInfo = RegisteredOperationName::lookup(
        arith::MinSIOp::getOperationName(), ctx);
    if (minInfo && !minInfo->hasInterface<ValueBoundsOpInterface>()) {
      // only attach the interface if it hasn't been attached by upstream mlir
      minInfo->attachInterface<MinSIOpInterface>();
    }
  });
}
