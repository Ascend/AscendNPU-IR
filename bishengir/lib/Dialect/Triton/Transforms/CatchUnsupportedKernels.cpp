//===- CatchUnsupportedKernels.cpp --------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Signals pass failure at the start of the pipeline if an unsupported kernel is
// detected. Does not make IR modifications.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"
#include <unordered_set>

namespace bishengir::triton {
#define GEN_PASS_DEF_CATCHUNSUPPORTEDKERNELS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

enum class ErrorType {
  F64Type,
};

bool hasNestedF64Type(Type type) {
  if (type.isF64()) {
    return true;
  }

  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return hasNestedF64Type(tensorType.getElementType());
  }

  if (auto pointerType = dyn_cast<triton::PointerType>(type)) {
    return hasNestedF64Type(pointerType.getPointeeType());
  }

  return false;
}

bool hasF64OperandOrResult(Operation *op) {
  Twine errorMsg;
  for (Value operand : op->getOperands()) {
    if (hasNestedF64Type(operand.getType())) {
      return true;
    }
  }

  for (Value result : op->getResults()) {
    if (hasNestedF64Type(result.getType())) {
      return true;
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument &blockArg : block.getArguments()) {
        if (hasNestedF64Type(blockArg.getType())) {
          return true;
        }
      }
    }
  }
  return false;
}

void addErrorIfPresent(Operation *op, std::unordered_set<ErrorType> &errors) {
  if (hasF64OperandOrResult(op)) {
    errors.insert(ErrorType::F64Type);
  }
}

Twine getErrorMsg(ErrorType error) {
  switch (error) {
  case ErrorType::F64Type:
    return "f64 types not supported in SIMT kernels!";
  }

  llvm_unreachable("Unhandled Error Type");
}

void attachFuncErrors(const std::unordered_set<ErrorType> &errors,
                      FuncOp &func) {
  for (ErrorType error : errors) {
    func.emitError(getErrorMsg(error));
  }
}

class CatchUnsupportedKernelsPass
    : public impl::CatchUnsupportedKernelsBase<CatchUnsupportedKernelsPass> {
public:
  using CatchUnsupportedKernelsBase::CatchUnsupportedKernelsBase;

  void runOnOperation() override {
    FuncOp func = getOperation();
    std::unordered_set<ErrorType> errors;

    func.walk([&](Operation *op) { addErrorIfPresent(op, errors); });

    if (!errors.empty()) {
      attachFuncErrors(errors, func);
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCatchUnsupportedKernelsPass() {
  return std::make_unique<CatchUnsupportedKernelsPass>();
}

} // namespace bishengir::triton
