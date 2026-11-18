//===- TensorViewDialect.cpp - TensorView dialect registration ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.cpp.inc"

namespace mlir::tv {

namespace {
struct TensorViewInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void TensorViewDialect::initialize() {
  // Types and attributes are registered in TensorViewTypes.cpp /
  // TensorViewAttrs.cpp, where their storage classes are complete (the
  // StorageUniquer requires the full storage definition).
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/TensorView/IR/TensorViewOps.cpp.inc"
      >();
  addInterfaces<TensorViewInlinerInterface>();
}

} // namespace mlir::tv
