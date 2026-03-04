//===- MathExtDialect.cpp - MLIR dialect for Math Ext implementation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::mathExt;

#include "bishengir/Dialect/MathExt/IR/MathExtOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with math
/// operations.
struct MathExtInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::mathExt::MathExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/MathExt/IR/MathExtOps.cpp.inc"
      >();
  addInterfaces<MathExtInlinerInterface>();
}