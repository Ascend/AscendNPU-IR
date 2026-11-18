//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace SelectOpInterfaceExt {
/// arith.select neither reads nor writes its operands, it only forwards one of
/// them. Force the true/false OpOperands to bufferize in-place so that no
/// `bufferization.alloc_tensor` copy is inserted for them.
bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                          const AnalysisState &state) {
  // OpOperand 0 is the condition. OpOperands 1 and 2 are the true/false
  // operands amd only need to bufferize these two operands in-place.
  return opOperand != op->getOpOperand(0);
}
} // namespace SelectOpInterfaceExt

RegisterOpInterfaceOverride(
    /*Op=*/arith::SelectOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/mustBufferizeInPlace,
    /*Impl=*/
    &SelectOpInterfaceExt::mustBufferizeInPlace);

void bishengir::arith::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {}
