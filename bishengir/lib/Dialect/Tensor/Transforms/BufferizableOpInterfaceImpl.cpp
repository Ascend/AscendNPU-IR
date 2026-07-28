//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace InsertSliceOpInterfaceForOpReuseInPlanMemory {

LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                               const AnalysisState &state) {
  auto bufferizableOp = cast<BufferizableOpInterface>(op);
  if (failed(bufferizableOp.resolveTensorOpOperandConflicts(rewriter, state)))
    return failure();

  auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
  auto sourceValue = insertSliceOp.getSource();
  auto result = insertSliceOp.getResult();
  auto &destinationOperand = insertSliceOp.getDestMutable();
  if (state.areAliasingBufferizedValues(sourceValue, result) &&
      !state.isInPlace(destinationOperand)) {
    // If the source value and result value are bufferized to alias buffer and
    // the destination operand is bufferized out-place, we need to allocate a
    // new buffer and copy the data from the alias buffer to the new buffer.
    // Because if the destination operand is bufferized out-place, it can't be
    // uWrite opOperand and cause some RaW conflicts misdected.
    rewriter.setInsertionPointAfter(insertSliceOp);
    FailureOr<Value> alloc = allocateTensorForShapedValue(
        rewriter, insertSliceOp.getLoc(), result, state.getOptions());
    if (failed(alloc))
      return failure();
    rewriter.replaceAllUsesExcept(result, *alloc, alloc->getDefiningOp());
  }
  return success();
}

} // namespace InsertSliceOpInterfaceForOpReuseInPlanMemory

RegisterOpInterfaceOverride(
    /*Op=*/tensor::InsertSliceOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/resolveConflicts,
    /*Impl=*/&InsertSliceOpInterfaceForOpReuseInPlanMemory::resolveConflicts);

void mlir::tensor_ext::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {}