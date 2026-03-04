//===- CommonFlatten.cpp - Common implementation of flatten interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#define DEBUG_TYPE "flatten-common"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils;
using namespace mlir::utils::debugger;

namespace mlir::hivm {
namespace detail {

FlattenResult computeAnnotationMarkedOp(FlattenResult payload) {
  SmallVector<Type> strideMarkedTypes;
  for (size_t i = 0; i < payload.operandOriginalVal.size(); ++i) {
    Value &operandVal = payload.operandOriginalVal[i];
    Type &operandType = payload.operandTypes[i].second;
    auto memrefType = dyn_cast<MemRefType>(operandType);
    if (memrefType == nullptr) {
      continue;
    }

    operandType = getAnnotationMarkByteAlignment(operandVal);
    strideMarkedTypes.push_back(operandType);
  }
  return payload;
}

FailureOr<FlattenResult> getFlattenedImpl(Operation *op,
                                          FlattenOptions &options) {
  bool isUniformReassociation =
      op->hasTrait<OpTrait::UniformReassociationFlattenTrait>();
  if (isUniformReassociation) {
    if (isa<hivm::VFlipOp>(op)) options.strictBarrierWithUnit = true;
    return getFlattenedUniformReassociation(cast<HIVMStructuredOp>(op),
                                            options);
  }
  LDBG(*op << " flatten is not implemented");
  FlattenResult result(op);
  if (auto hivmOp = dyn_cast<HIVMStructuredOp>(op)) {
    result.fillWithIdentity();
    return result;
  }
  LDBG(*op << "not HIVMStructuredOp and flatten is not implemented");
  return {};
}

FlattenResult getFlattenedElementwise(HIVMStructuredOp op,
                                      FlattenOptions &options) {
  // This operation is asserted to be elementwise
  if (op.existInlineBroadcastLoopDims())
    return getFlattenedBroadcastableOTF(op, options);
  if (op.existInlineTransposeLoopDims())
    return getFlattenedTransposableOTF(op, options);
  return collapseUniformReassociationPipeline(op, options, {});
}

FlattenResult composeFlattenResults(FlattenResult producer,
                                    FlattenResult consumer,
                                    MLIRContext *context) {
  LDBG(to_string(producer.getInputReassociation()));
  if (consumer.isIdentityCollapse())
    return producer;
  if (producer.isIdentityCollapse())
    return consumer;
  auto inputReassociation = mlir::composeReassociationIndices(
      producer.getInputReassociation(), consumer.getInputReassociation(),
      context);
  if (!inputReassociation.has_value()) {
    llvm::report_fatal_error("HIVM flatten interface failed to compose");
  }
  LDBG("Value fails to compose? "
       << to_string(producer.getInputReassociation()));
  LDBG("Value fails to compose? "
       << to_string(consumer.getInputReassociation()));
  FlattenResult composedFlattenResult = consumer;
  composedFlattenResult.originalTargetDims = producer.originalTargetDims;
  composedFlattenResult.reassociation = {inputReassociation.value()};
  if (!consumer.uniformReassociation()) {
    LDBG("This reassociation has init reassociation");
    // if its not uniform meaning it has input and init reassociation
    LDBG(to_string(consumer.getInitReassociation()));
    auto initReassociation = mlir::composeReassociationIndices(
        producer.getInitReassociation(), consumer.getInitReassociation(),
        context);
    if (!initReassociation.has_value()) {
      llvm::report_fatal_error("HIVM flatten interface failed to compose");
    }
    composedFlattenResult.reassociation.push_back(initReassociation.value());
  }
  return composedFlattenResult;
}

BitVector getInputConsistencyMask(ArrayRef<Type> shapedTypes) {
  BitVector consistencyMask;
  SmallVector<int64_t> pivotShape;
  bool pivotInitialized = false;

  for (const auto &type : shapedTypes) {
    auto memRefType = dyn_cast<MemRefType>(type);
    if (!memRefType) {
      continue;
    }

    // Initialize pivot shape with the first valid MemRefType
    if (!pivotInitialized) {
      pivotShape = llvm::to_vector(memRefType.getShape());
      consistencyMask = BitVector(memRefType.getRank(), true);
      pivotInitialized = true;
      continue;
    }

    // Skip types with different rank than pivot
    if (static_cast<int64_t>(pivotShape.size()) != memRefType.getRank()) {
      continue;
    }

    // Compare each dimension against the pivot shape
    auto currentShape = memRefType.getShape();
    for (const auto &[dimIndex, pivotDimSize, currentDimSize] :
         llvm::enumerate(pivotShape, currentShape)) {
      if (pivotDimSize != currentDimSize) {
        consistencyMask[dimIndex] = false;
      }
    }
  }

  return consistencyMask;
}

} // namespace detail
} // namespace mlir::hivm