//===- CommonFlatten.cpp - Common implementation of flatten interface -----===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
//============================================================================//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
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

static bool isRegBaseModule(Operation *op) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  return hacc::utils::hasTargetAttr(moduleOp) &&
         hacc::utils::isRegBasedArch(moduleOp);
}

FailureOr<FlattenResult> getFlattenedImpl(Operation *op,
                                          FlattenOptions &options) {
  // RegBase (A5) only: transpose-like flattening collapses unit dims
  // block-wise, trailing units included. A3 keeps the uniform-reassociation
  // flatten: mark-stride-align reads the flatten result and expects it.
  if (isa<hivm::VTransposeOp>(op) && isRegBaseModule(op)) {
    return getFlattenedTransposableOTF(cast<HIVMStructuredOp>(op), options);
  }
  bool isUniformReassociation =
      op->hasTrait<OpTrait::UniformReassociationFlattenTrait>();
  if (isUniformReassociation) {
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

static std::optional<SmallVector<ReassociationIndices>>
composeCollapseReassociationIndices(
    ArrayRef<ReassociationIndices> producerReassociations, // A -> B collapse
    ArrayRef<ReassociationIndices> consumerReassociations  // B -> C collapse
) {
  SmallVector<ReassociationIndices> composed;

  // C is rank-0 => empty reassociation list is valid.
  if (consumerReassociations.empty())
    return composed;

  // Consumer must index [0..rank(B)-1] in order: collapse groups are
  // contiguous.
  int64_t expectedB = 0;
  for (ReassociationIndicesRef group : consumerReassociations) {
    if (group.empty())
      return std::nullopt;
    for (int64_t idx : group) {
      if (idx != expectedB)
        return std::nullopt;
      ++expectedB;
    }
  }

  // Must consume exactly all B dims.
  if (expectedB != static_cast<int64_t>(producerReassociations.size()))
    return std::nullopt;

  // Compose: expand each C-group's B-dims into their A-groups.
  composed.reserve(consumerReassociations.size());
  for (ReassociationIndicesRef cGroup : consumerReassociations) {
    ReassociationIndices outGroup;
    for (int64_t bDim : cGroup)
      llvm::append_range(outGroup, producerReassociations[bDim]);
    composed.push_back(std::move(outGroup));
  }

  return composed;
}

FlattenResult composeFlattenResults(FlattenResult producer,
                                    FlattenResult consumer,
                                    MLIRContext *context) {
  LDBG(to_string(producer.getInputReassociation()));
  if (consumer.isIdentityCollapse())
    return producer;
  if (producer.isIdentityCollapse())
    return consumer;
  auto inputReassociation = composeCollapseReassociationIndices(
      producer.getInputReassociation(), consumer.getInputReassociation());
  LDBG("Value fails to compose? "
       << to_string(producer.getInputReassociation()));
  LDBG("Value fails to compose? "
       << to_string(consumer.getInputReassociation()));
  if (!inputReassociation.has_value()) {
    llvm::report_fatal_error("HIVM flatten interface failed to compose");
  }
  FlattenResult composedFlattenResult = consumer;
  composedFlattenResult.originalTargetDims = producer.originalTargetDims;
  composedFlattenResult.reassociation = {inputReassociation.value()};
  if (!consumer.uniformReassociation()) {
    LDBG("This reassociation has init reassociation");
    // if its not uniform meaning it has input and init reassociation
    LDBG("Value fails to compose? "
         << to_string(producer.getInitReassociation()));
    LDBG("Value fails to compose? "
         << to_string(consumer.getInitReassociation()));
    auto initReassociation = composeCollapseReassociationIndices(
        producer.getInitReassociation(), consumer.getInitReassociation());
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
