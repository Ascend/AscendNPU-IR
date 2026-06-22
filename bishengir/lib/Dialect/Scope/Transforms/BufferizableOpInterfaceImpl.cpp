//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
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
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/Transforms/BufferizableOpInterfaceImpl.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::scope;

namespace mlir {
namespace scope {
namespace {

static Value castBuffer(OpBuilder &b, Value buffer, Type type) {
  assert(isa<BaseMemRefType>(type) && "expected BaseMemRefType");
  assert(isa<BaseMemRefType>(buffer.getType()) && "expected BaseMemRefType");
  // If the buffer already has the correct type, no cast is needed.
  if (buffer.getType() == type)
    return buffer;
  return b.create<memref::CastOp>(buffer.getLoc(), type, buffer).getResult();
}

/// Bufferization of scope.scope.
struct ScopeOpInterface
    : public BufferizableOpInterface::ExternalModel<ScopeOpInterface,
                                                    scope::ScopeOp> {
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    OpBuilder::InsertionGuard g(rewriter);
    auto scopeOp = cast<scope::ScopeOp>(op);
    Region &region = scopeOp.getRegion();
    // Get the return operation inside the region
    Block &block = region.front();
    auto returnOp = dyn_cast<scope::ReturnOp>(block.getTerminator());
    if (!returnOp)
      return scopeOp.emitError("scope op must terminate with scope.return");
    // Step 1: Bufferize the return op's operands and collect new type
    SmallVector<Value> newReturnOperands;
    SmallVector<Type> newResultTypes;

    rewriter.setInsertionPoint(returnOp);
    for (Value operand : returnOp.getOperands()) {
      // Handle non-tensor type(pass)
      if (!operand.getType().isa<TensorType>()) {
        newReturnOperands.push_back(operand);
        newResultTypes.push_back(operand.getType());
        continue;
      }
      // Get buffer for the tensor operand
      FailureOr<Value> maybeBuffer = getBuffer(rewriter, operand, options);
      if (failed(maybeBuffer))
        return failure();
      Value buffer = *maybeBuffer;

      FailureOr<Type> maybeBufferType =
          bufferization::getBufferType(operand, options);
      if (failed(maybeBufferType))
        return failure();

      // Cast if necessary
      Value castedBuffer = castBuffer(rewriter, *maybeBuffer, *maybeBufferType);

      newReturnOperands.push_back(castedBuffer);
      newResultTypes.push_back(*maybeBufferType);
    }
    // Step 2: Create new scope op with bufferized result types
    rewriter.setInsertionPoint(scopeOp);
    auto newScopeOp =
        rewriter.create<scope::ScopeOp>(scopeOp.getLoc(), newResultTypes);
    newScopeOp->setAttrs(scopeOp->getAttrs());
    // Step 3: Move the region to the new scope op
    rewriter.inlineRegionBefore(region, newScopeOp.getRegion(),
                                newScopeOp.getRegion().end());
    // Step 4: Update returnOp inside the new region with bufferized operands
    Block &newBlock = newScopeOp.getRegion().front();
    auto newReturnOp = cast<scope::ReturnOp>(newBlock.getTerminator());
    rewriter.setInsertionPoint(newReturnOp);
    rewriter.replaceOpWithNewOp<scope::ReturnOp>(newReturnOp,
                                                 newReturnOperands);
    // Step 5: Replace op result with bufferized results
    replaceOpWithBufferizedValues(rewriter, scopeOp, newScopeOp->getResults());
    return success();
  }
};
} // namespace
} // namespace scope
} // namespace mlir

void mlir::scope::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scope::ScopeDialect *dialect) {
    ScopeOp::attachInterface<ScopeOpInterface>(*ctx);
  });
}
