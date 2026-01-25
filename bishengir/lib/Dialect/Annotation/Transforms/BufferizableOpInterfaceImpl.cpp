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

#include "bishengir/Dialect/Annotation/Transforms/BufferizableOpInterfaceImpl.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
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
using namespace mlir::annotation;

namespace mlir {
namespace annotation {
namespace {
/// Bufferization of annotation.mark.
struct MarkOpInterface
    : public BufferizableOpInterface::ExternalModel<MarkOpInterface,
                                                    annotation::MarkOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    // Mark operands always bufferize inplace. Otherwise, an alloc + copy
    // may be generated inside the block.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    // New operands for the cloned op.
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (OpOperand &opOperand : op->getOpOperands()) {
      if (!isa<TensorType>(opOperand.get().getType())) {
        newOperands.push_back(opOperand.get());
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, opOperand.get(), options);
      if (failed(buffer)) {
        return failure();
      }
      newOperands.push_back(*buffer);
    }

    DictionaryAttr newAttrs = op->getAttrDictionary();
    // Clone the op, but use the new operands.
    auto newOp =
        clone(rewriter, op, /*newResultTypes=*/TypeRange{}, newOperands);
    // Forward the old attributes to the new operation
    newOp->setAttrs(newAttrs);

    // Erase old annotation
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace
} // namespace annotation
} // namespace mlir

void mlir::annotation::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, annotation::AnnotationDialect *dialect) {
        MarkOp::attachInterface<MarkOpInterface>(*ctx);
      });
}
