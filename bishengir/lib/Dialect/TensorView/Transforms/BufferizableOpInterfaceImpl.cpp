//===- BufferizableOpInterfaceImpl.cpp - TensorView bufferization ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TensorView/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tv;

namespace {

static LogicalResult
bufferizeViewAccessOp(Operation *op, RewriterBase &rewriter,
                      const BufferizationOptions &options) {
  if (!llvm::any_of(op->getOperandTypes(),
                    [](Type type) { return isa<TensorType>(type); }))
    return success();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());
  for (OpOperand &opOperand : op->getOpOperands()) {
    Value value = opOperand.get();
    if (isa<TensorType>(value.getType())) {
      FailureOr<Value> buffer = getBuffer(rewriter, value, options);
      if (failed(buffer))
        return failure();
      value = *buffer;
    }
    newOperands.push_back(value);
  }

  SmallVector<Value> newOutputBuffers;
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op))
    for (OpResult result : op->getResults())
      newOutputBuffers.push_back(
          newOperands[dpsOp.getDpsInitOperand(result.getResultNumber())
                          ->getOperandNumber()]);
  rewriter.setInsertionPoint(op);
  clone(rewriter, op, /*newResultTypes=*/TypeRange{}, newOperands);
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);
  return success();
}

struct ViewLoadOpInterface
    : public DstBufferizableOpInterfaceExternalModel<ViewLoadOpInterface,
                                                     ViewLoadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The complete destination, including padding, is overwritten.
    return cast<DestinationStyleOpInterface>(op).isDpsInput(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeViewAccessOp(op, rewriter, options);
  }
};

struct ViewStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<ViewStoreOpInterface,
                                                    ViewStoreOp> {
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

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeViewAccessOp(op, rewriter, options);
  }
};

template <typename OpTy>
struct CoreTypeInterface
    : hivm::CoreTypeInterface::ExternalModel<CoreTypeInterface<OpTy>, OpTy> {
  std::optional<hivm::TCoreType> getCoreType(Operation *) const {
    return hivm::TCoreType::VECTOR;
  }
};
} // namespace

void tv::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, TensorViewDialect *) {
    context->getOrLoadDialect<hivm::HIVMDialect>();
    ViewLoadOp::attachInterface<ViewLoadOpInterface>(*context);
    ViewStoreOp::attachInterface<ViewStoreOpInterface>(*context);
    ViewLoadOp::attachInterface<CoreTypeInterface<ViewLoadOp>>(*context);
    ViewStoreOp::attachInterface<CoreTypeInterface<ViewStoreOp>>(*context);
  });
}
