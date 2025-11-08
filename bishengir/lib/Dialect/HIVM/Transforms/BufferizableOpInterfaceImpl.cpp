//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
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
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.cpp
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/BufferizableOpInterfaceImpl.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace hivm;
using namespace mlir::bufferization;

namespace {

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult bufferizeDestinationStyleOpInterface(
    RewriterBase &rewriter, DestinationStyleOpInterface op,
    const BufferizationOptions &options,
    bool supportMixedTensorBufferMode = false) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasPureBufferSemantics()) {
    return success();
  }

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasPureTensorSemantics() && !supportMixedTensorBufferMode) {
    return op->emitError() << "op does not have tensor semantics";
  }

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

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer)) {
      return failure();
    }
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands.
  clone(rewriter, op, /*newResultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

struct MmadL1OpInterface
    : public DstBufferizableOpInterfaceExternalModel<MmadL1OpInterface,
                                                     hivm::MmadL1Op> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct FixpipeOpInterface
    : public DstBufferizableOpInterfaceExternalModel<FixpipeOpInterface,
                                                     hivm::FixpipeOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    if (dpsOp.hasPureBufferSemantics()) {
      return success();
    }
    if (dpsOp.hasPureTensorSemantics()) {
      return bufferizeDestinationStyleOpInterface(rewriter, dpsOp, options);
    }
    // We only handle the case where fixpipe op's input is a tensor from
    // mmad and fixpipe op's output is a memref type.
    auto srcOp = dpsOp.getDpsInputOperand(0);
    auto dstOp = dpsOp.getDpsInitOperand(0);
    if (!isa<TensorType>(srcOp->get().getType()) ||
        !isa<MemRefType>(dstOp->get().getType())) {
      return op->emitError() << "src and dst op should have tensor and memref "
                                "type, respectively";
    }
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    FailureOr<Value> buffer = getBuffer(rewriter, srcOp->get(), options);
    if (failed(buffer)) {
      return failure();
    }
    // Set insertion point now that potential alloc/dealloc are introduced.
    rewriter.setInsertionPoint(op);
    // Clone the op, but use the new operands.
    auto newOp = cast<DestinationStyleOpInterface>(clone(
        rewriter, op, /*newResultTypes=*/TypeRange{}, {*buffer, dstOp->get()}));
    // We need to manually replace the old op because it has memory effects
    // and won't be deleted automatically.
    rewriter.replaceOp(op, newOp);
    return success();
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }
};

template <typename OpType>
struct NDNZConversionOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          NDNZConversionOpInterface<OpType>, OpType> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct HIVMCopyOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMCopyOpInterface,
                                                     hivm::CopyOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct HIVMLoadOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMLoadOpInterface,
                                                     hivm::LoadOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct HIVMStoreOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMStoreOpInterface,
                                                     hivm::StoreOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    if (dpsOp.hasPureBufferSemantics()) {
      return success();
    }
    if (dpsOp.hasPureTensorSemantics()) {
      return bufferizeDestinationStyleOpInterface(rewriter, dpsOp, options);
    }
    // We only handle the case where fixpipe op's input is a tensor from
    // mmad and fixpipe op's output is a memref type.
    auto srcOp = dpsOp.getDpsInputOperand(0);
    auto dstOp = dpsOp.getDpsInitOperand(0);
    if (!isa<TensorType>(srcOp->get().getType()) ||
        !isa<MemRefType>(dstOp->get().getType())) {
      return op->emitError() << "src and dst op should have tensor and memref "
                                "type, respectively";
    }
    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    FailureOr<Value> buffer = getBuffer(rewriter, srcOp->get(), options);
    if (failed(buffer)) {
      return failure();
    }
    // Set insertion point now that potential alloc/dealloc are introduced.
    rewriter.setInsertionPoint(op);
    // Clone the op, but use the new operands.
    auto newOp = cast<DestinationStyleOpInterface>(clone(
        rewriter, op, /*newResultTypes=*/TypeRange{}, {*buffer, dstOp->get()}));
    // We need to manually replace the old op because it has memory effects
    // and won't be deleted automatically.
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct HIVMMatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMMatmulOpInterface,
                                                     hivm::MatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

struct HIVMMixMatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HIVMMixMatmulOpInterface,
                                                     hivm::MixMatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // The `tilingParams` operand might be already bufferized.
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

struct HIVMMixGroupMatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          HIVMMixGroupMatmulOpInterface, hivm::MixGroupMatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // The `tilingParams` operand might be already bufferized.
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

template <typename OpTy>
struct VectorOpInterface
    : public DstBufferizableOpInterfaceExternalModel<VectorOpInterface<OpTy>,
                                                     OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Operand is read if it is used in the computation.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it is not an input/init.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    // Src0 and dst of elemwiseOp are not conflicting if the op bufferizes
    // to element-wise access.
    auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
    return hivmOp && hivmOp.isElemwiseNaryOp();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct DebugOpInterface
    : public BufferizableOpInterface::ExternalModel<DebugOpInterface,
                                                    hivm::DebugOp> {
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
    auto debugOp = cast<hivm::DebugOp>(op);

    auto debugtype = debugOp.getDebugtype();
    auto prefix = debugOp.getPrefix();
    auto hex = debugOp.getHex();

    Value newArg;
    const auto &arg = debugOp.getArg();
    Value value = arg;
    if (isa<TensorType>(value.getType())) {
      FailureOr<Value> maybeBuffer = getBuffer(rewriter, value, options);
      if (failed(maybeBuffer))
        return failure();
      Value buffer = *maybeBuffer;
      newArg = buffer;
    } else {
      newArg = value;
    }

    replaceOpWithNewBufferizedOp<hivm::DebugOp>(
        rewriter, op, debugtype, prefix, hex, newArg,
        hivm::TCoreTypeAttr::get(op->getContext(),
                                 hivm::TCoreType::CUBE_OR_VECTOR));

    return success();
  }
};

struct VPadOpInterface
    : public DstBufferizableOpInterfaceExternalModel<VPadOpInterface,
                                                     hivm::VPadOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // TODO
    return failure();
  }
};

struct VConcatOpInterface
    : public DstBufferizableOpInterfaceExternalModel<VConcatOpInterface,
                                                     hivm::VConcatOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /*supportMixedTensorBufferMode=*/true);
  }
};

/// Helper structure that iterates over all VectorOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops> struct VectorOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<VectorOpInterface<Ops>>(*ctx), ...);
  }
};

struct BitcastOpInterface
    : public BufferizableOpInterface::ExternalModel<BitcastOpInterface,
                                                    hivm::BitcastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto bitcastOp = dyn_cast<hivm::BitcastOp>(op);
    auto resultTensorType = dyn_cast<TensorType>(bitcastOp.getType());
    if (!resultTensorType)
      return success();

    FailureOr<Value> source = getBuffer(rewriter, bitcastOp.getSrc(), options);
    if (failed(source))
      return failure();
    auto sourceType = cast<BaseMemRefType>(source->getType());

    // Result type should have same layout and address space as the source type.
    BaseMemRefType resultType;
    if (auto rankedMemRefType = dyn_cast<MemRefType>(sourceType)) {
      resultType = MemRefType::get(
          rankedMemRefType.getShape(), resultTensorType.getElementType(),
          rankedMemRefType.getLayout(), rankedMemRefType.getMemorySpace());
    } else {
      auto unrankedMemrefType = cast<UnrankedMemRefType>(sourceType);
      resultType = UnrankedMemRefType::get(resultTensorType.getElementType(),
                                           unrankedMemrefType.getMemorySpace());
    }

    replaceOpWithNewBufferizedOp<hivm::BitcastOp>(rewriter, op, resultType,
                                                  *source);
    return success();
  }
};

} // namespace

void mlir::hivm::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, hivm::HIVMDialect *dialect) {
    FixpipeOp::attachInterface<FixpipeOpInterface>(*ctx);
    MmadL1Op::attachInterface<MmadL1OpInterface>(*ctx);
    ND2NZOp::attachInterface<NDNZConversionOpInterface<ND2NZOp>>(*ctx);
    NZ2NDOp::attachInterface<NDNZConversionOpInterface<NZ2NDOp>>(*ctx);
    CopyOp::attachInterface<HIVMCopyOpInterface>(*ctx);
    LoadOp::attachInterface<HIVMLoadOpInterface>(*ctx);
    StoreOp::attachInterface<HIVMStoreOpInterface>(*ctx);
    MatmulOp::attachInterface<HIVMMatmulOpInterface>(*ctx);
    MixMatmulOp::attachInterface<HIVMMixMatmulOpInterface>(*ctx);
    MixGroupMatmulOp::attachInterface<HIVMMixGroupMatmulOpInterface>(*ctx);
    DebugOp::attachInterface<DebugOpInterface>(*ctx);
    VConcatOp::attachInterface<VConcatOpInterface>(*ctx);
    BitcastOp::attachInterface<BitcastOpInterface>(*ctx);
    // Register all HIVM Vector Ops
    VectorOpInterfaceHelper<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
}
