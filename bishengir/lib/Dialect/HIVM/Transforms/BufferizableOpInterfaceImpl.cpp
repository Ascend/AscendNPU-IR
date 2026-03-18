//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
    auto fixPipeOp = static_cast<FixpipeOp>(op);
    SmallVector<Value> opr{*buffer, dstOp->get()};
    if (fixPipeOp.getQuantScale())
      opr.push_back(fixPipeOp.getQuantScale());
    auto newOp = cast<DestinationStyleOpInterface>(
        clone(rewriter, op, /*newResultTypes=*/TypeRange{}, opr));

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

template <typename CustomOpT>
struct HIVMCustomOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          HIVMCustomOpInterface<CustomOpT>, CustomOpT> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options,
        /* supportMixedTensorBufferMode */ true);
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

template <typename CopyOrStoreOpT>
struct HIVMCopyOrStoreOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          HIVMCopyOrStoreOpInterface<CopyOrStoreOpT>, CopyOrStoreOpT> {
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

struct EmbeddingGatherOpInterface
    : public BufferizableOpInterface::ExternalModel<EmbeddingGatherOpInterface,
                                                    hivm::EmbeddingGatherOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() < 2;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 2; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                     const AnalysisState &state) const {
    auto gatherOp = cast<EmbeddingGatherOp>(op);
    AliasingValueList result;

    if (opOperand.getOperandNumber() == 2) { // $dst
      // dst is alias of the result
      result.addAlias({AliasingValue(gatherOp->getResult(0),
                                    BufferRelation::Equivalent,
                                    /*isMustAlias=*/true)});
    }

    return result;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto gatherOp = cast<hivm::EmbeddingGatherOp>(op);

    auto srcBuffer = gatherOp.getSrc();

    FailureOr<Value> indexBuffer = getBuffer(rewriter, gatherOp.getIndex(), options);
    if (failed(indexBuffer))
      return failure();

    FailureOr<Value> dstBuffer = getBuffer(rewriter, gatherOp.getDst(), options);
    if (failed(dstBuffer))
      return failure();

    Value bound = gatherOp.getBound();
    auto offsets = gatherOp.getOffsets();
    auto numels = gatherOp.getNumels();

    rewriter.create<EmbeddingGatherOp>(gatherOp.getLoc(),
                                       /*resultType*/ TypeRange{},
                                       /*operands*/
                                       srcBuffer, *indexBuffer, *dstBuffer,
                                       bound, offsets, numels);

    if (gatherOp->getNumResults() > 0) {
      replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    }

    return success();
  }
};

struct IndirectLoadOpInterface
    : public BufferizableOpInterface::ExternalModel<IndirectLoadOpInterface,
                                                    hivm::IndirectLoadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() < 2 || opOperand.getOperandNumber() ==3 || opOperand.getOperandNumber() == 4;
  }
 
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 2; // $dst
  }
 
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                     const AnalysisState &state) const {
    auto indirectLoadOp = cast<IndirectLoadOp>(op);
    AliasingValueList result;
 
    if (opOperand.getOperandNumber() == 2) { // $dst
      result.addAlias({AliasingValue(indirectLoadOp->getResult(0),
                                    BufferRelation::Equivalent,
                                    /*isMustAlias=*/true)});
    }
 
    return result;
  }
 
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto indirectLoadOp = cast<hivm::IndirectLoadOp>(op);
 
    auto srcBuffer = indirectLoadOp.getSrc();
 
    FailureOr<Value> offsetBuffer = getBuffer(rewriter, indirectLoadOp.getOffsets(), options);
    if (failed(offsetBuffer))
      return failure();
 
    FailureOr<Value> dstBuffer = getBuffer(rewriter, indirectLoadOp.getDst(), options);
    if (failed(dstBuffer))
      return failure();
    
    FailureOr<Value> maskBuffer = failure();
    if (indirectLoadOp.getMask()) {
      maskBuffer = getBuffer(rewriter, indirectLoadOp.getMask(), options);
      if (failed(maskBuffer))
        return failure();
    }

    FailureOr<Value> otherBuffer = failure();
    if (indirectLoadOp.getOther()) {
      otherBuffer = getBuffer(rewriter, indirectLoadOp.getOther(), options);
      if (failed(otherBuffer))
        return failure();
    }

    auto mask = indirectLoadOp.getMask();
    auto other = indirectLoadOp.getOther();
 
    if (indirectLoadOp.getMask()) {
      if (indirectLoadOp.getOther()) {
        rewriter.create<IndirectLoadOp>(
        indirectLoadOp.getLoc(),
        /*resultType*/ TypeRange{},
        /*operands*/
        srcBuffer, *offsetBuffer, *dstBuffer, *maskBuffer, *otherBuffer);
      }else {
        rewriter.create<IndirectLoadOp>(
        indirectLoadOp.getLoc(),
        /*resultType*/ TypeRange{},
        /*operands*/
        srcBuffer, *offsetBuffer, *dstBuffer, *maskBuffer, other);
      }
    } else {
      rewriter.create<IndirectLoadOp>(
        indirectLoadOp.getLoc(),
        /*resultType*/ TypeRange{},
        /*operands*/
        srcBuffer, *offsetBuffer, *dstBuffer, mask, other);
    }
 
    if (indirectLoadOp->getNumResults() > 0) {
      replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    }
 
    return success();
  }
};

struct IndirectStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<IndirectStoreOpInterface,
                                                    hivm::IndirectStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() > 0;
  }
 
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }
 
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto indirectStoreOp = cast<hivm::IndirectStoreOp>(op);
 
    FailureOr<Value> srcBuffer = getBuffer(rewriter, indirectStoreOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();
 
    FailureOr<Value> offsetBuffer = getBuffer(rewriter, indirectStoreOp.getOffsets(), options);
    if (failed(offsetBuffer))
      return failure();
 
    auto dstBuffer = indirectStoreOp.getDst();
    
    FailureOr<Value> maskBuffer = failure();
    if (indirectStoreOp.getMask()) {
      maskBuffer = getBuffer(rewriter, indirectStoreOp.getMask(), options);
      if (failed(maskBuffer))
        return failure();
    }

    auto mask = indirectStoreOp.getMask();
    IndirectStoreOp newOp;
    if (mask) {
      newOp = rewriter.create<IndirectStoreOp>(
      indirectStoreOp.getLoc(),
      /*operands*/
      dstBuffer, *offsetBuffer, *srcBuffer, *maskBuffer);
    } else {
      newOp = rewriter.create<IndirectStoreOp>(
        indirectStoreOp.getLoc(),
        /*operands*/
        dstBuffer, *offsetBuffer, *srcBuffer, mask);
    }

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct GatherTOpInterface
    : public BufferizableOpInterface::ExternalModel<GatherTOpInterface,
                                                    hivm::GatherTOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0 || opOperand.getOperandNumber() == 1;
  }
 
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 2; // $dst
  }
 
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                     const AnalysisState &state) const {
    auto gatherOp = cast<GatherTOp>(op);
    AliasingValueList result;
 
    if (opOperand.getOperandNumber() == 2) { // $dst
      result.addAlias({AliasingValue(gatherOp->getResult(0),
                                    BufferRelation::Equivalent,
                                    /*isMustAlias=*/true)});
    }
 
    return result;
  }
 
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto gatherOp = cast<hivm::GatherTOp>(op);
 
    auto srcBuffer = gatherOp.getSrc();
 
    FailureOr<Value> indexBuffer = getBuffer(rewriter, gatherOp.getIndex(), options);
    if (failed(indexBuffer))
      return failure();
 
    FailureOr<Value> dstBuffer = getBuffer(rewriter, gatherOp.getDst(), options);
    if (failed(dstBuffer))
      return failure();

    Value dim = gatherOp.getDim();
    Value bound = gatherOp.getBound();
    auto src_stride = gatherOp.getSrcStride();
    auto index_shape = gatherOp.getIndexShape();
    auto offsets = gatherOp.getOffsets();

    rewriter.create<GatherTOp>(
        gatherOp.getLoc(),
        /*resultType*/ TypeRange{},
        /*operands*/
        srcBuffer, *indexBuffer, *dstBuffer, bound, dim, src_stride, index_shape, offsets);

    if (gatherOp->getNumResults() > 0) {
      replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    }
 
    return success();
  }
};

struct IndexPutOpInterface
    : public BufferizableOpInterface::ExternalModel<IndexPutOpInterface,
                                                    hivm::IndexPutOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1 || opOperand.getOperandNumber() == 2;
  }
 
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0; // $dst
  }
 
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                     const AnalysisState &state) const {
    return {};
  }
 
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto indexPutOp = cast<hivm::IndexPutOp>(op);
 
    auto dstBuffer = indexPutOp.getDst();
 
    FailureOr<Value> indexBuffer = getBuffer(rewriter, indexPutOp.getIndex(), options);
    if (failed(indexBuffer))
      return failure();
 
    FailureOr<Value> valueBuffer = getBuffer(rewriter, indexPutOp.getValue(), options);
    if (failed(valueBuffer))
      return failure();
    
    Value scatter_dim = indexPutOp.getScatterDim();
    Value bound = indexPutOp.getBound();
    auto end_offset = indexPutOp.getEndOffset();
    auto start_offset = indexPutOp.getStartOffset();
    auto dst_stride = indexPutOp.getDstStride();

    IndexPutOp newOp = rewriter.create<IndexPutOp>(
        indexPutOp.getLoc(),
        /*resultType*/ TypeRange{},
        /*operands*/
        dstBuffer, *indexBuffer, *valueBuffer, scatter_dim, bound, end_offset, start_offset, dst_stride);

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct ScatterTOpInterface
  : public BufferizableOpInterface::ExternalModel<ScatterTOpInterface,
    hivm::ScatterTOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1 || opOperand.getOperandNumber() == 2;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0; // $dst
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto scatterTOp = cast<hivm::ScatterTOp>(op);

    FailureOr<Value> valueBuffer = getBuffer(rewriter, scatterTOp.getValue(), options);
    if (failed(valueBuffer))
      return failure();

    FailureOr<Value> indexTileBuffer = getBuffer(rewriter, scatterTOp.getIndexTile(), options);
    if (failed(indexTileBuffer))
      return failure();

    auto ret = rewriter.create<ScatterTOp>(
      scatterTOp.getLoc(),
      /*resultType*/ TypeRange{},
      /*operands*/
      scatterTOp.getDst(), *valueBuffer, *indexTileBuffer, scatterTOp.getIndexBoundary(), scatterTOp.getDim(),
      scatterTOp.getDstStride(), scatterTOp.getIndexShape(), scatterTOp.getOffsets());

    rewriter.replaceOp(op, ret);

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
    L12UBOp::attachInterface<HIVMCopyOrStoreOpInterface<L12UBOp>>(*ctx);
    CopyOp::attachInterface<HIVMCopyOrStoreOpInterface<hivm::CopyOp>>(*ctx);
    CustomOp::attachInterface<HIVMCustomOpInterface<CustomOp>>(*ctx);
    CustomMacroOp::attachInterface<HIVMCustomOpInterface<CustomMacroOp>>(*ctx);
    LoadOp::attachInterface<HIVMLoadOpInterface>(*ctx);
    StoreOp::attachInterface<HIVMCopyOrStoreOpInterface<hivm::StoreOp>>(*ctx);
    MatmulOp::attachInterface<HIVMMatmulOpInterface>(*ctx);
    MixMatmulOp::attachInterface<HIVMMixMatmulOpInterface>(*ctx);
    MixGroupMatmulOp::attachInterface<HIVMMixGroupMatmulOpInterface>(*ctx);
    DebugOp::attachInterface<DebugOpInterface>(*ctx);
    VConcatOp::attachInterface<VConcatOpInterface>(*ctx);
    BitcastOp::attachInterface<BitcastOpInterface>(*ctx);
    EmbeddingGatherOp::attachInterface<EmbeddingGatherOpInterface>(*ctx);
    IndirectLoadOp::attachInterface<IndirectLoadOpInterface>(*ctx);
    IndirectStoreOp::attachInterface<IndirectStoreOpInterface>(*ctx);
    GatherTOp::attachInterface<GatherTOpInterface>(*ctx);
    IndexPutOp::attachInterface<IndexPutOpInterface>(*ctx);
    ScatterTOp::attachInterface<ScatterTOpInterface>(*ctx);
    // Register all HIVM Vector Ops
    VectorOpInterfaceHelper<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
}
