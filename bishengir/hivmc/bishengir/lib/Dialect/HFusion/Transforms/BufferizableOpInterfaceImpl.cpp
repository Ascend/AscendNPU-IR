//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
using namespace hfusion;
using namespace mlir::bufferization;

namespace {

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult
bufferizeDestinationStyleOpInterface(RewriterBase &rewriter,
                                     DestinationStyleOpInterface op,
                                     const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasPureBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasPureTensorSemantics())
    return op->emitError() << "op does not have pure tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumDpsInputs());
  for (OpOperand *opOperand : op.getDpsInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
    if (failed(buffer))
      return failure();
    newInputBuffers.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer))
      return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  assert(op->getNumRegions() == 1 && "expected that op has 1 region");
  OperationState state(op->getLoc(), op->getName(), newOperands, TypeRange{},
                       op->getAttrs());
  state.addRegion();
  Operation *newOp = Operation::create(state);
  newOp->getRegion(0).getBlocks().splice(newOp->getRegion(0).begin(),
                                         op->getRegion(0).getBlocks());

  // We don't want the rewriter tracks an incomplete operation, so insert new
  // operation after op was fully constructed.
  rewriter.insert(newOp);

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

/// Bufferization of hfusion.generic. Replace with a new hfusion.generic that
/// operates entirely on memrefs.
template <typename OpTy>
struct HFusionOpInterface
    : public DstBufferizableOpInterfaceExternalModel<HFusionOpInterface<OpTy>,
                                                     OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Operand is read if it is used in the computation.
    auto linalgOp = cast<linalg::LinalgOp>(op);
    return linalgOp.payloadUsesValueFromOperand(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it is not an input/init.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);

    // All loops must be parallel.
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
      return false;

    // All index maps of tensors must be identity maps.
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    assert(linalgOp->getNumOperands() == indexingMaps.size() &&
           "unexpected number of indexing maps");
    for (auto [operand, map] :
         llvm::zip(linalgOp->getOpOperands(), indexingMaps)) {
      // Non-tensors do not participate in bufferization, so they can be
      // ignored.
      if (!isa<RankedTensorType, MemRefType>(operand.get().getType()))
        continue;
      // Only consider operands in `opOperands`.
      if (!llvm::is_contained(opOperands, &operand))
        continue;
      // TODO: This could be generalized to other indexing maps. (All indexing
      // must be the same.)
      if (!map.isIdentity())
        return false;
    }

    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

/// Helper structure that iterates over all HFusionOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops> struct HFusionOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<HFusionOpInterface<Ops>>(*ctx), ...);
  }
};
} // namespace

void mlir::hfusion::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, hfusion::HFusionDialect *dialect) {
        // Register all HFusion structured ops, which implements `linalgOp`
        // interface.
        // `LinalgOp` is an interface and it is not possible to attach an
        // external interface to an existing interface. Therefore, attach the
        // `BufferizableOpInterface` to all ops one-by-one.
        HFusionOpInterfaceHelper<
#define GET_OP_LIST
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc"
            >::registerOpInterface(ctx);
      });
}
