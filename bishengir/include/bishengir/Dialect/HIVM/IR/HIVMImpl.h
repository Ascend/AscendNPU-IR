//===- HIVMImpl.h - HIVM implementation -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace hivm {
/// find v in vector valueVec
std::optional<int> findIdx(SmallVector<Value> valueVec, Value v);

int64_t getUsersNum(Value v);

bool isLocalMatmulInit(Operation *op, Value v);

/// to trace op in isMatchedOp way and check whether op is single chain
bool traceSingleChainUser(
    Value v, const std::function<bool(Operation *, Value v)> &isMatchedOp);

template <typename OpType>
std::optional<Operation *> traceDefOp(Value v, bool isSingleChain = false) {
  if (isSingleChain && getUsersNum(v) != 1)
    return std::nullopt;
  if (Operation *definingOp = v.getDefiningOp<OpType>()) {
    return definingOp;
  } else if (auto reshapeOp = v.getDefiningOp<tensor::ReshapeOp>()) {
    return traceDefOp<OpType>(reshapeOp.getSource(), isSingleChain);
  } else if (auto memrefCollapseShape =
                 v.getDefiningOp<memref::CollapseShapeOp>()) {
    return traceDefOp<OpType>(memrefCollapseShape.getViewSource(),
                              isSingleChain);
  } else if (auto tensorCollapseShape =
                 v.getDefiningOp<tensor::CollapseShapeOp>()) {
    return traceDefOp<OpType>(tensorCollapseShape.getSrc(), isSingleChain);
  } else if (auto subViewOp = v.getDefiningOp<memref::SubViewOp>()) {
    return traceDefOp<OpType>(subViewOp.getViewSource(), isSingleChain);
  } else if (auto toMemrefOp = v.getDefiningOp<bufferization::ToMemrefOp>()) {
    return traceDefOp<OpType>(toMemrefOp.getOperand(), isSingleChain);
  } else if (auto toTensorOp = v.getDefiningOp<bufferization::ToTensorOp>()) {
    return traceDefOp<OpType>(toTensorOp.getOperand(), isSingleChain);
  } else if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
    return traceDefOp<OpType>(viewOp.getViewSource(), isSingleChain);
  } else if (auto reshapeOp = v.getDefiningOp<memref::ReshapeOp>()) {
    return traceDefOp<OpType>(reshapeOp.getViewSource(), isSingleChain);
  } else if (auto expandShapeOp = v.getDefiningOp<memref::ExpandShapeOp>()) {
    return traceDefOp<OpType>(expandShapeOp.getViewSource(), isSingleChain);
  } else if (auto tensorExpandShapeOp =
                 v.getDefiningOp<tensor::ExpandShapeOp>()) {
    return traceDefOp<OpType>(tensorExpandShapeOp->getOperand(0),
                              isSingleChain);
  } else if (auto extractStridedMetadataOp =
                 v.getDefiningOp<memref::ExtractStridedMetadataOp>()) {
    return traceDefOp<OpType>(extractStridedMetadataOp.getViewSource(),
                              isSingleChain);
  } else if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
    return traceDefOp<OpType>(castOp.getViewSource(), isSingleChain);
  } else if (auto reinterpretCastOp =
                 v.getDefiningOp<memref::ReinterpretCastOp>()) {
    return traceDefOp<OpType>(reinterpretCastOp.getViewSource(), isSingleChain);
  } else if (auto blockArg = dyn_cast_if_present<BlockArgument>(v)) {
    if (auto scfForOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (OpOperand *iterArgOperand = scfForOp.getTiedLoopInit(blockArg))
        return traceDefOp<OpType>(iterArgOperand->get(), isSingleChain);
    }
  } else if (auto forOp = v.getDefiningOp<scf::ForOp>()) {
    const unsigned int index = cast<OpResult>(v).getResultNumber();
    Value yieldedValue = forOp.getYieldedValues()[index];
    return traceDefOp<OpType>(yieldedValue, isSingleChain);
  } else if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
    const unsigned int index = cast<OpResult>(v).getResultNumber();
    Block &thenBlock = ifOp.getThenRegion().front();
    Value yieldedValue = thenBlock.getTerminator()->getOperand(index);
    return traceDefOp<OpType>(yieldedValue, isSingleChain);
  } else if (auto extractSliceOp = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    return traceDefOp<OpType>(extractSliceOp.getSource(), isSingleChain);
  } else if (auto insertSliceOp = v.getDefiningOp<tensor::InsertSliceOp>()) {
    // In the tensor.insertSlice dialect, both source and Dest are data sources.
    // Preferentially return the source.
    auto traceOp = traceDefOp<OpType>(insertSliceOp.getSource(), isSingleChain);
    if (traceOp != std::nullopt)
      return traceOp;
    else
      return traceDefOp<OpType>(insertSliceOp.getDest(), isSingleChain);
  } else if (auto memSpaceCastOp =
                 v.getDefiningOp<memref::MemorySpaceCastOp>()) {
    return traceDefOp<OpType>(memSpaceCastOp.getSource(), isSingleChain);
  }
  return std::nullopt;
}


template <typename OpType>
void traceDefOpsImpl(Value v, bool isSingleChain,
                                           SmallVectorImpl<Operation *> &results,
                                           SmallVectorImpl<Value> &recursionStack) {
  if (!v)
    return;

  if (isSingleChain && getUsersNum(v) != 1)
    return;

  // Avoid infinite recursion without a visited set
  for (Value stacked : recursionStack) {
    if (stacked == v)
      return;
  }
  recursionStack.push_back(v);

  struct RecursionStackGuard {
    SmallVectorImpl<Value> &recursionStack;
    ~RecursionStackGuard() { recursionStack.pop_back(); }
  } guard{recursionStack};

  if (Operation *definingOp = v.getDefiningOp<OpType>()) {
    results.push_back(definingOp);
    return;
  } else if (auto reshapeOp = v.getDefiningOp<tensor::ReshapeOp>()) {
    traceDefOpsImpl<OpType>(reshapeOp.getSource(), isSingleChain, results, recursionStack);
  } else if (auto memrefCollapseShape =
                 v.getDefiningOp<memref::CollapseShapeOp>()) {
    traceDefOpsImpl<OpType>(memrefCollapseShape.getViewSource(),
                              isSingleChain, results, recursionStack);
  } else if (auto tensorCollapseShape =
                 v.getDefiningOp<tensor::CollapseShapeOp>()) {
    traceDefOpsImpl<OpType>(tensorCollapseShape.getSrc(), isSingleChain, results, recursionStack);
  } else if (auto subViewOp = v.getDefiningOp<memref::SubViewOp>()) {
    traceDefOpsImpl<OpType>(subViewOp.getViewSource(), isSingleChain, results, recursionStack);
  } else if (auto toMemrefOp = v.getDefiningOp<bufferization::ToMemrefOp>()) {
    traceDefOpsImpl<OpType>(toMemrefOp.getOperand(), isSingleChain, results, recursionStack);
  } else if (auto toTensorOp = v.getDefiningOp<bufferization::ToTensorOp>()) {
    traceDefOpsImpl<OpType>(toTensorOp.getOperand(), isSingleChain, results, recursionStack);
  } else if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
    traceDefOpsImpl<OpType>(viewOp.getViewSource(), isSingleChain, results, recursionStack);
  } else if (auto reshapeOp = v.getDefiningOp<memref::ReshapeOp>()) {
    traceDefOpsImpl<OpType>(reshapeOp.getViewSource(), isSingleChain, results, recursionStack);
  } else if (auto expandShapeOp = v.getDefiningOp<memref::ExpandShapeOp>()) {
    traceDefOpsImpl<OpType>(expandShapeOp.getViewSource(), isSingleChain, results, recursionStack);
  } else if (auto tensorExpandShapeOp =
                 v.getDefiningOp<tensor::ExpandShapeOp>()) {
    traceDefOpsImpl<OpType>(tensorExpandShapeOp->getOperand(0), isSingleChain, results, recursionStack);
  } else if (auto extractStridedMetadataOp =
                 v.getDefiningOp<memref::ExtractStridedMetadataOp>()) {
    traceDefOpsImpl<OpType>(extractStridedMetadataOp.getViewSource(), isSingleChain, results, recursionStack);
  } else if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
    traceDefOpsImpl<OpType>(castOp.getViewSource(), isSingleChain, results, recursionStack);
  } else if (auto reinterpretCastOp =
                 v.getDefiningOp<memref::ReinterpretCastOp>()) {
    traceDefOpsImpl<OpType>(reinterpretCastOp.getViewSource(), isSingleChain, results, recursionStack);
  } else if (auto blockArg = dyn_cast_if_present<BlockArgument>(v)) {
    // trace both scf.for iter args and region args
    if (auto scfForOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (blockArg.getOwner() == scfForOp.getBody()) {
        unsigned numInduction = scfForOp.getNumInductionVars();
        unsigned argNo = blockArg.getArgNumber();
        if (argNo < numInduction)
          return;

        unsigned iterIdx = argNo - numInduction;
        if (Operation *terminator = scfForOp.getBody()->getTerminator()) {
          if (auto yieldOp = dyn_cast_if_present<scf::YieldOp>(terminator)) {
            Value yielded = yieldOp.getOperand(iterIdx);
            // avoid tracing to the same yield operand again
            if (yielded != v)
              traceDefOpsImpl<OpType>(yielded, isSingleChain, results, recursionStack);
          }
        }
        if (OpOperand *iterInit = scfForOp.getTiedLoopInit(blockArg))
          traceDefOpsImpl<OpType>(iterInit->get(), isSingleChain, results, recursionStack);
      }
    }
  } else if (auto forOp = v.getDefiningOp<scf::ForOp>()) {
    const unsigned int index = cast<OpResult>(v).getResultNumber();
    Value yieldedValue = forOp.getYieldedValues()[index];
    traceDefOpsImpl<OpType>(yieldedValue, isSingleChain, results, recursionStack);
  } else if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
    // trace both then and else regions
    const unsigned int index = cast<OpResult>(v).getResultNumber();
    if (!ifOp.getThenRegion().empty()) {
      Block &thenBlock = ifOp.getThenRegion().front();
      Value yieldedValue = thenBlock.getTerminator()->getOperand(index);
      traceDefOpsImpl<OpType>(yieldedValue, isSingleChain, results, recursionStack);
    }
    if (!ifOp.getElseRegion().empty()) {
      Block &elseBlock = ifOp.getElseRegion().front();
      Value yieldedValue = elseBlock.getTerminator()->getOperand(index);
      traceDefOpsImpl<OpType>(yieldedValue, isSingleChain, results, recursionStack);
    }
  } else if (auto extractSliceOp = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    traceDefOpsImpl<OpType>(extractSliceOp.getSource(), isSingleChain, results, recursionStack);
  } else if (auto insertSliceOp = v.getDefiningOp<tensor::InsertSliceOp>()) {
    // In the tensor.insertSlice dialect, both source and Dest are data sources.
    // Preferentially return the source.
    traceDefOpsImpl<OpType>(insertSliceOp.getSource(), isSingleChain, results, recursionStack);
    traceDefOpsImpl<OpType>(insertSliceOp.getDest(), isSingleChain, results, recursionStack);
  } else if (auto memSpaceCastOp =
                 v.getDefiningOp<memref::MemorySpaceCastOp>()) {
    traceDefOpsImpl<OpType>(memSpaceCastOp.getSource(), isSingleChain, results, recursionStack);
  }
}

/// to trace ops in traceDefOp way and collect all matched ops
///
/// Compared with `traceDefOp`, this function additionally traces both branches
/// of `scf.if`, both yielded values and init operands for `scf.for`, and both
/// source and dest for `tensor.insert_slice`.
template <typename OpType>
SmallVector<Operation *> traceDefOps(Value v, bool isSingleChain = false) {
  SmallVector<Operation *> results;
  SmallVector<Value> recursionStack;
  traceDefOpsImpl<OpType>(v, isSingleChain, results, recursionStack);
  return results;
}

template <typename MmadLikeOpType>
typename std::enable_if<std::is_same_v<MmadLikeOpType, hivm::MmadL1Op> ||
                            std::is_same_v<MmadLikeOpType, hivm::BatchMmadL1Op>,
                        bool>::type
isSingleChainMmadToMmad(MmadLikeOpType op) {
  auto maybeMmadLikeOp =
      traceDefOp<MmadLikeOpType>(op.getC(), /*isSingleChain=*/true);
  return maybeMmadLikeOp.has_value();
}

/// Broadcast Scalar.
hivm::VBrcOp brcScalar(RewriterBase &rewriter, Location loc,
                       TypedAttr initValue, Value targetTensor);

/// Infer funcOp core type.
std::optional<TFuncCoreType> queryFuncCoreType(Operation *funcOp);

/// get operation core type
FailureOr<TCoreType> getCoreType(Operation *op);

// get is scalar like
bool isScalarLike(Type type);

/// Checks if a MemRefType has identity strides.
///
/// Identity strides represent the default memory layout where elements are
/// stored contiguously in row-major order
///
/// @param shapedType The MemRefType to check
/// @return true if the type has no layout or has an identity strided layout,
///         false otherwise
bool isIdentityStrides(MemRefType shapedType);

using AlignInfoMap = SmallVector<int64_t>;
/// Computes aligned sizes by rounding up each dimension to its alignment
/// requirement.
///
/// For each dimension, calculates the smallest size that is a multiple of the
/// corresponding alignment value. This ensures memory accesses respect
/// alignment constraints.
///
/// @param baseSizes Original sizes for each dimension
/// @param alignInfo Alignment requirements for each dimension (in elements)
/// @return Vector of aligned sizes where alignedSizes[i] = ceil(baseSizes[i] /
/// alignInfo[i]) * alignInfo[i]
SmallVector<int64_t> getAlignedSizes(ArrayRef<int64_t> baseSizes,
                                     AlignInfoMap &alignInfo);

/// Extracts byte alignment requirements from annotation marks and computes
/// aligned type.
///
/// Analyzes all StrideAlignDims annotations on a value to determine byte
/// alignment requirements for each dimension. When multiple annotations specify
/// alignment for the same dimension, computes the LCM to satisfy all
/// constraints. Returns a MemRefType with sizes aligned to these requirements.
///
/// @param value The value to analyze for alignment annotations
/// @return MemRefType with dimensions aligned according to the annotations
Type getAnnotationMarkByteAlignment(Value value);

/// Create eltwise vv operation according to atomic kind.
std::optional<Operation *>
createEltwiseOpByAtomicKind(OpBuilder &builder, Location loc,
                            TypeRange resTypeRange, ValueRange src,
                            ValueRange dst, hivm::AtomicKind atomicKind);

/// Create castOP to specified element type.
mlir::hivm::VCastOp castTo(OpBuilder &builder, Location loc, Value src,
                           hivm::RoundModeAttr roundMode, Type targetElemType);

/// To retrieve real mmad perChannel bias from implicit broadcast and so on
Value extractMmadBiasFromPotentialUnitDimExpand(Value bias);

std::pair<bool, bool> analyzeCoreTypes(Block *block);

/// To judge whether yields of if have the same core type
bool needsSplit(scf::IfOp ifOp);

namespace util {
/// Returns if the reassociations are identity that each indices group only
/// contains a single dimension. e.g. `[[0], [1], [3]]` is indentity collapse.
bool isIdentityCollapse(ArrayRef<ReassociationIndices> reassociations);
bool isTransposeWithLastAxis(ArrayRef<int64_t> permutation);
SmallVector<int64_t> getTransposeAxes(ArrayRef<int64_t> permutation);
bool isTransposeAdjacentAxes(SmallVector<int64_t> transposeAxes);

/// Return the ConstantOp IntValue.
FailureOr<std::string> stringfyConstantIntOpValue(Value value);
} // namespace util

/// Creates alloc.op based on a buffer size and element type
Value allocExtraBuffer(Operation *op, const SmallVector<int64_t> &bufSize,
                       Type elemType);

bool isCopytoL1(Operation *op);
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H