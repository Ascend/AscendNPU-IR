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

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace hivm {

/// Enum to control trace result matching behavior for multi-source cases
enum class TraceResultMode {
  // Default behaviour, insert every matched op to final result for all traced
  // sources, which means as long as one of the sources produces matched op,
  // final result is not empty.
  Default,
  // StrictSame consideres multi-source cases valid only if every traced source
  // produces the same matched op set; otherwise this function returns an empty
  // result.
  StrictSame,
  // Compared with Default, TypeMatch additionally requires the core type of
  // every traced source matches the core type of tracing type.
  // Note that currently we only check if sources matches non-vector core type,
  // cases may expand later.
  TypeMatch
};

using hivm::detail::queryCoreTypeHelper;

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
void traceDefOpsImpl(Value v,
                     bool isSingleChain,
                     TraceResultMode traceMode,
                     SmallVectorImpl<Operation *> &results,
                     SmallVectorImpl<Value> &recursionStack,
                     bool &traceFailed) {
  if (traceFailed || !v)
    return;

  if (isSingleChain && getUsersNum(v) != 1) {
    traceFailed = true;
    return;
  }

  // Currently we only check if current source is defined by vector while
  // tracing type is MmadL1 or BatchMmadL1. If so, the result treats invalid to
  // avoid missing matmul decompose.
  auto defOp = v.getDefiningOp();
  if (defOp && traceMode == TraceResultMode::TypeMatch) {
    auto core = queryCoreTypeHelper(defOp).value_or(TCoreType::CUBE_OR_VECTOR);
    if ((std::is_same_v<OpType, hivm::MmadL1Op> ||
         std::is_same_v<OpType, hivm::BatchMmadL1Op>) && (core ==
        TCoreType::VECTOR)) {
          traceFailed = true;
          return;  
    }
  }

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

  auto traceSource = [isSingleChain, traceMode, &results,
                      &recursionStack, &traceFailed](Value source) {
    traceDefOpsImpl<OpType>(source, isSingleChain, traceMode, results,
                            recursionStack, traceFailed);
  };

  auto traceSameTraceResultSources =
      [isSingleChain, traceMode, &results, &recursionStack,
       &traceFailed](ArrayRef<Value> sources) {
    if (sources.empty())
      return;

    SmallPtrSet<Operation *, 4> matchedResultSet;
    bool initialized = false;
    for (Value source : sources) {
      SmallVector<Operation *> branchResults;
      SmallPtrSet<Operation *, 4> branchResultSet;
      SmallVector<Value> branchRecursionStack(recursionStack.begin(),
                                              recursionStack.end());
      bool branchTraceFailed = false;
      traceDefOpsImpl<OpType>(
          source,
          /*isSingleChain=*/isSingleChain,
          /*traceMode=*/traceMode,
          branchResults,
          branchRecursionStack,
          branchTraceFailed);

      if (branchTraceFailed) {
        traceFailed = true;
        return;
      }

      for (Operation *op : branchResults)
        branchResultSet.insert(op);

      if (!initialized) {
        initialized = true;
        for (Operation *op : branchResultSet)
          matchedResultSet.insert(op);
        continue;
      }

      if (matchedResultSet.size() != branchResultSet.size()) {
        traceFailed = true;
        return;
      }
      for (Operation *op : branchResultSet) {
        if (!matchedResultSet.contains(op)) {
          traceFailed = true;
          return;
        }
      }
    }

    results.append(matchedResultSet.begin(), matchedResultSet.end());
  };

  if (Operation *definingOp = v.getDefiningOp<OpType>()) {
    results.push_back(definingOp);
    return;
  } else if (auto reshapeOp = v.getDefiningOp<tensor::ReshapeOp>()) {
    traceSource(reshapeOp.getSource());
  } else if (auto memrefCollapseShape = v.getDefiningOp<memref::CollapseShapeOp>()) {
    traceSource(memrefCollapseShape.getViewSource());
  } else if (auto tensorCollapseShape = v.getDefiningOp<tensor::CollapseShapeOp>()) {
    traceSource(tensorCollapseShape.getSrc());
  } else if (auto subViewOp = v.getDefiningOp<memref::SubViewOp>()) {
    traceSource(subViewOp.getViewSource());
  } else if (auto toMemrefOp = v.getDefiningOp<bufferization::ToMemrefOp>()) {
    traceSource(toMemrefOp.getOperand());
  } else if (auto toTensorOp = v.getDefiningOp<bufferization::ToTensorOp>()) {
    traceSource(toTensorOp.getOperand());
  } else if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
    traceSource(viewOp.getViewSource());
  } else if (auto reshapeOp = v.getDefiningOp<memref::ReshapeOp>()) {
    traceSource(reshapeOp.getViewSource());
  } else if (auto expandShapeOp = v.getDefiningOp<memref::ExpandShapeOp>()) {
    traceSource(expandShapeOp.getViewSource());
  } else if (auto tensorExpandShapeOp = v.getDefiningOp<tensor::ExpandShapeOp>()) {
    traceSource(tensorExpandShapeOp->getOperand(0));
  } else if (auto extractStridedMetadataOp = v.getDefiningOp<memref::ExtractStridedMetadataOp>()) {
    traceSource(extractStridedMetadataOp.getViewSource());
  } else if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
    traceSource(castOp.getViewSource());
  } else if (auto reinterpretCastOp = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    traceSource(reinterpretCastOp.getViewSource());
  } else if (auto blockArg = dyn_cast_if_present<BlockArgument>(v)) {
    if (auto scfForOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (blockArg.getOwner() == scfForOp.getBody()) {
        unsigned numInduction = scfForOp.getNumInductionVars();
        unsigned argNo = blockArg.getArgNumber();
        if (argNo < numInduction)
          return;

        unsigned iterIdx = argNo - numInduction;
        SmallVector<Value, 2> sources;
        if (Operation *terminator = scfForOp.getBody()->getTerminator()) {
          if (auto yieldOp = dyn_cast_if_present<scf::YieldOp>(terminator)) {
            Value yielded = yieldOp.getOperand(iterIdx);
            // avoid tracing to the same yield operand again
            if (yielded != v)
              sources.push_back(yielded);
          }
        }
        if (OpOperand *iterInit = scfForOp.getTiedLoopInit(blockArg)) {
          if (Value initValue = iterInit->get())
            sources.push_back(initValue);
        }

        if (traceMode == TraceResultMode::StrictSame) {
          traceSameTraceResultSources(sources);
          return;
        }

        for (Value source : sources)
          traceSource(source);
      }
    }
  } else if (auto forOp = v.getDefiningOp<scf::ForOp>()) {
    const unsigned index = cast<OpResult>(v).getResultNumber();
    Value yieldedValue = forOp.getYieldedValues()[index];
    traceSource(yieldedValue);
  } else if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
    const unsigned index = cast<OpResult>(v).getResultNumber();
    SmallVector<Value, 2> sources;
    if (!ifOp.getThenRegion().empty()) {
      Block &thenBlock = ifOp.getThenRegion().front();
      sources.push_back(thenBlock.getTerminator()->getOperand(index));
    }
    if (!ifOp.getElseRegion().empty()) {
      Block &elseBlock = ifOp.getElseRegion().front();
      sources.push_back(elseBlock.getTerminator()->getOperand(index));
    }
    if (traceMode == TraceResultMode::StrictSame) {
      traceSameTraceResultSources(sources);
      return;
    }
    for (Value source : sources)
      traceSource(source);
  } else if (auto extractSliceOp = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    traceSource(extractSliceOp.getSource());
  } else if (auto insertSliceOp = v.getDefiningOp<tensor::InsertSliceOp>()) {
    SmallVector<Value, 2> sources{insertSliceOp.getSource(),
                                  insertSliceOp.getDest()};
    if (traceMode == TraceResultMode::StrictSame) {
      traceSameTraceResultSources(sources);
      return;
    }
    for (Value source : sources)
      traceSource(source);
  } else if (auto memSpaceCastOp = v.getDefiningOp<memref::MemorySpaceCastOp>()) {
    traceSource(memSpaceCastOp.getSource());
  }
}

/// to trace ops in traceDefOp way and collect all matched ops
///
/// Compared with `traceDefOp`, this function additionally traces both branches
/// of `scf.if`, both yielded values and init operands for `scf.for`, and both
/// source and dest for `tensor.insert_slice`.
///
/// When `isSingleChain` is true, every recursively traced value is required to
/// stay on a single-use chain. If any traced path violates this requirement,
/// the whole trace is treated as failed and this function returns an empty
/// result.
///
/// 'traceMode' controls trace result matching behaviour under multi-source
/// cases(eg. result of scf.if, argument of scf.for ...). We require at
/// least of one the source produces valid matched op. If traceMode is not
/// Default, all paths will be checked jiontly, details can be referred in
/// TraceResultMode's description.
template <typename OpType>
SmallVector<Operation *>
traceDefOps(Value v, bool isSingleChain = false,
            TraceResultMode traceMode = TraceResultMode::Default) {
  SmallVector<Operation *> results;
  SmallVector<Value> recursionStack;
  bool traceFailed = false;
  traceDefOpsImpl<OpType>(v, isSingleChain, traceMode, results,
                          recursionStack, traceFailed);
  if (traceFailed)
    results.clear();
  return results;
}

/// Returns whether the mmadlike operand traces back to the same mmad-like op set through a single chain.
template <typename MmadLikeOpType>
typename std::enable_if<std::is_same_v<MmadLikeOpType, hivm::MmadL1Op> ||
                            std::is_same_v<MmadLikeOpType, hivm::BatchMmadL1Op>,
                        bool>::type
isSingleChainMmadToMmad(MmadLikeOpType op) {
  auto maybeMmadLikeOps =
      traceDefOps<MmadLikeOpType>(op.getC(),
                                  /*isSingleChain=*/true,
                                  /*traceMode=*/TraceResultMode::TypeMatch);
  return !maybeMmadLikeOps.empty();
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

bool hasOnlyIfRegionOperations(Block *block);

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