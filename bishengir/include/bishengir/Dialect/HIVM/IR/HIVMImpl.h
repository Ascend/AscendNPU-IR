//===- HIVMImpl.h - HIVM implementation -----------------------------------===//
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
    return traceDefOp<OpType>(insertSliceOp.getSource(), isSingleChain);
  }
  return std::nullopt;
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
                           hivm::RoundModeAttr roundMode, Type targetElemType,
                           hivm::TypeFnAttr typeFnAttr = {});

/// To retrieve real mmad perChannel bias from implicit broadcast and so on
Value extractMmadBiasFromPotentialUnitDimExpand(Value bias);

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
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H