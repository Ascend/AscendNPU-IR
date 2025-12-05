//===- HIVMTraits.cpp - HIVM dialect traits implementation ----------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-traits"

using namespace mlir;
using namespace hivm;

// Returns true if input type is a shaped type with known rank.
inline bool hasRank(const Type &type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.hasRank();
  return false;
}

// Returns rank of shaped type with known rank.
inline int64_t getRank(const Type &type) {
  return cast<ShapedType>(type).getRank();
}

// Returns stride of shaped memref type.
inline SmallVector<int64_t> getStride(const Type &type) {
  auto memrefType = cast<MemRefType>(type);
// TODO : release different version of ascendnpu ir and remove the macro
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
  auto [strides, offset] = getStridesAndOffset(cast<MemRefType>(type));
#else
  auto [strides, offset] = memrefType.getStridesAndOffset();
#endif
  return strides;
}

// Returns shape of shaped type with known rank.
inline SmallVector<int64_t> getShape(const Type &type) {
  return SmallVector<int64_t>(cast<ShapedType>(type).getShape());
}

// Returns true if none of the stride is dynamic.
inline bool isStrideFullyStatic(const SmallVector<int64_t> &strides) {
  return llvm::all_of(strides,
                      [](int64_t s) { return s != ShapedType::kDynamic; });
}

LogicalResult OpTrait::impl::verifyElementwiseNaryOpTrait(Operation *op,
                                                          int numOperands) {
  auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
  if (!hivmOp)
    return op->emitOpError() << "ElementwiseNaryOpTrait expect op to follow "
                                "HIVMStructuredOp";

  auto inputs = hivmOp.getDpsInputs();
  size_t extraBufferSize = 0;
  if (auto extraBufferOp = dyn_cast<ExtraBufferOpInterface>(op))
    extraBufferSize = extraBufferOp.getExtraBuffers().size();

  if (inputs.size() - extraBufferSize != static_cast<unsigned>(numOperands))
    return op->emitOpError() << "elementwise op expected " << numOperands
                             << " inputs, but found " << inputs.size();

  SmallVector<Type, 8> types =
      hivmOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
  types.append(llvm::to_vector<1>(op->getResultTypes()));
  auto rankedTypes = llvm::make_filter_range(types, hasRank);
  if (std::distance(rankedTypes.begin(), rankedTypes.end()) < 1)
    return op->emitOpError() << "at least one operands should be ranked";

  if (hivmOp.hasPureTensorSemantics())
    return success();

  // If Op is broadcastable or transposable on-the-fly, verify shape later on.
  if ((hivmOp.isInlineBroadcastable() &&
       !op->getAttrOfType<DenseI64ArrayAttr>("broadcast").empty()) ||
      (hivmOp.isInlineTransposable() &&
       !op->getAttrOfType<DenseI64ArrayAttr>("transpose").empty()))
    return success();

  if (failed(verifyCompatibleShapes(llvm::to_vector(rankedTypes))))
    return op->emitOpError() << "operands' shape are inconsistent";

  return success();
}

LogicalResult
OpTrait::impl::verifyHIVMOpSameOperandsAndResultRank(Operation *op) {
  auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
  if (!hivmOp)
    return op->emitOpError() << "HIVMOpSameOperandsAndResultRankTrait expect "
                                "op to follow HIVMStructuredOp";

  SmallVector<Type, 8> types =
      hivmOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
  types.append(llvm::to_vector<1>(op->getResultTypes()));
  auto rankedTypes = llvm::make_filter_range(types, hasRank);
  // If all operands and results are unranked, then no further verification.
  if (rankedTypes.empty())
    return success();

  // delegate function that returns rank of shaped type with known rank
  auto getRank = [](const Type type) {
    return cast<ShapedType>(type).getRank();
  };

  auto rank = getRank(*rankedTypes.begin());

  for (const auto type : rankedTypes) {
    if (rank != getRank(type)) {
      return op->emitOpError("operands must have same rank");
    }
  }

  return success();
}

LogicalResult OpTrait::impl::verifyBroadcastableOTF(Operation *op) {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);

  // (B1) Op has buffer semantic.
  if (dpsOp.hasPureTensorSemantics())
    return success();

  auto broadcastDimsAttr = dpsOp->getAttrOfType<DenseI64ArrayAttr>("broadcast");
  if (!broadcastDimsAttr)
    return op->emitOpError() << "BroadcastableOTF Trait expect ops to have"
                                "an DenseI64ArrayAttr named 'broadcast'";

  auto broadcastDims = broadcastDimsAttr.asArrayRef();
  if (broadcastDims.empty())
    return success();

  auto outputs = dpsOp.getDpsInits();
  if (outputs.size() != 1)
    return op->emitOpError()
           << "BroadcastableOTF Op expect 1 output, but found "
           << outputs.size();

  // For DPS Op with buffer semantic, the init operands are ranked memrefs.
  auto outputType = cast<ShapedType>(outputs.front().getType());
  // Ranked operands have the same rank because we've verified
  // HIVMOpSameOperandsAndResultRank trait.
  auto opRank = outputType.getRank();
  // (B3) For all `d` in `broadcast`, `0 <= d < rank(dst)`.
  if (!llvm::all_of(broadcastDims,
                    [&](const int64_t d) { return (d >= 0) && (d < opRank); }))
    return op->emitOpError() << "broadcast dim exceeds op's rank";

  auto hivmOp = cast<HIVMStructuredOp>(op);
  SmallVector<Value> inputs;
  // Not to check the temp buffer
  for (OpOperand *opOperand : hivmOp.getHIVMInputOperands(false))
    inputs.push_back(opOperand->get());

  auto inputTypes = llvm::map_to_vector<8>(
      inputs, [&](const Value &input) { return input.getType(); });
  auto rankedTypes = llvm::make_filter_range(inputTypes, hasRank);
  auto shapes = llvm::map_to_vector<8>(
      rankedTypes, [&](const Type &t) { return cast<ShapedType>(t); });

  auto broadcastDimsSet = llvm::SmallSet<int64_t, 10>();
  broadcastDimsSet.insert(broadcastDims.begin(), broadcastDims.end());

  for (int64_t d = 0; d < opRank; ++d) {
    // Relationship between rank, shape and index:
    //    Rank  = R
    //    Shape = [dim0, dim1, ..., dimR]
    //    Index = [0,    1,    ..., R-1 ]
    auto dims = llvm::map_to_vector(
        shapes, [&](const ShapedType &shape) { return shape.getDimSize(d); });
    if (broadcastDimsSet.contains(d)) {
      // (B4) If `d` is in `broadcast`,
      //      `(dim(src1, d) = 1 || dim(src1, d) = dim(dst, d)) & ... &
      //       (dim(srcN, d) = 1 || dim(srcN, d) = dim(dst, d)) = 1`.
      auto dst_dim_d = outputType.getDimSize(d);
      if (!llvm::all_of(dims, [&](int64_t d) {
            return d == 1 || d == dst_dim_d || d == ShapedType::kDynamic;
          }))
        return op->emitOpError() << "input operand's broadcast dim is not 1";
    } else {
      // (B5) If `d` is not in `broadcast`,
      //      `dim(src1, d) & ... & dim(srcN, d) = dim(dst, d)`.
      dims.push_back(outputType.getDimSize(d));
      if (verifyCompatibleDims(dims).failed())
        return op->emitOpError() << "input operand's non-broadcast dim does "
                                    "not match with output";
    }
  }
  return success();
}

LogicalResult OpTrait::impl::verifyTransposableOTF(Operation *op) {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);

  // (T1) Op has buffer semantic.
  if (dpsOp.hasPureTensorSemantics())
    return success();

  auto transposeDimsAttr = dpsOp->getAttrOfType<DenseI64ArrayAttr>("transpose");
  if (!transposeDimsAttr)
    return op->emitOpError() << "TransposableOTF Trait expect ops to have"
                                "an DenseI64ArrayAttr named 'transpose'";

  auto transposeDims = transposeDimsAttr.asArrayRef();
  if (transposeDims.empty())
    return success();

  auto outputs = dpsOp.getDpsInits();
  if (outputs.size() != 1)
    return op->emitOpError() << "TransposableOTF Op expect 1 output, but found "
                             << outputs.size();

  // For DPS Op with buffer semantic, the init operands are ranked memrefs.
  auto outputType = cast<ShapedType>(outputs.front().getType());
  // Ranked operands have the same rank because we've verified
  // HIVMOpSameOperandsAndResultRank trait.
  auto opRank = outputType.getRank();

  // (T2) `transpose` is a permutation of `range(rank(dst))`.
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, opRank));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           transposeDims.begin(), transposeDims.end())) {
    return op->emitOpError()
           << "expects 'transpose' to be a permutation, found "
           << transposeDims;
  }

  // (T3) `transpose[rank(dst) - 1] = rank(dst) - 1`.
  if (transposeDims.back() != opRank - 1)
    return op->emitOpError() << "transpose dim is the last dimension";

  auto inputs = dpsOp.getDpsInputs();
  auto inputTypes = llvm::map_to_vector<8>(
      inputs, [&](const Value &input) { return input.getType(); });
  auto rankedTypes = llvm::make_filter_range(inputTypes, hasRank);
  auto shapes = llvm::map_to_vector<8>(
      rankedTypes, [&](const Type &t) { return cast<ShapedType>(t); });

  // (T4) `dim(dst, d) = dim(src1, transpose[d]) = ... = dim(srcN,
  // transpose[d]).` check if each dims are equal or dynamic for the sources and
  // transposed dim
  for (int64_t d = 0; d < opRank; ++d) {
    auto idx = opRank - 1 - d;
    auto transposeIdx = transposeDims[idx];
    auto dims = llvm::map_to_vector(shapes, [&](const ShapedType &shape) {
      return shape.getDimSize(transposeIdx);
    });
    dims.push_back(outputType.getDimSize(idx));
    if (verifyCompatibleDims(llvm::to_vector(dims)).failed()) {
      LLVM_DEBUG(for (const auto dim : dims) { llvm::dbgs() << dim << " "; });
      LLVM_DEBUG(llvm::dbgs() << "\n";);
      return op->emitOpError() << "failed to verify transpose behavior at "
                               << idx << " " << transposeIdx;
    }
  }
  return success();
}

LogicalResult OpTrait::impl::verifyVectorOnlyTrait(Operation *op, int idx) {
  auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
  if (!hivmOp)
    return op->emitOpError() << "ElementwiseNaryOpTrait expect op to follow "
                                "HIVMStructuredOp";
  if (!(hivmOp->getNumOperands() > static_cast<unsigned>(idx))) {
    return hivmOp.emitOpError()
           << "failed to verify that operand at index " << idx << " exists";
  }
  if (!isa<ShapedType>(hivmOp->getOperand(idx).getType())) {
    return hivmOp.emitOpError() << "failed to verify that operand at index "
                                << idx << " is vector-only";
  }
  return success();
}

LogicalResult OpTrait::impl::verifyScalarOnlyHWTrait(Operation *op, int idx) {
  auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
  if (!hivmOp)
    return op->emitOpError() << "ElementwiseNaryOpTrait expect op to follow "
                                "HIVMStructuredOp";
  if (!(hivmOp->getNumOperands() > static_cast<unsigned>(idx))) {
    return hivmOp.emitOpError()
           << "failed to verify that operand at index " << idx << " exists";
  }
  return success();
}

bool hivm::detail::isElemwiseNaryOpImpl(Operation *op) {
  bool isUnary = op->hasTrait<OpTrait::ElementwiseNaryOpTrait<1>::Impl>();
  bool isBinary = op->hasTrait<OpTrait::ElementwiseNaryOpTrait<2>::Impl>();
  bool isTernaryOp = op->hasTrait<OpTrait::ElementwiseNaryOpTrait<3>::Impl>();
  return isUnary || isBinary || isTernaryOp;
}

LogicalResult hivm::detail::setIteratorTypesArrayForElemwiseOp(
    Operation *op, const IteratorType &iteratorType,
    const DenseI64ArrayAttr &arrayAttr) {
  auto hivmOp = cast<HIVMStructuredOp>(op);
  if (!hivmOp.isInlineBroadcastable() && !hivmOp.isInlineTransposable())
    return failure();
  assert(iteratorType == IteratorType::kTranspose ||
         iteratorType == IteratorType::kBroadcast ||
         iteratorType == IteratorType::kParallel);
  hivmOp->setAttr(stringifyIteratorType(iteratorType), arrayAttr);
  return success();
}

SmallVector<IteratorType>
hivm::detail::getIteratorTypesArrayForElemwiseOp(Operation *op) {
  auto hivmOp = cast<HIVMStructuredOp>(op);
  auto shapedType =
      cast<ShapedType>(hivmOp.getDpsInitOperand(0)->get().getType());
  auto numLoops = shapedType.getRank();
  auto iteratorTypes =
      SmallVector<IteratorType>(numLoops, IteratorType::kParallel);

  if (!hivmOp.isInlineBroadcastable() && !hivmOp.isInlineTransposable())
    return iteratorTypes;

  // Broadcast inline dims.
  auto broadcastDimsAttr = hivmOp->getAttrOfType<DenseI64ArrayAttr>(
      stringifyIteratorType(IteratorType::kBroadcast));
  for (auto broadcastDim : broadcastDimsAttr.asArrayRef()) {
    iteratorTypes[broadcastDim] = IteratorType::kBroadcast;
  }

  // Transpose inline dims.
  auto transposeDimsAttr = hivmOp->getAttrOfType<DenseI64ArrayAttr>(
      stringifyIteratorType(IteratorType::kTranspose));
  for (auto [dim, permutedDim] :
       llvm::enumerate(transposeDimsAttr.asArrayRef())) {
    if (static_cast<int64_t>(dim) != permutedDim)
      iteratorTypes[permutedDim] = IteratorType::kTranspose;
  }
  return iteratorTypes;
}

template <typename SpecificOpType>
static ArrayRef<int64_t>
getArrayFromOp(Operation *op, IteratorType iteratorType,
               ArrayRef<int64_t> (SpecificOpType::*getter)()) {
  if (auto specificOp = dyn_cast<SpecificOpType>(op))
    return (specificOp.*getter)();

  auto hivmOp = cast<HIVMStructuredOp>(op);
  auto attr = hivmOp->getAttrOfType<DenseI64ArrayAttr>(
      stringifyIteratorType(iteratorType));
  if (!attr)
    return {};
  return attr.asArrayRef();
}

ArrayRef<int64_t> hivm::detail::getPermutationArray(Operation *op) {
  return getArrayFromOp<VTransposeOp>(op, IteratorType::kTranspose,
                                      &VTransposeOp::getPermutation);
}

ArrayRef<int64_t> hivm::detail::getBroadcastArray(Operation *op) {
  return getArrayFromOp<hivm::VBrcOp>(op, IteratorType::kBroadcast,
                                      &VBrcOp::getBroadcastDims);
}

SmallVector<int64_t>
hivm::detail::getInlinedBroadcastableAxes(const Operation *op,
                                          const OpOperand *opOperand) {
  auto hivmOp = cast<HIVMStructuredOp>(op);
  if (!hivmOp.isInlineBroadcastable())
    return {};
  auto opBrcIndexes =
      hivmOp->getAttrOfType<DenseI64ArrayAttr>(hivmOp.getBroadcastAttrString())
          .asArrayRef();
  if (opOperand == nullptr) {
    return SmallVector<int64_t>(opBrcIndexes);
  }
  auto operandType = dyn_cast<ShapedType>(opOperand->get().getType());
  if (!operandType)
    return {};
  llvm::SmallVector<int64_t> ret;
  for (auto idx : opBrcIndexes)
    if (operandType.getShape()[idx] == 1)
      ret.push_back(idx);
  return ret;
}
