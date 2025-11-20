//===- Loops.cpp - HIVM Structure Op Interface Loop-related impl. ---------===//
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
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {

constexpr size_t kDimTwo = 2;
constexpr size_t kDimFour = 4;

int64_t getRankFromShapedTypeValue(Value val) {
  return cast<ShapedType>(val.getType()).getRank();
}

template <typename HIVMOP>
SmallVector<hivm::IteratorType> getCumOpIteratorTypesArray(HIVMOP op) {
  int64_t rank = getRankFromShapedTypeValue(op.getDst());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  for (int64_t cumDim : op.getCumDims())
    iteratorTypes[cumDim] = hivm::IteratorType::kCumulative;
  return iteratorTypes;
}

SmallVector<hivm::IteratorType> getIteratorTypesArrayForGlobalMatmulOps() {
  // Currently only support ND layout.
  // For ND layout, assuming axes are (M, N, K).
  return SmallVector<hivm::IteratorType>{hivm::IteratorType::kParallel,
                                         hivm::IteratorType::kParallel,
                                         hivm::IteratorType::kReduction};
}

} // namespace

#define ENABLE_NO_DEFAULT_GET_ITERATOR_TYPES_ARRAY(OP_NAME)                    \
  SmallVector<hivm::IteratorType> OP_NAME::getIteratorTypesArray() {           \
    llvm_unreachable("get iterator not implemented");                          \
  }

ENABLE_NO_DEFAULT_GET_ITERATOR_TYPES_ARRAY(NZ2NDOp)
ENABLE_NO_DEFAULT_GET_ITERATOR_TYPES_ARRAY(ND2NZOp)
#undef ENABLE_NO_DEFAULT_GET_ITERATOR_TYPES_ARRAY

#define ENABLE_COMMON_INDEXING_MAPS(OP_NAME)                                   \
  ArrayAttr OP_NAME::getIndexingMaps() {                                       \
    return mlir::hivm::detail::getIndexingMapsImpl(*this);                     \
  }

ENABLE_COMMON_INDEXING_MAPS(LoadOp)
ENABLE_COMMON_INDEXING_MAPS(StoreOp)
ENABLE_COMMON_INDEXING_MAPS(CopyOp)
#undef ENABLE_COMMON_INDEXING_MAPS

//===----------------------------------------------------------------------===//
// VArangeOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VArangeOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  // Note: the arange dim cannot be merged, use opaque to avoid
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kOpaque);
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VBrcOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VBrcOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  for (int64_t broadcastDim : getBroadcastDims())
    iteratorTypes[broadcastDim] = hivm::IteratorType::kBroadcast;
  return iteratorTypes;
}

ArrayAttr VBrcOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getRankFromShapedTypeValue(getDpsInits()[0]);
  SmallVector<AffineExpr, 4> broadcastExprs;
  if (isa<ShapedType>(getDpsInputs()[0].getType())) {
    DenseSet<int64_t> broadcastDims(getBroadcastDims().begin(),
                                    getBroadcastDims().end());
    for (int i = 0; i < rank; i++) {
      if (broadcastDims.contains(i)) {
        broadcastExprs.push_back(builder.getAffineConstantExpr(0));
      } else {
        broadcastExprs.push_back(builder.getAffineDimExpr(i));
      }
    }
  }
  AffineMap broadcastMap =
      AffineMap::get(rank, 0, broadcastExprs, builder.getContext());
  AffineMap outputMap = builder.getMultiDimIdentityMap(rank);
  return builder.getAffineMapArrayAttr({broadcastMap, outputMap});
}

//===----------------------------------------------------------------------===//
// VConcatOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VConcatOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  for (int64_t i = 0; i < rank; i++) {
    if (static_cast<size_t>(i) == getDim()) {
      iteratorTypes[i] = hivm::IteratorType::kConcat;
    }
  }
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VConcatOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VCumprodOp::getIteratorTypesArray() {
  return getCumOpIteratorTypesArray(*this);
}

SmallVector<hivm::IteratorType> VCumsumOp::getIteratorTypesArray() {
  return getCumOpIteratorTypesArray(*this);
}

//===----------------------------------------------------------------------===//
// VDeinterleaveOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VDeinterleaveOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getSrc());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  iteratorTypes.back() = hivm::IteratorType::kDeinterleave;
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VGatherOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VGatherOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  // hivm gather op only support last gather axis now
  iteratorTypes.back() = hivm::IteratorType::kGather;
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VFlip
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VFlipOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  iteratorTypes[rank - 1] = hivm::IteratorType::kInverse;
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VInterleaveOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VInterleaveOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  iteratorTypes[rank - 1] = hivm::IteratorType::kInterleave;
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VMulExtendedOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VMulextendedOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst()[0]);
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VPadOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VPadOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  // conservative default choice: kPad
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kPad);
  // `getConstantIntValue` is declared in
  // "mlir/Dialect/Utils/StaticValueUtils.h" If a dimension's low and high
  // padding lengths are both (statically known) zeros, then optimize it to
  // hivm::IteratorType::kParallel.
  SmallVector<OpFoldResult> lowPadLengths = getMixedLowPad();
  SmallVector<OpFoldResult> highPadLengths = getMixedHighPad();
  for (int i = 0; i < rank; i++) {
    if (getConstantIntValue(lowPadLengths[i]) == static_cast<int64_t>(0) &&
        getConstantIntValue(highPadLengths[i]) == static_cast<int64_t>(0)) {
      iteratorTypes[i] = hivm::IteratorType::kParallel;
    }
  }
  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VReduceOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDstValue());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  for (int64_t reductionDim : getReduceDims())
    iteratorTypes[reductionDim] = hivm::IteratorType::kReduction;
  return iteratorTypes;
}

ArrayAttr VReduceOp::getIndexingMaps() {
  Builder builder(getContext());
  int64_t rank = getRankFromShapedTypeValue(getDpsInputs()[0]);
  AffineMap inputMap = builder.getMultiDimIdentityMap(rank);
  SmallVector<AffineExpr, 4> outputExprs;
  DenseSet<int64_t> reduceDims(getReduceDims().begin(), getReduceDims().end());
  for (int i = 0; i < rank; i++) {
    if (reduceDims.contains(i)) {
      outputExprs.push_back(builder.getAffineConstantExpr(0));
    } else {
      outputExprs.push_back(builder.getAffineDimExpr(i));
    }
  }
  AffineMap outputMap =
      AffineMap::get(rank, 0, outputExprs, builder.getContext());
  SmallVector<AffineMap> maps(getIndices() ? 2 : 1, inputMap);
  maps.append(getNumDpsInits() - (getTempBuffer() ? 1 : 0), outputMap);
  return builder.getAffineMapArrayAttr(maps);
}

//===----------------------------------------------------------------------===//
// VTransposeOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> VTransposeOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  auto iteratorTypes =
      SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
  for (auto [dimIdx, permutedIdx] : llvm::enumerate(getPermutation())) {
    if (static_cast<int64_t>(dimIdx) != permutedIdx)
      iteratorTypes[dimIdx] = hivm::IteratorType::kTranspose;
  }
  return iteratorTypes;
}

LogicalResult
VTransposeOp::setIteratorTypesArray(const IteratorType iteratorType,
                                    const DenseI64ArrayAttr &arrayAttr) {
  assert(iteratorType == hivm::IteratorType::kTranspose);
  getOperation()->setAttr(stringifyIteratorType(iteratorType), arrayAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> CopyOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  return SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> LoadOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  return SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
}

//===----------------------------------------------------------------------===//
// FixpipeOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> FixpipeOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  return SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
}

ArrayAttr FixpipeOp::getIndexingMaps() {
  // If src/dst rank is 2, the indexing map is parallel.
  int64_t srcRank = getRankFromShapedTypeValue(getSrc());
  int64_t dstRank = getRankFromShapedTypeValue(getDst());
  // TODO: Handle ND2NZ indexing map.
  if (dstRank != 2 || srcRank != dstRank)
    llvm_unreachable("Not implemented");

  SmallVector<AffineMap> affineMaps(
      getNumDpsInputs(),
      AffineMap::getMultiDimIdentityMap(srcRank, getContext()));
  AffineMap resultMap =
      AffineMap::getMultiDimIdentityMap(dstRank, getContext());
  for (int64_t i = 0, e = getNumDpsInits(); i < e; ++i)
    affineMaps.push_back(resultMap);

  return Builder(getContext()).getAffineMapArrayAttr(affineMaps);
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> StoreOp::getIteratorTypesArray() {
  int64_t rank = getRankFromShapedTypeValue(getDst());
  return SmallVector<hivm::IteratorType>(rank, hivm::IteratorType::kParallel);
}

//===----------------------------------------------------------------------===//
// BatchMmadL1Op
//===----------------------------------------------------------------------===//

SmallVector<hivm::IteratorType> BatchMmadL1Op::getIteratorTypesArray() {
  auto rank = static_cast<size_t>(getRankFromShapedTypeValue(getC()));
  // Deduct one which means batch dimension, and then consider left dimension
  switch (rank - 1) {
  case kDimTwo:
    // Assume that current axes are (Batch, M, N, K)
    return SmallVector<hivm::IteratorType>{
        hivm::IteratorType::kParallel, hivm::IteratorType::kParallel,
        hivm::IteratorType::kParallel, hivm::IteratorType::kReduction};
  case kDimFour:
    // Assume that current axes are (Batch, M1, N1, N0, M0, K)
    return SmallVector<hivm::IteratorType>{
        hivm::IteratorType::kParallel, hivm::IteratorType::kParallel,
        hivm::IteratorType::kParallel, hivm::IteratorType::kParallel,
        hivm::IteratorType::kParallel, hivm::IteratorType::kReduction};
  default:
    break;
  }
  // Ops, unknown rank
  return {hivm::IteratorType::kOpaque};
}

//===----------------------------------------------------------------------===//
// GlobalMatmulOp
//===----------------------------------------------------------------------===//

#define ENABLE_GLOBAL_MATMUL_GET_ITERATOR_TYPES_ARRAY(OP_NAME)                 \
  SmallVector<hivm::IteratorType> OP_NAME::getIteratorTypesArray() {           \
    return getIteratorTypesArrayForGlobalMatmulOps();                          \
  }

ENABLE_GLOBAL_MATMUL_GET_ITERATOR_TYPES_ARRAY(MatmulOp)
ENABLE_GLOBAL_MATMUL_GET_ITERATOR_TYPES_ARRAY(MixGroupMatmulOp)
ENABLE_GLOBAL_MATMUL_GET_ITERATOR_TYPES_ARRAY(MixMatmulOp)
#undef ENABLE_GLOBAL_MATMUL_GET_ITERATOR_TYPES_ARRAY

//===----------------------------------------------------------------------===//
// MmadL1Op
//===----------------------------------------------------------------------===//

ArrayAttr MmadL1Op::getIndexingMaps() {
  auto cLayoutAttr = getOperandCLayout();
  if (failed(cLayoutAttr) ||
      cLayoutAttr.value().getDataLayout() != DataLayout::DOTC_ND) {
    llvm_unreachable("Unknown/unsupported layout");
    return ArrayAttr();
  }

  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  // Initialize all with scalar map
  SmallVector<AffineMap> indexingMaps(getNumOperands(), scalarMap);
  // Indexing map of C is (M,   N,  K) -> (M,  N)
  //                      (d0, d1, d2) -> (d0, d1)
  AffineMap cMap = parseAffineMap("(d0, d1, d2) -> (d0, d1)", context);
  indexingMaps[getCMutable().getOperandNumber()] = cMap;

  // Indexing map of A is (M, N, K) -> (M, K) or (K, M)
  AffineMap aMap = getATranspose().has_value()
                       ? parseAffineMap("(d0, d1, d2) -> (d2, d0)", context)
                       : parseAffineMap("(d0, d1, d2) -> (d0, d2)", context);
  indexingMaps[getAMutable().getOperandNumber()] = aMap;

  // Indexing map of B is (M, N, K) -> (K, N) or (N, K)
  AffineMap bMap = getBTranspose().has_value()
                       ? parseAffineMap("(d0, d1, d2) -> (d1, d2)", context)
                       : parseAffineMap("(d0, d1, d2) -> (d2, d1)", context);
  indexingMaps[getBMutable().getOperandNumber()] = bMap;
  return Builder(context).getAffineMapArrayAttr(indexingMaps);
}

SmallVector<hivm::IteratorType> MmadL1Op::getIteratorTypesArray() {
  auto cLayoutAttr = getOperandCLayout();
  if (failed(cLayoutAttr))
    return {hivm::IteratorType::kOpaque};

  if (cLayoutAttr.value().getDataLayout() == DataLayout::DOTC_ND) {
    // For ND layout, assuming axes are (M, N, K).
    return SmallVector<hivm::IteratorType>{hivm::IteratorType::kParallel,
                                           hivm::IteratorType::kParallel,
                                           hivm::IteratorType::kReduction};
  }
  if (cLayoutAttr.value().getDataLayout() == DataLayout::zN) {
    // For zN layout, assuming axes are (N1, M1, M0, N0, K).
    return SmallVector<hivm::IteratorType>{
        hivm::IteratorType::kParallel, hivm::IteratorType::kParallel,
        hivm::IteratorType::kParallel, hivm::IteratorType::kParallel,
        hivm::IteratorType::kReduction};
  }

  if (cLayoutAttr.value().getDataLayout() == DataLayout::nZ) {
    // For nZ layout, assuming axes are (M1, N1, N0, M0, K).
    return SmallVector<hivm::IteratorType>{
        hivm::IteratorType::kParallel, hivm::IteratorType::kParallel,
        hivm::IteratorType::kParallel, hivm::IteratorType::kParallel,
        hivm::IteratorType::kReduction};
  }
  return {hivm::IteratorType::kOpaque};
}
