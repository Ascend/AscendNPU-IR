//===------------------ Helper.cpp - HIVM implementation ------------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMVectorize.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <type_traits>
#include <utility>

#define DEBUG_TYPE "hivm-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir::hivm {
namespace {

template <typename OpT, typename = void>
struct HasArrayRefMultiDimReductionBuild : std::false_type {};

template <typename OpT>
struct HasArrayRefMultiDimReductionBuild<OpT, std::void_t<decltype(
    OpT::build(
        std::declval<OpBuilder &>(), std::declval<OperationState &>(),
        std::declval<vector::CombiningKind>(), std::declval<Value>(),
        std::declval<Value>(), std::declval<ArrayRef<int64_t>>()))>>
    : std::true_type {};

template <typename RewriterT,
          bool SupportsArrayRef =
              HasArrayRefMultiDimReductionBuild<
                  vector::MultiDimReductionOp>::value,
          std::enable_if_t<SupportsArrayRef, int> = 0>
static Value createMultiDimReductionOp(RewriterT &rewriter, Location loc,
                                       vector::CombiningKind combiningKind,
                                       Value source, Value acc,
                                       ArrayRef<int64_t> reduceDims) {
  return rewriter.template create<vector::MultiDimReductionOp>(
      loc, combiningKind, source, acc, reduceDims);
}

template <typename RewriterT,
          bool SupportsArrayRef =
              HasArrayRefMultiDimReductionBuild<
                  vector::MultiDimReductionOp>::value,
          std::enable_if_t<!SupportsArrayRef, int> = 0>
static Value createMultiDimReductionOp(RewriterT &rewriter, Location loc,
                                       vector::CombiningKind combiningKind,
                                       Value source, Value acc,
                                       ArrayRef<int64_t> reduceDims) {
  return rewriter.template create<vector::MultiDimReductionOp>(
      loc, combiningKind, source, acc, rewriter.getI64ArrayAttr(reduceDims));
}

} // namespace

LogicalResult VReduceOp::vectorize(RewriterBase &rewriter,
                                   ArrayRef<int64_t> vectorSizes) {
  if (failed(checkVectorizePreconditions(*this, vectorSizes)))
    return failure();

  Location loc = getLoc();
  Value src = getSrc();
  Value dst = getDstValue();

  Type elementType = getElementTypeOrSelf(src);
  VectorType vectorType = VectorType::get(vectorSizes, elementType);
  int64_t rank = (int64_t)vectorSizes.size();

  // Map HIVM reduction op to vector combining kind and arithmetic kind
  VectorArithKind arithKind;
  vector::CombiningKind combiningKind;
  auto reduceOpArith = getArithAttr();
  auto reduceOpAttr = reduceOpArith.getReduceOp();

  switch (reduceOpAttr) {
  case hivm::ReduceOperation::sum:
    arithKind = VectorArithKind::ADD;
    combiningKind = vector::CombiningKind::ADD;
    break;
  case hivm::ReduceOperation::prod:
    arithKind = VectorArithKind::MUL;
    combiningKind = vector::CombiningKind::MUL;
    break;
  case hivm::ReduceOperation::max:
    arithKind = VectorArithKind::MAX;
    if (isa<FloatType>(elementType)) {
      combiningKind = vector::CombiningKind::MAXIMUMF;
    } else {
      combiningKind = getUnsignedSrc() ? vector::CombiningKind::MAXUI
                                       : vector::CombiningKind::MAXSI;
    }
    break;
  case hivm::ReduceOperation::min:
    arithKind = VectorArithKind::MIN;
    if (isa<FloatType>(elementType)) {
      combiningKind = vector::CombiningKind::MINIMUMF;
    } else {
      combiningKind = getUnsignedSrc() ? vector::CombiningKind::MINUI
                                       : vector::CombiningKind::MINSI;
    }
    break;
  default:
    return failure();
  }

  // Create base indices (all zeros)
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(rank, zero);

  // Get dimension sizes for creating mask
  SmallVector<Value> dimSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (isa<TensorType>(src.getType())) {
      dimSizes.push_back(rewriter.create<tensor::DimOp>(loc, src, i));
    } else {
      dimSizes.push_back(rewriter.create<memref::DimOp>(loc, src, i));
    }
  }

  Value mask = rewriter.create<vector::CreateMaskOp>(
      loc, vectorType.clone(rewriter.getI1Type()), dimSizes);

  // Get identity element for padding
  Value padding = getIdentityElement(rewriter, loc, elementType, arithKind);

  SmallVector<bool> inBounds(rank, true);
  AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
  Value vectorData = rewriter.create<vector::TransferReadOp>(
      loc, vectorType, src, indices, identityMap, padding, mask,
      rewriter.getBoolArrayAttr(inBounds));

  // Get reduction dimensions
  SmallVector<int64_t> reduceDims(getReduceDims());
  if (reduceDims.empty() ||
      llvm::any_of(reduceDims, [&](int64_t dim) {
        return dim < 0 || dim >= rank;
      }))
    return failure();

  // Compute the result vector shape (reduced dims removed)
  SmallVector<int64_t> reducedShape;
  SmallVector<int64_t> outputShape;
  SmallVector<bool> isReducedDim(rank, false);
  for (int64_t dim : reduceDims) {
    isReducedDim[dim] = true;
  }
  for (int64_t i = 0; i < rank; ++i) {
    if (isReducedDim[i]) {
      outputShape.push_back(1); // Reduced dims become 1
    } else {
      reducedShape.push_back(vectorSizes[i]); // Non-reduced dims kept
      outputShape.push_back(vectorSizes[i]);
    }
  }

  // Create vector types
  VectorType reducedVectorType = VectorType::get(reducedShape, elementType);
  VectorType outputVectorType = VectorType::get(outputShape, elementType);

  // Read initial accumulator from dst
  Value initAccum = rewriter.create<vector::TransferReadOp>(
      loc, outputVectorType, dst, indices, identityMap, padding,
      /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));

  // Reshape accumulator to match multi_reduction output shape (remove size-1
  // dims)
  Value vectorOut =
      rewriter.create<vector::ShapeCastOp>(loc, reducedVectorType, initAccum);

  // Perform multi-dimensional reduction with accumulator
  Value reduced = createMultiDimReductionOp(rewriter, loc, combiningKind,
                                            vectorData, vectorOut, reduceDims);

  // Reshape result back to have size-1 dimensions
  Value finalResult =
      rewriter.create<vector::ShapeCastOp>(loc, outputVectorType, reduced);

  // Write into the DPS init; for tensor semantics replace the op result.
  Value destOperand = getDstValue();
  auto writeOp = rewriter.create<vector::TransferWriteOp>(
      loc, TypeRange(destOperand.getType()), finalResult, destOperand,
      indices, identityMap, /*mask=*/Value(),
      rewriter.getBoolArrayAttr(inBounds));
  if (getNumResults() > 0) {
    rewriter.replaceOp(*this, writeOp.getResult());
  } else {
    rewriter.eraseOp(*this);
  }

  return success();
}
} // namespace mlir::hivm
