//===-------------- VectorizeTranspose.cpp - HIVM implementation ----------===//
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

namespace mlir::hivm {

LogicalResult VTransposeOp::vectorize(RewriterBase &rewriter,
                                      ArrayRef<int64_t> vectorSizes) {
  if (failed(checkVectorizePreconditions(*this, vectorSizes)) ||
      failed(checkShapedInputs(*this, vectorSizes)))
    return failure();

  ArrayRef<int64_t> permutation = getPermutation();
  int64_t rank = static_cast<int64_t>(vectorSizes.size());
  if (permutation.empty() || static_cast<int64_t>(permutation.size()) != rank)
    return failure();

  // Ensure permutation is valid.
  AffineMap readMap =
      AffineMap::getPermutationMap(permutation, rewriter.getContext());
  if (!readMap.isPermutation())
    return failure();

  Location loc = getLoc();
  Value dst = getDst();
  auto srcType = cast<ShapedType>(getSrc().getType());

  Value mask = createShapeMask(rewriter, loc, dst, vectorSizes);

  // Transfer masks use the source orientation.
  SmallVector<int64_t> readMaskSizes(rank);
  for (int64_t i = 0; i < rank; ++i)
    readMaskSizes[permutation[i]] = vectorSizes[i];
  Value readMask = createShapeMask(rewriter, loc, getSrc(), readMaskSizes);

  // Out-of-bounds lanes of the read are discarded with default zero value
  Value padding = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(srcType.getElementType()));

  // HIVM's input indexing map maps destination loops to source dimensions.
  // A transfer map has the opposite direction, as in linalg vectorization.
  Value vector = createMaskedTransferRead(
      rewriter, loc, VectorType::get(vectorSizes, srcType.getElementType()),
      getSrc(), padding, readMask, readMap);

  auto writeOp = createMaskedTransferWrite(rewriter, loc, vector, dst, mask);

  if (getNumResults() > 0)
    rewriter.replaceOp(*this, writeOp.getResult());
  else
    rewriter.eraseOp(*this);

  return success();
}

} // namespace mlir::hivm
