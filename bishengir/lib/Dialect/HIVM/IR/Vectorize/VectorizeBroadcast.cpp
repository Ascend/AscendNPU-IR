//===- VectorizeBroadcast.cpp - Vectorize HIVM broadcasts ----------------===//
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

LogicalResult VBrcOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {

  if (failed(checkVectorizePreconditions(*this, vectorSizes)))
    return failure();

  Location loc = getLoc();
  Value dst = getDst();
  Type elementType = getElementTypeOrSelf(getSrc());

  // The mask follows the destination: it is the op's iteration domain.
  Value mask = createShapeMask(rewriter, loc, dst, vectorSizes);

  // Handles both scalar and shaped sources.
  Value padding = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(elementType));
  Value vector =
      readOperand(rewriter, loc, getSrc(), vectorSizes, padding, mask);

  auto writeOp = createMaskedTransferWrite(rewriter, loc, vector, dst, mask);

  if (getNumResults() > 0)
    rewriter.replaceOp(*this, writeOp.getResult());
  else
    rewriter.eraseOp(*this);

  return success();
}

} // namespace mlir::hivm
