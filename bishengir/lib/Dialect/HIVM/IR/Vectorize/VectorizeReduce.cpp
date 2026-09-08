//===- VectorizeReduce.cpp - Vectorize HIVM reductions --------------------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::hivm {

LogicalResult VReduceOp::vectorize(RewriterBase &rewriter,
                                   ArrayRef<int64_t> vectorSizes) {
  if (failed(checkVectorizePreconditions(*this, vectorSizes)) ||
      failed(checkShapedInputs(*this, vectorSizes)))
    return failure();

  Location loc = getLoc();
  Value src = getSrc();
  Value dst = getDstValue();

  Type elementType = getElementTypeOrSelf(src);
  VectorType vectorType = VectorType::get(vectorSizes, elementType);
  int64_t rank = (int64_t)vectorSizes.size();
  SmallVector<int64_t> reduceDims(getReduceDims());
  if (reduceDims.empty()) {
    return failure();
  }

  arith::AtomicRMWKind rmwKind;
  vector::CombiningKind combiningKind;
  auto reduceOpArith = getArithAttr();
  auto reduceOpAttr = reduceOpArith.getReduceOp();

  switch (reduceOpAttr) {
  case hivm::ReduceOperation::sum:
    combiningKind = vector::CombiningKind::ADD;
    rmwKind = isa<FloatType>(elementType) ? arith::AtomicRMWKind::addf
                                          : arith::AtomicRMWKind::addi;
    break;
  case hivm::ReduceOperation::prod:
    combiningKind = vector::CombiningKind::MUL;
    rmwKind = isa<FloatType>(elementType) ? arith::AtomicRMWKind::mulf
                                          : arith::AtomicRMWKind::muli;
    break;
  case hivm::ReduceOperation::max:
    if (isa<FloatType>(elementType)) {
      combiningKind = vector::CombiningKind::MAXIMUMF;
      rmwKind = arith::AtomicRMWKind::maximumf;
    } else {
      combiningKind = getUnsignedSrc() ? vector::CombiningKind::MAXUI
                                       : vector::CombiningKind::MAXSI;
      rmwKind = getUnsignedSrc() ? arith::AtomicRMWKind::maxu
                                 : arith::AtomicRMWKind::maxs;
    }
    break;
  case hivm::ReduceOperation::min:
    if (isa<FloatType>(elementType)) {
      combiningKind = vector::CombiningKind::MINIMUMF;
      rmwKind = arith::AtomicRMWKind::minimumf;
    } else {
      combiningKind = getUnsignedSrc() ? vector::CombiningKind::MINUI
                                       : vector::CombiningKind::MINSI;
      rmwKind = getUnsignedSrc() ? arith::AtomicRMWKind::minu
                                 : arith::AtomicRMWKind::mins;
    }
    break;
  case hivm::ReduceOperation::any:
  case hivm::ReduceOperation::ori:
    combiningKind = vector::CombiningKind::OR;
    rmwKind = arith::AtomicRMWKind::ori;
    break;
  case hivm::ReduceOperation::all:
  case hivm::ReduceOperation::andi:
    combiningKind = vector::CombiningKind::AND;
    rmwKind = arith::AtomicRMWKind::andi;
    break;
  case hivm::ReduceOperation::xori:
    combiningKind = vector::CombiningKind::XOR;
    // XOR and OR have the same zero identity.
    rmwKind = arith::AtomicRMWKind::ori;
    break;
  default:
    return failure();
  }

  Value mask = createShapeMask(rewriter, loc, src, vectorSizes);

  Value padding = arith::getIdentityValue(rmwKind, elementType, rewriter, loc);

  Value vectorData =
      createMaskedTransferRead(rewriter, loc, vectorType, src, padding, mask);

  // reducedShape: drops the reduced dims
  // outputShape: keeps the reduced dims and collapses them to one
  // e.g.
  // vectorSizes [2,4,3] with reduce_dims [1] -> reduced [2,3], output [2,1,3].
  SmallVector<int64_t> reducedShape;
  SmallVector<int64_t> outputShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (llvm::is_contained(reduceDims, i)) {
      outputShape.push_back(1);
    } else {
      reducedShape.push_back(vectorSizes[i]);
      outputShape.push_back(vectorSizes[i]);
    }
  }

  VectorType reducedVectorType = VectorType::get(reducedShape, elementType);
  VectorType outputVectorType = VectorType::get(outputShape, elementType);

  Value outputMask = createShapeMask(rewriter, loc, dst, outputShape);
  Value initAccum = createMaskedTransferRead(rewriter, loc, outputVectorType,
                                             dst, padding, outputMask);

  Value vectorOut =
      rewriter.create<vector::ShapeCastOp>(loc, reducedVectorType, initAccum);

  // `vector::MultiDimReductionOp` takes the reduction dims as an `ArrayAttr`
  // in the AscendNPU-IR baseline LLVM, but as an `ArrayRef<int64_t>` when
  // building against newer LLVM 22.
  Value reduced;
#if defined(__LLVM_MAJOR_VERSION_20_COMPATIBLE__) ||                           \
    defined(__LLVM_MAJOR_VERSION_21_COMPATIBLE__) ||                           \
    defined(__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
  reduced = rewriter.create<vector::MultiDimReductionOp>(
      loc, combiningKind, vectorData, vectorOut, reduceDims);
#else
  reduced = rewriter.create<vector::MultiDimReductionOp>(
      loc, combiningKind, vectorData, vectorOut,
      rewriter.getI64ArrayAttr(reduceDims));
#endif

  Value finalResult =
      rewriter.create<vector::ShapeCastOp>(loc, outputVectorType, reduced);

  auto writeOp =
      createMaskedTransferWrite(rewriter, loc, finalResult, dst, outputMask);
  if (getNumResults() > 0) {
    rewriter.replaceOp(*this, writeOp.getResult());
  } else {
    rewriter.eraseOp(*this);
  }

  return success();
}
} // namespace mlir::hivm
