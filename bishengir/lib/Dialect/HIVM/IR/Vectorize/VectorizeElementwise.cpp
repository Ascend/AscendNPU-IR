//===------------------ Helper.cpp - HIVM implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMVectorize.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#define DEBUG_TYPE "hivm-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir::hivm {

//===----------------------------------------------------------------------===//
// Vectorization Helpers
//===----------------------------------------------------------------------===//

static LogicalResult
vectorizeElementwiseOp(Operation *op, RewriterBase &rewriter,
                       ArrayRef<int64_t> vectorSizes, VectorArithKind kind,
                       function_ref<Value(ValueRange, Value)> computeBuilder) {
  auto hivmOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!hivmOp) return failure();
  Location loc = hivmOp.getLoc();
  SmallVector<Value> inputs = hivmOp.getDpsInputs();
  Value firstOperand = inputs[0];
  Type elementType = getElementTypeOrSelf(firstOperand);

  VectorType vectorType = VectorType::get(vectorSizes, elementType);
  int64_t rank = (int64_t)vectorSizes.size();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(rank, zero);

  SmallVector<Value> dimSizes;

  for (int64_t i = 0; i < rank; ++i) {
    if (isa<TensorType>(firstOperand.getType())) {
      dimSizes.push_back(rewriter.create<tensor::DimOp>(loc, firstOperand, i));
    } else {
      dimSizes.push_back(rewriter.create<memref::DimOp>(loc, firstOperand, i));
    }
  }

  Value mask = rewriter.create<vector::CreateMaskOp>(
      loc, vectorType.clone(rewriter.getI1Type()), dimSizes);

  Value padding = getIdentityElement(rewriter, loc, elementType, kind);

  SmallVector<Value> vectorOperands;
  SmallVector<bool> inBounds(rank, true);

  for (Value input : inputs) {
    AffineMap map = rewriter.getMultiDimIdentityMap(rank);
    Value read = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, input, indices, map, padding, mask,
        rewriter.getBoolArrayAttr(inBounds));
    vectorOperands.push_back(read);
  }

  Value resultVector = computeBuilder(vectorOperands, mask);
  bool isTensorSemantics = hivmOp->getNumResults() > 0;
  Value dest = hivmOp.getDpsInitOperand(0)->get();
  AffineMap map = rewriter.getMultiDimIdentityMap(rank);
  auto writeOp = rewriter.create<vector::TransferWriteOp>(
      loc, TypeRange(dest.getType()), resultVector, dest, indices, map, mask,
      rewriter.getBoolArrayAttr(inBounds));

  if (isTensorSemantics) {
    rewriter.replaceAllUsesExcept(op->getResult(0), writeOp.getResult(),
                                  writeOp);
  }
  rewriter.eraseOp(op);

  return success();
}

template <VectorArithKind Kind>
static LogicalResult vectorizeElementwiseBinary(Operation *op,
                                                RewriterBase &rewriter,
                                                ArrayRef<int64_t> vectorSizes) {
  auto computer = [&](ValueRange vecOps, Value mask) -> Value {
    return createVectorArithOp(rewriter, op->getLoc(), Kind, vecOps[0],
                               vecOps[1]);
  };

  return vectorizeElementwiseOp(op, rewriter, vectorSizes, Kind, computer);
}

template <typename UnaryOpF, typename UnaryOpI>
static LogicalResult vectorizeElementwiseUnary(Operation *op,
                                               RewriterBase &rewriter,
                                               ArrayRef<int64_t> vectorSizes) {
  auto computer = [&](ValueRange vecOps, Value mask) -> Value {
    Type elemType = getElementTypeOrSelf(vecOps[0].getType());
    if (isa<FloatType>(elemType))
      return rewriter.create<UnaryOpF>(op->getLoc(), vecOps[0]);
    return rewriter.create<UnaryOpI>(op->getLoc(), vecOps[0]);
  };

  // Unary ops typically use 0 as neutral element
  return vectorizeElementwiseOp(op, rewriter, vectorSizes,
                                VectorArithKind::ADD, computer);
}

//===----------------------------------------------------------------------===//
// VAddOp Vectorization (Update existing implementation)
//===----------------------------------------------------------------------===//

LogicalResult VAddOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeElementwiseBinary<VectorArithKind::ADD>(*this, rewriter,
    vectorSizes);
}

LogicalResult VSubOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeElementwiseBinary<VectorArithKind::SUB>(*this, rewriter,
    vectorSizes);
}

LogicalResult VMulOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeElementwiseBinary<VectorArithKind::MUL>(*this, rewriter,
    vectorSizes);
}

LogicalResult VDivOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeElementwiseBinary<VectorArithKind::DIV>(*this, rewriter,
    vectorSizes);
}

LogicalResult VMaxOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeElementwiseBinary<VectorArithKind::MAX>(*this, rewriter,
    vectorSizes);
}

LogicalResult VMinOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeElementwiseBinary<VectorArithKind::MIN>(*this, rewriter,
    vectorSizes);
}

LogicalResult VAbsOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeElementwiseUnary<math::AbsFOp, math::AbsIOp>(*this, rewriter,
    vectorSizes);
}
}