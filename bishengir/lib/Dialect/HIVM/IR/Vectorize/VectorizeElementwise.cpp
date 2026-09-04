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
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/TypeSwitch.h"
#define DEBUG_TYPE "hivm-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir::hivm {
namespace {

enum class BitwiseKind { And, Or, Xor };

static Value createSplatVector(OpBuilder &builder, Location loc,
                               VectorType vectorType, Attribute value) {
  return builder.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(vectorType, value));
}

static Value createVectorBitwiseOp(OpBuilder &builder, Location loc,
                                   BitwiseKind kind, Value lhs, Value rhs) {
  auto applyInt = [&](Value lhsInt, Value rhsInt) -> Value {
    switch (kind) {
    case BitwiseKind::And:
      return builder.create<arith::AndIOp>(loc, lhsInt, rhsInt);
    case BitwiseKind::Or:
      return builder.create<arith::OrIOp>(loc, lhsInt, rhsInt);
    case BitwiseKind::Xor:
      return builder.create<arith::XOrIOp>(loc, lhsInt, rhsInt);
    }
    llvm_unreachable("unhandled bitwise kind");
  };

  Type elemType = getElementTypeOrSelf(lhs.getType());
  if (isa<IntegerType>(elemType))
    return applyInt(lhs, rhs);

  auto floatType = dyn_cast<FloatType>(elemType);
  if (!floatType)
    llvm::report_fatal_error("unsupported element type for bitwise op");

  auto srcVec = cast<VectorType>(lhs.getType());
  auto intType = IntegerType::get(builder.getContext(), floatType.getWidth());
  auto intVec = VectorType::get(srcVec.getShape(), intType);
  Value lhsInt = builder.create<arith::BitcastOp>(loc, intVec, lhs);
  Value rhsInt = builder.create<arith::BitcastOp>(loc, intVec, rhs);
  Value resultInt = applyInt(lhsInt, rhsInt);
  return builder.create<arith::BitcastOp>(loc, srcVec, resultInt);
}

} // namespace

//===----------------------------------------------------------------------===//
// Vectorization Helpers
//===----------------------------------------------------------------------===//

static LogicalResult
vectorizeElementwiseOp(Operation *op, RewriterBase &rewriter,
                       ArrayRef<int64_t> vectorSizes, VectorArithKind kind,
                       function_ref<Value(ValueRange, Value)> computeBuilder) {
  if (failed(checkVectorizePreconditions(op, vectorSizes)))
    return failure();

  auto hivmOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!hivmOp)
    return failure();
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
  if (!resultVector)
    return failure();
  bool isTensorSemantics = hivmOp->getNumResults() > 0;
  Value dest = hivmOp.getDpsInitOperand(0)->get();
  AffineMap map = rewriter.getMultiDimIdentityMap(rank);
  auto writeOp = rewriter.create<vector::TransferWriteOp>(
      loc, TypeRange(dest.getType()), resultVector, dest, indices, map, mask,
      rewriter.getBoolArrayAttr(inBounds));

  if (isTensorSemantics) {
    rewriter.replaceOp(op, writeOp.getResult());
  } else {
    rewriter.eraseOp(op);
  }

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
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, VectorArithKind::ADD,
                                computer);
}

template <typename UnaryOpF>
static LogicalResult
vectorizeElementwiseFloatUnary(Operation *op, RewriterBase &rewriter,
                               ArrayRef<int64_t> vectorSizes) {
  Type elemType = getElementTypeOrSelf(
      cast<DestinationStyleOpInterface>(op).getDpsInputs().front());
  if (!isa<FloatType>(elemType))
    return failure();
  auto computer = [&](ValueRange vecOps, Value mask) -> Value {
    return rewriter.create<UnaryOpF>(op->getLoc(), vecOps[0]);
  };
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, VectorArithKind::ADD,
                                computer);
}

static LogicalResult vectorizeElementwiseBitwise(Operation *op,
                                                 RewriterBase &rewriter,
                                                 ArrayRef<int64_t> vectorSizes,
                                                 BitwiseKind kind) {
  auto computer = [&](ValueRange vecOps, Value mask) -> Value {
    return createVectorBitwiseOp(rewriter, op->getLoc(), kind, vecOps[0],
                                 vecOps[1]);
  };
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, VectorArithKind::ADD,
                                computer);
}

LogicalResult vectorizeElementwise(Operation *op, RewriterBase &rewriter,
                                   ArrayRef<int64_t> vectorSizes) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<VAddOp>([&](auto op) {
        return vectorizeElementwiseBinary<VectorArithKind::ADD>(op, rewriter,
                                                                vectorSizes);
      })
      .Case<VSubOp>([&](auto op) {
        return vectorizeElementwiseBinary<VectorArithKind::SUB>(op, rewriter,
                                                                vectorSizes);
      })
      .Case<VMulOp>([&](auto op) {
        return vectorizeElementwiseBinary<VectorArithKind::MUL>(op, rewriter,
                                                                vectorSizes);
      })
      .Case<VDivOp>([&](auto op) {
        return vectorizeElementwiseBinary<VectorArithKind::DIV>(op, rewriter,
                                                                vectorSizes);
      })
      .Case<VMaxOp>([&](auto op) {
        return vectorizeElementwiseBinary<VectorArithKind::MAX>(op, rewriter,
                                                                vectorSizes);
      })
      .Case<VMinOp>([&](auto op) {
        return vectorizeElementwiseBinary<VectorArithKind::MIN>(op, rewriter,
                                                                vectorSizes);
      })
      .Case<VAbsOp>([&](auto op) {
        return vectorizeElementwiseUnary<math::AbsFOp, math::AbsIOp>(
            op, rewriter, vectorSizes);
      })
      .Case<VExpOp>([&](auto op) {
        return vectorizeElementwiseFloatUnary<math::ExpOp>(op, rewriter,
                                                           vectorSizes);
      })
      .Case<VLnOp>([&](auto op) {
        return vectorizeElementwiseFloatUnary<math::LogOp>(op, rewriter,
                                                           vectorSizes);
      })
      .Case<VSqrtOp>([&](auto op) {
        return vectorizeElementwiseFloatUnary<math::SqrtOp>(op, rewriter,
                                                            vectorSizes);
      })
      .Case<VRsqrtOp>([&](auto op) {
        return vectorizeElementwiseFloatUnary<math::RsqrtOp>(op, rewriter,
                                                             vectorSizes);
      })
      .Case<VRecOp>([&](auto recOp) {
        auto computer = [&](ValueRange vecOps, Value) -> Value {
          auto vectorType = cast<VectorType>(vecOps[0].getType());
          auto floatType = dyn_cast<FloatType>(vectorType.getElementType());
          if (!floatType)
            return Value();
          Value one = createSplatVector(
              rewriter, recOp.getLoc(), vectorType,
              rewriter.getFloatAttr(floatType, 1.0));
          return rewriter.create<arith::DivFOp>(recOp.getLoc(), one,
                                                vecOps[0]);
        };
        return vectorizeElementwiseOp(recOp, rewriter, vectorSizes,
                                      VectorArithKind::DIV, computer);
      })
      .Case<VReluOp>([&](auto reluOp) {
        auto computer = [&](ValueRange vecOps, Value) -> Value {
          auto vectorType = cast<VectorType>(vecOps[0].getType());
          Type elemType = vectorType.getElementType();
          Value zero = createSplatVector(rewriter, reluOp.getLoc(), vectorType,
                                         rewriter.getZeroAttr(elemType));
          if (isa<FloatType>(elemType))
            return rewriter.create<arith::MaximumFOp>(reluOp.getLoc(),
                                                      vecOps[0], zero);
          if (isa<IntegerType>(elemType))
            return rewriter.create<arith::MaxSIOp>(reluOp.getLoc(), vecOps[0],
                                                   zero);
          return Value();
        };
        return vectorizeElementwiseOp(reluOp, rewriter, vectorSizes,
                                      VectorArithKind::ADD, computer);
      })
      .Case<VNotOp>([&](auto notOp) {
        auto computer = [&](ValueRange vecOps, Value) -> Value {
          auto srcVec = cast<VectorType>(vecOps[0].getType());
          Type elemType = srcVec.getElementType();
          if (auto intType = dyn_cast<IntegerType>(elemType)) {
            Value ones = createSplatVector(
                rewriter, notOp.getLoc(), srcVec,
                rewriter.getIntegerAttr(intType, -1));
            return rewriter.create<arith::XOrIOp>(notOp.getLoc(), vecOps[0],
                                                  ones);
          }
          auto floatType = dyn_cast<FloatType>(elemType);
          if (!floatType)
            return Value();
          auto intType =
              IntegerType::get(rewriter.getContext(), floatType.getWidth());
          auto intVec = VectorType::get(srcVec.getShape(), intType);
          Value srcInt = rewriter.create<arith::BitcastOp>(notOp.getLoc(),
                                                           intVec, vecOps[0]);
          Value ones = createSplatVector(rewriter, notOp.getLoc(), intVec,
                                         rewriter.getIntegerAttr(intType, -1));
          Value resultInt = rewriter.create<arith::XOrIOp>(notOp.getLoc(),
                                                           srcInt, ones);
          return rewriter.create<arith::BitcastOp>(notOp.getLoc(), srcVec,
                                                   resultInt);
        };
        return vectorizeElementwiseOp(notOp, rewriter, vectorSizes,
                                      VectorArithKind::ADD, computer);
      })
      .Case<VAndOp>([&](auto op) {
        return vectorizeElementwiseBitwise(op, rewriter, vectorSizes,
                                           BitwiseKind::And);
      })
      .Case<VOrOp>([&](auto op) {
        return vectorizeElementwiseBitwise(op, rewriter, vectorSizes,
                                           BitwiseKind::Or);
      })
      .Case<VXorOp>([&](auto op) {
        return vectorizeElementwiseBitwise(op, rewriter, vectorSizes,
                                           BitwiseKind::Xor);
      })
      .Default([](Operation *) { return failure(); });
}

bool canVectorizeHIVMOp(Operation *op) {
  return isa<VAddOp, VSubOp, VMulOp, VDivOp, VMaxOp, VMinOp, VAbsOp, VExpOp,
             VLnOp, VSqrtOp, VRsqrtOp, VRecOp, VReluOp, VNotOp, VAndOp, VOrOp,
             VXorOp, VReduceOp>(op);
}
} // namespace mlir::hivm
