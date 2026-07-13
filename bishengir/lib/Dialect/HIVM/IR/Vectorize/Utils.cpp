//===------------------ Utils.cpp - HIVM Vectorize Utils-   ------------------===//
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
#define DEBUG_TYPE "hivm-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir::hivm {
Value getIdentityElement(OpBuilder &builder, Location loc, Type elemType,
                         VectorArithKind kind) {
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
    case VectorArithKind::SUB:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(floatType, 0.0));
    case VectorArithKind::MUL:
    case VectorArithKind::DIV:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(floatType, 1.0));
    case VectorArithKind::MAX:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(
              floatType, APFloat::getInf(floatType.getFloatSemantics(),
                                         /*Negative=*/true)));
    case VectorArithKind::MIN:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(
              floatType, APFloat::getInf(floatType.getFloatSemantics(),
                                         /*Negative=*/false)));
    }
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
    case VectorArithKind::SUB:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(intType, 0));
    case VectorArithKind::MUL:
    case VectorArithKind::DIV:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(intType, 1));
    case VectorArithKind::MAX:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(
              intType, APInt::getSignedMinValue(intType.getWidth())));
    case VectorArithKind::MIN:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(
              intType, APInt::getSignedMaxValue(intType.getWidth())));
    }
  }
  llvm_unreachable("unsupported element type for neutral element");
}

Value createVectorArithOp(OpBuilder &builder, Location loc,
                          VectorArithKind kind, Value lhs, Value rhs) {
  Type elemType = getElementTypeOrSelf(lhs.getType());

  if (isa<FloatType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
      return builder.create<arith::AddFOp>(loc, lhs, rhs);
    case VectorArithKind::SUB:
      return builder.create<arith::SubFOp>(loc, lhs, rhs);
    case VectorArithKind::MUL:
      return builder.create<arith::MulFOp>(loc, lhs, rhs);
    case VectorArithKind::DIV:
      return builder.create<arith::DivFOp>(loc, lhs, rhs);
    case VectorArithKind::MAX:
      return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
    case VectorArithKind::MIN:
      return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
    }
  } else if (isa<IntegerType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
      return builder.create<arith::AddIOp>(loc, lhs, rhs);
    case VectorArithKind::SUB:
      return builder.create<arith::SubIOp>(loc, lhs, rhs);
    case VectorArithKind::MUL:
      return builder.create<arith::MulIOp>(loc, lhs, rhs);
    case VectorArithKind::DIV:
      return builder.create<arith::DivSIOp>(loc, lhs, rhs); // or DivUIOp
    case VectorArithKind::MAX:
      return builder.create<arith::MaxSIOp>(loc, lhs, rhs);
    case VectorArithKind::MIN:
      return builder.create<arith::MinSIOp>(loc, lhs, rhs);
    }
  }

  llvm_unreachable("unsupported element type for vector arithmetic");
}
}