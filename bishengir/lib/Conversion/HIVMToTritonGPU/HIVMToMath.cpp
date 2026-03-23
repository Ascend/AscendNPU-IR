//===- HIVMToMath.cpp - conversion from HIVM to Math dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
 
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToMath.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
 
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <functional>


 
using namespace mlir;
using namespace mlir::hivm;

static bool operateOnShaped(Operation *op) {
    if (auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op)) {
        return llvm::all_of(structuredOp.getHIVMOperandTypes(false),
                            [](Type type) { return isa<VectorType, RankedTensorType>(type); });
    }
    return false;
}

static SmallVector<Value> getHIVMVectorOperands(Operation *op) {
    if (auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op)) {
        SmallVector<Value> hivmOperands;
        for (OpOperand *operand :
                structuredOp.getHIVMOperands(false)) {
            hivmOperands.push_back(operand->get());
        }
        return hivmOperands;
    }
    return llvm::SmallVector<mlir::Value>();
}

template<typename... types>
static bool operateOnTypes(Operation *op) {
    if (auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op)) {
        return llvm::all_of(structuredOp.getHIVMOperandTypes(true),
                            [](Type type){ return llvm::isa<types...>(getElementTypeOrSelf(type)); });
    }
    return false;

}

using sst = IntegerType::SignednessSemantics;
template<sst signed_require>
static bool operateOnSigned(Operation *op) {
    if (auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op)) {
        switch (signed_require) {
            case IntegerType::Signless: {
                return llvm::all_of(structuredOp.getHIVMOperandTypes(true),
                                    [](Type type) { return getElementTypeOrSelf(type).isSignlessInteger(); });
            }
            default: {
                if (!op->hasAttr("isSigned")) {
                    return signed_require == IntegerType::Signed;
                }
                mlir::BoolAttr validAttr = op->getAttr("isSigned").cast<mlir::BoolAttr>();
                bool validValue = validAttr.getValue();
                if (validValue) {
                    return signed_require == IntegerType::Signed;
                } else {
                    return signed_require == IntegerType::Unsigned;
                }
            }
        }
    }

    return false;
}

static bool broadcast_check(Operation *op) {
    if (auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op)) {
        if (!structuredOp.isInlineBroadcastable()) {
            return true;
        }
        SmallVector<int64_t> brcDims;
        structuredOp.getBroadcastLoopDims(brcDims);
        if (brcDims.empty()) {
            return true;
        }
    }
    return false;
}

static bool transpose_check(Operation *op) {
    if (auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op)) {
        if (!structuredOp.isInlineTransposable()) {
            return true;
        }

        auto trnDims = structuredOp.getPermutationArray();
        if (trnDims.empty()) {
            return true;
        }
    }
    return false;
}
 
template<bool legal_or_not, bool sign_check, sst signed_require, typename... types>
static bool entryCondition(Operation *op) {
    // conversion would be executed only as oper is used on operand with shape.
    if (!operateOnShaped(op)) {
        return false;
    }
 
    // conversion would be executed only as transpose is empty.
    if (!transpose_check(op)) {
        return false;
    }
 
    // conversion would be executed only as broadcast is empty.
    if (!broadcast_check(op)) {
        return false;
    }
 
    // conversion would be executed only as operand type subject the type constraint.
    bool type_check = legal_or_not ? (operateOnTypes<types...>(op)) : !operateOnTypes<types...>(op);
    if (!type_check) {
        return false;
    }
 
    // conversion would be executed only as sign subject the sign constraint.
    if (sign_check) {
        return operateOnSigned<signed_require>(op);
    }
 
    return true;
}
 

template <typename MathUnaryOp, typename HIVMVectorOp,
        bool legal_or_not, bool sign_check, sst signed_require, typename... types>
struct VectorOpToMathUnary : public OpRewritePattern<HIVMVectorOp> {
  using OpRewritePattern<HIVMVectorOp>::OpRewritePattern;
 
  LogicalResult matchAndRewrite(HIVMVectorOp op,
                                PatternRewriter &rewriter) const final {
    auto  condition_entry = entryCondition<legal_or_not, sign_check, signed_require, types...>;
    if (!condition_entry(op)) {
        return failure();
    }
 
    SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
    if (hivmOperands.size() < 2) {
        return failure();
    }
    Value lhs = hivmOperands[0];
    Value dst_val = hivmOperands[1];
    auto resType = op.getResult().getType();
 
    auto result = rewriter.create<MathUnaryOp>(op.getLoc(), resType, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
};



template <typename MathBinaryOp, typename HIVMVectorOp,
        bool legal_or_not, bool sign_check, sst signed_require, typename... types>
struct VectorOpToMathBinary : public OpRewritePattern<HIVMVectorOp> {
  using OpRewritePattern<HIVMVectorOp>::OpRewritePattern;
 
  LogicalResult matchAndRewrite(HIVMVectorOp op,
                                PatternRewriter &rewriter) const final {
    auto  condition_entry = entryCondition<legal_or_not, sign_check, signed_require, types...>;
    if (!condition_entry(op)) {
        return failure();
    }
 
    SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
    if (hivmOperands.size() < 3) {
        return failure();
    }
    Value lhs = hivmOperands[0];
    Value rhs = hivmOperands[1];
    Value dst_val = hivmOperands[2];
    auto resType = op.getResult().getType();
 
    auto result = rewriter.create<MathBinaryOp>(op.getLoc(), resType, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
};


// VAbsOp → math.absf/absi  RewritePattern
template <bool legal_or_not, bool sign_check, sst signed_require, typename... types>
struct HIVMVAbsToMathAbsPattern : public OpRewritePattern<VAbsOp> {
  using OpRewritePattern<VAbsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VAbsOp vabsOp, PatternRewriter &rewriter) const override {
    auto  condition_entry = entryCondition<legal_or_not, sign_check, signed_require, types...>;
    if (!condition_entry(vabsOp)) {
        return failure();
    }

    // 1.1 Verify that the src/dst operands are not null
    if (vabsOp.getSrc().empty() || vabsOp.getDst().empty()) {
      return vabsOp.emitError("VAbsOp must have at least one src and one dst operand");
    }
    Value src = vabsOp.getSrc()[0];
    Value dst = vabsOp.getDst()[0];

    // Verify that the input is of RankedTensorType (vector/tensor, compliant with VAbsOp constraints)
    auto srcTensorType = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTensorType) {
      return vabsOp.emitError("VAbsOp only supports ranked tensor input (src)");
    }

    // Extract element type and verify supported types (F16/F32/I16/I32/I64)
    Type elemType = srcTensorType.getElementType();
    bool isFloatType = elemType.isF16() || elemType.isF32();
    bool isIntType = elemType.isInteger(16) || elemType.isInteger(32) || elemType.isInteger(64);
    if (!isFloatType && !isIntType) {
      return vabsOp.emitError("VAbsOp only supports F16/F32/I16/I32/I64, got ") << elemType;
    }

    // conversion process (generate absf/absi according to type)
    Value absResult;
    Location loc = vabsOp.getLoc();
    Type resultType = srcTensorType;

    if (isFloatType) {
      // float abs → math.absf
      absResult = rewriter.create<math::AbsFOp>(loc, resultType, src);
    } else {
      // int abs → math.absi
      absResult = rewriter.create<math::AbsIOp>(loc, resultType, src);
    }

    // Replace the original VAbsOp operator with the abs result
    rewriter.replaceOp(vabsOp, absResult);

    return success();
  }
};

void mlir::hivm::populateHIVMToMathConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<
        VectorOpToMathUnary<math::ExpOp, hivm::VExpOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathUnary<math::LogOp, hivm::VLnOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathUnary<math::SqrtOp, hivm::VSqrtOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathUnary<math::RsqrtOp, hivm::VRsqrtOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathUnary<math::TanhOp, hivm::VTanhOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathUnary<math::SinOp, hivm::VSinOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathUnary<math::CosOp, hivm::VCosOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathUnary<math::ErfOp, hivm::VErfOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToMathBinary<math::IPowIOp, hivm::VPowOp, true, false, IntegerType::Signless, IntegerType>,
        HIVMVAbsToMathAbsPattern<true, false, IntegerType::Signless, FloatType, IntegerType>         
    >(patterns.getContext());
}