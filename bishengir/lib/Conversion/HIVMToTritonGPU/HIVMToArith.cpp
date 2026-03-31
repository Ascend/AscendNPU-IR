//===- HIVMToArith.cpp - conversion from HIVM to Arith dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToArith.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <functional>

using namespace mlir;
using namespace mlir::hivm;

static bool operateOnShaped(Operation *op) {
  auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op);
  if (!structuredOp) {
    return false;
  }
  return llvm::all_of(structuredOp.getHIVMOperandTypes(false),
                      [](Type type) { return isa<VectorType, RankedTensorType>(type); });
}

static SmallVector<Value> getHIVMVectorOperands(Operation *op) {
    auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op);
    SmallVector<Value> hivmOperands = {};
    if (!structuredOp) {
        return hivmOperands;
    }
    for (OpOperand *operand :
         structuredOp.getHIVMOperands(false)) {
      hivmOperands.push_back(operand->get());
    }
    return hivmOperands;
}

template<typename... types>
static bool operateOnTypes(Operation *op) {
    auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op);
    if (!structuredOp) {
        return false;
    }
    return llvm::all_of(structuredOp.getHIVMOperandTypes(false),
                        [](Type type){ return llvm::isa<types...>(getElementTypeOrSelf(type)); });
}

using sst = IntegerType::SignednessSemantics;
template<sst signed_require>
static bool operateOnSigned(Operation *op) {
    auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op);
    if (!structuredOp) {
        return false;
    }
    switch (signed_require) {
        case IntegerType::Signless: {
            return llvm::all_of(structuredOp.getHIVMOperandTypes(false),
                            [](Type type) { return getElementTypeOrSelf(type).isSignlessInteger(); });
        }
        default: {
            if (!op->hasAttr("isSigned")) {
                return signed_require == IntegerType::Signed;
            }
            mlir::BoolAttr validAttr = mlir::cast<mlir::BoolAttr>(op->getAttr("isSigned"));
            bool validValue = validAttr.getValue();
            if (validValue) {
                return signed_require == IntegerType::Signed;
            } else {
                return signed_require == IntegerType::Unsigned;
            }
        }
    }
    return false;
}

static bool broadcast_check(Operation *op) {
    auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op);
    if (!structuredOp.isInlineBroadcastable()) {
        return true;
    }
    SmallVector<int64_t> brcDims;
    structuredOp.getBroadcastLoopDims(brcDims);
    if (brcDims.empty()) {
        return true;
    }
    return false;
}

static bool transpose_check(Operation *op) {
    auto structuredOp = dyn_cast_if_present<HIVMStructuredOp>(op);
    if (!structuredOp) {
        return false;
    }
    if (!structuredOp.isInlineTransposable()) {
      return true;
    }

    auto trnDims = structuredOp.getPermutationArray();
    if (trnDims.empty()) {
        return true;
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

using EntryConditonFunc = std::function<bool(Operation*)>;
template <typename ArithBinaryOp, typename HIVMVectorOp,
        bool legal_or_not, bool sign_check, sst signed_require, typename... types>
struct VectorOpToArithBinary : public OpRewritePattern<HIVMVectorOp> {
  using OpRewritePattern<HIVMVectorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HIVMVectorOp op,
                                PatternRewriter &rewriter) const final {
    auto  condition_entry = entryCondition<legal_or_not, sign_check, signed_require, types...>;
    if (!condition_entry(op)) {
        return failure();
    }

    SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
    Value lhs = hivmOperands[0];
    Value rhs = hivmOperands[1];
    Value dst_val = hivmOperands[2];
    auto resType = op.getResult().getType();

    auto newOp = rewriter.create<ArithBinaryOp>(
      op.getLoc(), resType, lhs, rhs);
    dst_val.replaceAllUsesWith(newOp.getResult());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/*
* vcast -> arith.extsi
* The round_mode attribute value of the hivm.hir.vcast Op must be rint.
* The cast attribute value of the hivm.hir.vcast Op must be cast_signed.
* The bit width of the src operand must be less than that of the dst operand.
*/
static bool extsi_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    if (roundMode != hivm::RoundMode::RINT) {
        return false;
    }

    const bool is_signed = (casting == hivm::TypeFn::cast_signed);

    if (!src.isInteger() || !dst.isInteger()) {
        return false;
    }
    if (!is_signed) {
        return false;
    }
    
    auto src_width = src.getIntOrFloatBitWidth();
    auto dst_width = dst.getIntOrFloatBitWidth();
    if (src_width >= dst_width) {
        return false;
    }

    if ((src.isInteger(8) || src.isInteger(16) || src.isInteger(32)) \
        && (dst.isInteger(16) || dst.isInteger(32) || dst.isInteger(64))) {
        return true;
    }

    return false;
}

/*
* vcast -> extui
* The round_mode attribute value of the hivm.hir.vcast Op must be rint.
* The cast attribute value of the hivm.hir.vcast Op must be cast_unsigned.
* The bit width of the src operand must be less than that of the dst operand.
*/
static bool extui_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    if (roundMode != hivm::RoundMode::RINT) {
        return false;
    }

    if (!src.isInteger() || !dst.isInteger()) {
        return false;
    }

    auto src_width = src.getIntOrFloatBitWidth();
    auto dst_width = dst.getIntOrFloatBitWidth();
    if (src_width >= dst_width) {
        return false;
    }

    const bool isInt8ToInt16 = src.isInteger(8) && dst.isInteger(16);
    const bool isInt8ToInt32 = src.isInteger(8) && dst.isInteger(32);
    const bool isInt8ToInt64 = src.isInteger(8) && dst.isInteger(64);
    const bool isInt16ToInt32 = src.isInteger(16) && dst.isInteger(32);
    const bool isInt16ToInt64 = src.isInteger(16) && dst.isInteger(64);
    const bool isInTypeI8  = isInt8ToInt16 || isInt8ToInt32 || isInt8ToInt64;
    const bool isInTypeI16 = isInt16ToInt32 || isInt16ToInt64;
    const bool isInTypeI32 = src.isInteger(32) && dst.isInteger(64);
    const bool is_unsigned = (casting == hivm::TypeFn::cast_unsigned);

    if (is_unsigned && (isInTypeI8 || isInTypeI16 ||isInTypeI32)) {
        return true;
    }
    
    const bool isInType1 = src.isInteger(1) && (dst.isInteger(8) || dst.isInteger(16) || dst.isInteger(32));
    if (!is_unsigned) {
        return isInType1;
    }

    return false;
}

/*
* vcast -> arith.extf
* The round_mode attribute value of the hivm.hir.vcast Op must be rint.
* the element type of all operand are float.
* The bit width of the src operand must be less than that of the dst operand.
*/
static bool extf_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    if (roundMode != hivm::RoundMode::RINT) {
        return false;
    }

    if (!llvm::isa<FloatType>(src) || !llvm::isa<FloatType>(dst)) {
        return false;
    }

    auto src_width = src.getIntOrFloatBitWidth();
    auto dst_width = dst.getIntOrFloatBitWidth();
    if (src_width >= dst_width) {
        return false;
    }

    if (src.isF16() && dst.isF32()) {
        return true;
    }

    if (src.isBF16() && dst.isF32()) {
        return true;
    }

    if ((src.isF32() || src.isF16() || src.isBF16()) && (dst.isFloat8E4M3FN() || dst.isFloat8E5M2())) {
        return true;
    }
    return false;
}

static bool isOverFlowMode(Type inType, Type outType) {
  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  return (isI16ToI8 || isI32ToI16 || isI32ToI8 || isF32ToI16 || isF32ToI8 ||
          isF16ToI8);
}

/*
* vcast -> fptosi
* The source operand of the hivm.hir.vcast Op must be a floating-point type,
* and the destination operand must be an integer type."
* The round_mode attribute value of the hivm.hir.vcast Op must be trunc or truncwithoverflow.
* The cast attribute value of the hivm.hir.vcast Op must be cast_signed.
* The bit width of the src operand must be less than that of the dst operand.
*/
static bool fptosi_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    const bool is_src_float = llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(src);
    const bool is_dst_integer = dst.isInteger();
    const bool is_signed = (casting == hivm::TypeFn::cast_signed);
    if (!is_signed) {
        return false;
    }

    if (!is_src_float || !is_dst_integer) {
        return false;
    }

    if (isOverFlowMode(src, dst)) {
        return roundMode == hivm::RoundMode::TRUNCWITHOVERFLOW;
    } else {
        return roundMode == hivm::RoundMode::TRUNC;
    }
}

/*
* vcast -> arith.fptoui
* The source operand of the hivm.hir.vcast Op must be a floating-point type,
* and the destination operand must be an integer type."
* The round_mode attribute value of the hivm.hir.vcast Op must be trunc or truncwithoverflow.
* The cast attribute value of the hivm.hir.vcast Op must be cast_unsigned.
* The bit width of the src operand must be less than that of the dst operand.
*/
static bool fptoui_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    const bool is_src_float = llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(src);
    const bool is_dst_integer = dst.isInteger();
    const bool is_unsigned = (casting == hivm::TypeFn::cast_unsigned);
    if (!is_unsigned) {
        return false;
    }
    if (!is_src_float || !is_dst_integer) {
        return false;
    }

    if (isOverFlowMode(src, dst)) {
        return roundMode == hivm::RoundMode::TRUNCWITHOVERFLOW;
    } else {
        return roundMode == hivm::RoundMode::TRUNC;
    }
}

/*
* vcast -> arith.sitofp
* The source operand of the hivm.hir.vcast Op must be a integer type, and the destination operand must be an floating-point type."
* The round_mode attribute value of the hivm.hir.vcast Op must be trunc or truncwithoverflow.
* The cast attribute value of the hivm.hir.vcast Op must be cast_signed.
* The bit width of the src operand must be less than that of the dst operand.
*/
static bool sitofp_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    const bool is_dst_float = llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(dst);
    const bool is_src_integer = src.isInteger();
    const bool is_signed = (casting == hivm::TypeFn::cast_signed);

    if (!is_signed) {
        return false;
    }

    if (!is_src_integer || !is_dst_float) {
        return false;
    }

    return roundMode == hivm::RoundMode::RINT;
}

/*
* vcast -> arith.uitofp
* The source operand of the hivm.hir.vcast Op must be a integer type,
* and the destination operand must be an floating-point type."
* The round_mode attribute value of the hivm.hir.vcast Op must be trunc or truncwithoverflow.
* The cast attribute value of the hivm.hir.vcast Op must be cast_unsigned.
* The bit width of the src operand must be less than that of the dst operand.
*/
static bool uitofp_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    const bool is_dst_float = llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(dst);
    const bool is_src_integer = src.isInteger();
    const bool is_unsigned = (casting == hivm::TypeFn::cast_unsigned);
    if (!is_unsigned) {
        return false;
    }
    if (!is_src_integer || !is_dst_float) {
        return false;
    }
    return roundMode == hivm::RoundMode::RINT;
}

/*
* vcast -> trunci
* The round_mode attribute value of the hivm.hir.vcast Op must be rint or truncwithoverflow.
* The bit width of the src operand must be larger than that of the dst operand.
*/
static bool trunci_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    if (!src.isInteger() || !dst.isInteger()) {
        return false;
    }
    
    auto src_width = src.getIntOrFloatBitWidth();
    auto dst_width = dst.getIntOrFloatBitWidth();
    if (src_width <= dst_width) {
        return false;
    }

    if (isOverFlowMode(src, dst)) {
        return roundMode == hivm::RoundMode::TRUNCWITHOVERFLOW;
    } else {
        return roundMode == hivm::RoundMode::RINT;
    }
}

/*
* vcast -> truncf
* The round_mode attribute value of the hivm.hir.vcast Op must be rint.
* The bit width of the src operand must be larger than that of the dst operand.
*/
static bool truncf_cast_condition(Type src, Type dst, hivm::TypeFn casting, hivm::RoundMode roundMode) {
    if (!llvm::isa<FloatType>(src) || !llvm::isa<FloatType>(dst)) {
        return false;
    }
    auto src_width = src.getIntOrFloatBitWidth();
    auto dst_width = dst.getIntOrFloatBitWidth();
    if (src_width <= dst_width) {
        return false;
    }

    if (!(src.isF32() || src.isF16() || src.isBF16())) {
        return false;
    }

    if (!(dst.isF16() || dst.isBF16() || dst.isF32() || dst.isFloat8E4M3FN() || dst.isFloat8E5M2())) {
        return false;
    }

    return roundMode == hivm::RoundMode::RINT;
}

struct HIVMToArithCastOp: public OpRewritePattern<hivm::VCastOp> {
    using OpRewritePattern<hivm::VCastOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VCastOp op,
                                PatternRewriter &rewriter) const final {
        if (!operateOnShaped(op)) {
            return failure();
        }
        
        // conversion would be executed only as transpose is empty.
        if (!transpose_check(op)) {
            return failure();
        }

        // conversion would be executed only as broadcast is empty.
        if (!broadcast_check(op)) {
            return failure();
        }

        HIVMStructuredOp structuredOp = op;
        SmallVector<Type> types = structuredOp.getHIVMOperandTypes(false);
        Type src_type = getElementTypeOrSelf(types[0]);
        Type dst_type = getElementTypeOrSelf(types[1]);
        hivm::TypeFn casting = op.getCast();
        hivm::RoundMode roundMode = op.getRoundMode();
       
        SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
        if (hivmOperands.size() < 2) {
            return failure();
        }
        Value src = hivmOperands[0];
        Value dst_val = hivmOperands[1];
        auto resType = op.getResult().getType();
        if (truncf_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::TruncFOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (trunci_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::TruncIOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (extf_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::ExtFOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (extsi_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::ExtSIOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (extui_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::ExtUIOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (fptosi_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::FPToSIOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (fptoui_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::FPToUIOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (sitofp_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::SIToFPOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (uitofp_cast_condition(src_type, dst_type, casting, roundMode)) {
            auto result = rewriter.create<arith::UIToFPOp>(op.getLoc(), resType, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else {
            return failure();
        }
        return success();
    }
};

/*
* bitcast -> arith.bitcast
* The bit widths of all operand types must be the same.
* the bit width of src operand must be one of 16, 32, 64, which is the constraint of hivm.bitcast op.
*/
struct HIVMToArithBitcastOp: public OpRewritePattern<hivm::BitcastOp> {
    using OpRewritePattern<hivm::BitcastOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::BitcastOp op, PatternRewriter &rewriter) const final {
        auto src = op.getSrc();
        Type src_type = src.getType();
        if (!isa<VectorType, RankedTensorType>(src_type)) {
            return failure();
        }
        auto resType = op.getResult().getType();
        auto result = rewriter.create<arith::BitcastOp>(op.getLoc(), resType, src);
        rewriter.replaceOp(op, result);
        return success();
    }
};

static arith::CmpFPredicate selectFPredicate(hivm::VCmpOp op) {
    switch (op.getCompareMode()) {
        case hivm::CompareMode::EQ:
            return arith::CmpFPredicate::OEQ;
        case hivm::CompareMode::NE:
            return arith::CmpFPredicate::ONE;
        case hivm::CompareMode::LT:
            return arith::CmpFPredicate::OLT;
        case hivm::CompareMode::GT:
            return arith::CmpFPredicate::OGT;
        case hivm::CompareMode::LE:
            return arith::CmpFPredicate::OLE;
        case hivm::CompareMode::GE:
            return arith::CmpFPredicate::OGE;
    }
}

static arith::CmpIPredicate selectIPredicate(hivm::VCmpOp op) {
    switch (op.getCompareMode()) {
        case hivm::CompareMode::EQ:
            return arith::CmpIPredicate::eq;
        case hivm::CompareMode::NE:
            return arith::CmpIPredicate::ne;
        case hivm::CompareMode::LT:
            return arith::CmpIPredicate::slt;
        case hivm::CompareMode::GT:
            return arith::CmpIPredicate::sgt;
        case hivm::CompareMode::LE:
            return arith::CmpIPredicate::sle;
        case hivm::CompareMode::GE:
            return arith::CmpIPredicate::sge;
    }
}

static arith::CmpIPredicate selectI1Predicate(hivm::VCmpOp op) {
    switch (op.getCompareMode()) {
        case hivm::CompareMode::LT:
            return arith::CmpIPredicate::ult;
        case hivm::CompareMode::GT:
            return arith::CmpIPredicate::ugt;
        case hivm::CompareMode::LE:
            return arith::CmpIPredicate::ule;
        case hivm::CompareMode::GE:
            return arith::CmpIPredicate::uge;
        case hivm::CompareMode::EQ:
            return arith::CmpIPredicate::eq;
        case hivm::CompareMode::NE:
            return arith::CmpIPredicate::ne;
    }
}

static arith::CmpIPredicate selectPredicate(hivm::VCmpOp op) {
    HIVMStructuredOp structuredOp = op;
    SmallVector<Type> types = structuredOp.getHIVMOperandTypes(false);
    auto elemType = getElementTypeOrSelf(types[0]);
    auto width = elemType.getIntOrFloatBitWidth();
    if (width == 1) {
        return selectI1Predicate(op);
    } else {
        return selectIPredicate(op);
    }
}

/*
* vcmp -> arith.cmpf
* hivm.hir.vcmp cannot distinguish between signed and unsigned integer comparisons.
* Currently, all are uniformly converted to signed comparisons, which may lead to precision issues.
* vcmp -> arith.cmpi
* Ordered and unordered comparisons cannot be distinguished in the current conversion of
* floating-point comparison ops.
* Currently, unordered comparison has no practical use cases in the Triton DSL,
* and floating-point comparison ops are uniformly converted to ordered comparison.
* the type of all operand are tensor or vector.
* the attribute of broadcast is empty.
* the attribute of transpose is empty.
* the element type of all operand are float.
*/
struct HIVMToArithCmpOp: public OpRewritePattern<hivm::VCmpOp> {
    using OpRewritePattern<hivm::VCmpOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VCmpOp op, PatternRewriter &rewriter) const final {
        // conversion would be executed only as transpose is empty.
        if (!transpose_check(op)) {
            return failure();
        }

        // conversion would be executed only as broadcast is empty.
        if (!broadcast_check(op)) {
            return failure();
        }
        if (!operateOnShaped(op)) {
            return failure();
        }

        SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
        Value lhs = hivmOperands[0];
        Value rhs = hivmOperands[1];
        Value dst_val = hivmOperands[2];
        auto resType = op.getResult().getType();
        Type elem_type = getElementTypeOrSelf(resType[0]);
        if (!elem_type.isSignlessInteger(1)) {
            return failure();
        }
        if (isa<FloatType>(getElementTypeOrSelf(lhs.getType()))) {
            arith::CmpFPredicate pred = selectFPredicate(op);
            auto result = rewriter.create<arith::CmpFOp>(op.getLoc(), resType, pred, lhs, rhs);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else {
            arith::CmpIPredicate pred = selectPredicate(op);
            auto result = rewriter.create<arith::CmpIOp>(op.getLoc(), resType, pred, lhs, rhs);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
            // todo: support for unsigned integer comparisons in hivm.cmp.
        }
        return success();
    }
};

/*
* vsel -> arith.select
* the type of all operand are tensor or vector.
* the attribute of broadcast is empty.
* the attribute of transpose is empty.
* The type of the condition operand must be i1, which is a constraint of the arith.select op.
*/
struct HIVMToArithSelOp: public OpRewritePattern<hivm::VSelOp> {
    using OpRewritePattern<hivm::VSelOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VSelOp op,
                                PatternRewriter &rewriter) const final {
        if (!transpose_check(op)) {
            return failure();
        }

        // conversion would be executed only as broadcast is empty.
        if (!broadcast_check(op)) {
            return failure();
        }
        if (!operateOnShaped(op)) {
            return failure();
        }

        SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
        Value condition = hivmOperands[0];
        Value trueValue = hivmOperands[1];
        Value falseValue = hivmOperands[2];
        Value dst_val = hivmOperands[3];
        auto result = rewriter.create<arith::SelectOp>(op.getLoc(), condition, trueValue, falseValue);
        dst_val.replaceAllUsesWith(result.getResult());
        rewriter.replaceOp(op, result);
        return success();
    }
};

/*
* vrelu -> arith.maximumf
* All operand element types are required to be F16, F32 or I32.
* the type of all operand are tensor or vector.
* the attribute of broadcast is empty.
* the attribute of transpose is empty.
* the element type of all operand are f32 or f16 or i32.
* Currently, the vrelu operator only supports signless i32 integers.
* It cannot distinguish between signed and unsigned operations based on the operand type,
* nor does it carry additional sign-related attribute information.
* When the operand type of the vrelu operator is not ui32, the vrelu op is uniformly converted to the maxsi op.
*/
struct HIVMToArithReluOp: public OpRewritePattern<hivm::VReluOp> {
    using OpRewritePattern<hivm::VReluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VReluOp op,
                                PatternRewriter &rewriter) const final {
        if (!transpose_check(op)) {
            return failure();
        }

        // conversion would be executed only as broadcast is empty.
        if (!broadcast_check(op)) {
            return failure();
        }
        if (!operateOnShaped(op)) {
            return failure();
        }

        SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
        Value src = hivmOperands[0];
        Value dst_val = hivmOperands[1];
        auto tensorType = cast<RankedTensorType>(src.getType());
        Type elem_type = getElementTypeOrSelf(src.getType());
        auto const_type = RankedTensorType::get(tensorType.getShape(), elem_type);
        DenseElementsAttr onesAttr;
        assert(elem_type.isF16() || elem_type.isF32() || elem_type.isSignlessInteger(32));
        if (elem_type.isF16() || elem_type.isF32()) {
            const llvm::fltSemantics &semantics = mlir::cast<FloatType>(elem_type).getFloatSemantics();
            llvm::APFloat initVal(semantics);
            initVal = llvm::APFloat::getZero(semantics);
            onesAttr = DenseElementsAttr::get(const_type, rewriter.getFloatAttr(elem_type, initVal));
            Value newConst = rewriter.create<arith::ConstantOp>(
            op.getLoc(),
                const_type,
                onesAttr
            ).getResult();
            auto resType = op.getResult().getType();
            auto result = rewriter.create<arith::MaximumFOp>(op.getLoc(), resType, newConst, src);
            dst_val.replaceAllUsesWith(result.getResult());
            rewriter.replaceOp(op, result);
        } else if (elem_type.isInteger(32)) {
            unsigned width = elem_type.getIntOrFloatBitWidth();
            llvm::APInt initVal = llvm::APInt::getZero(width);
            onesAttr = DenseElementsAttr::get(const_type, rewriter.getIntegerAttr(elem_type, initVal));
            Value newConst = rewriter.create<arith::ConstantOp>(
            op.getLoc(),
                const_type,
                onesAttr
            ).getResult();
            auto resType = op.getResult().getType();
            // to do: The vrelu operand in the hivm dialect does not currently support unsigned integer types.
            if (elem_type.isUnsignedInteger()) {
                auto result = rewriter.create<arith::MaxUIOp>(op.getLoc(), resType, newConst, src);
                dst_val.replaceAllUsesWith(result.getResult());
                rewriter.replaceOp(op, result);
            } else {
                auto result = rewriter.create<arith::MaxSIOp>(op.getLoc(), resType, newConst, src);
                dst_val.replaceAllUsesWith(result.getResult());
                rewriter.replaceOp(op, result);
            }
        }
        return success();
    }
};

/*
* vrec -> arith.divf
* The type of all operand are tensor or vector.
* The attribute of broadcast is empty.
* The attribute of transpose is empty.
* All operand element types are required to be F16 or F32.
*/
struct HIVMToArithRecOp: public OpRewritePattern<hivm::VRecOp> {
    using OpRewritePattern<hivm::VRecOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VRecOp op,
                                PatternRewriter &rewriter) const final {
        if (!transpose_check(op)) {
            return failure();
        }

        // conversion would be executed only as broadcast is empty.
        if (!broadcast_check(op)) {
            return failure();
        }
        if (!operateOnShaped(op)) {
            return failure();
        }
        SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
        Value src = hivmOperands[0];
        Value dst_val = hivmOperands[1];
        auto tensorType = cast<RankedTensorType>(src.getType());
        Type elem_type = getElementTypeOrSelf(src.getType());
        auto const_type = RankedTensorType::get(tensorType.getShape(), elem_type);
        DenseElementsAttr onesAttr;
        assert(elem_type.isF16() || elem_type.isF32());
        const llvm::fltSemantics &semantics = mlir::cast<FloatType>(elem_type).getFloatSemantics();
        llvm::APFloat initVal(semantics);
        initVal = llvm::APFloat::getOne(semantics);
        onesAttr = DenseElementsAttr::get(const_type, rewriter.getFloatAttr(elem_type, initVal));
        Value newConst = rewriter.create<arith::ConstantOp>(
            op.getLoc(),
            const_type,
            onesAttr
        ).getResult();
        auto resType = op.getResult().getType();
        auto result = rewriter.create<arith::DivFOp>(op.getLoc(), resType, newConst, src);
        dst_val.replaceAllUsesWith(result.getResult());
        rewriter.replaceOp(op, result);
        return success();
    }
};

/*
* vmulext -> arith.mulsi_extended
* The type of all operand are tensor or vector.
* The attribute of broadcast is empty.
* The attribute of transpose is empty.
* At present, it is not possible to distinguish the symbolic information of hivm.hir.vmulext,
* and it can only be uniformly converted to arith.mulsi_extended.
* In the Triton DSL, there is only one interface umulhi; after TA conversion,
* it generates arith.mulsi_extended, and no usage scenarios for arith.mului_extended have been identified.
* arith.mului_extended is only generated after the execution of the arith-emulate-wide-int pass;
* this pass has not been found to be invoked in SIMD pipelines.
*/
struct HIVMToArithMulExtOp: public OpRewritePattern<hivm::VMulExtOp> {
    using OpRewritePattern<hivm::VMulExtOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VMulExtOp op,
                                PatternRewriter &rewriter) const final {
        if (!transpose_check(op)) {
            return failure();
        }

        // conversion would be executed only as broadcast is empty.
        if (!broadcast_check(op)) {
            return failure();
        }
        if (!operateOnShaped(op)) {
            return failure();
        }
        SmallVector<Value> hivmOperands = getHIVMVectorOperands(op);
        Value lhs = hivmOperands[0];
        Value rhs = hivmOperands[1];
        Value low_res = hivmOperands[2];
        Value high_res = hivmOperands[3];
        auto result = rewriter.create<arith::MulSIExtendedOp>(
            op.getLoc(),
            op.getResult().getType()[0],
            op.getResult().getType()[1],
            lhs,
            rhs
        );
        low_res.replaceAllUsesWith(result.getLow());
        high_res.replaceAllUsesWith(result.getHigh());
        rewriter.replaceOp(op, result);
        return success();
    }
};

void mlir::hivm::populateHIVMToArithConversionPatterns(RewritePatternSet &patterns) {
    patterns.add<
        VectorOpToArithBinary<arith::AndIOp, hivm::VAndOp, true, false, IntegerType::Signless, IntegerType>,
        VectorOpToArithBinary<arith::OrIOp, hivm::VOrOp, true, false, IntegerType::Signless, IntegerType>,
        VectorOpToArithBinary<arith::AddFOp, hivm::VAddOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToArithBinary<arith::AddIOp, hivm::VAddOp, true, false, IntegerType::Signless, IntegerType>,
        VectorOpToArithBinary<arith::SubFOp, hivm::VSubOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToArithBinary<arith::SubIOp, hivm::VSubOp, true, false, IntegerType::Signless, IntegerType>,
        VectorOpToArithBinary<arith::MulFOp, hivm::VMulOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToArithBinary<arith::MulIOp, hivm::VMulOp, true, false, IntegerType::Signless, IntegerType>,
        VectorOpToArithBinary<arith::DivFOp, hivm::VDivOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToArithBinary<arith::DivSIOp, hivm::VDivOp, true, true, IntegerType::Signed, IntegerType>,
        VectorOpToArithBinary<arith::DivUIOp, hivm::VDivOp, true, true, IntegerType::Unsigned, IntegerType>,
        VectorOpToArithBinary<arith::MaximumFOp, hivm::VMaxOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToArithBinary<arith::MaxSIOp, hivm::VMaxOp, true, true, IntegerType::Signless, IntegerType>,
        VectorOpToArithBinary<arith::MinimumFOp, hivm::VMinOp, true, false, IntegerType::Signless, FloatType>,
        VectorOpToArithBinary<arith::MinSIOp, hivm::VMinOp, true, true, IntegerType::Signless, IntegerType>,
        VectorOpToArithBinary<arith::RemUIOp, hivm::VModUIOp, true, false, IntegerType::Signless, IntegerType>,
        HIVMToArithCastOp,
        HIVMToArithCmpOp,
        HIVMToArithBitcastOp,
        HIVMToArithSelOp,
        HIVMToArithRecOp,
        HIVMToArithReluOp,
        HIVMToArithMulExtOp
    >(patterns.getContext());
}