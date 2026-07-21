//===-------- DecomposeFRem.cpp - Decompose llvm.frem op ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <cassert>

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSEFREM
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// Decompose llvm.frem to sub(mul(trunc(div)))
// a % b = a - b * trunc( a / b )
struct FRemOpPattern : public OpRewritePattern<LLVM::FRemOp> {
public:
  using OpRewritePattern<LLVM::FRemOp>::OpRewritePattern;
  Value emitOrCreateLibCall(OpBuilder &builder, Location loc,
                            llvm::StringRef funcName, ValueRange operands,
                            Type returnType) const {
    ModuleOp module;
    if (auto *defOp = operands[0].getDefiningOp()) {
      module = defOp->getParentOfType<mlir::ModuleOp>();
    } else {
      module = operands[0].getParentRegion()->getParentOfType<mlir::ModuleOp>();
    }
    LLVM::LLVMFuncOp func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);

    if (!func) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());

      SmallVector<Type> types;
      for (auto type : operands.getTypes()) {
        types.push_back(type);
      }
      auto funcType =
          LLVM::LLVMFunctionType::get(returnType, ArrayRef<Type>(types), false);

      func = builder.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);

      auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
          hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
      func->setAttr(haccAlwaysInlineAttr, builder.getUnitAttr());
      auto llvmEmitCAttr = LLVM::LLVMDialect::getEmitCWrapperAttrName();
      func->setAttr(llvmEmitCAttr, builder.getUnitAttr());
      func->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                    builder.getStringAttr("private"));
      auto ctx = builder.getContext();
      func->setAttr(
          mlir::hivm::TFuncCoreTypeAttr::name,
          hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
      func->setAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName,
                    hivm_regbaseintrins::SIMT_CallableAttr::get(ctx));
    }

    auto calleeAttr = mlir::SymbolRefAttr::get(builder.getContext(), funcName);
    auto call =
        builder.create<LLVM::CallOp>(loc, returnType, calleeAttr, operands);
    return call.getResult();
  }

  LogicalResult matchAndRewrite(LLVM::FRemOp fremOp,
                                PatternRewriter &rewriter) const override {
    auto lhs = fremOp.getLhs();
    auto rhs = fremOp.getRhs();
    Value fdivOp;
    auto inType = llvm::cast<FloatType>(lhs.getType());
    assert(inType.getWidth() <= 32 && "simt currently only support FP32 and lower bitwidth");
    if (inType.isF32()) {
      fdivOp = emitOrCreateLibCall(rewriter, fremOp->getLoc(),
                                   "_mlir_ciface_simt_divrn_float",
                                   ValueRange{lhs, rhs}, lhs.getType());
    } else {
      fdivOp = rewriter.create<LLVM::FDivOp>(fremOp->getLoc(), lhs, rhs);
    }

    auto cst = [&](int32_t val) -> Value {
      return rewriter.create<LLVM::ConstantOp>(fremOp->getLoc(),
                                               rewriter.getI32Type(),
                                               rewriter.getI32IntegerAttr(val));
    };
    Type i32 = rewriter.getI32Type();
    auto f32 = rewriter.getF32Type();
    if(inType.getWidth() < 32){
      fdivOp = rewriter.create<LLVM::FPExtOp>(fremOp->getLoc(), f32, fdivOp);
    }

    // Bitwise Operation Simulate Trunc
    // 1. bitcast float -> i32
    Value bits =
        rewriter.create<LLVM::BitcastOp>(fremOp->getLoc(), i32, fdivOp);

    // 2. exp = (bits >> 23) & 0xFF - 127
    Value c23 = cst(23);
    Value c127 = cst(127);
    Value c255 = cst(255);

    Value exp_shifted =
        rewriter.create<LLVM::LShrOp>(fremOp->getLoc(), bits, c23);
    Value exp_biased =
        rewriter.create<LLVM::AndOp>(fremOp->getLoc(), exp_shifted, c255);
    Value exp =
        rewriter.create<LLVM::SubOp>(fremOp->getLoc(), exp_biased, c127);
    
    // 3. condition judge
    //    is_lt0   : exp < 0
    //    is_lt23  : exp < 23
    Value c0 = cst(0);
    Value is_lt0 = rewriter.create<LLVM::ICmpOp>(
        fremOp->getLoc(), LLVM::ICmpPredicate::slt, exp, c0);
    Value is_lt23 = rewriter.create<LLVM::ICmpOp>(
        fremOp->getLoc(), LLVM::ICmpPredicate::slt, exp, c23);

    // 4. case_zero
    Value sign_mask = cst(0x80000000);
    Value sign =
        rewriter.create<LLVM::AndOp>(fremOp->getLoc(), bits, sign_mask);
    Value zero_f =
        rewriter.create<LLVM::BitcastOp>(fremOp->getLoc(), f32, sign);

    // 5. case_partial
    Value mantissa = cst(0x007FFFFF);
    Value frac_mask =
        rewriter.create<LLVM::LShrOp>(fremOp->getLoc(), mantissa, exp);
    Value cm1 = cst(-1);
    Value clear_mask =
        rewriter.create<LLVM::XOrOp>(fremOp->getLoc(), frac_mask, cm1);
    Value cleared =
        rewriter.create<LLVM::AndOp>(fremOp->getLoc(), bits, clear_mask);
    Value partial_f =
        rewriter.create<LLVM::BitcastOp>(fremOp->getLoc(), f32, cleared);

    //    select is_lt23 ? partial_f : x      -> integer_or_partial
    //    select is_lt0  ? zero_f : above     -> result
    Value integer_or_partial = rewriter.create<LLVM::SelectOp>(
        fremOp->getLoc(), is_lt23, partial_f, fdivOp);
    Value result = rewriter.create<LLVM::SelectOp>(fremOp->getLoc(), is_lt0,
                                                   zero_f, integer_or_partial);
    if(inType.getWidth() < 32){
      result = rewriter.create<LLVM::FPTruncOp>(fremOp->getLoc(), lhs.getType(), result);
    }
    auto fmulOp = rewriter.create<LLVM::FMulOp>(fremOp->getLoc(), result, rhs);
    auto fsubOp = rewriter.create<LLVM::FSubOp>(fremOp->getLoc(), lhs, fmulOp);
    rewriter.replaceOp(fremOp, fsubOp);
    return success();
  }
};

struct DecomposeFRemPass : public impl::DecomposeFRemBase<DecomposeFRemPass> {
  void runOnOperation() override {
    auto mod = getOperation();
    auto context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addIllegalOp<LLVM::FRemOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    patterns.add<FRemOpPattern>(context);
    (void)applyPartialConversion(mod, target, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> bishengir::triton::createDecomposeFRemPass() {
  return std::make_unique<DecomposeFRemPass>();
}
