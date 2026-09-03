//===- AppendSanitizerData.cpp - Append and use args ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to append allocated space for sanitizer.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HIVMToLLVM/AppendSanitizerData.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_APPENDSANITIZERDATA
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct AppendSanitizerData
    : public impl::AppendSanitizerDataBase<AppendSanitizerData> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Add a function argument for the mssanitizer to use
// Refer to AppendUsePrintDebugData pass
struct AppendSanitizerArg : public mlir::OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp funcOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!hacc::utils::isDevice(funcOp) ||
        funcOp->hasAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
      return rewriter.notifyMatchFailure(funcOp, "Target only kernel function");
    }

    // update ArgAttrs.
    auto &body = funcOp.getBody();
    size_t numOrigArg = funcOp.getNumArguments();
    unsigned newArgIdx = numOrigArg;
    if (hacc::utils::isKernelArg(funcOp, newArgIdx - 1,
                                 hacc::KernelArgType::kSanitizerAddr))
      return failure();

    OpBuilder opBuilder(funcOp.getContext());
    // add argument
    auto newArgType = LLVM::LLVMPointerType::get(
        rewriter.getContext(), static_cast<unsigned>(hivm::AddressSpace::GM));
    body.insertArgument(newArgIdx, newArgType, funcOp.getLoc());

    // update llvmtype.
    mlir::LLVM::LLVMFunctionType llvmType = funcOp.getFunctionType();
    llvm::ArrayRef<Type> argumentTypes = funcOp.getArgumentTypes();
    llvm::SmallVector<Type, 32> newArgumentTypes; // 32 should be enough
    for (size_t idx = 0; idx < newArgIdx; idx++) {
      newArgumentTypes.push_back(argumentTypes[idx]);
    }
    newArgumentTypes.push_back(newArgType);
    Type newllvmType = mlir::LLVM::LLVMFunctionType::get(
        llvmType.getReturnType(), llvm::ArrayRef(newArgumentTypes),
        llvmType.getVarArg());
    funcOp.setType(newllvmType);

    // update ArgAttrs.
    NamedAttribute sanitizerAddrAttr = hacc::createHACCKernelArgAttr(
        opBuilder.getContext(), hacc::KernelArgType::kSanitizerAddr);
    DictionaryAttr sanitizerDictAttrs = opBuilder.getDictionaryAttr(
        SmallVector<NamedAttribute>{sanitizerAddrAttr});

    SmallVector<Attribute> newArgAttrs(numOrigArg + 1);

    auto oldArgAttrs = funcOp.getAllArgAttrs();
    if (!oldArgAttrs) {
      for (unsigned j = 0; j < numOrigArg; ++j)
        newArgAttrs[j] = DictionaryAttr::get(rewriter.getContext(), {});
    } else {
      for (unsigned j = 0; j < numOrigArg; ++j)
        newArgAttrs[j] = oldArgAttrs[j];
    }

    newArgAttrs[newArgIdx] = sanitizerDictAttrs;
    funcOp.setAllArgAttrs(rewriter.getArrayAttr(newArgAttrs));

    return success();
  }
};

void AppendSanitizerData::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.insert<AppendSanitizerArg>(patterns.getContext());
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createAppendSanitizerDataPass() {
  return std::make_unique<AppendSanitizerData>();
}
