//===- AppendUsePrintDebugData.cpp - Append and use args -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to append and use print debug data.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HIVMToLLVM/AppendUsePrintDebugData.h"
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
#define GEN_PASS_DEF_APPENDUSEPRINTDEBUGDATA
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "append-use-print-debug-data"

static constexpr StringRef printArgAttrName = "print_arg_idx";

namespace {
struct AppendUsePrintDebugData
    : public impl::AppendUsePrintDebugDataBase<AppendUsePrintDebugData> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// TODO: Fix func call-site on host side
// TODO: Fix multi dev func and the case where one dev func calls another
struct AppendPrintDebugData : public mlir::OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp llvmFuncOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!hacc::utils::isDevice(llvmFuncOp)) {
      return rewriter.notifyMatchFailure(llvmFuncOp,
                                         "Target only kernel function");
    }

    // Check if already set
    if (llvmFuncOp->getAttr(printArgAttrName)) {
      return failure();
    }
    // Only when _mlir_ciface_init_debug* or _mlir_ciface_finish_debug* exists,
    // append argument to the kernel.
    bool printHelperExist = false;
    llvmFuncOp.walk([&](Operation *op) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
        auto symName = callOp.getCallee().value();
        if (symName.starts_with("_mlir_ciface_init_debug") ||
            symName.starts_with("_mlir_ciface_finish_debug")) {
          printHelperExist = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (!printHelperExist) {
      return rewriter.notifyMatchFailure(
          llvmFuncOp, "No act for kernel without calling helpers of print");
    }

    size_t numOrigArg = llvmFuncOp.getArguments().size();
    for (unsigned i = 0; i < numOrigArg; i++) {
      assert(!hacc::utils::isKernelArg(llvmFuncOp, i,
                                       hacc::KernelArgType::kSanitizerAddr) &&
             "sanitizer & device print can't be enabled at the same time!");
    }

    auto &body = llvmFuncOp.getBody();
    unsigned newArgIdx = numOrigArg;
    auto newArgType = LLVM::LLVMPointerType::get(
        rewriter.getContext(), static_cast<unsigned>(hivm::AddressSpace::GM));
    body.insertArgument(newArgIdx, newArgType, llvmFuncOp.getLoc());

    // update llvmtype.
    mlir::LLVM::LLVMFunctionType llvmType = llvmFuncOp.getFunctionType();
    llvm::ArrayRef<Type> argumentTypes = llvmFuncOp.getArgumentTypes();
    llvm::SmallVector<Type, 32> newArgumentTypes; // 32 should be enough
    for (size_t idx = 0; idx < newArgIdx; idx++) {
      newArgumentTypes.push_back(argumentTypes[idx]);
    }
    newArgumentTypes.push_back(newArgType);
    Type newllvmType = mlir::LLVM::LLVMFunctionType::get(
        llvmType.getReturnType(), llvm::ArrayRef(newArgumentTypes),
        llvmType.getVarArg());
    llvmFuncOp.setType(newllvmType);

    // update ArgAttrs.
    unsigned numParam =
        cast<LLVM::LLVMFunctionType>(newllvmType).getNumParams();
    SmallVector<Attribute> newArgAttrs(numParam);
    auto oldArgAttrs = llvmFuncOp.getAllArgAttrs();
    if (!oldArgAttrs) {
      for (unsigned j = 0; j < numParam; ++j)
        newArgAttrs[j] = DictionaryAttr::get(rewriter.getContext(), {});
    } else {
      for (unsigned j = 0; j < oldArgAttrs.size(); ++j)
        newArgAttrs[j] = oldArgAttrs[j];
      newArgAttrs[numParam - 1] =
          DictionaryAttr::get(rewriter.getContext(), {});
    }
    llvmFuncOp.setAllArgAttrs(rewriter.getArrayAttr(newArgAttrs));

    llvmFuncOp->setAttr(printArgAttrName,
                        rewriter.getI64IntegerAttr(newArgIdx));

    return success();
  }
};

struct UsePrintDebugData : public mlir::OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(LLVM::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto symName = callOp.getCallee().value();
    if (!(symName.starts_with("_mlir_ciface_init_debug") ||
          symName.starts_with("_mlir_ciface_finish_debug"))) {
      return failure();
    }

    unsigned numOrigArg = callOp.getNumOperands();
    // Check if already set
    if (numOrigArg > 0) {
      return failure();
    }
    auto kernelFuncOp = callOp->getParentOfType<LLVM::LLVMFuncOp>();
    assert(kernelFuncOp->hasAttr(printArgAttrName));
    auto printArgIdx =
        kernelFuncOp->getAttrOfType<IntegerAttr>(printArgAttrName).getInt();
    assert(printArgIdx < kernelFuncOp.getBody().getNumArguments());
    auto printArg = kernelFuncOp.getBody().getArgument(printArgIdx);

    OpBuilder builder(callOp);
    auto funcOp =
        mlir::utils::getCalledFunction<LLVM::LLVMFuncOp, LLVM::CallOp>(callOp);
    LLVM::LLVMFunctionType funcType = funcOp.getFunctionType();
    builder.setInsertionPoint(funcOp);
    auto newFuncOp =
        cast<LLVM::LLVMFuncOp>(builder.clone(*(funcOp.getOperation())));
    newFuncOp.setType(LLVM::LLVMFunctionType::get(funcType.getReturnType(),
                                                  {printArg.getType()}));
    rewriter.replaceOp(funcOp, newFuncOp);

    auto newCallOp =
        rewriter.create<LLVM::CallOp>(callOp.getLoc(), newFuncOp, printArg);
    rewriter.replaceOp(callOp, newCallOp);

    return success();
  }
};

void AppendUsePrintDebugData::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.insert<AppendPrintDebugData>(patterns.getContext());
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
  // UsePrintDebugData must be run after AppendPrintDebugData
  {
    RewritePatternSet patternsLater(&getContext());
    patternsLater.insert<UsePrintDebugData>(patternsLater.getContext());
    if (failed(applyPatternsGreedily(op, std::move(patternsLater)))) {
      signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> mlir::createAppendUsePrintDebugDataPass() {
  return std::make_unique<AppendUsePrintDebugData>();
}
