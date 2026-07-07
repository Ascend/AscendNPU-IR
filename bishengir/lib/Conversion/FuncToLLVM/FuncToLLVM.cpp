//===- FuncToLLVM.cpp - Func to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR Func and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

static constexpr StringRef barePtrAttrName = "llvm.bareptr";

/// Return `true` if the `op`'s argument at `argIdx` should use bare pointer
/// calling convention.
static bool
inputArgumentShouldUseBarePtrCallConv(FunctionOpInterface op, unsigned argIdx,
                                      const LLVMTypeConverter *typeConverter) {
  assert(typeConverter->getOptions().onDemandBarePtrCallConv);
  return op && op.getArgAttr(argIdx, barePtrAttrName);
}

static SmallVector<bool>
inputArgumentsShouldUseBarePtrCallConv(FunctionOpInterface op,
                                       const LLVMTypeConverter *typeConverter) {
  auto funcTy = cast<FunctionType>(op.getFunctionType());
  return llvm::map_to_vector(llvm::seq<unsigned>(0, funcTy.getNumInputs()),
                             [&](unsigned argIdx) -> bool {
                               return inputArgumentShouldUseBarePtrCallConv(
                                   op, argIdx, typeConverter);
                             });
}

static void modifyFuncOpToUseBarePtrCallingConv(
    ConversionPatternRewriter &rewriter, Location loc,
    const LLVMTypeConverter &typeConverter, LLVM::LLVMFuncOp funcOp,
    TypeRange oldArgTypes,
    const SmallVector<bool> &useBarePtrCallConvForArguments) {
  if (funcOp.getBody().empty())
    return;

  // Promote bare pointers from memref arguments to memref descriptors at the
  // beginning of the function so that all the memrefs in the function have a
  // uniform representation.
  Block *entryBlock = &funcOp.getBody().front();
  auto blockArgs = entryBlock->getArguments();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(entryBlock);

  size_t blockArgOffset = 0;
  for (auto it : llvm::zip(useBarePtrCallConvForArguments, oldArgTypes)) {
    bool isBarePtrCallConv = std::get<0>(it);
    Type argTy = std::get<1>(it);

    // Unranked memrefs are not supported in the bare pointer calling
    // convention. We should have bailed out before in the presence of
    // unranked memrefs.
    assert(!isa<UnrankedMemRefType>(argTy) &&
           "Unranked memref is not supported");
    auto memrefTy = dyn_cast<MemRefType>(argTy);
    if (!memrefTy) {
      assert(!isBarePtrCallConv);
      blockArgOffset++;
      continue;
    }

    if (!isBarePtrCallConv) {
      // For memrefs that are not converted to bare pointers, advance by the
      // size of the memref descriptor struct.
      blockArgOffset += MemRefDescriptor::getNumUnpackedValues(memrefTy);
      continue;
    }

    assert(blockArgOffset < blockArgs.size());
    auto arg = blockArgs[blockArgOffset];
    // Replace barePtr with a placeholder (undef), promote barePtr to a ranked
    // or unranked memref descriptor and replace placeholder with the last
    // instruction of the memref descriptor.
    // TODO: The placeholder is needed to avoid replacing barePtr uses in the
    // MemRef descriptor instructions. We may want to have a utility in the
    // rewriter to properly handle this use case.
    auto placeholder = rewriter.create<LLVM::UndefOp>(
        loc, typeConverter.convertType(memrefTy));
    rewriter.replaceUsesOfBlockArgument(arg, placeholder);

    Value desc = MemRefDescriptor::fromStaticShape(rewriter, loc, typeConverter,
                                                   memrefTy, arg);
    rewriter.replaceOp(placeholder, {desc});

    blockArgOffset++;
  }
}

namespace {

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
///
/// Similar to the `FuncOpConversion` pattern, except that it supports lowering
/// with on-demand bare ptr call convention (Only argument/results with
/// `llvm.bareptr` attributes will be lowered using the bare-ptr convention)
struct FuncOpWithOnDemandBarePtrConversion
    : public ConvertOpToLLVMPattern<func::FuncOp> {
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  // This pattern is favored over the regular FuncOpConversion Pattern.
  explicit FuncOpWithOnDemandBarePtrConversion(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(
            converter, /*benefit*/ 10) { // use 10 high benefits above all
  }

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  FailureOr<LLVM::LLVMFuncOp>
  convertFuncOpToLLVMFuncOp(func::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    return mlir::convertFuncOpToLLVMFuncOp(
        cast<FunctionOpInterface>(funcOp.getOperation()), rewriter,
        *getTypeConverter());
  }

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!getTypeConverter()->getOptions().onDemandBarePtrCallConv)
      return rewriter.notifyMatchFailure(
          funcOp,
          "Can only convert with on-demand-bare-ptr-memref-call-conv option");

    if (getTypeConverter()->getOptions().useBarePtrCallConv)
      return rewriter.notifyMatchFailure(
          funcOp, "When lowering with bare ptr calling convention, all "
                  "inputs/results are lowered in the same way.");

    auto shouldUseBarePtrCallConv =
        inputArgumentsShouldUseBarePtrCallConv(funcOp, getTypeConverter());
    if (llvm::all_of(shouldUseBarePtrCallConv, [](bool v) { return !v; }))
      return rewriter.notifyMatchFailure(
          funcOp,
          "Couldn't find any arguments/results to apply on-demand bare ptr "
          "calling convention. Lower using the regular func to llvm pattern.");

    FailureOr<LLVM::LLVMFuncOp> newFuncOp =
        convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (failed(newFuncOp))
      return rewriter.notifyMatchFailure(funcOp, "Could not convert func op");

    modifyFuncOpToUseBarePtrCallingConv(
        rewriter, funcOp->getLoc(), *getTypeConverter(), *newFuncOp,
        funcOp.getFunctionType().getInputs(), shouldUseBarePtrCallConv);

    rewriter.eraseOp(funcOp);
    return success();
  }
};

} // namespace

void bishengir::populateFuncToLLVMFuncOpConversionPattern(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<FuncOpWithOnDemandBarePtrConversion>(converter);
}
