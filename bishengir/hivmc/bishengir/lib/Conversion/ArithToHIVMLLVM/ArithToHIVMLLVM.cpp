//===-- ArithToHIVMLLVM.cpp - Code to convert arith select into llvm ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file deals with operations transformed by arith-to-llvm that does not
// the requirement of hivm backend, therefore, perform special partial
// conversion from arith to llvm. Please use this pass before ArithToLLVM pass.
// Supported ops are the following:
//
// 1. arith::select operations that have memref type as operand and convert them
// into llvm select that uses pointers as operands
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToHIVMLLVM/ArithToHIVMLLVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOHIVMLLVMCONVERSIONPASS
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "convert-arith-to-hivm-llvm"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// SelectOpLowering
//===----------------------------------------------------------------------===//
struct SelectOpLowering : public ConvertOpToLLVMPattern<arith::SelectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

LogicalResult
SelectOpLowering::matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  // Check if select result is memref
  Type resultType = op.getResult().getType();
  auto memrefResult = dyn_cast<MemRefType>(resultType);
  if (!memrefResult) {
    LLVM_DEBUG(llvm::dbgs() << "Select non-memref operands\n");
    return failure();
  }

  // Get memref descriptor of select's trueValue and falseValue
  MemRefDescriptor trueDescriptor(adaptor.getTrueValue());
  MemRefDescriptor falseDescriptor(adaptor.getFalseValue());

  // Obtain the pointer of true and false value, get select condition
  Location loc = op->getLoc();
  Value condition = adaptor.getCondition();
  Value trueAllocatedPtr = trueDescriptor.allocatedPtr(rewriter, loc);
  Value trueAlignedPtr = trueDescriptor.alignedPtr(rewriter, loc);
  Value falseAllocatedPtr = falseDescriptor.allocatedPtr(rewriter, loc);
  Value falseAlignedPtr = falseDescriptor.alignedPtr(rewriter, loc);

  // Create new selectOp
  Value resAllocatedPtr = rewriter.create<LLVM::SelectOp>(
      loc, condition, trueAllocatedPtr, falseAllocatedPtr);
  Value resAlignedPtr = rewriter.create<LLVM::SelectOp>(
      loc, condition, trueAlignedPtr, falseAlignedPtr);

  // Build memref descriptor from the result
  auto convertedType = typeConverter->convertType(memrefResult);
  auto resDescr = MemRefDescriptor::undef(rewriter, loc, convertedType);

  resDescr.setAllocatedPtr(rewriter, loc, resAllocatedPtr);
  resDescr.setAlignedPtr(rewriter, loc, resAlignedPtr);
  auto offset = trueDescriptor.offset(rewriter, loc);
  resDescr.setOffset(rewriter, loc, offset);
  for (int64_t i = 0, e = memrefResult.getRank(); i != e; ++i) {
    auto size = trueDescriptor.size(rewriter, loc, i);
    auto stride = trueDescriptor.stride(rewriter, loc, i);
    resDescr.setSize(rewriter, loc, i, size);
    resDescr.setStride(rewriter, loc, i, stride);
  }

  // Replace old selectOp with new result desctiptor
  rewriter.replaceOp(op, {resDescr});
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
namespace {
struct ArithToHIVMLLVMPass
    : public impl::ArithToHIVMLLVMConversionPassBase<ArithToHIVMLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());

    LowerToLLVMOptions options(&getContext());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter converter(&getContext(), options);

    // Patterns for lowering HIVM address space.
    mlir::hivm::populateHIVMAddressSpaceAttributeConversions(converter);

    RewritePatternSet patterns(&getContext());
    mlir::arith::populateArithToHIVMLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::arith::populateArithToHIVMLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<SelectOpLowering>(converter);
}

std::unique_ptr<Pass> mlir::createArithToHIVMLLVMConversionPass() {
  return std::make_unique<ArithToHIVMLLVMPass>();
}
