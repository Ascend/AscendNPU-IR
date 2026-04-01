//===--TritonOpToDPX.cpp - Triton Op to DPX Conversion ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/TritonOpToDPX.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

struct GetProgramIdOpConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto axis = op.getAxis();

    Operation *newOp = nullptr;
    switch (axis) {
    case triton::ProgramIDDim::X:
      newOp =
          rewriter.create<ascend_dpx::BlockIdxXOp>(op.getLoc(), op.getType());
      break;
    case triton::ProgramIDDim::Y:
      newOp =
          rewriter.create<ascend_dpx::BlockIdxYOp>(op.getLoc(), op.getType());
      break;
    case triton::ProgramIDDim::Z:
      newOp =
          rewriter.create<ascend_dpx::BlockIdxZOp>(op.getLoc(), op.getType());
      break;
    }
    if (!newOp)
      return failure();
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct GetNumProgramsOpConversion
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  using OpConversionPattern<triton::GetNumProgramsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto axis = op.getAxis();

    Operation *newOp = nullptr;
    switch (axis) {
    case triton::ProgramIDDim::X:
      newOp =
          rewriter.create<ascend_dpx::GridDimXOp>(op.getLoc(), op.getType());
      break;
    case triton::ProgramIDDim::Y:
      newOp =
          rewriter.create<ascend_dpx::GridDimYOp>(op.getLoc(), op.getType());
      break;
    case triton::ProgramIDDim::Z:
      newOp =
          rewriter.create<ascend_dpx::GridDimZOp>(op.getLoc(), op.getType());
      break;
    }
    if (!newOp)
      return failure();
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};


} // namespace

namespace mlir::triton::ascend {

void populateTritonOpToDPXPatterns(LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion, GetNumProgramsOpConversion>(
      converter, patterns.getContext(), benefit);
}

} // namespace mlir::triton::ascend