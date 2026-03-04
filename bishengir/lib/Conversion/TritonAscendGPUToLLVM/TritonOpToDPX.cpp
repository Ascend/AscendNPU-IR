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

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {

  AtomicRMWOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto valElems = unpackLLElements(loc, adaptor.getVal(), rewriter);

    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        getTypeConverter()->convertType(tensorTy.getElementType());

    unsigned elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    SmallVector<Value> resultVals(elemsPerThread);
    auto rmwOp = op.getAtomicRmwOp();

    for (size_t i = 0; i < elemsPerThread; i++) {
      Value ptr = ptrElems[i];
      Value val = valElems[i];

      Value result;
      switch (rmwOp) {
      case triton::RMWOp::AND:
        result =
            rewriter.create<ascend_dpx::AtomicAndOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::OR:
        result =
            rewriter.create<ascend_dpx::AtomicOrOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::XOR:
        result =
            rewriter.create<ascend_dpx::AtomicXorOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::ADD:
      case triton::RMWOp::FADD:
        result =
            rewriter.create<ascend_dpx::AtomicAddOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::MAX:
        result =
            rewriter.create<ascend_dpx::AtomicMaxOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::MIN:
        result =
            rewriter.create<ascend_dpx::AtomicMinOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::UMAX:
        result =
            rewriter
                .create<ascend_dpx::AtomicUMaxOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::UMIN:
        result =
            rewriter
                .create<ascend_dpx::AtomicUMinOp>(loc, valueElemTy, ptr, val)
                .getRes();
        break;
      case triton::RMWOp::XCHG:
        result = rewriter
                     .create<ascend_dpx::AtomicExchangeOp>(loc, valueElemTy,
                                                           ptr, val)
                     .getRes();
        break;
      }

      resultVals[i] = result;
    }

    Value result =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, tensorTy);
    rewriter.replaceOp(op, result);
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
  patterns.add<AtomicRMWOpConversion>(converter, benefit);
}

} // namespace mlir::triton::ascend