//===--GPUOpToDPX.cpp - GPU Op to DPX Conversion ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/GPUToDPX/GPUOpToDPX.h"
#include "bishengir/Conversion/GPUToDPX/IndexIntrinsicsOpLowering.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

struct BarrierOPConversion : public OpConversionPattern<gpu::BarrierOp> {
  using OpConversionPattern<gpu::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ascend_dpx::SyncThreadsOp>(op);
    return success();
  }
};

namespace mlir::triton::ascend {
void populateGPUOpToDPXPatterns(LLVMTypeConverter &converter,
                                RewritePatternSet &patterns,
                                PatternBenefit benefit) {
  patterns.add<
      gpu::index_lowering::OpLowering<gpu::ThreadIdOp, ascend_dpx::ThreadIdXOp,
                                      ascend_dpx::ThreadIdYOp,
                                      ascend_dpx::ThreadIdZOp>,
      gpu::index_lowering::OpLowering<gpu::BlockDimOp, ascend_dpx::BlockDimXOp,
                                      ascend_dpx::BlockDimYOp,
                                      ascend_dpx::BlockDimZOp>,
      gpu::index_lowering::OpLowering<gpu::BlockIdOp, ascend_dpx::BlockIdxXOp,
                                      ascend_dpx::BlockIdxYOp,
                                      ascend_dpx::BlockIdxZOp>>(converter);
  patterns.add<BarrierOPConversion>(converter, patterns.getContext(), benefit);
}
} // namespace mlir::triton::ascend