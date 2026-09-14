//===-- DotOpToLLVM.cpp - Dot Op to LLVM Conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/FMADotUtility.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/PatternTritonAscendGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

static constexpr llvm::StringLiteral kFMAConvertedAttr = "fma.converted";
namespace {

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  DotOpConversion(LLVMTypeConverter &converter,
                  bool enableCGroupingDotTileLowering, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        enableCGroupingDotTileLowering(enableCGroupingDotTileLowering) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // C-group lowering must own the whole group/chain of dots so it can lower
    // and interleave the FMAs more closely, reduce register lifetimes, avoid
    // intermediate pack/unpack, and preserve accumulator flow.
    if (enableCGroupingDotTileLowering && ascend::isGroupedDotAnchor(op))
      return ascend::convertGroupedFMADots(op, adaptor, getTypeConverter(),
                                           rewriter);

    if (op->hasAttr(kFMAConvertedAttr) ||
        isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(op.getResult().getType()).getEncoding())) {
      return ascend::convertFMADot(op, adaptor, getTypeConverter(), rewriter);
    }
    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

  bool enableCGroupingDotTileLowering;
};

} // namespace

void mlir::triton::ascend::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    bool enableCGroupingDotTileLowering,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter,
                                enableCGroupingDotTileLowering, benefit);
}
