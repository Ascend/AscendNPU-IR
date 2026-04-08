//===- FractalMemoryOpToLLVM.cpp - Fractal shared memory lowering -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Custom lowering for local_store / local_load / local_alloc when the shared
// memory descriptor uses a FractalSharedEncodingAttr (from the ttgext dialect).
//
// The upstream TritonGPU patterns call toLinearLayout(memDescTy) which
// dispatches through TritonGPUDialect::toLinearLayout(). That dispatch chain
// does not know about ttgext attributes and will assert. These patterns
// intercept those ops, compute the fractal linear layout directly via
// fractalSharedToLinearLayout(), and delegate the actual load/store emission
// to the same lowerLocalLdSt helper used by the upstream code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

#include "bishengir/Dialect/TritonExt/IR/FractalLinearLayoutConversions.h"
#include "bishengir/Dialect/TritonExt/IR/TritonExtAttrs.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

/// Check if the memdesc uses a fractal shared encoding from the ttgext dialect.
static bool hasFractalEncoding(MemDescType memDescTy) {
  return isa<bishengir::triton_ext::FractalSharedEncodingAttr>(
      memDescTy.getEncoding());
}

/// Compute the linear layout for a fractal-encoded memdesc.
/// Mirrors the logic in toLinearLayout(MemDescType) but calls our own
/// fractalSharedToLinearLayout instead of the generic dispatch.
static LinearLayout
getFractalSharedLinearLayout(MemDescType memDescTy) {
  auto shape = memDescTy.getAllocShape().take_back(memDescTy.getRank());
  auto fractal =
      cast<bishengir::triton_ext::FractalSharedEncodingAttr>(
          memDescTy.getEncoding());
  return bishengir::triton_ext::fractalSharedToLinearLayout(shape, fractal);
}

/// Lower a local store into fractal shared memory.
/// This mirrors the upstream lowerLocalStore() but uses getFractalSharedLinearLayout().
static LogicalResult
lowerFractalLocalStore(Location loc, MLIRContext *ctx, Value regVal,
                       MemDescType memDescTy, SharedMemoryObject smemObj,
                       ArrayRef<Value> inVals,
                       const LLVMTypeConverter *typeConverter,
                       ConversionPatternRewriter &rewriter,
                       const TargetInfoBase &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto kBlock = str_attr("block");

  auto regLayout = toLinearLayout(regTy);
  auto sharedLayout = getFractalSharedLinearLayout(memDescTy);
  auto cvt = regLayout.invertAndCompose(sharedLayout);

  if (!cvt.isTrivialOver({kBlock}))
    return failure();

  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);
  return success();
}

/// Custom LocalStoreOp conversion for fractal shared memory.
struct FractalLocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  FractalLocalStoreOpConversion(const LLVMTypeConverter &converter,
                                const TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memDescTy = cast<MemDescType>(op.getDst().getType());
    if (!hasFractalEncoding(memDescTy))
      return failure();

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto typeConverter = getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getDst(), llvmElemTy, rewriter);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    if (failed(lowerFractalLocalStore(loc, ctx, op.getSrc(), memDescTy, smemObj,
                                      inVals, typeConverter, rewriter,
                                      targetInfo)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

/// Custom LocalLoadOp conversion for fractal shared memory.
struct FractalLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
  FractalLocalLoadOpConversion(const LLVMTypeConverter &converter,
                               const TargetInfoBase &targetInfo,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memDescTy = cast<MemDescType>(op.getSrc().getType());
    if (!hasFractalEncoding(memDescTy))
      return failure();

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto typeConverter = getTypeConverter();
    auto regTy = cast<RankedTensorType>(op.getResult().getType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), llvmElemTy, rewriter);

    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto kBlock = str_attr("block");

    auto regLayout = toLinearLayout(regTy);
    auto sharedLayout = getFractalSharedLinearLayout(memDescTy);
    auto cvt = regLayout.invertAndCompose(sharedLayout);

    if (!cvt.isTrivialOver({kBlock}))
      return failure();

    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op);

    Value result =
        packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

/// Custom LocalAllocOp conversion for fractal shared memory.
/// Handles the case where LocalAllocOp has an initial tensor that gets stored.
struct FractalLocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  FractalLocalAllocOpConversion(const LLVMTypeConverter &converter,
                                const TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();

    auto memDescTy = cast<MemDescType>(op.getType());
    if (!hasFractalEncoding(memDescTy))
      return failure();

    auto loc = op->getLoc();
    Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                               op.getOperation());
    auto typeConverter = getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy,
                                      memDescTy.getRank(), loc, rewriter);

    if (op.getSrc()) {
      auto *ctx = op.getContext();
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      if (failed(lowerFractalLocalStore(loc, ctx, op.getSrc(), memDescTy,
                                        smemObj, inVals, typeConverter,
                                        rewriter, targetInfo)))
        return failure();
    }

    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // anonymous namespace

namespace mlir::triton::ascend {

void populateFractalMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfoBase &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit) {
  patterns.add<FractalLocalStoreOpConversion>(typeConverter, targetInfo,
                                              benefit);
  patterns.add<FractalLocalLoadOpConversion>(typeConverter, targetInfo,
                                             benefit);
  patterns.add<FractalLocalAllocOpConversion>(typeConverter, targetInfo,
                                              benefit);
}

} // namespace mlir::triton::ascend
