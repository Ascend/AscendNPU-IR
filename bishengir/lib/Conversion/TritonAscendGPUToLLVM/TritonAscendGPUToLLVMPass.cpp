//===--TritonAscendGPUToLLVMPass.cpp - AscendGPU Conversions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/TritonAscendGPUToLLVM.h"

#include "bishengir/Analysis/AscendAllocation.h"
#include "bishengir/Conversion/GPUToDPX/GPUOpToDPX.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/LoadStoreOpToLLVM.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/PatternTritonAscendGPUOpToLLVM.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/TargetInfo.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/TritonOpToDPX.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

#define GEN_PASS_DEF_CONVERTTRITONASCENDGPUTOLLVM
#include "bishengir/Conversion/Passes.h.inc"

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<ascend_dpx::AscendDPXDialect>();
    addIllegalDialect<mlir::arith::ArithDialect>();
    addIllegalDialect<mlir::math::MathDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonAscendGPUToLLVMPass
    : public impl::ConvertTritonAscendGPUToLLVMBase<
          ConvertTritonAscendGPUToLLVMPass> {

  void runOnOperation() override {
    MLIRContext *const context = &getContext();
    ModuleOp mod = getOperation();
    if (!triton::util::getPassColumnDigit(mod, "convert-triton-gpu-to-llvm")) {
      return;
    }

    // Higher benefit to prioritize these patterns during conversion
    constexpr int kDefaultPatternBenefit = 10;
    ascend::TargetInfo targetInfo{};
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Run shared-memory allocation and barrier insertion before the func op
    // conversion. Func conversion rewrites memdesc-typed arguments through the
    // type converter, leaving them re-materialized via
    // `builtin.unrealized_conversion_cast` — which SharedMemoryAliasAnalysis
    // does not recognize as a memdesc producer and asserts on. Matching the
    // NVIDIA pipeline's ordering avoids that.
    ModuleAllocation allocation(
        mod, mlir::triton::ascend::AscendAllocationAnalysisScratchSizeFn,
        mlir::triton::ascend::AscendAllocationSharedMemCheckFn);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Lower functions
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, kDefaultPatternBenefit);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // Subsequent passes may require shared-memory usage, so it is important to
    // initialize a global object to facilitate shared memory. Do this before
    // any of the conversion passes run.
    initSharedMemory(typeConverter);

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    TritonLLVMConversionTarget convTarget(*context);
    RewritePatternSet patterns(context);
    mlir::triton::populateMakeRangeOpToLLVMPattern(
        typeConverter, targetInfo, patterns, kDefaultPatternBenefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, kDefaultPatternBenefit);
    mlir::triton::populateScanOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, kDefaultPatternBenefit);
    mlir::triton::populateGatherOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, kDefaultPatternBenefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, kDefaultPatternBenefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               kDefaultPatternBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(
        typeConverter, patterns, targetInfo, kDefaultPatternBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(
        typeConverter, patterns, targetInfo, kDefaultPatternBenefit);
    triton::ascend::populateGPUOpToDPXPatterns(typeConverter, patterns,
                                               kDefaultPatternBenefit);
    triton::ascend::populateTritonOpToDPXPatterns(typeConverter, patterns,
                                                  kDefaultPatternBenefit);
    // Compute capability 61 means devices do not support MMA
    triton::ascend::populateDotOpToLLVMPatterns(typeConverter, patterns,
                                        kDefaultPatternBenefit);
    triton::ascend::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis,
        kDefaultPatternBenefit);
    triton::ascend::populateAscendReduceOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, kDefaultPatternBenefit);
    triton::ascend::populateAscendElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo,
        kDefaultPatternBenefit);
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           kDefaultPatternBenefit);
    // Fractal shared memory patterns override upstream memory ops.
    // Higher benefit ensures these match first when the memdesc uses
    // FractalSharedEncodingAttr (ttgext dialect).
    triton::ascend::populateFractalMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, kDefaultPatternBenefit + 1);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Need to preserve tensor types until after other conversions are done
    TritonLLVMConversionTarget cfTarget(*context);
    cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             context->getLoadedDialect<cf::ControlFlowDialect>();
    });
    RewritePatternSet cfPatterns(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          cfPatterns);
    if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
      return signalPassFailure();
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        static_cast<unsigned>(ascend_dpx::AscendDPXAddressSpace::SHARED_MEM));
  }
};

std::unique_ptr<Pass> createConvertTritonAscendGPUToLLVMPass() {
  return std::make_unique<ConvertTritonAscendGPUToLLVMPass>();
}

namespace ascend {

#define GEN_PASS_DEF_ALLOCATEASCENDSHAREDMEMORY
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h.inc"

struct AllocateAscendSharedMemory
    : public impl::AllocateAscendSharedMemoryBase<AllocateAscendSharedMemory> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(
        mod, mlir::triton::ascend::AscendAllocationAnalysisScratchSizeFn,
        mlir::triton::ascend::AscendAllocationSharedMemCheckFn);

    // Memory allocation needs to run anyways, but membar analysis is for DPX
    // path only.
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);

    // Multiply memory allocated by superblock factor
    // If superblocking is turned off (superblock factor is 1), does nothing
    unsigned superBlockFactor = 1;
    if (auto superBlockFactorAttr = mod->getAttrOfType<IntegerAttr>(
            triton::gpu::AttrSuperBlockFactor))
      superBlockFactor = superBlockFactorAttr.getUInt();
    
    size_t newSharedMemAmount = allocation.getSharedMemorySize();
    
    if (superBlockFactor > 1) {
      // Pad memory for each block up to the nearest multiple of 16
      newSharedMemAmount = ((newSharedMemAmount + 15) / 16 * 16) * superBlockFactor;
    }
    mod->setAttr("ttg.shared",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 32), newSharedMemAmount));

    if (triton::util::getPassColumnDigit(mod, "convert-triton-gpu-to-llvm")) {
      ModuleMembarOrFenceAnalysis<MembarAnalysis> analyzer(&allocation);
      analyzer.run();
    }
  }
};

} // namespace ascend

} // namespace mlir::triton
