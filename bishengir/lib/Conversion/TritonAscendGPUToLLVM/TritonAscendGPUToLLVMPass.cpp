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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
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
#include "llvm/Support/Casting.h"

#include <numeric>

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
    addIllegalOp<ascend_dpx::PrintOp>();
    addIllegalOp<ascend_dpx::AssertOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonAscendGPUToLLVMPass
    : public impl::ConvertTritonAscendGPUToLLVMBase<
          ConvertTritonAscendGPUToLLVMPass> {
  using Base::Base;

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
    triton::ascend::populateDebugOpToLLVMPattern(
        typeConverter, patterns, targetInfo, kDefaultPatternBenefit);
    triton::ascend::populateGPUOpToDPXPatterns(typeConverter, patterns,
                                               kDefaultPatternBenefit);
    triton::ascend::populateTritonOpToDPXPatterns(typeConverter, patterns,
                                                  kDefaultPatternBenefit);
    // Compute capability 61 means devices do not support MMA
    triton::ascend::populateDotOpToLLVMPatterns(typeConverter, patterns,
                                        this->enableCGroupingDotTileLowering,
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
    if (mod.lookupSymbol("global_smem"))
      return;

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

std::unique_ptr<Pass> createConvertTritonAscendGPUToLLVMPass(
    const ConvertTritonAscendGPUToLLVMOptions &options) {
  return std::make_unique<ConvertTritonAscendGPUToLLVMPass>(options);
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

namespace ascend {

#define GEN_PASS_DEF_ERASETRITONDEBUGOPS
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h.inc"

struct EraseTritonDebugOps
    : public impl::EraseTritonDebugOpsBase<EraseTritonDebugOps> {
  void runOnOperation() override {
    SmallVector<Operation *> debugOps;
    getOperation().walk([&](Operation *op) {
      if (isa<triton::PrintOp, triton::AssertOp>(op))
        debugOps.push_back(op);
    });
    for (Operation *op : debugOps)
      op->erase();
  }
};

} // namespace ascend

namespace {
triton::gpu::MemDescType
getUnswizzledSharedMemDescType(Value val, RankedTensorType rankedTy) {
  auto ctx = val.getContext();
  auto shape = rankedTy.getShape();
  auto elemType = rankedTy.getElementType();

  unsigned rank = shape.size();

  // [rank-1, rank-2, ..., 0]  →  row-major, innermost dim varies fastest
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);

  auto ctaLayout = triton::gpu::CTALayoutAttr::get(
      ctx,
      /*CTAsPerCGA=*/SmallVector<unsigned>(rank, 1),
      /*CTASplitNum=*/SmallVector<unsigned>(rank, 1),
      /*CTAOrder=*/order);

    // Unswizzled = vec=1, perPhase=1, maxPhase=1, row-major order
    auto sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
      ctx,
      /*vec=*/1,
      /*perPhase=*/1,
      /*maxPhase=*/1,
      order,
      ctaLayout);

    auto memDescType = triton::gpu::MemDescType::get(
        shape,
        elemType,
        sharedEnc,
        triton::gpu::SharedMemorySpaceAttr::get(ctx));

    return memDescType;
  }
} // namespace

namespace ascend {

#define GEN_PASS_DEF_CONVERTDEBUGOPTOASCENDDPX
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h.inc"

// SIMT tt.device_print / device_assert design.
//
// In SIMT mode only warp-0/thread-0 prints/asserts; it reads each tensor out
// of shared memory and writes it to the debug-tunnel log (print) or checks
// the assertion (assert). Tensors in registers carry a swizzled TritonGPU
// layout, so a linear read would be out of logical order. This pass therefore,
// for each tensor operand:
//   1. ttg.local_alloc into an *unswizzled* (row-major) shared encoding — all
//      threads cooperatively write their shard in natural order;
//   2. emits a block barrier (ascend_dpx.sync_threads) after the write;
//   3. hands the memdesc to ascend_dpx.print/assert, which thread-0 consumes.
// Scalars/ptrs bypass shared memory and are passed through unchanged.
//
// Sync model: there is no pipe_barrier on the SIMT path (pipe_barrier is
// no-op'd only in the M300/C310 scalar-debug template). The write is
// cooperative (each thread stores its shard) and only thread-0 reads, so a
// barrier between write and read is required for visibility. We insert it
// manually because ascend_dpx.print/assert declare no memory effects (see the
// op def TODO), so the membar in AllocateAscendSharedMemory cannot see them
// as shared readers and won't insert one. The *write-after-read* hazard
// across successive debug ops — whose buffers alias because
// AllocateAscendSharedMemory reuses the same shared offset — is handled by
// that same membar, which inserts a gpu.barrier when it reuses a slot, so no
// leading barrier is needed here. (Giving ascend_dpx.print/assert a
// MemRead<SharedMemory> effect would let membar insert both barriers and
// remove the need for this manual one.)
// ascend_dpx.print/assert -> LLVM is lowered in DebugOpToLLVM.cpp; the
// thread-0-guard scf.if it emits is lowered by a later SCFToControlFlow in the
// pipeline.

static bool isSupportedPrintFP8Type(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  return elementType.isFloat8E4M3FN() || elementType.isFloat8E5M2();
}

static Value convertPrintFP8ToF32(IRRewriter &rewriter, Location loc,
                                  Value value) {
  Type resultType = rewriter.getF32Type();
  if (auto rankedType = dyn_cast<RankedTensorType>(value.getType()))
    resultType = rankedType.clone(rewriter.getF32Type());
  return rewriter.create<triton::FpToFpOp>(loc, resultType, value,
                                           triton::RoundingModeAttr());
}

// Stages a single operand into unswizzled shared memory for the thread-0
// reader: i1 tensors are widened to i8 first; scalars/ptrs pass through.
// Emits the cooperative-write barrier the reader needs for visibility.
static Value stageToSharedMemory(IRRewriter &rewriter, Location loc,
                                 Value val) {
  auto rankedTy = dyn_cast<RankedTensorType>(val.getType());
  if (!rankedTy)
    return val;

  if (rankedTy.getElementType().isInteger(1)) {
    auto i8Ty = IntegerType::get(rewriter.getContext(), 8);
    auto i8RankedTy = RankedTensorType::get(rankedTy.getShape(), i8Ty,
                                            rankedTy.getEncoding());
    val = rewriter.create<arith::ExtUIOp>(loc, i8RankedTy, val);
    rankedTy = i8RankedTy;
  }

  auto memDescType = getUnswizzledSharedMemDescType(val, rankedTy);
  auto allocOp =
      rewriter.create<triton::gpu::LocalAllocOp>(loc, memDescType, val);
  rewriter.create<ascend_dpx::SyncThreadsOp>(loc);
  return allocOp.getResult();
}

static bool isBoolValue(Value val) {
  Type elementType = getElementTypeOrSelf(val.getType());
  if (elementType.isInteger(1))
    return true;
  if (!elementType.isInteger(8))
    return false;

  // Loads through !tt.ptr<i1> are represented as i8 values; the defining load
  // carries their logical type after this widening.
  Operation *defOp = val.getDefiningOp();
  auto wasBool =
      defOp ? defOp->getAttrOfType<BoolAttr>("was_bool_to_int8") : BoolAttr();
  return wasBool && wasBool.getValue();
}

struct ConvertDebugOpToAscendDPX
    : public impl::ConvertDebugOpToAscendDPXBase<ConvertDebugOpToAscendDPX> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(mod.getContext());

    WalkResult printWalkResult = mod.walk([&](triton::PrintOp op) {
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();

      // For each tensor arg, alloc unswizzled shared memory and store into it.
      // Non-tensor args (scalars, ptrs) are passed through unchanged.
      SmallVector<Value> newArgs;
      SmallVector<int32_t> isBool;
      newArgs.reserve(op.getArgs().size());
      isBool.reserve(op.getArgs().size());
      for (Value val : op.getArgs()) {
        bool boolVal = isBoolValue(val);
        isBool.push_back(boolVal ? 1 : 0);

        // Scalar bool loads use i8 storage. Restore their logical i1 type so
        // scalar runtime dispatch selects the bool ABI; tensors remain byte
        // buffers because the SIMT tensor runtime reads one byte per element.
        if (boolVal && val.getType().isInteger(8))
          val =
              rewriter.create<arith::TruncIOp>(loc, rewriter.getI1Type(), val);

        if (isSupportedPrintFP8Type(val.getType())) {
          // The debug runtime has no native FP8 entry point. Match the
          // established SIMD path by widening both A5 FP8 formats to f32.
          val = convertPrintFP8ToF32(rewriter, loc, val);
        } else if (!isa<triton::PointerType>(val.getType()) &&
                   !isSupportedPrintRuntimeType(
                       getElementTypeOrSelf(val.getType()))) {
          op.emitError("device print does not support element type ")
              << getElementTypeOrSelf(val.getType());
          return WalkResult::interrupt();
        }

        newArgs.push_back(stageToSharedMemory(rewriter, loc, val));
      }

      // Replace tt.print with ascenddpx.print
      rewriter.replaceOpWithNewOp<ascend_dpx::PrintOp>(
          op, op.getPrefixAttr(), op.getHexAttr(), newArgs,
          op.getIsSignedAttr(),
          DenseI32ArrayAttr::get(rewriter.getContext(), isBool));
      return WalkResult::advance();
    });
    if (printWalkResult.wasInterrupted())
      return signalPassFailure();

    mod->walk([&](triton::AssertOp op) {
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();

      Value cond = op.getCondition();
      Type condTy = cond.getType();
      if (auto rankedTy = dyn_cast<RankedTensorType>(condTy)) {
        if (!rankedTy.getElementType().isInteger(1)) {
          op->emitError("AssertOp condition must have bool type");
          return;
        }
      } else if (!condTy.isInteger(1)) {
        op->emitError("AssertOp condition must have bool type");
        return;
      }

      Value newCond = stageToSharedMemory(rewriter, loc, cond);
      rewriter.replaceOpWithNewOp<ascend_dpx::AssertOp>(op, newCond,
                                                        op.getMessageAttr());
    });
  }
};

} // namespace ascend

} // namespace mlir::triton
