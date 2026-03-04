//===- HIVMToTritonGPU.cpp - conversion from HIVM to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToArith.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMTOTRITONGPU
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct HIVMToTritonGPUConversionPass
    : public impl::ConvertHIVMToTritonGPUBase<HIVMToTritonGPUConversionPass> {
  void runOnOperation() override;
};
} // namespace

Type mlir::hivm::HIVMToTritonTypeConvert(Type ty) {
  if (auto memrefTy = dyn_cast<MemRefType>(ty)) {
    auto elemTy = memrefTy.getElementType();
    // Note: If the address space of memref type is not specified, treat it as
    //       in global memory.
    if (auto attr = memrefTy.getMemorySpace()) {
      if (auto ASAttr = dyn_cast<AddressSpaceAttr>(attr)) {
        auto AS = static_cast<uint32_t>(ASAttr.getAddressSpace());
        return triton::getPointerType(elemTy, AS);
      }
      llvm::report_fatal_error("Invalid memory space");
    } else {
      return triton::getPointerType(elemTy);
    }
  }
  return ty;
}

void HIVMToTritonGPUConversionPass::runOnOperation() {
  auto module = getOperation();
  auto &ctx = getContext();

  // Stage1: Convert operators in function body
  ConversionTarget stage1Target(ctx);
  stage1Target
      .addLegalDialect<arith::ArithDialect, math::MathDialect,
                       tensor::TensorDialect, triton::TritonDialect,
                       triton::gpu::TritonGPUDialect, func::FuncDialect>();
  stage1Target.addIllegalDialect<hivm::HIVMDialect>();
  stage1Target.addIllegalOp<bufferization::ToTensorOp, memref::ReinterpretCastOp>();
  stage1Target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  RewritePatternSet stage1Patterns(&ctx);

  // TODO: Add more HIVMToXXPatterns
  populateHIVMToArithConversionPatterns(stage1Patterns);
  populateHIVMToTensorPatterns(stage1Patterns);
  populateBufferizationToTritonPatterns(stage1Patterns);
  populateReinterpretCastToUnrealizedCastPatterns(stage1Patterns);
  populateHIVMToTritonPatterns(stage1Patterns);

  if (failed(applyPartialConversion(module, stage1Target,
                                    std::move(stage1Patterns)))) {
    module->emitError("Stage1 failed: HIVM/Bufferization Op conversion failed");
    signalPassFailure();
  }

  // Stage2: Convert FuncOp alone
  // Note: Return values are allowed only in test scenarios.
  if (!allowReturnValue) {
    RewritePatternSet stage2Patterns(&ctx);
    ConversionTarget stage2Target(ctx);
    stage2Target.addLegalDialect<
        triton::TritonDialect, triton::gpu::TritonGPUDialect,
        arith::ArithDialect, mlir::BuiltinDialect, tensor::TensorDialect>();
    stage2Target.addIllegalOp<func::FuncOp>();
    stage2Target.addLegalOp<UnrealizedConversionCastOp>();
    populateFuncToTritonPatterns(stage2Patterns);

    if (failed(applyPartialConversion(module, stage2Target,
                                      std::move(stage2Patterns)))) {
      module->emitError("Stage2 failed: FuncOp to Triton conversion failed");
      signalPassFailure();
    }
  }

  // Stage3: Clean up redundant UnrealizedCastsOp
  OpPassManager dynPM;
  dynPM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(runPipeline(dynPM, module))) {
    module->emitError("Stage3 failed: Reconcile unrealized casts failed");
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createHIVMToTritonGPUConversionPass() {
  return std::make_unique<HIVMToTritonGPUConversionPass>();
}
