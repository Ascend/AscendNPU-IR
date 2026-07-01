//===- HIVMPipelines.cpp - HIVM pipelines ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/ArithToHIVMAVE/ArithToHIVMAVE.h"
#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Conversion/LowerMemRefExt/LowerMemRefExt.h"
#include "bishengir/Conversion/TensorToHIVM/TensorToHIVM.h"
#include "bishengir/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.h"
#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/Arith/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hivmave {

static void hivmAVEOptimizationPipeline(
    OpPassManager &pm, const HIVMAVEPipelineOptions &hivmAVEPipelineOptions) {
  if (!hivmAVEPipelineOptions.enableDirectHIVMLowering &&
      hacc::utils::isRegBasedArch(
          hacc::symbolizeTargetDeviceEnum(hivmAVEPipelineOptions.target))) {
    ConvertVectorToHIVMAVEOptions vectorOptions;
    vectorOptions.enableTritonCompile =
        hivmAVEPipelineOptions.enableTritonKernelCompile;
    pm.addPass(createVectorToHIVMAVEConversionPass(vectorOptions));
    pm.nest<func::FuncOp>().addPass(createArithToHIVMAVEConversionPass());
    pm.nest<func::FuncOp>().addPass(hivmave::createI1opSoftImplPass());
    pm.nest<func::FuncOp>().addPass(
        hivmave::createComplexReductionIntermediateLoweringPass());
    pm.addPass(createCanonicalizerPass());
    OptimizeReductionLoopHIVMAVEOptions optimizeReductionLoopOptions;
    optimizeReductionLoopOptions.maxSplit =
        hivmAVEPipelineOptions.maxReductionSplitNum;
    // Vsstb packing depends on adjacent store order; reduction splitting may
 	// pair non-adjacent IVs (e.g. i and i + half) and hide that pattern.
    pm.nest<func::FuncOp>().addPass(hivmave::createProcessVsstbPass());
    pm.nest<func::FuncOp>().addPass(
        hivmave::createOptimizeReductionLoopHIVMAVEPass(
            optimizeReductionLoopOptions));
    pm.nest<func::FuncOp>().addPass(hivmave::createAveLoopOptimizePass());
    pm.nest<func::FuncOp>().addPass(hivmave::createLegalizeOptHIVMAVEPass());
    pm.nest<func::FuncOp>().addPass(
        hivmave::createReplaceWithVectorScalarPass());
    pm.nest<func::FuncOp>().addPass(hivmave::createProcessMembarPass());
    pm.nest<func::FuncOp>().addPass(annotation::createAnnotationLoweringPass());
    CombineAVEOPsOptions combineAVEOPsOptions;
    combineAVEOPsOptions.enableFusedMultiplyAdd =
        hivmAVEPipelineOptions.enableFusedMultiplyAdd;
    pm.nest<func::FuncOp>().addPass(
        hivmave::createCombineAVEOPsPass(combineAVEOPsOptions));
    pm.nest<func::FuncOp>().addPass(
        hivmave::createScalarBroadcastToVLoadPass());
    pm.nest<func::FuncOp>().addPass(hivmave::createPLTToPLTMPass());
    pm.nest<func::FuncOp>().addPass(scf::createLegalizeLoopIterArgsPass());
    pm.nest<func::FuncOp>().addPass(hivmave::createAnalyzeVectorLayoutPass());
    pm.nest<func::FuncOp>().addPass(hivmave::createAnalyzeAlignmentBitwidthPass());
    pm.nest<func::FuncOp>().addPass(createCanonicalizerPass());
    pm.nest<func::FuncOp>().addPass(hivmave::createAVENormalizeOpsPass());
    pm.nest<func::FuncOp>().addPass(hivmave::createRemoveVectorLayoutAttrPass());
  }
}

void buildLowerAVEPipelines(
    OpPassManager &pm, const HIVMAVEPipelineOptions &hivmAVEPipelineOptions) {
  hivmAVEOptimizationPipeline(pm, hivmAVEPipelineOptions);
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerLowerAVEPipelines() {
  PassPipelineRegistration<HIVMAVEPipelineOptions>(
      "lower-ave-pipeline", "lower ave pipeline",
      [](OpPassManager &pm, const HIVMAVEPipelineOptions &options) {
        buildLowerAVEPipelines(pm, options);
      });
}

} // namespace hivmave
} // namespace mlir
