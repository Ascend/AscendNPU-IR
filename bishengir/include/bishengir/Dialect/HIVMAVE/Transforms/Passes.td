//===-- Passes.td - HIVMAVE dialect pass definition file ------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_AVE_PASSES
#define BISHENGIR_DIALECT_AVE_PASSES

include "mlir/Pass/PassBase.td"

def AVENormalizeOps : Pass<"ave-normalize-ops", "func::FuncOp"> {
  let summary = "Normalize AVE-Ops to adapt the preg/vector register bitwidth.";
  let constructor = "mlir::hivmave::createAVENormalizeOpsPass()";
  let dependentDialects = ["hivmave::AVEDialect", "arith::ArithDialect", "hivm::HIVMDialect"];
}

def ReplaceWithVectorScalar : Pass<"ave-replace-with-vector-scalar", "func::FuncOp"> {
  let summary = "Replace vector-vector op with vector-scalar op if possible";
  let constructor = "mlir::hivmave::createReplaceWithVectorScalarPass()";
  let dependentDialects = ["arith::ArithDialect", "vector::VectorDialect","hivmave::AVEDialect"];
}

def PLTToPLTM : Pass<"ave-plt-to-pltm", "func::FuncOp"> {
  let summary = "Convert PLT op on tail block to PLTM.";
  let constructor = "mlir::hivmave::createPLTToPLTMPass()";
  let dependentDialects = ["arith::ArithDialect","hivmave::AVEDialect"];
}

def ProcessVsstb : Pass<"ave-process-vsstb", "func::FuncOp"> {
  let summary = "Process VSSTB related SCF For loops (unroll 64F32 loops and rewrite vector store)";
  let description = [{
    This pass matches specific SCF For loops with VFTruncFOp and VFStoreWithStrideOp,
    unrolls them by factor 2, and rewrites the vector store logic with DeInterleaveOp.
  }];
  let constructor = "mlir::hivmave::createProcessVsstbPass()";
    let dependentDialects = [
    "hivmave::AVEDialect",
    "arith::ArithDialect",
    "hivm::HIVMDialect"
  ];
}

def ProcessWideVextf : Pass<"ave-process-wide-vextf", "func::FuncOp"> {
  let summary = "Process wide vextf related SCF For loops (unroll 64F16 loops and rewrite vextf)";
  let description = [{
    This pass matches specific SCF For loops with vextf,
    unrolls them by factor 2, and rewrites the vector load logic with VfInterleaveOp.
  }];
  let constructor = "mlir::hivmave::createProcessWideVextfPass()";
    let dependentDialects = [
    "hivmave::AVEDialect",
    "arith::ArithDialect",
    "hivm::HIVMDialect",
    "mlir::annotation::AnnotationDialect"
  ];
}

def I1opSoftImpl : Pass<"ave-i1op-soft-impl", "func::FuncOp"> {
  let summary = "Implement I1 op";
  let description = [{
    Implement the i1-op by vector-ave-op combination.
  }];
  let constructor = "mlir::hivmave::createI1opSoftImplPass()";
    let dependentDialects = [
    "hivmave::AVEDialect",
    "arith::ArithDialect"
  ];
}

def ProcessMembar : Pass<"ave-process-membar", "func::FuncOp"> {
  let summary = "Process membar interface to ave";
  let description = [{
    Lowring the annotation dialect.
  }];
  let constructor = "mlir::hivmave::createProcessMembarPass()";
    let dependentDialects = [
    "hivmave::AVEDialect",
    "arith::ArithDialect",
    "hivm::HIVMDialect"
  ];
}

def CombineAVEOPs : Pass<"combine-ave-ops", "func::FuncOp"> {
  let summary = "combine ave ops.";
  let constructor = "mlir::hivmave::createCombineAVEOPsPass()";
  let dependentDialects = ["arith::ArithDialect","hivmave::AVEDialect"];
  let options = [Option<"enableFusedMultiplyAdd", "enable-fused-multiply-add", "bool",
                        /*default=*/"false", "Enable fused multiply add.">];
}

def LegalizeOptHIVMAVE : Pass<"legalize-opt-hivmave", "func::FuncOp"> {
  let summary = "Legalize and optimize ave op";
  let constructor = "mlir::hivmave::createLegalizeOptHIVMAVEPass()";
  let dependentDialects = ["hivmave::AVEDialect"];
}

def OptimizeReductionLoopHIVMAVE : Pass<"optimize-reduction-loop", "func::FuncOp"> {
  let summary = "Optimize reductionLoop";
  let constructor = "mlir::hivmave::createOptimizeReductionLoopHIVMAVEPass()";
  let dependentDialects = ["hivmave::AVEDialect", "scf::SCFDialect"];
  let options = [
    Option<"maxSplit", "max-split", "int", /*default=*/"1",
           "Maximum split depth">
  ];
}

def SIMTLoopNestsScheduling : Pass<"simt-loop-nests-scheduling"> {
  let summary = "Optimize SIMT loop nests";
  let constructor = "mlir::hivmave::createSIMTLoopNestsSchedulingPass()";
  let dependentDialects = ["transform::TransformDialect"];
}

def SIMTGridBlockPartition : Pass<"simt-grid-block-partition", "func::FuncOp"> {
  let summary = "Perform for-to-forall, forall-to-parallel, parallel-loop-tiling";
  let constructor = "::mlir::hivmave::createSIMTGridBlockPartitionPass()";
  let dependentDialects = ["linalg::LinalgDialect",
    "transform::TransformDialect", "hivm_regbaseintrins::HIVMRegbaseIntrinsDialect"];
}

def ScalarBroadcastToVLoad : Pass<"hivmave-scalar-broadcast-to-vload", "func::FuncOp"> {
  let summary = "Convert scalar broadcast (VFBroadcastScalarOp) to vector load (VFLoadOp)";
  let description = "Eliminate redundant scalar load + broadcast by replacing with direct vector load.";
  let constructor = "mlir::hivmave::createScalarBroadcastToVLoadPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect", "hivm::HIVMDialect", "hivmave::AVEDialect"
  ];
}

def HoistVstas : Pass<"hoist-vstas",  "func::FuncOp"> {
  let summary = "Hoist vstas op out of loops";
  let constructor = "::mlir::hivmave::createHoistVstasPass()";
  let dependentDialects = ["hivmave::AVEDialect"];
}

def AnalyzeVectorLayout : Pass<"analyze-vector-layout", "func::FuncOp"> {
  let summary = "Analyze vector layout";
  let constructor = "::mlir::hivmave::createAnalyzeVectorLayoutPass()";
  let dependentDialects = ["hivmave::AVEDialect"];
}

def AnalyzeAlignmentBitwidth : Pass<"analyze-alignment-bitwidth", "func::FuncOp"> {
  let summary = "Analyze alignment bitwidth";
  let constructor = "::mlir::hivmave::createAnalyzeAlignmentBitwidthPass()";
  let dependentDialects = ["hivmave::AVEDialect"];
}

def RemoveVectorLayoutAttr : Pass<"remove-vector-layout-attr", "func::FuncOp"> {
  let summary = "Remove vector layout attribute before lowering to ave-intrin";
  let constructor = "::mlir::hivmave::createRemoveVectorLayoutAttrPass()";
  let dependentDialects = ["hivmave::AVEDialect"];
}

def HIVMComplexReductionIntermediateLowering
    : Pass<"hivm-complex-reduction-intermediate-lowering", "func::FuncOp"> {
  let summary = "Lower complex hivmave reduction operations (without a direct one-to-one reduction primitive) into a sequence of elementwise operations.";
  let description = [{
    This pass further lowers  hivmave reduction operations without reduction intrinsics into an explicit loop-based reduction using vector slides and element-wise operations
  }];
  let constructor = "mlir::hivmave::createComplexReductionIntermediateLoweringPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "hivmave::AVEDialect"
  ];
}

#endif // BISHENGIR_DIALECT_AVE_PASSES
