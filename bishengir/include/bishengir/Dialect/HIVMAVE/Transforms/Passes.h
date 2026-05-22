//===- Passes.h - Pass Entrypoints --------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_AVE_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_AVE_TRANSFORMS_PASSES_H

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"

namespace hivmave {

/// Create a pass to normalize arith ops for simd backend
std::unique_ptr<Pass> createReplaceWithVectorScalarPass();

/// Unroll and add deintlv for vsstb
std::unique_ptr<Pass> createProcessVsstbPass();

/// Process membar before avetointrin
std::unique_ptr<Pass> createProcessMembarPass();

/// Implement the i1-op by vector-ave-op combination
std::unique_ptr<Pass> createI1opSoftImplPass();

/// Combine multiple AVE ops into more efficient fused ops.
std::unique_ptr<Pass>
createCombineAVEOPsPass(const CombineAVEOPsOptions &options = {});

/// Convert PLT to PLTM op
std::unique_ptr<Pass> createPLTToPLTMPass();

/// Create a pass to legalize and optimize the ave op
std::unique_ptr<Pass> createLegalizeOptHIVMAVEPass();

/// Create a pass to optimize reductionLoop
std::unique_ptr<Pass> createOptimizeReductionLoopHIVMAVEPass(
    const OptimizeReductionLoopHIVMAVEOptions &options = {});

/// Optimize SIMT loop nests
std::unique_ptr<Pass> createSIMTLoopNestsSchedulingPass();

// Create a pass to perform loop tiling for VF SIMT threads
std::unique_ptr<Pass> createSIMTGridBlockPartitionPass();

// Create a pass to convert scalar brc to vload
std::unique_ptr<Pass> createScalarBroadcastToVLoadPass();

// Create a pass to hoist vstas
std::unique_ptr<Pass> createHoistVstasPass();

/// Create a pass to analyze data layout
std::unique_ptr<Pass> createDataLayoutAnalysisPass();

/// Soft implement for reduction <xor>
std::unique_ptr<Pass> createComplexReductionIntermediateLoweringPass();

// Create a pass to analyze vector layout attribute
std::unique_ptr<Pass> createAnalyzeVectorLayoutPass();

// Create a pass to analyze alignment bitiwidth
std::unique_ptr<Pass> createAnalyzeAlignmentBitwidthPass();

// Create a pass to remove vector layout attribute
std::unique_ptr<Pass> createRemoveVectorLayoutAttrPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"

} // namespace hivmave
} // namespace mlir

#endif // BISHENGIR_DIALECT_AVE_TRANSFORMS_PASSES_H
