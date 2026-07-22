//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace bishengir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace triton {

/// Creates wrappers and attributes for SIMT functions
std::unique_ptr<mlir::Pass>
createAdaptGPUKernelPass(TritonRemapOptions options = {});
/// Create a pass to convert llvm.frem to sub(mul(trunc(fdiv)))
/// IR.
std::unique_ptr<mlir::Pass>
createDecomposeFRemPass();
/// Create a pass to convert triton-generated LLVM IR into NPU-compatible LLVM
/// IR.
std::unique_ptr<mlir::Pass> createTritonRemapPass(const TritonRemapOptions &options = {});

/// Create a pass to add SIMT opt attribution.
std::unique_ptr<mlir::Pass> createSetBishengirSimtOptAttrPass(
    const SetBishengirSimtOptAttrOptions &options = {});

/// Create a pass to adapt Triton IR kernel.
std::unique_ptr<mlir::Pass> createAdaptTritonIRKernelPass(
    const AdaptTritonIRKernelOptions &options = {});

/// Create a pass to batch logical Triton programs onto a limited number of
/// physical SIMT cores.
std::unique_ptr<mlir::Pass> createSIMTAutoBlockifyPass(unsigned factor = 1);

std::unique_ptr<mlir::Pass> createEnableAscendDPXMMAPass();

std::unique_ptr<mlir::Pass> createConvertDotInputToLinearLayoutPass();


/// Create a pass to convert f16 operations to f32 operations
std::unique_ptr<mlir::Pass> createLegalizeF16ForTritonPass();

/// Create a pass to convert slice-based concatenation to select based.
std::unique_ptr<mlir::Pass> createFixFusedCatPass();

/// Create a pass to rewrite a restricted form of `tensor.extract_slice` and
/// `tensor.insert_slice` (single-axis power-of-two block index along any one
/// axis, with the offset a multiple of the block size) into Triton dialect
/// ops: `tt.trans`, `tt.reshape`, `tt.split`, `tt.join`.
std::unique_ptr<mlir::Pass> createRewriteSliceOpToTritonPass();

/// Create a pass that erases all `annotation.mark` ops in the module.
std::unique_ptr<mlir::Pass> createRemoveAnnotationMarkPass();

/// Create a pass to decompose reduction.
std::unique_ptr<mlir::Pass> createDecomposeReductionPass();

/// Create a pass to convert remove unnecessary layout conversion to imporve
/// performance
std::unique_ptr<mlir::Pass> createOptimizeLayoutsPass();

/// Create a pass to optimize loads
std::unique_ptr<mlir::Pass> createOptimizeLoadsPass();

/// Create a pass to split loops up and tailor arange ranges
std::unique_ptr<mlir::Pass> createLoopRestructureArangeOptimizationPass();

std::unique_ptr<mlir::Pass> createGetTritonMetadataPass(const GetTritonMetadataOptions &options = {});

/// Create a pass that prints the fractal zN layout mapping for a given offset.
std::unique_ptr<mlir::Pass>
createDumpFractalLayoutPass(const DumpFractalLayoutOptions &options = {});

/// Create a pass to optimize division operations for SIMT execution.
std::unique_ptr<mlir::Pass> createSIMTFastDivPass();

/// Create a pass that converts tt.load/tt.store with ptr<6> to
/// ttg.local_load/ttg.local_store with memdesc types.
std::unique_ptr<mlir::Pass> createConvertSharedPtrToMemDescPass();

/// Create a pass to flatten memdesc struct args to bare pointers.
std::unique_ptr<mlir::Pass> createFlattenMemDescArgsPass();

/// Create a pass that propagates shared memory offsets from local_alloc to
/// call_scalar and removes ordering-only local_load ops.
std::unique_ptr<mlir::Pass> createPopulateSharedMemoryOffsetToDPXPass();

/// Create a pass to tile tt.dot load inputs to reduce register spill.

std::unique_ptr<mlir::Pass>
createTileDotLoadsPass(const TileDotLoadsOptions &options = {});

/// Create a pass that performs cheap math rewrites: math.exp -> math.exp2
/// with a log2(e) factor (downstream canonicalize folds it into any
/// adjacent constant-splat scale)
std::unique_ptr<mlir::Pass> createOptimizeMathPass();

/// Run two complementary transformations on K-tile chain loops:
///   1. Hoist any tt.trans whose source is loop-invariant out of every
///      scf.for (targeted LICM specialization).
///   2. Fuse adjacent scf.for ops with identical bounds when their
///      chains are independent AND their combined SMEM footprint stays
///      under 70% of the kernel's SMEM budget, so the instruction
///      scheduler sees both chains in one body and can interleave their
///      issue.
std::unique_ptr<mlir::Pass>
createHoistAndFuseDotChainsPass(const HoistAndFuseDotChainsOptions &options = {});

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace triton

} // namespace bishengir

#endif // BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_H
