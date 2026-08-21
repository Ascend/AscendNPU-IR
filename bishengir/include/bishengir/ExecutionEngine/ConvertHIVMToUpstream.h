//===- ConvertHIVMToUpstream.h - HIVM to upstream pattern sets -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the pattern sets shared by the two HIVM-to-upstream
// conversions: the strict full conversion used by the CPU runner and the
// partial vector-side conversion used by the delayed RegBase
// re-vectorization pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_EXECUTION_ENGINE_CONVERTHIVMTOUPSTREAM_H
#define BISHENGIR_EXECUTION_ENGINE_CONVERTHIVMTOUPSTREAM_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace execution_engine {

/// Populate the strict full conversion patterns: all HIVM ops, including
/// load/store, are rewritten; the caller is expected to mark the HIVM dialect
/// illegal. Used by the CPU runner.
void populateConvertHIVMToUpstreamPatterns(RewritePatternSet &patterns,
                                           bool convertToNamedOp);

/// Populate the partial conversion patterns for the RegBase pipeline:
/// vector-side ops only, hivm.hir.load preserved, bitwise logic ops converted
/// to named ops when requested.
void populateConvertHIVMToHFusionPatterns(RewritePatternSet &patterns,
                                          bool convertToNamedOp);

} // namespace execution_engine
} // namespace mlir

#endif // BISHENGIR_EXECUTION_ENGINE_CONVERTHIVMTOUPSTREAM_H
