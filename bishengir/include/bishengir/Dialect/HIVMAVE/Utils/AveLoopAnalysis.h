//===- AveLoopAnalysis.h - AVE loop analysis utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVMAVE_UTILS_AVELOOPANALYSIS_H
#define BISHENGIR_DIALECT_HIVMAVE_UTILS_AVELOOPANALYSIS_H

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace hivmave {

enum class LoopAccessContinuity {
  Contiguous,
  NonContiguous,
  Unknown,
};

/// Prove whether adjacent loop iterations access adjacent vectors. Memref
/// strides and the resulting address delta are measured in elements.
LoopAccessContinuity analyzeLoopAccessContinuity(scf::ForOp loop, Value base,
                                                 ValueRange indices,
                                                 MemRefType memrefType,
                                                 VectorType vectorType);

/// Relative pressure on the three independently scheduled vector pipelines.
/// The values are throughput weights, not hardware cycle estimates.
struct AVEPipelineCost {
  float load = 0.0f;
  float execute = 0.0f;
  float store = 0.0f;

  AVEPipelineCost &operator+=(const AVEPipelineCost &other);
  AVEPipelineCost scaled(float factor) const;
  float bottleneck() const;
};

enum class AVEPipelineBound {
  Load,
  Execute,
  Store,
  Balanced,
};

/// Estimate one AVE operation or one loop iteration. Loop analysis ignores
/// non-AVE address calculation operations. Unclassified AVE operations do not
/// participate in the cost calculation.
AVEPipelineCost estimateAVEPipelineCost(const Operation &op);
AVEPipelineCost estimateLoopPipelineCost(scf::ForOp loop);

AVEPipelineBound classifyAVEPipelineBound(const AVEPipelineCost &cost);

/// Cost deltas for replacing factor original operations with one merged group.
AVEPipelineCost estimateLoadMergeDelta(unsigned factor);
AVEPipelineCost estimateNarrowChainMergeDelta(unsigned factor,
                                              float elementwiseChainCost,
                                              unsigned packTreeCount);

/// Compare the bottleneck before and after applying delta.
bool isAVEPipelinePlanProfitable(const AVEPipelineCost &before,
                                 const AVEPipelineCost &delta,
                                 float minimumGain = 0.0f);

} // namespace hivmave
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVMAVE_UTILS_AVELOOPANALYSIS_H
