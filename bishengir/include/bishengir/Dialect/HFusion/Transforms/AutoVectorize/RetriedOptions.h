//===- RetriedOptions.h - Auto vectorization retry options -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_RETRIEDOPTIONS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_RETRIEDOPTIONS_H

namespace mlir {
namespace hfusion {

/// Pass options that may be retried with different values after a failure.
/// Lives across retry attempts; PlanContext is created from it per-attempt.
struct RetriedOptions {
  unsigned maxFusedOps;
  bool enableMultipleConsumerFusion;
  bool enableCrossIfFusion;
  bool enableVFStackLimit;
  unsigned vectorLength;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_CONTEXT_H
