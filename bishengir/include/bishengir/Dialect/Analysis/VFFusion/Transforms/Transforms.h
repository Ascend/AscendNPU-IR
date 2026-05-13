//===- Transforms.h ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_TRANSFORM_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_TRANSFORM_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace analysis {

void populateEmptifyReduceInitPatterns(RewritePatternSet &patterns);

} // namespace analysis
} // namespace mlir

#endif // BISHENGIR_DIALECT_ANALYSIS_VFFUSION_TRANSFORM_H