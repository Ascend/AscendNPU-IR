//===- Transforms.h - HFusion Dialect Transformation Entrypoints *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_H

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;

namespace hfusion {

/// Collect a set of patterns to Flatten HFusion/Linalg ops
void populateFlattenOpsPattern(RewritePatternSet &patterns);

void populateTreeReducePattern(RewritePatternSet &patterns);

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_H
