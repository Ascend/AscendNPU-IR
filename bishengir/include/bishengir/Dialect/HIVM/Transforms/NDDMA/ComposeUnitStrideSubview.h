//===- ComposeUnitStrideSubview.h - Fuse unit-stride subviews ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for folding chains of unit-stride subviews, including dynamic sizes.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_COMPOSEUNITSTRIDESUBVIEW_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_COMPOSEUNITSTRIDESUBVIEW_H

namespace mlir {
class MLIRContext;
class RewritePatternSet;

namespace hivm {
namespace nddma {

void populateComposeUnitStrideSubviewPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context);

} // namespace nddma
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_COMPOSEUNITSTRIDESUBVIEW_H
