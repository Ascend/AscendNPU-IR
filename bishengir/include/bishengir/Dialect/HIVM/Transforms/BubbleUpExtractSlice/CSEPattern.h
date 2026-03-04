//===- CSEPattern.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_CSE_EXTRACT_SLICE_PATTERN_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_CSE_EXTRACT_SLICE_PATTERN_H

namespace mlir {
class RewritePatternSet;
namespace hivm::detail {
using namespace mlir;
void populateCSEPattern(RewritePatternSet &patterns);
} // namespace hivm::detail
} // namespace mlir
#endif
