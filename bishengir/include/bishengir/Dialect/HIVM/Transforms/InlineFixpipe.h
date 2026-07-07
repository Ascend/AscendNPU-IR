//===- InlineFixpipe.h - HIVM Fixpipe transform patterns ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPE_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPE_H

namespace mlir {

class RewritePatternSet;

namespace hivm {

struct InsertFixpipePatternOptions {
  bool inferFixpipeDmaMode = false;
};

struct InlineFixpipePatternOptions {
  bool inlineQuantScale = false;
};

/// Populate pattern that insert `hivm.fixpipe` operation for MMAD result
/// yielded from scf.for as iter_arg which is used as input argument of
/// MMAD operation itself.
void populateInsertFixpipeForIterArgMMADPattern(
    RewritePatternSet &patterns,
    InsertFixpipePatternOptions options = InsertFixpipePatternOptions{});

/// Populate patterns that insert `hivm.fixpipe` ops.
void populateInsertFixpipePatterns(
    RewritePatternSet &patterns,
    InsertFixpipePatternOptions options = InsertFixpipePatternOptions{});

/// Populate patterns that inline / fold around `hivm.fixpipe` ops.
void populateInlineFixpipePatterns(
    RewritePatternSet &patterns,
    InlineFixpipePatternOptions options = InlineFixpipePatternOptions{});

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPE_H
