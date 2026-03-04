//===- NormalizeLoopIterator.h - Process memory conflicts in loop iterator-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hivm {

// As plan memory will consider that iterArg and yield value use same one space
// and if there exists use of iterArg after yield value memory initialization,
// iterArg has been 'dirty'.
// Here consider above state and separate original yield value memory from
// iteration memmory
void populateNormalizeLoopIneratorPattern(RewritePatternSet &patterns);

} // namespace hivm
} // namespace mlir