//===- FixCallUnknownLoc.h - Fix UnknownLoc on call ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_FIXCALLUNKNOWNLOC_FIXCALLUNKNOWNLOC_H_
#define BISHENGIR_CONVERSION_FIXCALLUNKNOWNLOC_FIXCALLUNKNOWNLOC_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_FIXCALLUNKNOWNLOC
#include "bishengir/Conversion/Passes.h.inc"

/// Create a pass to fix UnknownLoc on call ops by inheriting location from
/// users or parent ops.
std::unique_ptr<Pass> createFixCallUnknownLocPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_FIXCALLUNKNOWNLOC_FIXCALLUNKNOWNLOC_H_
