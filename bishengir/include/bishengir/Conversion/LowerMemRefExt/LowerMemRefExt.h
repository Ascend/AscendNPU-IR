//===----- LowerMemRefExt.h - Lower Extended MemRef Dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the MemRefExt dialect to the HIVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_LOWERMEMREFEXT_LOWERMEMREFEXT_H
#define BISHENGIR_CONVERSION_LOWERMEMREFEXT_LOWERMEMREFEXT_H

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_LOWERMEMREFEXT
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createMemrefExtLoweringPass();
} // namespace mlir
#endif
