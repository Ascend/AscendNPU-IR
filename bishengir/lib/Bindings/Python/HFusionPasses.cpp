//===- HFusionPasses.cpp - Pybind module for the HFusion passes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir-c/Dialect/HFusion.h"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_bishengirHFusionPasses, m) {
  m.doc() = "MLIR HFusion Dialect Passes";

  // Register all HFusion passes on load.
  mlirRegisterHFusionPasses();
}
