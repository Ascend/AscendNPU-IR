//===- HIVMPasses.cpp - Pybind module for the HIVM passes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir-c/Dialect/HIVM.h"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_bishengirHIVMPasses, m) {
  m.doc() = "MLIR HIVM Dialect Passes";

  // Register all HIVM passes on load.
  mlirRegisterHIVMPasses();
}
