//===- AnnotationPasses.cpp - Pybind module for the Annotation passes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir-c/Dialect/Annotation.h"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_bishengirAnnotationPasses, m) {
  m.doc() = "MLIR Annotation Dialect Passes";

  // Register all Annotation passes on load.
  mlirRegisterAnnotationPasses();
}
