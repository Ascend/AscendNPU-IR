//===- RegisterEverything.cpp - API to register all dialects/passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/Pass/PassManager.h"

#include "bishengir-c/Dialect/Annotation.h"
#include "bishengir-c/Dialect/HFusion.h"
#include "bishengir-c/Dialect/HIVM.h"

PYBIND11_MODULE(_bishengirRegisterEverything, m) {
  m.doc() =
      "BiShengIR All Upstream Dialects, Extensions and Passes Registration";

  // register dialects of bishengir i.e. hivm, hfusion. annotation
  m.def("register_dialects",
        [](MlirContext context) { bishengirRegisterAllDialects(context); });

  m.def("register_translations",
        [](MlirContext context) { bishengirRegisterAllTranslations(context); });

  // Register all passes on load.
  bishengirRegisterAllPasses();
}
