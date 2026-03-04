//===- RegisterEverything.cpp - Register all BiShengIR entities -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir-c/RegisterEverything.h"

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "mlir/CAPI/IR.h"

void bishengirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  bishengir::registerAllDialects(registry);
  bishengir::registerAllExtensions(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void bishengirRegisterAllPasses() {
  bishengir::registerAllPasses();
  bishengir::registerBiShengIRCompilePass();
}
