//===- bishengir-opt.cpp - BiShengIR Optimizer Driver -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for bishengir-opt built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "bishengir/Version/Version.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

#ifdef MLIR_INCLUDE_TESTS
#include "Test/InitTestDialect.h"
#include "Test/TestPasses.h"
#endif

namespace mlir {
namespace test {
void registerTestTransformDialectEraseSchedulePass();
} // namespace test
} // namespace mlir

namespace test {
void registerTestDialect(::mlir::DialectRegistry &registry);
void registerTestTransformDialectExtension(::mlir::DialectRegistry &registry);
} // namespace test

static void printVersion(llvm::raw_ostream &os) {
  os << bishengir::getBiShengIRToolFullVersion("bishengir-opt") << '\n';
}

int main(int argc, char **argv) {
  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);

  // Register passes.
  mlir::registerAllPasses();
  bishengir::registerAllPasses();

  // Register dialect extensions.
  mlir::registerAllExtensions(registry);
  bishengir::registerAllExtensions(registry);

#ifdef MLIR_INCLUDE_TESTS
  ::bishengir_test::registerTestDialect(registry);
  ::bishengir_test::registerAllTestPasses();
  ::mlir::test::registerTestTransformDialectEraseSchedulePass();
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
#endif

  // Register version printer
  llvm::cl::SetVersionPrinter(printVersion);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "BiShengIR optimizer driver\n", registry));
}
