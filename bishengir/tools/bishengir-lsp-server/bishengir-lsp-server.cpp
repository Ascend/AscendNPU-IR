//===- bishengir-lsp-server.cpp - BiShengIR Language Server -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

#ifdef MLIR_INCLUDE_TESTS
namespace bishengir_test {
void registerTestDialect(::mlir::DialectRegistry &registry);
} // namespace bishengir_test

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test
#endif

int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);

#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
  ::bishengir_test::registerTestDialect(registry);
#endif

  return failed(MlirLspServerMain(argc, argv, registry));
}
