//===- bishengir-translate.cpp - BiShengIR Translate Driver -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllTranslations.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  bishengir::registerAllTranslations();

  return failed(mlir::mlirTranslateMain(argc, argv,
                                        "HIVM MLIR Translation Testing Tool"));
}
