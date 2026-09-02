//===- bishengir-target-spec-tblgen.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for BiShengIR Target Spec TableGen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

// Generator to invoke.
static const mlir::GenInfo *generator;

// Generator that prints records.
static mlir::GenRegistration
    printRecords("print-records", "Print all records to stdout",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   os << records;
                   return false;
                 });

static bool bishengirTargetSpecGenMain(raw_ostream &os, RecordKeeper &records) {
  if (!generator) {
    os << records;
    return false;
  }
  return generator->invoke(records, os);
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<const mlir::GenInfo *, true, mlir::GenNameParser> generator(
      "", llvm::cl::desc("Generator to run"), cl::location(::generator));
  cl::ParseCommandLineOptions(argc, argv, "BiShengIR Target Spec Generator");

  return TableGenMain(argv[0], &bishengirTargetSpecGenMain);
}
