//===- bishengir-target-spec-tblgen.cpp -----------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for BiShengIR Target Spec TableGen.
//
//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/mlir-tblgen.cpp
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

#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
static bool bishengirTargetSpecGenMain(raw_ostream &os, RecordKeeper &records) {
#else
static bool bishengirTargetSpecGenMain(raw_ostream &os, const RecordKeeper &records) {
#endif
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
