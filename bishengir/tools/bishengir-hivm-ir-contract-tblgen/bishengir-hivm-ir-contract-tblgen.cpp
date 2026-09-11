//===- bishengir-hivm-ir-contract-tblgen.cpp --------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// This file contains the main function for BiShengIR HIVM IR Contract
// TableGen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace llvm;

// Generator to invoke.
static const GenInfo *generator;

// Generator that prints records.
static GenRegistration
    printRecords("print-records", "Print all records to stdout",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   os << records;
                   return false;
                 });

#if defined(__LLVM_MAJOR_VERSION_20_COMPATIBLE__) || defined(__LLVM_MAJOR_VERSION_21_COMPATIBLE__)
static bool bishengirHivmIrContractGenMain(raw_ostream &os,
                                           const RecordKeeper &records) {
#else
static bool bishengirHivmIrContractGenMain(raw_ostream &os,
                                           RecordKeeper &records) {
#endif
  if (!generator) {
    os << records;
    return false;
  }
  return generator->invoke(records, os);
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::opt<const GenInfo *, true, GenNameParser> generator(
      "", cl::desc("Generator to run"), cl::location(::generator));
  cl::ParseCommandLineOptions(argc, argv, "BiShengIR HIVM IR Contract Gen");

  return TableGenMain(argv[0], &bishengirHivmIrContractGenMain);
}
