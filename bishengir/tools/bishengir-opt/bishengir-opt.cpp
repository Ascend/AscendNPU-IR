//===- bishengir-opt.cpp - BiShengIR Optimizer Driver -----------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
// Main entry function for bishengir-opt built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Version/Version.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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

/// Check if arg is --target=VALUE or -target=VALUE where VALUE matches
/// Ascend910_95* or Ascend950*.
static bool isAscend910_95TargetArg(llvm::StringRef arg) {
  llvm::StringRef target;
  if (arg.starts_with("--target="))
    target = arg.drop_front(9);
  else if (arg.starts_with("-target="))
    target = arg.drop_front(8);
  else
    return false;
  return target.starts_with("Ascend910_95") || target.starts_with("Ascend950");
}

/// Check if any argv is --target=VALUE or -target=VALUE where VALUE matches
/// Ascend910_95* (Ascend910_950z, Ascend910_9579, ...) or Ascend950*.
static bool hasAscend910_95Target(int argc, char **argv) {
  for (int i = 1; i < argc; ++i)
    if (isAscend910_95TargetArg(argv[i]))
      return true;
  return false;
}

static int runBishengirOptA5(int argc, char **argv) {
  llvm::SmallVector<llvm::StringRef> arguments;
  arguments.push_back(""); // placeholder, replaced by execute with full path
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (isAscend910_95TargetArg(arg))
      continue; // skip --target and -target options when delegating to A5
    arguments.push_back(arg);
  }
  if (failed(bishengir::execute("bishengir-opt-a5",
                                bishengir::getBiShengInstallPath(), arguments)))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

static void printVersion(llvm::raw_ostream &os) {
  os << bishengir::getBiShengIRToolFullVersion("bishengir-opt") << '\n';
}

int main(int argc, char **argv) {
  // If --target=Ascend910_95* or --target=Ascend950* is specified, delegate to
  // bishengir-opt-91095.
  // TODO: this will be removed after bishengir-opt and bishengir-opt-a5
  // are merged.
  if (hasAscend910_95Target(argc, argv))
    return runBishengirOptA5(argc, argv);

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
