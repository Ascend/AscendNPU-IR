//===- bishengir-compile.cpp - BiShengIR Compile Driver ---------*- C++ -*-===//
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
// Main entry function for bishengir-compile built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Version/Version.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

static void printVersion(llvm::raw_ostream &os) {
  os << bishengir::getBiShengIRToolFullVersion("bishengir-compile") << '\n';
}

void registerAndParseCLIOptions(int argc, char **argv) {
  // Register any command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  bishengir::BiShengIRCompileMainConfig::registerCLOptions();
  bishengir::registerPassManagerCLOptions();
  // Enable full pass management abilities.
  mlir::registerPassManagerCLOptions();

  // Register version printer
  llvm::cl::SetVersionPrinter(printVersion);
  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv, "BiShengIR Compile Tool\n");
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

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

  // Parse command line.
  registerAndParseCLIOptions(argc, argv);

  // Create config from command line options.
  bishengir::BiShengIRCompileMainConfig config =
      bishengir::BiShengIRCompileMainConfig::createFromCLOptions();
  // Check the validity of intput/output options
  if (failed(checkInOutOptionsValidity(config))) {
    return EXIT_FAILURE;
  }

  std::string errorMessage;
  std::string inputFile = config.getInputFile();
  auto file = mlir::openInputFile(inputFile, &errorMessage);
  if (!file) {
    llvm::errs() << "[ERROR] Failed to open input file: "
                 << (inputFile == "-" ? "stdin" : inputFile)
                 << " error message: " << errorMessage << '\n';
    return EXIT_FAILURE;
  }

  // create context
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects(config.getAllowUnregisteredDialects());

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), mlir::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> moduleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "[ERROR] Failed to parse input file:  "
                 << (inputFile == "-" ? "stdin" : inputFile) << '\n';
    return EXIT_FAILURE;
  }
  mlir::ModuleOp module = moduleRef.release();

  auto res = runBiShengIRPipeline(module, config);
  if (failed(res)) {
    llvm::errs() << "[ERROR] Failed to run BiShengIR pipeline\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
