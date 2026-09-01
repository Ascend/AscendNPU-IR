//===- hivmc.cpp - BiShengIR Compile Driver ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for hivmc built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Config/bishengir-config.h"
#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"

#include "bishengir/InitAllPasses.h"
#include "bishengir/InitAllPassesA3.h"

#include "bishengir/InitAllTranslations.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Target/LLVMIR/Dialect/All.h"
#include "bishengir/Tools/hivmc/Utility.h"
#include "bishengir/Version/Version.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

static void printVersion(llvm::raw_ostream &os) {
  std::string verStr = bishengir::getHivmcFullVersion();
  os << verStr;
}

std::string registerAndParseCLIOptions(int argc, char **argv) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  // Register any command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  bishengir::HIVMCMainConfig::registerCLOptions();
  bishengir::registerPassManagerCLOptions();
#if BISHENGIR_ENABLE_PM_CL_OPTIONS
  // Enable full pass management abilities.
  mlir::registerPassManagerCLOptions();
#endif
  // Register version printer
  llvm::cl::SetVersionPrinter(printVersion);
  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv, "HIVMC Compile Tool\n");

  StringTmpPath path(inputFilename.getValue());
  llvm::cantFail(llvm::errorCodeToError(hivmcCanonicalizePath(path)),
                 "failed to canonicalize input file path.");
  inputFilename.setValue(path.str().str());
  return inputFilename.getValue();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);

  // Register passes.
  mlir::registerAllPasses();

  // Register dialect extensions.
  mlir::registerAllExtensions(registry);
  bishengir::registerAllExtensions(registry);

  // Register translations.
  mlir::registerAllToLLVMIRTranslations(registry);
  bishengir::registerAllTranslations();
  bishengir::registerAllToLLVMIRTranslations(registry);

  // Parse command line.
  auto inputFile = registerAndParseCLIOptions(argc, argv);

  // Create config from command line options.
  bishengir::HIVMCMainConfig config =
      bishengir::HIVMCMainConfig::createFromCLOptions();

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFile, &errorMessage);
  if (!file) {
    llvm::errs() << "[ERROR] Failed to open input file: "
                 << (inputFile == "-" ? "stdin" : inputFile)
                 << " error message: " << errorMessage << '\n';
    return EXIT_FAILURE;
  }

  // bad! // bad! // bad!
  if (config.shouldCompileA5()) {
    bishengir::registerAllPasses();
  } else if (config.shouldCompileA3()) {
    bishengir::registerAllPassesA3();
  }
  // bad! // bad! // bad!

  // create context
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), mlir::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> moduleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "[ERROR] Failed to parse input file:  "
                 << (inputFile == "-" ? "stdin" : inputFile) << '\n';
    return EXIT_FAILURE;
  }

  mlir::ModuleOp module = *moduleRef;
  if (failed(bishengir::runHIVMCCompile(module, config))) {
    llvm::errs() << "[ERROR] Failed to run HIVMC pipeline\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
