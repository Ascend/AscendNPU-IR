//===- bishengir-compile.cpp - BiShengIR Compile Driver ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/Utility.h"
#include "bishengir/Version/Version.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"

namespace {
// getMainExecutable needs an address inside this binary; ISO C++ forbids using
// `main` as a normal function pointer.
static void bishengirCompileExecutableAnchor() {}
} // namespace

static void printVersion(llvm::raw_ostream &os) {
  os << bishengir::getBiShengIRToolFullVersion("bishengir-compile") << '\n';
}

std::string registerAndParseCLIOptions(int argc, char **argv) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  // Register any command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  bishengir::BiShengIRCompileMainConfig::registerCLOptions();
  bishengir::registerPassManagerCLOptions();
#if BISHENGIR_ENABLE_PM_CL_OPTIONS
  // Enable full pass management abilities.
  mlir::registerPassManagerCLOptions();
#endif

  // Register version printer
  llvm::cl::SetVersionPrinter(printVersion);
  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv, "BiShengIR Compile Tool\n");

  StringTmpPath path(inputFilename.getValue());
  llvm::cantFail(llvm::errorCodeToError(canonicalizePath(path)),
                 "failed to canonicalize input file path.");
  inputFilename.setValue(path.str().str());
  return inputFilename.getValue();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Save original CLI arguments before parsing so they can be forwarded to
  // the downstream hivmc toolchain.
  std::vector<std::string> originalCLArgs;
  for (int i = 1; i < argc; ++i)
    originalCLArgs.emplace_back(argv[i]);

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

  // Register translations.
  mlir::registerAllToLLVMIRTranslations(registry);

  // Parse command line.
  auto inputFile = registerAndParseCLIOptions(argc, argv);

  // Create config from command line options.
  bishengir::BiShengIRCompileMainConfig config =
      bishengir::BiShengIRCompileMainConfig::createFromCLOptions();

  config.setExecutablePath(getExecutablePath(
      argv[0], reinterpret_cast<void *>(&bishengirCompileExecutableAnchor)));

  // Forward original CLI arguments as fallback for downstream hivmc toolchain.
  // Filter out arguments that are handled purely by bishengir-compile itself.
  {
    std::vector<std::string> filteredArgs;
    for (const auto &arg : originalCLArgs) {
      llvm::StringRef argRef(arg);
      // Skip positional (input file) and output file arguments.
      if (!argRef.starts_with("-"))
        continue;
      // Skip arguments handled directly by bishengir-compile, not hivmc.
      if (argRef.starts_with("-o=") || argRef == "-o")
        continue;
      if (argRef.starts_with("--hivmc-args"))
        continue;
      if (argRef.contains("enable-lir-compile"))
        continue;
      filteredArgs.push_back(arg);
    }
    config.setClArgs(std::move(filteredArgs));
  }

  std::string errorMessage;
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

  mlir::ModuleOp module = *moduleRef;
  if (failed(inferLayoutOptimization(module, config))) {
    llvm::errs() << "[ERROR] Failed to infer layout optimization\n";
    return EXIT_FAILURE;
  }
  if (config.getEnableTritonKernelCompile()) {
    if (failed(inferMixedCV(module, config))) {
      llvm::errs() << "[ERROR] Failed to infer mix mode\n";
      return EXIT_FAILURE;
    }
  }

  if (failed(bishengir::runBiShengIRPipeline(module, config))) {
    llvm::errs() << "[ERROR] Failed to run BiShengIR pipeline\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
