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

#include <fstream>
#include <optional>

#ifdef __GLIBC__
#include <gnu/libc-version.h> // gnu_get_libc_version()
#endif

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "bishengir/InitAllTranslations.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/regbase/Driver.h"
#include "bishengir/Tools/hivmc/BishengLlcBridge.h" // resolveBishengBackendMode, resolveBishengLlcLibrary
#include "bishengir/Tools/hivmc/CompileScope.h" // CompileScope RAII owner
#include "bishengir/Version/Version.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VersionTuple.h"

namespace {
// getMainExecutable needs an address inside this binary; ISO C++ forbids using
// `main` as a normal function pointer.
static void bishengirCompileExecutableAnchor() {}
} // namespace

/// Check if any argv is --target VALUE, --target=VALUE, -target VALUE, or
/// -target=VALUE where VALUE matches Ascend910_95* (Ascend910_950z,
/// Ascend910_9579, ...) or Ascend950*.
static bool hasRegBaseTarget(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    llvm::StringRef target;
    if (arg.starts_with("--target="))
      target = arg.drop_front(9);
    else if (arg.starts_with("-target="))
      target = arg.drop_front(8);
    else if ((arg == "--target" || arg == "-target") && i + 1 < argc)
      target = argv[++i];
    else
      continue;
    if (mlir::hacc::utils::isRegBasedArch(target))
      return true;
  }
  return false;
}

static int runBishengirCompile91095(int argc, char **argv) {
  llvm::SmallVector<llvm::StringRef> arguments;
  arguments.push_back(""); // placeholder, replaced by execute with full path
  for (int i = 1; i < argc; ++i)
    arguments.push_back(argv[i]);
  if (failed(bishengir::execute("bishengir-compile-a5",
                                bishengir::getBiShengInstallPath(), arguments)))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

/// Runtime version of the C library this process is running on, read via
/// gnu_get_libc_version().  Returns std::nullopt when the platform is not
/// glibc (the call does not exist) or when the reported string cannot be
/// parsed.  The version MUST be queried at runtime: the compile-time
/// __GLIBC__/__GLIBC_MINOR__ macros reflect the host that produced the binary,
/// not the host that runs it, and would misfire for a wheel built on a newer
/// glibc than the machine it is installed on.
static std::optional<llvm::VersionTuple> getRuntimeGlibcVersion() {
#ifdef __GLIBC__
  const char *verStr = gnu_get_libc_version();
  llvm::VersionTuple version;
  // Note the inverted polarity: tryParse() returns true on FAILURE.
  if (!verStr || version.tryParse(verStr))
    return std::nullopt;
  return version;
#else
  return std::nullopt;
#endif
}

static void printVersion(llvm::raw_ostream &os) {
  os << bishengir::getBiShengIRToolFullVersion("bishengir-compile") << '\n';
}

static void registerAndParseCLIOptions(int argc, char **argv) {
  // Register any command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
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

  // For regbase targets (Ascend910_95* / Ascend950*):
  //  - set BISHENGIR_LEGACY_A5_REGBASE=1 to use the legacy
  //    A5 regbase pipeline;
  //    without it, run merged native pipeline.
  // TODO: delegation will be removed after bishengir-compile and
  // bishengir-compile-a5 are merged.
  if (hasRegBaseTarget(argc, argv)) {
    if (getenv("BISHENGIR_LEGACY_A5_REGBASE")) {
      return runBishengirCompile91095(argc, argv);
    }
    llvm::errs() << "[INFO] Using merged native A5 regbase pipeline\n";
  }

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
  bishengir::registerAllTranslations();
  bishengir::registerAllToLLVMIRTranslations(registry);

  // Parse command line.
  registerAndParseCLIOptions(argc, argv);

  // Create config from command line options.
  bool regbase = hasRegBaseTarget(argc, argv);
  bishengir::BiShengIRCompileMainConfig config =
      bishengir::BiShengIRCompileMainConfig::createFromCLOptions(regbase);
  config.setExecutablePath(bishengir::getExecutablePath(
      argv[0], reinterpret_cast<void *>(&bishengirCompileExecutableAnchor)));
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

  // ---- CompileScope: RAII owner of the true-async runtime ----
  // Native mode (default) requires libbisheng-llc.so.  Resolve the SO path
  // (BISHENG_LLC_LIBRARY env, else
  // <bisheng-install>/bin/../lib/libbisheng-llc.so located via
  // getBiShengCompilerInstallPath); if it cannot be found, fall back to the
  // external `bisheng` process so the baseline invocation keeps working without
  // the SO installed.
  bishengir::CompileScope compileScope;
  {
    bool isNative = (bishengir::resolveBishengBackendMode() ==
                     bishengir::BishengBackendMode::Native);
    bool isCpuRunner = config.shouldEnableCPURunner();
    std::string soPath;
    if (isNative && !isCpuRunner) {
      // The Native path (vendor SO, pure LLVM backend) cannot reproduce the
      // full driver translation of `--cce-enable-sanitizer`/`-g`/
      // `--enable-debug-variables`: those reach the vendor only via cc1
      // front-end flags and -mllvm options that the SO does not accept.
      // Force the external `bisheng` process so the user-visible behavior of
      // these options stays identical across backends.
      if (config.getEnableSanitizer() || config.getEnableDebugInfo() ||
          config.getEnableDebugVariables()) {
        llvm::errs() << "[INFO] debug/sanitizer options enabled; using the"
                        " external `bisheng` process (Native path does not"
                        " reproduce the driver translation).\n";
        isNative = false;
      } else {
        // The dual-lane Native path dlmopen()s libbisheng-llc.so into TWO
        // separate LM_ID_NEWLM namespaces (one per AIC/AIV worker) so the
        // vendor's LLVM15 cl::opt globals stay isolated from the host LLVM19
        // and from each other.  On glibc < 2.35 the dynamic loader mishandles
        // the SECOND dlmopen of a SO whose static initializer has already run
        // in the first namespace: the second namespace's init dereferences a
        // NULL pointer (SEGV_MAPERR, si_addr=NULL) during `calling init`.
        // glibc 2.35+ fixed the dl_tls/namespace-static-init interaction.
        // Require glibc >= 2.35 for the Native path; otherwise fall back to
        // the external `bisheng` process (Process mode) so old distros
        // (Ubuntu 20.04 / glibc 2.31, etc.) keep working.
        //
        // BISHENG_NATIVE_DLMOPEN_FORCE bypasses the glibc gate for
        // environments that report < 2.35 but are known-good for
        // dual-namespace dlmopen.  It only skips the version check; the SO
        // path is still resolved below, because an empty soPath makes
        // CompileScope::maybeCreateRuntime() return nullptr and silently
        // drops the process to Process mode instead of enabling Native.
        static const llvm::VersionTuple requiredGlibcVersion(2, 35);
        std::optional<llvm::VersionTuple> glibcVersion =
            getRuntimeGlibcVersion();
        // A glibc that reports a version we cannot parse is treated as
        // unsupported, i.e. the same conservative outcome as "too old".
        bool glibcTooOld =
            !glibcVersion || *glibcVersion < requiredGlibcVersion;
        if (glibcTooOld && getenv("BISHENG_NATIVE_DLMOPEN_FORCE")) {
          llvm::errs() << "[WARN] BISHENG_NATIVE_DLMOPEN_FORCE is set; skipping"
                          " the glibc >= 2.35 requirement for dual-namespace"
                          " dlmopen even though this process runs on glibc "
                       << (glibcVersion ? glibcVersion->getAsString() : "?")
                       << ".  Proceeding at the risk of a startup SEGV"
                          " (SEGV_MAPERR) in the second namespace.\n";
        } else if (glibcTooOld) {
          llvm::errs() << "[INFO] glibc "
                       << (glibcVersion ? glibcVersion->getAsString() : "?")
                       << " < 2.35 cannot safely dlmopen libbisheng-llc.so"
                          " into two namespaces (second-namespace static"
                          " init crashes with SEGV_MAPERR); falling back to"
                          " external `bisheng` process.  Set"
                          " BISHENG_NATIVE_DLMOPEN_FORCE=1 to override on a"
                          " known-good newer glibc.\n";
          isNative = false;
        }
        if (isNative) {
          soPath = bishengir::resolveBishengLlcLibrary();
          if (soPath.empty() || !llvm::sys::fs::exists(soPath)) {
            llvm::errs() << "[INFO] libbisheng-llc.so not found"
                         << (soPath.empty() ? "" : (" at " + soPath))
                         << "; falling back to external `bisheng` process.\n";
            isNative = false;
          }
        }
      }
    }
    (void)compileScope.maybeCreateRuntime(/*isNative=*/isNative,
                                          /*isCpuRunner=*/isCpuRunner, soPath);
  }
  bishengir::BishengAsyncRuntime *runtime = compileScope.runtime();

  bishengir::CompileTiming timing; // shared between reg- and membased pipeline
  // regbase pipeline entry
  if (regbase)
    return bishengir::regbase::runRegBaseCompile(module, config, originalCLArgs,
                                                 runtime);

  auto res = runBiShengIRPipeline(module, config, runtime);
  if (failed(res)) {
    llvm::errs() << "[ERROR] Failed to run BiShengIR pipeline\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
