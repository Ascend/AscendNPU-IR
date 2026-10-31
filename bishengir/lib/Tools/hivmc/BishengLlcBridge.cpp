//===- BishengLlcBridge.cpp - In-process bridge to libbisheng-llc.so -------===//
//
// See header.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/hivmc/BishengLlcBridge.h"
#include "bishengir/Tools/hivmc/Utility.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <dlfcn.h>
#include <link.h>

#ifndef LM_ID_NEWLM
#define LM_ID_NEWLM ((long)-1)
#endif

#define BRIDGE_LOG "[BishengLlcBridge] "

namespace bishengir {

BishengBackendMode resolveBishengBackendMode() {
  const char *env = getenv("BISHENG_BACKEND");
  if (!env || env[0] == '\0')
    return BishengBackendMode::Native;
  llvm::StringRef v(env);
  if (v == "process")
    return BishengBackendMode::Process;
  if (v == "native" || v == "llc" || v == "direct")
    return BishengBackendMode::Native;
  llvm::errs() << BRIDGE_LOG "Unknown BISHENG_BACKEND='" << v
               << "'; using Native.\n";
  return BishengBackendMode::Native;
}

std::string defaultBishengLlcLibraryPath() {
  // libbisheng-llc.so ships with the BiSheng (ccec) compiler, NOT with
  // bishengir-compile: it lives at <bisheng-install>/bin/../lib/libbisheng-llc.so
  // (e.g. ccec_toolkits/<ver>/bisheng_compiler/lib/libbisheng-llc.so).
  // Resolve the BiSheng install `bin` directory via the shared
  // getBiShengCompilerInstallPath helper (BISHENG_INSTALL_PATH → PATH lookup of
  // `bisheng` → ASCEND_HOME_PATH/bin), then append ../lib/libbisheng-llc.so.
  //
  // The `bisheng` entry under that `bin` directory may itself be a symlink
  // (e.g. dev/ccec_compiler/bin/bisheng -> .../BiSheng-CCE/build/install/bin/ccec).
  // The SO ships in the REAL installation tree, not necessarily alongside the
  // symlink, so resolve the symlink to its real path and derive `../lib` from
  // the real binary's location.  Falls back to the unresolved `bin` directory
  // if the real-path lookup fails (e.g. bisheng missing or dangling symlink).
  std::string bishengBin = bishengir::getBiShengCompilerInstallPath();
  if (bishengBin.empty())
    return "";
  llvm::SmallString<128> bishengFile(bishengBin);
  llvm::sys::path::append(bishengFile, "bisheng");
  llvm::SmallString<128> realBin;
  if (!llvm::sys::fs::real_path(bishengFile, realBin, /*expand_tilde=*/false) &&
      !realBin.empty()) {
    // realBin = <real-install>/bin/bisheng -> bin dir = parent_path.
    bishengBin = llvm::sys::path::parent_path(realBin).str();
  }
  llvm::SmallString<128> libPath(bishengBin);
  llvm::sys::path::append(libPath, "..", "lib", "libbisheng-llc.so");
  return libPath.str().str();
}

std::string resolveBishengLlcLibrary() {
  if (const char *env = getenv("BISHENG_LLC_LIBRARY")) {
    if (env[0] != '\0')
      return std::string(env);
  }
  return defaultBishengLlcLibraryPath();
}

std::string laneTargetCpu(llvm::StringRef archBase, bool isCube) {
  // archBase is "dav-c220" / "dav-c310" / "dav-920r1"; append -vec/-cube.
  // Vendor v2 accepts only the four documented CPUs (dav-c220-vec/cube,
  // dav-c310-vec/cube).  dav-920r1 is not yet supported by the vendor.
  std::string cpu = archBase.str();
  cpu += isCube ? "-cube" : "-vec";
  return cpu;
}

std::string bishengLlcDumpDir() {
  const char *env = getenv("BISHENG_LLC_DUMP_DIR");
  if (!env || env[0] == '\0')
    return "";
  return std::string(env);
}

llvm::StringRef bishengLlcStatusName(int status) {
  switch (status) {
  case BISHENG_LLC_OK:
    return "OK";
  case BISHENG_LLC_BAD_ARGUMENT:
    return "BAD_ARGUMENT";
  case BISHENG_LLC_BAD_STATE:
    return "BAD_STATE";
  case BISHENG_LLC_AUX_ERROR:
    return "AUX_ERROR";
  case BISHENG_LLC_BACKEND_ERROR:
    return "BACKEND_ERROR";
  case BISHENG_LLC_IO_ERROR:
    return "IO_ERROR";
  default:
    return "UNKNOWN";
  }
}

BishengLlcBridge::BishengLlcBridge() = default;

BishengLlcBridge::~BishengLlcBridge() {
  // If init() created a context but compile() was never called, destroy it
  // here.  If compile() ran, it already destroyed the context and set ctx_
  // null, so this is a no-op.  Per the vendor contract, dlclose must come
  // AFTER destroy (the SO's code may still touch the context during destroy).
  destroyCtx_();
  if (handle_) {
    dlclose(handle_);
    handle_ = nullptr;
  }
}

bool BishengLlcBridge::load(llvm::StringRef soPath) {
  std::string path = soPath.str();
  if (path.empty()) {
    const char *env = getenv("BISHENG_LLC_LIBRARY");
    if (env && env[0] != '\0')
      path = env;
  }

  // dlmopen into a new namespace so LLVM15 cl::opt globals are isolated per
  // lane.  RTLD_NOW resolves all symbols up front; RTLD_LOCAL hides them.
  handle_ = dlmopen(LM_ID_NEWLM, path.c_str(),
                    RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    llvm::errs() << BRIDGE_LOG "dlmopen failed for '" << path
                 << "': " << dlerror() << "\n";
    return false;
  }

  // Record the LMID for isolation verification (dlinfo RTLD_DI_LMID).
  lmid_ = 0;
  dlinfo(handle_, RTLD_DI_LMID, &lmid_);

// Resolve exactly the four v2 symbols.
#define RESOLVE(name, field, type)                                            \
  do {                                                                        \
    dlerror(); /* clear */                                                    \
    field = reinterpret_cast<type>(dlsym(handle_, name));                     \
    const char *err = dlerror();                                              \
    if (err || !field) {                                                      \
      llvm::errs() << BRIDGE_LOG "dlsym('" name "') failed: "                 \
                   << (err ? err : "symbol is NULL") << "\n";                 \
      field = nullptr;                                                        \
      dlclose(handle_);                                                       \
      handle_ = nullptr;                                                      \
      return false;                                                           \
    }                                                                         \
  } while (0)

  RESOLVE("bishengLlcInit", fnInit_, int (*)(BishengLlcContext **));
  RESOLVE("bishengLlcPreload", fnPreload_,
          int (*)(BishengLlcContext *, const char *));
  RESOLVE("bishengLlcCompile", fnCompile_,
          int (*)(BishengLlcContext *, const BishengLlcConfig *,
                  const void *, size_t, const char *const *, size_t,
                  const char *));
  RESOLVE("bishengLlcDestroy", fnDestroy_, void (*)(BishengLlcContext *));
#undef RESOLVE

  return true;
}

bool BishengLlcBridge::init() {
  if (!handle_ || !fnInit_) {
    llvm::errs() << BRIDGE_LOG "init() called before successful load().\n";
    return false;
  }
  if (ctx_) {
    llvm::errs() << BRIDGE_LOG "init() called twice.\n";
    return false;
  }
  int st = fnInit_(&ctx_);
  if (st != BISHENG_LLC_OK || !ctx_) {
    llvm::errs() << BRIDGE_LOG "bishengLlcInit failed: "
                 << bishengLlcStatusName(st) << " (" << st << ")\n";
    ctx_ = nullptr;
    return false;
  }
  initCount_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void BishengLlcBridge::destroyCtx_() {
  if (!ctx_)
    return;
  if (fnDestroy_)
    fnDestroy_(ctx_);
  ctx_ = nullptr;
  destroyCount_.fetch_add(1, std::memory_order_relaxed);
}

bool BishengLlcBridge::preloadAux(llvm::StringRef path) {
  if (!ctx_ || !fnPreload_) {
    llvm::errs() << BRIDGE_LOG "preloadAux() before init().\n";
    return false;
  }
  int st = fnPreload_(ctx_, path.str().c_str());
  if (st != BISHENG_LLC_OK) {
    llvm::errs() << BRIDGE_LOG "bishengLlcPreload('" << path
                 << "') failed: " << bishengLlcStatusName(st) << "\n";
    return false;
  }
  return true;
}

BishengLlcCompileResult
BishengLlcBridge::compile(llvm::StringRef irBytes,
                          const llvm::SmallVectorImpl<std::string> &auxPaths,
                          llvm::StringRef targetCpu, uint32_t optLevel,
                          const std::vector<std::string> &extraLlvmOpts,
                          llvm::StringRef outputPath) {
  BishengLlcCompileResult result;
  if (compiled_) {
    result.diagnostic = "compile() called more than once on one bridge";
    return result;
  }
  compiled_ = true;

  if (!ctx_ || !fnCompile_) {
    result.diagnostic = "compile() before init()/load()";
    return result;
  }

  // Build stable backing storage for the C ABI pointers.  The vendor reads
  // all strings only during the call, but we keep them alive until the call
  // returns regardless.
  std::vector<const char *> libCPtrs;
  libCPtrs.reserve(auxPaths.size());
  for (const auto &p : auxPaths)
    libCPtrs.push_back(p.c_str());

  std::vector<const char *> optCPtrs;
  optCPtrs.reserve(extraLlvmOpts.size());
  for (const auto &o : extraLlvmOpts)
    optCPtrs.push_back(o.c_str());

  // Keep the CPU string alive across the fnCompile_ call.  `targetCpu.str()`
  // returns a temporary std::string; binding `.c_str()` directly to
  // `config.TargetCpu` would dangle at the end of the full expression.
  std::string targetCpuStr = targetCpu.str();
  std::string outPath = outputPath.str();

  BishengLlcConfig config{};
  config.TargetCpu = targetCpuStr.c_str();
  config.OptLevel = optLevel;
  config.ExtraLlvmOpts = optCPtrs.empty() ? nullptr : optCPtrs.data();
  config.ExtraOptCount = optCPtrs.size();

  // Vendor v2 ABI: (Ctx, Config, Kernel, KernelSize, LibPaths, LibCount,
  // ObjectPath).  Kernel is the in-memory LLVM IR buffer; LibPaths are the
  // ordered aux bitcode paths (preloaded paths reuse the vendor cache).
  int st = fnCompile_(ctx_, &config,
                      irBytes.data(), irBytes.size(),
                      libCPtrs.empty() ? nullptr : libCPtrs.data(),
                      libCPtrs.size(), outPath.c_str());
  result.vendorStatus = st;

  // The vendor contract is Init -> [Preload]* -> Compile -> Destroy: compile
  // CONSUMES the context (sets Compiled=true, refuses a second compile) but
  // does NOT free it.  Destroy is the only release path.  Call destroy here
  // for every outcome (success, vendor error, post-OK integrity error) so the
  // LLVM15 LLVMContext + cached aux modules are released promptly and exactly
  // once per Init.  After this, ctx_ is null and the bridge is in the
  // Consumed state; resetForRetry must reset() + load() + init() again.
  destroyCtx_();

  if (st != BISHENG_LLC_OK) {
    result.diagnostic =
        ("bishengLlcCompile failed: " + bishengLlcStatusName(st).str() +
         " (" + std::to_string(st) + ")");
    return result;
  }

  // Verify the object file exists and is non-empty (vendor guarantees this
  // on OK, but double-check for robustness).
  uint64_t size = 0;
  if (llvm::sys::fs::file_size(outPath, size) || size == 0) {
    result.diagnostic = "bishengLlcCompile returned OK but object is missing/empty";
    return result;
  }

  result.success = true;
  result.outputPath = outPath;
  return result;
}

void BishengLlcBridge::destroy() {
  destroyCtx_();
}

void BishengLlcBridge::reset() {
  // Tear down the vendor context (if still alive — e.g. init() succeeded but
  // compile() was never called, or a preload failed before compile) and
  // dlclose the dlmopen namespace so the next `load()` produces a fresh LMID
  // + a fresh Vendor context.  After compile() ran, ctx_ is already null
  // (compile() destroys the context), so this only destroys a never-compiled
  // context.  dlclose must come AFTER destroy.  Used by
  // `BishengLlcWorker::resetForRetry()` to recreate a lane after the previous
  // Compile consumed the context (workerState_ == Consumed).  The caller
  // must re-`load()` + `init()` before any preload/compile.
  destroyCtx_();
  if (handle_) {
    dlclose(handle_);
    handle_ = nullptr;
  }
  fnInit_ = nullptr;
  fnPreload_ = nullptr;
  fnCompile_ = nullptr;
  fnDestroy_ = nullptr;
  lmid_ = 0;
  compiled_ = false;
}

} // namespace bishengir