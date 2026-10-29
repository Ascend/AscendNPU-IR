//===- BishengLlcBridge.h - In-process bridge to libbisheng-llc.so -*- C++ -*-===//
//
// Loads the Vendor v2 in-process IR-to-object compiler service
// (libbisheng-llc.so, 4-symbol C ABI) via dlmopen and compiles one device
// kernel per lane.  Used by the Native backend (default); the external
// `bisheng` process is the fallback when the SO is unavailable.
//
//   - dlmopen(LM_ID_NEWLM, RTLD_NOW|RTLD_LOCAL) inside the owning worker so
//     LLVM15 cl::opt state is isolated per lane (AIC/AIV get separate LMIDs).
//   - Exactly four dlsym lookups: bishengLlcInit/Preload/Compile/Destroy.
//   - One BishengLlcContext per worker; compile consumes it (success or
//     failure).  destroy() before dlclose(); worker join before freeing.
//   - IR is passed as raw bytes (the .ll file already written by saveToFiles);
//     no IR content modification.
//   - Aux modules passed as the complete ordered path list to compile();
//     preload() is an optional cache hint only.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_BISHENG_LLC_BRIDGE_H
#define BISHENGIR_TOOLS_HIVMC_BISHENG_LLC_BRIDGE_H

#include "bisheng-llc.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace bishengir {

/// Backend mode used by compileDeviceKernel.
enum class BishengBackendMode {
  Native,   ///< In-process libbisheng-llc.so v2 (default).
  Process,  ///< External `bisheng` process (opt-out via BISHENG_BACKEND=process).
};

/// Resolve the requested backend mode from BISHENG_BACKEND env.
/// Default (unset/empty) is Native.  "process" selects Process.
/// Any other value keeps Native (with a warning).
BishengBackendMode resolveBishengBackendMode();

/// Default location of libbisheng-llc.so relative to this executable:
/// `<bindir>/../lib/libbisheng-llc.so`.  Empty on failure.
std::string defaultBishengLlcLibraryPath();

/// Resolve the SO path for the Native backend: BISHENG_LLC_LIBRARY env if
/// set, otherwise defaultBishengLlcLibraryPath().  Empty if neither is
/// available.
std::string resolveBishengLlcLibrary();

/// Full target CPU string for a lane (e.g. "dav-c220-vec").
/// Mirrors the driver's --cce-aicore-arch mapping used by A3/A5.
std::string laneTargetCpu(llvm::StringRef archBase, bool isCube);

/// Per-lane observer dump directory (from BISHENG_LLC_DUMP_DIR, forwarded to
/// the vendor as-is).  Empty if unset.
std::string bishengLlcDumpDir();

/// Result of a single Native compile attempt.
struct BishengLlcCompileResult {
  bool success = false;
  int vendorStatus = -1;        ///< Raw BishengLlcStatus return value.
  std::string outputPath;       ///< Object file path on success.
  std::string diagnostic;       ///< Human-readable error on failure.
};

/// BishengLlcBridge — owns one dlmopen handle + one context per instance.
///
/// Lifecycle: create → (preload aux)* → compile (exactly once) → destroy.
/// One instance per lane worker; not thread-safe (single owning thread).
class BishengLlcBridge {
public:
  BishengLlcBridge();
  ~BishengLlcBridge();

  BishengLlcBridge(const BishengLlcBridge &) = delete;
  BishengLlcBridge &operator=(const BishengLlcBridge &) = delete;

  /// Load libbisheng-llc.so via dlmopen and resolve the 4 symbols.
  /// \p soPath overrides BISHENG_LLC_LIBRARY env if non-empty.
  /// Returns false if the SO is missing or any symbol is absent.
  bool load(llvm::StringRef soPath = "");

  bool isLoaded() const { return handle_ != nullptr; }

  /// Create the vendor context (bishengLlcInit).  Must follow load().
  bool init();

  /// Preload one aux module into the vendor cache (bishengLlcPreload).
  /// Optional; the file is always reparsed and replaces any cached entry
  /// under the same path.  Call at most once per distinct path.
  bool preloadAux(llvm::StringRef path);

  /// Compile one kernel.  Consumes AND destroys the vendor context (call at
  /// most once).  Per the vendor contract (Init -> [Preload]* -> Compile ->
  /// Destroy), compile marks the context consumed (Compiled=true) but does
  /// NOT free it; this method calls bishengLlcDestroy immediately after
  /// bishengLlcCompile returns (success or failure) so the LLVM15 LLVMContext
  /// + cached aux modules are released exactly once per Init.  After this
  /// returns, ctx_ is null and the bridge is in the Consumed state; a retry
  /// must reset() + load() + init() again.
  ///
  /// \p irBytes   raw LLVM IR text (the .ll file contents).
  /// \p auxPaths  complete ordered aux bitcode path list.
  /// \p targetCpu full CPU (e.g. "dav-c220-vec").
  /// \p optLevel  0 or 2.
  /// \p extraLlvmOpts  extra -mllvm-style option VALUES (no "-mllvm" prefix),
  ///                   applied after the CPU's driver-default table.
  /// \p outputPath  object file to write.
  BishengLlcCompileResult
  compile(llvm::StringRef irBytes,
          const llvm::SmallVectorImpl<std::string> &auxPaths,
          llvm::StringRef targetCpu, uint32_t optLevel,
          const std::vector<std::string> &extraLlvmOpts,
          llvm::StringRef outputPath);

  /// Destroy the vendor context and dlclose the handle.
  void destroy();

  /// Reset the bridge to its unloaded state so `load()` can be called
  /// again on the same instance, producing a NEW dlmopen namespace (new
  /// LMID) and a NEW Vendor context on the next `init()`.  Used by
  /// `BishengLlcWorker::resetForRetry()` to recreate a lane after a Compile
  /// consumed the previous context (workerState_ == Consumed) or after a
  /// failure (Failed), without leaking the old namespace/context.
  /// Calls `destroy()` (frees the vendor context if still alive) then
  /// `dlclose()` (unloads the dlmopen namespace), clears the symbol
  /// pointers, and resets `compiled_` / `lmid_`.  After reset, `isLoaded()`
  /// is false and `load()` must precede `init()` again.
  void reset();

  /// The LMID of this bridge's dlmopen namespace (for isolation verification).
  /// Returns 0 (base) if not loaded or dlinfo unavailable.
  long lmid() const { return lmid_; }

  /// Observability: number of successful bishengLlcInit calls on this bridge
  /// (each creates a vendor context that must be paired with exactly one
  /// bishengLlcDestroy).  Single-threaded access (the owning Worker thread),
  /// but atomic so the runtime can read it from the main thread without a
  /// lock for diagnostics.  Reset to 0 by reset().
  uint64_t initCount() const { return initCount_.load(); }
  /// Observability: number of bishengLlcDestroy calls on this bridge.  Must
  /// equal initCount() at quiescence (after the bridge is destroyed/reset).
  uint64_t destroyCount() const { return destroyCount_.load(); }

private:
  void *handle_ = nullptr;          ///< dlmopen handle (RTLD_LOCAL).
  long lmid_ = 0;                   ///< link-map ID (dlinfo RTLD_DI_LMID).
  BishengLlcContext *ctx_ = nullptr; ///< vendor context (owned).

  // Resolved function pointers (exactly four).
  int (*fnInit_)(BishengLlcContext **) = nullptr;
  int (*fnPreload_)(BishengLlcContext *, const char *) = nullptr;
  int (*fnCompile_)(BishengLlcContext *, const BishengLlcConfig *,
                    const void *, size_t, const char *const *, size_t,
                    const char *) = nullptr;
  void (*fnDestroy_)(BishengLlcContext *) = nullptr;

  bool compiled_ = false; ///< compile() called at most once.

  // Init/Destroy pairing counters for lifecycle observability.  At
  // quiescence initCount_ == destroyCount_; a mismatch indicates a leak
  // (initCount_ > destroyCount_) or a double-destroy (initCount_ <
  // destroyCount_).  Atomic for cross-thread diagnostic reads; mutated only
  // by the owning Worker thread.
  std::atomic<uint64_t> initCount_{0};
  std::atomic<uint64_t> destroyCount_{0};

  /// Call fnDestroy_ on ctx_ if non-null, then null ctx_.  Centralises the
  /// destroy + counter increment so every destroy site (compile, destroy,
  /// reset, ~BishengLlcBridge) is paired with exactly one counter increment.
  void destroyCtx_();
};

/// Map a BishengLlcStatus integer to a diagnostic string.
llvm::StringRef bishengLlcStatusName(int status);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_BISHENG_LLC_BRIDGE_H