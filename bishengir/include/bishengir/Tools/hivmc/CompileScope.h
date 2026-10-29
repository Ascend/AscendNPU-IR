//===- CompileScope.h - RAII owner of BishengAsyncRuntime -------*- C++ -*-===//
//
// Compile-scope RAII wrapper owning the true-async runtime.  Created at the
// CLI main() compile scope; owns the runtime and ensures drain+join on
// destruction.  A null runtime (Process mode, CPU Runner mode, or Native
// mode with no available libbisheng-llc.so) causes compileDeviceKernel to
// fall back to the external `bisheng` process.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_COMPILE_SCOPE_H
#define BISHENGIR_TOOLS_HIVMC_COMPILE_SCOPE_H

#include "bishengir/Tools/hivmc/BishengAsyncRuntime.h"

#include <memory>

namespace bishengir {

/// RAII owner of BishengAsyncRuntime, created at the CLI main() compile scope.
class CompileScope {
public:
  CompileScope() = default;
  ~CompileScope() = default;

  CompileScope(const CompileScope &) = delete;
  CompileScope &operator=(const CompileScope &) = delete;
  CompileScope(CompileScope &&) noexcept = default;
  CompileScope &operator=(CompileScope &&) noexcept = default;

  /// Create the async runtime iff Native mode is requested, CPU Runner is
  /// off, and \p soPath is non-empty (i.e. libbisheng-llc.so was located).
  /// Returns the owned runtime pointer (nullptr when not created — caller
  /// falls back to the external `bisheng` process).
  BishengAsyncRuntime *maybeCreateRuntime(bool isNative, bool isCpuRunner,
                                          llvm::StringRef soPath) {
    if (!isNative || isCpuRunner || soPath.empty())
      return nullptr;
    runtime_ = std::make_unique<BishengAsyncRuntime>(soPath);
    return runtime_.get();
  }

  /// Accessor; nullptr when the runtime was not created.
  BishengAsyncRuntime *runtime() const { return runtime_.get(); }

private:
  std::unique_ptr<BishengAsyncRuntime> runtime_;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_COMPILE_SCOPE_H
