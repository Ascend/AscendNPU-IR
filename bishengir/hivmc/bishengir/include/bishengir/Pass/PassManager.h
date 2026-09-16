//===- PassManager.h - Pass Management Interface ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_PASS_PASSMANAGER_H
#define BISHENGIR_PASS_PASSMANAGER_H

#include "mlir/Pass/PassManager.h"

namespace bishengir {
/// Register a set of useful command-line options that can be used to configure
/// a pass manager. The values of these options can be applied via the
/// 'applyPassManagerCLOptions' method below.
void registerPassManagerCLOptions();

/// Apply any values provided to the pass manager options that were registered
/// with 'registerPassManagerOptions'.
llvm::LogicalResult applyPassManagerCLOptions(mlir::PassManager &pm);

// A pass manager that allows filtering the passes before running. It's more
// expensive to use with compared to mlir::PassManager.
class BiShengIRPassManager : public mlir::PassManager {
public:
  using PassManager::PassManager;
#ifdef BISHENGIR_ENABLE_EXECUTION_ENGINE
  mlir::LogicalResult run(mlir::Operation *op);

private:
  void filterCPURunnerPasses(mlir::OpPassManager &originalPM);
#endif // BISHENGIR_ENABLE_EXECUTION_ENGINE
};

} // namespace bishengir

#endif // BISHENGIR_PASS_PASSMANAGER_H
