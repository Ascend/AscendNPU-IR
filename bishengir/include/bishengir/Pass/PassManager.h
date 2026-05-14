//===- PassManager.h - Pass Management Interface ----------------*- C++ -*-===//
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

#ifndef BISHENGIR_PASS_PASSMANAGER_H
#define BISHENGIR_PASS_PASSMANAGER_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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
  BiShengIRCompileConfigBase config;

  BiShengIRPassManager(const BiShengIRCompileConfigBase &config,
                       mlir::MLIRContext *ctx, llvm::StringRef operationName,
                       Nesting nesting)
      : PassManager(ctx, operationName, nesting), config(&config) {
    ctx->registerActionHandler([](llvm::function_ref<void()> execute,
                                  const mlir::tracing::Action &action) {
      auto *passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&action);
      if (!passAction) {
        execute();
        return;
      }

      mlir::Operation *op = passAction->getOp();
      llvm::StringRef passArg = passAction->getPass().getArgument();

      // Adaptor passes (empty argument) orchestrate nested pipelines — always
      // let them through so nested ops still get processed.
      if (passArg.empty()) {
        execute();
        return;
      }

      // Helper: returns true if passArg is excluded by a FilterPassesAttr.
      auto isFiltered = [&](mlir::Operation *candidate) -> bool {
        auto attr =
            candidate->getAttrOfType<mlir::annotation::FilterPassesAttr>(
                mlir::annotation::FilterPassesAttr::name);
        if (!attr)
          return false;
        llvm::SmallVector<llvm::StringRef> allowed;
        attr.getPasses().getValue().split(allowed, ',');
        for (llvm::StringRef entry : allowed)
          if (entry.trim() == passArg)
            return false;
        return true; // attr present but passArg not listed
      };

      // Case 1: the op itself is filtered — skip entirely.
      if (isFiltered(op))
        return;

      // Case 2: op is a module — temporarily remove child ops that are
      // filtered for this pass, execute, then restore them in order.
      if (auto mod = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        mlir::Block *body = mod.getBody();

        // Collect ops to hide.
        llvm::SmallVector<mlir::Operation *> hidden;
        for (mlir::Operation &childOp : llvm::make_early_inc_range(*body)) {
          if (isFiltered(&childOp)) {
            hidden.push_back(&childOp);
            childOp.remove();
          }
        }

        execute();

        // Restore in original order: insert each op after its predecessor.
        for (auto *op : llvm::reverse(hidden)) {
          body->push_front(op);
        }
        return;
      }

      execute();
    });
  }

#if MLIR_ENABLE_EXECUTION_ENGINE
  mlir::LogicalResult run(mlir::Operation *op);

private:
  void filterCPURunnerPasses(mlir::OpPassManager &originalPM);
#endif // MLIR_ENABLE_EXECUTION_ENGINE
};

} // namespace bishengir

#endif // BISHENGIR_PASS_PASSMANAGER_H
