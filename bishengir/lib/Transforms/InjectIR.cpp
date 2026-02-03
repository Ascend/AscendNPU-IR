//===--------------------- InjectIR.cpp -----------------------------------===//
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

#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"

namespace bishengir {
#define GEN_PASS_DEF_INJECTIR
#include "bishengir/Transforms/Passes.h.inc"
} // namespace bishengir

namespace {
using namespace mlir;
using namespace bishengir;
using namespace bishengir::impl;

#define DEBUG_TYPE "inject-ir"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

struct InjectIR : InjectIRBase<InjectIR> {
  using InjectIRBase::InjectIRBase;

  explicit InjectIR(const InjectIROptions &options)
      : InjectIRBase::InjectIRBase(options) {}

  void runOnOperation() override {
    LDBG("filePath = " << filePath);
    ModuleOp module = getOperation();
    if (filePath.empty())
      return;

    if (!llvm::sys::fs::exists(filePath)) {
      module.emitError() << "inject IR file does not exist: " << filePath;
      signalPassFailure();
      return;
    }

    mlir::ParserConfig config(module.getContext());
    auto loadedRef = mlir::parseSourceFile<mlir::ModuleOp>(filePath, config);
    if (!loadedRef) {
      module.emitError() << "failed to parse inject IR file: " << filePath;
      signalPassFailure();
      return;
    }

    ModuleOp loadedModule = loadedRef.get();
    for (func::FuncOp loadedFunc :
         loadedModule.getBody()->getOps<func::FuncOp>()) {
      StringRef name = loadedFunc.getSymName();
      if (name.empty())
        continue;
      func::FuncOp currentFunc = module.lookupSymbol<func::FuncOp>(name);
      if (!currentFunc)
        continue;

      Region &currentBody = currentFunc.getBody();
      Region &loadedBody = loadedFunc.getBody();

      while (!currentBody.empty())
        currentBody.front().erase();

      IRMapping mapper;
      loadedBody.cloneInto(&currentBody, mapper);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
bishengir::createInjectIRPass(llvm::StringRef filePath) {
  InjectIROptions options;
  options.filePath = filePath.str();
  return std::make_unique<InjectIR>(options);
}
