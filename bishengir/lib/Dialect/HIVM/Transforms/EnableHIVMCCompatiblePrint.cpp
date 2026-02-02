//===- EnableHIVMCCompatiblePrint.cpp - Convert ops to HIVMC Compatible Ops -===//
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
//===------------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
#define GEN_PASS_DEF_ENABLEHIVMCCOMPATIBLEPRINT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;
using namespace bishengir;

struct EnableHIVMCCompatiblePrintPass
    : public impl::EnableHIVMCCompatiblePrintBase<
          EnableHIVMCCompatiblePrintPass> {
  void runOnOperation() {
    auto *ctx = &getContext();
    Operation *moduleOp = getOperation();
    moduleOp->setAttr(hacc::HIVMCCompatiblePrintAttr::name,
                      BoolAttr::get(ctx, true));
  }
};

std::unique_ptr<Pass> mlir::hivm::createEnableHIVMCCompatiblePrintPass() {
  return std::make_unique<EnableHIVMCCompatiblePrintPass>();
}
