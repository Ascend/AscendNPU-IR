//===- InsertAllocBasePlaceholder.cpp - CoreType Inference Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "insert-alloc-base-placeholder"

namespace mlir {
#define GEN_PASS_DEF_INSERTALLOCBASEPLACEHOLDER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
class InsertAllocBasePlaceholderPass
    : public impl::InsertAllocBasePlaceholderBase<InsertAllocBasePlaceholderPass> {
public:
  void runOnOperation() override;
};
} // namespace

void InsertAllocBasePlaceholderPass::runOnOperation() {
  static constexpr int64_t PlaceholderSize = 1024;
  auto module = getOperation();
  module->walk([&](func::FuncOp funcOp) {
    if (util::isSIMTVF(funcOp)) {
      auto symUses = SymbolTable::getSymbolUses(funcOp, module);
      if (!symUses || symUses->empty()) {
        llvm::report_fatal_error("Expect at least one simt vf callsite");
      }

      auto *ctx = &getContext();
      auto loc = UnknownLoc::get(ctx);
      OpBuilder builder(ctx);
      auto ty = MemRefType::get({PlaceholderSize}, builder.getI8Type());
      auto sharedAttr = DictionaryAttr::get(
          ctx, {NamedAttribute(SharedMemoryAttr::name, builder.getUnitAttr())});
      funcOp.insertArgument(funcOp.getNumArguments(), ty, sharedAttr, loc);

      for (auto use : *symUses) {
        auto callOp = cast<func::CallOp>(use.getUser());
        builder.setInsertionPoint(callOp);
        auto shared = builder.create<memref::AllocOp>(callOp.getLoc(), ty);
        shared->setAttr(SharedMemoryAttr::name, builder.getUnitAttr());
        auto operands = callOp.getArgOperandsMutable();
        operands.append({shared});
      }
    }
  });
}

std::unique_ptr<Pass> mlir::hivm::createInsertAllocBasePlaceholderPass() {
  return std::make_unique<InsertAllocBasePlaceholderPass>();
}
