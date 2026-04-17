//===- FlattenMemDescArgs.cpp - Flatten memdesc struct args -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After ConvertTritonAscendGPUToLLVM, memdesc function arguments become
// llvm.struct<(ptr<3>, i32, ...)>.  This pass replaces each such argument
// with a bare ptr<6> (UB address space), rewriting extractvalue %arg[0]
// to use the pointer directly, and propagating the address space change
// (ptr<3> → ptr<6>) through the def-use chain.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_FLATTENMEMDESCARGS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

/// Check if a type is a memdesc-lowered struct: struct<(ptr<AS>, i32, ...)>.
static bool isMemDescStruct(Type ty) {
  auto structTy = dyn_cast<LLVM::LLVMStructType>(ty);
  if (!structTy || structTy.isIdentified() || structTy.getBody().empty())
    return false;
  // First element must be a pointer.
  if (!isa<LLVM::LLVMPointerType>(structTy.getBody().front()))
    return false;
  // Remaining elements must be i32 (offsets).
  for (size_t i = 1; i < structTy.getBody().size(); ++i) {
    auto intTy = dyn_cast<IntegerType>(structTy.getBody()[i]);
    if (!intTy || intTy.getWidth() != 32)
      return false;
  }
  return true;
}

/// Propagate an address space change through the def-use chain.
/// Starting from `root`, for every op that uses a ptr<oldAS> value
/// transitively reachable from root, rewrite its ptr results to ptr<newAS>.
static void propagateAddressSpace(Value root, unsigned oldAS, unsigned newAS) {
  auto oldPtrTy = LLVM::LLVMPointerType::get(root.getContext(), oldAS);
  auto newPtrTy = LLVM::LLVMPointerType::get(root.getContext(), newAS);

  SmallVector<Value> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (OpOperand &use : val.getUses()) {
      Operation *user = use.getOwner();
      // Update any results that have the old pointer type.
      for (OpResult result : user->getResults()) {
        if (result.getType() == oldPtrTy) {
          result.setType(newPtrTy);
          worklist.push_back(result);
        }
      }
    }
  }
}

class FlattenMemDescArgsPass
    : public impl::FlattenMemDescArgsBase<FlattenMemDescArgsPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod.walk([&](LLVM::LLVMFuncOp funcOp) {
      SmallVector<unsigned> structArgIndices;
      auto funcTy = funcOp.getFunctionType();

      // Identify memdesc struct args.
      for (unsigned i = 0; i < funcTy.getNumParams(); ++i) {
        if (isMemDescStruct(funcTy.getParamType(i)))
          structArgIndices.push_back(i);
      }

      if (structArgIndices.empty())
        return;

      // UB address space.
      constexpr unsigned ubAS = 6;
      Type ubPtrTy = LLVM::LLVMPointerType::get(funcOp.getContext(), ubAS);

      // For each struct arg, rewrite uses and change the arg type.
      for (unsigned argIdx : structArgIndices) {
        auto structTy =
            cast<LLVM::LLVMStructType>(funcTy.getParamType(argIdx));
        // Get the original address space from the struct's pointer field.
        auto origPtrTy =
            cast<LLVM::LLVMPointerType>(structTy.getBody().front());
        unsigned origAS = origPtrTy.getAddressSpace();

        BlockArgument arg = funcOp.getArgument(argIdx);

        // Collect extractvalue users before modifying.
        SmallVector<LLVM::ExtractValueOp> toErase;
        bool hasError = false;

        for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
          auto extractOp =
              dyn_cast<LLVM::ExtractValueOp>(use.getOwner());
          if (!extractOp) {
            // Non-extractvalue use of the struct arg — skip (could be
            // an unrealized_conversion_cast or similar).
            continue;
          }

          auto position = extractOp.getPosition();
          if (position.size() != 1) {
            extractOp->emitError("unexpected nested extractvalue on "
                                 "memdesc struct arg");
            hasError = true;
            continue;
          }

          int64_t idx = position[0];
          if (idx == 0) {
            // extractvalue %arg[0] → use the pointer arg directly.
            toErase.push_back(extractOp);
          } else if (extractOp.getResult().use_empty()) {
            // Dead extractvalue at non-zero index — safe to erase.
            toErase.push_back(extractOp);
          } else {
            extractOp->emitError("unexpected live extractvalue at index ")
                << idx << " on memdesc struct arg (only index 0 expected)";
            hasError = true;
          }
        }

        if (hasError) {
          signalPassFailure();
          return;
        }

        // Change the arg type from struct to ptr<6>.
        arg.setType(ubPtrTy);

        // Replace extractvalue[0] uses with the arg directly; erase dead ones.
        for (auto extractOp : toErase) {
          if (extractOp.getPosition()[0] == 0) {
            extractOp.getResult().replaceAllUsesWith(arg);
          }
          extractOp->erase();
        }

        // Propagate address space change (ptr<origAS> → ptr<6>) through
        // the def-use chain so that getelementptr, store, etc. all use <6>.
        if (origAS != ubAS)
          propagateAddressSpace(arg, origAS, ubAS);
      }

      // Update the function type.
      SmallVector<Type> newParamTypes;
      for (unsigned i = 0; i < funcTy.getNumParams(); ++i) {
        BlockArgument arg = funcOp.getArgument(i);
        newParamTypes.push_back(arg.getType());
      }
      auto newFuncTy = LLVM::LLVMFunctionType::get(
          funcTy.getReturnType(), newParamTypes, funcTy.isVarArg());
      funcOp.setFunctionType(newFuncTy);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFlattenMemDescArgsPass() {
  return std::make_unique<FlattenMemDescArgsPass>();
}

} // namespace triton
} // namespace bishengir
