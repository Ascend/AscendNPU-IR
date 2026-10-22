//===- SplitKernelUtils.h - MIX split helpers ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_SPLITKERNELUTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_SPLITKERNELUTILS_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace hivm {

/// Emit the private MIX declaration used by host callers. No declaration is
/// created when the kernel has no callers.
inline LogicalResult generateMixKernelDecl(func::FuncOp funcOp) {
  auto module = funcOp->getParentOfType<ModuleOp>();
  std::optional<SymbolTable::UseRange> maybeUses = funcOp.getSymbolUses(module);
  if (!maybeUses || maybeUses.value().empty())
    return success();

  if (!llvm::all_of(maybeUses.value(), [&](SymbolTable::SymbolUse use) {
        auto call = cast<func::CallOp>(use.getUser());
        auto caller = call->getParentOfType<hacc::HACCFunction>();
        return caller.isHost();
      })) {
    return funcOp.emitError()
           << "Currently, MIX kernels can only be called by host functions!";
  }

  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  auto funcDeclOp = builder.create<func::FuncOp>(
      funcOp->getLoc(), funcOp.getSymName(), funcOp.getFunctionType());
  funcDeclOp.setPrivate();
  funcDeclOp->setAttr(hacc::HACCFuncTypeAttr::name,
                      hacc::HACCFuncTypeAttr::get(funcOp.getContext(),
                                                  hacc::HACCFuncType::DEVICE));
  funcDeclOp->setAttr(hivm::TFuncCoreTypeAttr::name,
                      hivm::TFuncCoreTypeAttr::get(funcOp.getContext(),
                                                   hivm::TFuncCoreType::MIX));
  if (auto coreRatio = hivm::getCoreRatioAttr(funcOp))
    funcDeclOp->setAttr(hivm::TCoreRatioAttr::name, coreRatio);
  if (hacc::utils::isDeviceEntry(funcOp))
    funcDeclOp->setAttr(
        hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::MIX_ENTRY),
        UnitAttr::get(funcOp.getContext()));
  return success();
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_SPLITKERNELUTILS_H
