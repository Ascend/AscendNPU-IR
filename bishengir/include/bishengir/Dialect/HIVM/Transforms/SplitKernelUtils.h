//===- SplitKernelUtils.h - MIX split helpers -------------------*- C++ -*-===//
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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/TypeSwitch.h"

#include <utility>

namespace mlir {
namespace hivm {
class TCoreRatioAttr;

inline std::optional<TCoreType> getExplicitCoreType(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<TCoreType>>(op)
      .Case<CustomOp, CustomMacroOp>(
          [](auto customOp) { return customOp.getCoreType(); })
      .Case<DebugOp, LoadOp>([](auto nativeOp) -> std::optional<TCoreType> {
        if (auto attr = nativeOp.getTcoretypeAttr())
          return attr.getTcoretype();
        return std::nullopt;
      })
      .Case<SyncBlockSetOp, SyncBlockWaitOp, SyncIntraBlockSetOp,
            SyncIntraBlockWaitOp>(
          [](auto nativeOp) -> std::optional<TCoreType> {
            return nativeOp.getTcoreTypeAttr().getTcoretype();
          })
      .Default(
          [](Operation *) -> std::optional<TCoreType> { return std::nullopt; });
}

/// Update the explicit placement field read by getExplicitCoreType.
inline void setExplicitCoreType(Operation *op, TCoreType core) {
  auto attr = TCoreTypeAttr::get(op->getContext(), core);
  llvm::TypeSwitch<Operation *>(op)
      .Case<CustomOp, CustomMacroOp>(
          [&](auto customOp) { customOp.setCoreType(core); })
      .Case<DebugOp, LoadOp>(
          [&](auto nativeOp) { nativeOp.setTcoretypeAttr(attr); })
      .Case<SyncBlockSetOp, SyncBlockWaitOp, SyncIntraBlockSetOp,
            SyncIntraBlockWaitOp>(
          [&](auto nativeOp) { nativeOp.setTcoreTypeAttr(attr); });
}

/// Resolve explicitly flexible placement after function classification. Query
/// operation-specific inference before applying any function-wide fallbacks:
/// notify, load and debug inference must still see the original dependencies.
/// With specializeBoth, also specialize both-core annotations in split halves.
inline void resolveFlexibleCoreTypes(
    func::FuncOp func, bool specializeBoth = false,
    TFuncCoreType defaultCoreType = TFuncCoreType::AIC_OR_AIV) {
  auto funcAttr =
      func->getAttrOfType<TFuncCoreTypeAttr>(TFuncCoreTypeAttr::name);
  if (hacc::utils::isHost(func))
    return;
  auto funcCore = funcAttr ? funcAttr.getFuncCoreType() : defaultCoreType;
  TCoreType fallback;
  switch (funcCore) {
  case TFuncCoreType::AIC:
    fallback = TCoreType::CUBE;
    break;
  case TFuncCoreType::AIV:
    fallback = TCoreType::VECTOR;
    break;
  case TFuncCoreType::MIX:
    fallback = TCoreType::CUBE_AND_VECTOR;
    // A single-core MIX task must not acquire a requirement for the absent
    // core solely because it contains flexible operations.
    if (auto ratio = getCoreRatioAttr(func)) {
      if (ratio.getCube() == 0)
        fallback = TCoreType::VECTOR;
      else if (ratio.getVector() == 0)
        fallback = TCoreType::CUBE;
    }
    break;
  default:
    return;
  }

  struct Resolution {
    Operation *op;
    TCoreType core;
  };
  SmallVector<Resolution> resolutions;
  func.walk([&](Operation *op) {
    if (op == func.getOperation())
      return;
    // Builtin declarations have a fixed placement contract checked by their
    // verifier, even when the surrounding function has a concrete core type.
    if (llvm::TypeSwitch<Operation *, bool>(op)
            .Case<CustomOp, CustomMacroOp>(
                [](auto customOp) { return customOp.isBuiltin(); })
            .Default(false))
      return;
    auto explicitCore = getExplicitCoreType(op);
    if (!explicitCore)
      return;
    auto core = *explicitCore;
    bool flexible = core == TCoreType::CUBE_OR_VECTOR;
    if (!flexible && !(specializeBoth && core == TCoreType::CUBE_AND_VECTOR))
      return;
    if (auto inferred = detail::queryCoreTypeHelper(op))
      core = *inferred;
    if (core == TCoreType::CUBE_OR_VECTOR ||
        (specializeBoth && funcCore != TFuncCoreType::MIX &&
         core == TCoreType::CUBE_AND_VECTOR))
      core = fallback;
    resolutions.push_back({op, core});
  });
  for (const Resolution &resolution : resolutions)
    setExplicitCoreType(resolution.op, resolution.core);
}

/// Verify that a mixed function or split core half has no required work when
/// its core_ratio reserves zero instances of that core.
LogicalResult verifyCoreRatio(func::FuncOp func);

bool isSingleCoreRatio(TCoreRatioAttr ratio);

/// Rename the original function to AIC and clone an AIV function. Validate
/// symbol uses and preserve a private MIX declaration for host callers.
/// Return the AIC/AIV pair; each caller must prune the two bodies.
FailureOr<std::pair<func::FuncOp, func::FuncOp>>
createSplitMixFunctions(func::FuncOp func);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_SPLITKERNELUTILS_H
