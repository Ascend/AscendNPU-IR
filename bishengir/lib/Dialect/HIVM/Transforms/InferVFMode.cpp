//===-------------- InferVFMode.cpp - Infer VF Mode for Ops ---------------===//
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
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "hivm-infer-vf-mode"

namespace mlir {
#define GEN_PASS_DEF_INFERVFMODE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

namespace mlir::hivm {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const std::optional<VFMode> &vfMode) {
  if (!vfMode) {
    OS << " { } ";
    return OS;
  }
  switch (*vfMode) {
  case VFMode::MIX: {
    OS << " { MIX } ";
    break;
  }
  case VFMode::SIMD: {
    OS << " { SIMD } ";
    break;
  }
  case VFMode::SIMT: {
    OS << " { SIMT } ";
    break;
  }
  }
  return OS;
}

static inline std::optional<VFMode> getVFMode(Operation *op) {
  if (const auto vfModeAttr = op->getAttrOfType<VFModeAttr>(VFModeAttr::name))
    return vfModeAttr.getValue();
  return std::nullopt;
}

static inline bool isMIX(Operation *op) {
  if (const auto vfMode = getVFMode(op))
    return vfMode == VFMode::MIX;

  return false;
}

template <typename... OpTypes> static bool isVectorOp(Operation *op) {
  return (... || isa<OpTypes>(op));
}

static inline bool isSIMD(Operation *op) {
  if (const auto vfMode = getVFMode(op))
    return vfMode == VFMode::SIMD;

  if (isVectorOp<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
          >(op))
    return true;

  return op->hasAttr(VectorFunctionAttr::name);
}

static inline bool isSIMT(Operation *op) {
  return getVFMode(op) == VFMode::SIMT;
}

using Op2VFModeMap = llvm::DenseMap<Operation *, std::optional<VFMode>>;

static std::optional<VFMode> inferVFMode(Operation *rootOp,
                                         Op2VFModeMap &op2VfMode);

/// Classifies the VFMode of an operation based on its immediate properties.
static std::optional<VFMode> classifyRootOperation(Operation *op,
                                                   Op2VFModeMap &op2VfMode) {
  if (isMIX(op))
    return VFMode::MIX;
  if (isSIMD(op))
    return VFMode::SIMD;
  if (isSIMT(op))
    return VFMode::SIMT;
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    SymbolTable symtab(op->getParentOfType<ModuleOp>());
    if (auto callee = symtab.lookup<func::FuncOp>(callOp.getCallee())) {
      return inferVFMode(callee, op2VfMode);
    }
  }
  return std::nullopt;
}

/// Resolves the final VFMode by walking nested operations and detecting
/// conflicts.
///
/// If any nested operation has a different mode than the current mode, the
/// result becomes MIX. Early exits when MIX mode is already determined.
static std::optional<VFMode>
resolveNestedModes(Operation *rootOp, std::optional<VFMode> currentMode,
                   Op2VFModeMap &op2VfMode) {
  std::optional<VFMode> vfMode = currentMode;

  rootOp->walk([&](Operation *op) {
    // Skip the root operation itself
    if (op == rootOp)
      return WalkResult::advance();

    // Early exit if already determined to be MIX
    if (vfMode && *vfMode == VFMode::MIX)
      return WalkResult::interrupt();

    // Recursively infer mode for nested operation
    std::optional<VFMode> nestedMode = inferVFMode(op, op2VfMode);

    // Ops are SIMD defaulty. SIMT ops are specified.
    if (!nestedMode && !isa<func::ReturnOp>(op)) {
      nestedMode = VFMode::SIMD;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      op->print(llvm::dbgs());
      llvm::dbgs() << " Query: " << nestedMode << "\n";
    });

    // Detect mode conflict
    if (vfMode && nestedMode && *vfMode != *nestedMode) {
      vfMode = VFMode::MIX;
      return WalkResult::interrupt();
    }

    // Update mode if this is the first nested operation with a mode
    if (nestedMode)
      vfMode = nestedMode;

    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      op->print(llvm::dbgs());
      llvm::dbgs() << " Current: " << vfMode << "\n";
    });

    return WalkResult::advance();
  });

  return vfMode;
}

/// Infers the vectorization/parallelization mode (VFMode) for an operation.
///
/// This function determines the VFMode by examining the operation and its
/// nested operations. The inference follows these rules:
/// - Direct classification: Check if the operation is MIX, SIMD, or SIMT
/// - Function calls: Recursively infer mode from the callee function
/// - Nested operations: Walk the operation tree to find conflicting modes
/// - Mixed mode: If nested operations have different modes, result is MIX
///
/// For function operations, the inferred mode is stored as an attribute.
static std::optional<VFMode> inferVFMode(Operation *rootOp,
                                         Op2VFModeMap &op2VfMode) {
  if (!rootOp)
    return std::nullopt;

  if (auto it = op2VfMode.find(rootOp); it != op2VfMode.end())
    return it->second;

  if (auto existingMode = getVFMode(rootOp)) {
    op2VfMode[rootOp] = existingMode;
    return existingMode;
  }

  std::optional<VFMode> vfMode = classifyRootOperation(rootOp, op2VfMode);

  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    rootOp->print(llvm::dbgs());
    llvm::dbgs() << " Current: " << vfMode << "\n";
  });

  vfMode = resolveNestedModes(rootOp, vfMode, op2VfMode);

  op2VfMode[rootOp] = vfMode;

  return vfMode;
}

struct InferVFModePass : public impl::InferVFModeBase<InferVFModePass> {
  void runOnOperation() override;
};

void InferVFModePass::runOnOperation() {
  Operation *op = getOperation();
  if (!hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>()))
    return;

  if (!hacc::utils::isDeviceEntry(op))
    return;

  Op2VFModeMap op2VfMode;
  if (auto vfMode = inferVFMode(op, op2VfMode))
    op->setAttr(VFModeAttr::name,
                VFModeAttr::get(op->getContext(), *vfMode));
}

} // namespace mlir::hivm

std::unique_ptr<mlir::Pass> mlir::hivm::createInferVFModePass() {
  return std::make_unique<InferVFModePass>();
}
