//===- FusibleBlock.h --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <queue>

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCK_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCK_H

namespace mlir {
namespace hfusion {
namespace opfusion {
class FusibleBlock {
  using InclusionCheck = std::function<bool(Operation *, Operation *)>;

public:
  explicit FusibleBlock(const llvm::ArrayRef<Operation *> ops,
                        const FusibleHelper *fusibleHelper)
      : fusibleHelper_(fusibleHelper), ops_(ops.begin(), ops.end()){};
  explicit FusibleBlock(const llvm::ArrayRef<Operation *> ops,
                        const FusibleHelper *fusibleHelper,
                        const llvm::ArrayRef<Operation *> mod)
      : fusibleHelper_(fusibleHelper), ops_(ops.begin(), ops.end()),
        outsModification_(mod.begin(), mod.end()){};

  Operation *getLastOp() { return getOutputs().back().getDefiningOp(); }
  template <typename T> T getParentOfType() const {
    return getOps().back()->getParentOfType<T>();
  }
  Location getLoc() const { return getOps().back()->getLoc(); }

  llvm::ArrayRef<Operation *> getOps() const { return ops_.getArrayRef(); }

  llvm::ArrayRef<Value> getInputs() {
    if (ins_.empty())
      visitInValues();
    return ins_.getArrayRef();
  }

  llvm::ArrayRef<Value> getOutputs() {
    if (outs_.empty())
      visitOutValues();
    return outs_.getArrayRef();
  }

  llvm::ArrayRef<Operation *> getOpWithAuxs() {
    if (opWithAuxs_.empty())
      visitAuxiliaryOps();
    return opWithAuxs_.getArrayRef();
  }
  void dump();

  const FusibleHelper *fusibleHelper_;

private:
  void visitOutValues();
  void fillNonEdgeOps();
  void visitAuxiliaryOps();
  void visitInValues();
  void processOperandForBFS(const Value &operand, Operation *pivotOp,
                            DenseSet<Operation *> &visited,
                            std::queue<Operation *> &workQueue,
                            const InclusionCheck &shouldInclude,
                            const DenseSet<Operation *> &blocker);
  void auxBFS(const SetVector<Operation *> &initialOps,
              DenseSet<Operation *> &visited,
              const InclusionCheck &shouldInclude,
              const DenseSet<Operation *> &blocker);
  void auxBFSDown(const SetVector<Operation *> &initialOps,
                  DenseSet<Operation *> &visited,
                  const InclusionCheck &shouldInclude,
                  const DenseSet<Operation *> &blocker);

  bool isPossibleCountingAux(Operation *defOp);
  bool isValidAuxOrBuffer(Operation *defOp, Operation *pivotOp);
  bool isValidBuffer(Operation *defOp, Operation *pivotOp);

  mutable llvm::SetVector<Operation *> ops_;
  mutable llvm::SetVector<Operation *> outsModification_;
  mutable llvm::SetVector<Operation *> opWithAuxs_;
  mutable llvm::SetVector<Operation *> nonEdgeOps_;
  mutable llvm::SetVector<Value> ins_;
  mutable llvm::SetVector<Value> outs_;
};
} // namespace opfusion
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCK_H
