//===- IntermediateValuesAnalyzer.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_ANALYSIS_INTERMEDIATEVALUESANALYZER_H
#define BISHENGIR_DIALECT_HFUSION_ANALYSIS_INTERMEDIATEVALUESANALYZER_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace hfusion {
namespace detail {

bool isAllowedOp(Operation *op);

template <typename Derived>
struct Mark : public dataflow::AbstractSparseLattice {
  using AbstractSparseLattice::AbstractSparseLattice;

  ChangeResult mark();

  ChangeResult join(const AbstractSparseLattice &other) override {
    const auto &otherMark = static_cast<const Derived &>(other);
    if (!otherMark.isMarked)
      return ChangeResult::NoChange;
    return mark();
  }

  ChangeResult meet(const AbstractSparseLattice &other) override {
    return join(other);
  }

  bool isMarked = false;
};

template <typename Derived> ChangeResult Mark<Derived>::mark() {
  bool wasMarked = isMarked;
  isMarked = true;
  return wasMarked ? ChangeResult::NoChange : ChangeResult::Change;
}

struct ConsumedMark : public Mark<ConsumedMark> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConsumedMark)
  using Mark::Mark;

  void print(raw_ostream &os) const override;
};

class ConsumedValuesAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<ConsumedMark> {
public:
  ConsumedValuesAnalysis(DataFlowSolver &solver,
                         SymbolTableCollection &symbolTable)
      : dataflow::SparseBackwardDataFlowAnalysis<ConsumedMark>(solver,
                                                               symbolTable) {}

  void setToExitState(ConsumedMark *mark) override { (void)mark; }

  void visitBranchOperand(OpOperand &operand) override { (void)operand; }

  void visitCallOperand(OpOperand &operand) override { (void)operand; }

  LogicalResult visitOperation(Operation *op, ArrayRef<ConsumedMark *> operands,
                               ArrayRef<const ConsumedMark *> results) override;
};

struct ProducedMark : public Mark<ProducedMark> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProducedMark)
  using Mark::Mark;

  void print(raw_ostream &os) const override;
};

class ProducedValuesAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ProducedMark> {
public:
  ProducedValuesAnalysis(DataFlowSolver &solver)
      : dataflow::SparseForwardDataFlowAnalysis<ProducedMark>(solver) {}

  void setToEntryState(ProducedMark *mark) override { (void)mark; }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const ProducedMark *> operands,
                               ArrayRef<ProducedMark *> results) override;
};

class IntermediateValuesAnalyzer {
public:
  explicit IntermediateValuesAnalyzer(Operation *root);

  bool isIntermediateTransfer(Value val);
  bool isIntermediateTransfer(Operation *op);

  void registerNewValue(Value val, bool isMarked = true);
  bool wasVisited(Value val);

  void invalidate(Operation *op);
  void invalidate(Block *block);

private:
  DataFlowSolver solver;
};

} // namespace detail
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_ANALYSIS_INTERMEDIATEVALUESANALYZER_H
