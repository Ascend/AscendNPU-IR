//===- IntermediateValuesAnalyzer.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Analysis/IntermediateValuesAnalyzer.h"
#include "bishengir/Dialect/HFusion/Utils/ScalarOptimizationsUtils.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::dataflow;

namespace mlir {
namespace hfusion {
namespace detail {

bool isAllowedOp(Operation *op) {
  auto isIntOrFloat = [](Type type) { return type.isIntOrFloat(); };

  if (!llvm::all_of(op->getOperandTypes(), isIntOrFloat))
    return false;

  if (op->getResults().empty() || !isIntOrFloat(op->getResult(0).getType()))
    return false;

  // In case if operation has non-trivial rounding mode we should skip it,
  // since the tensor lowering pipeline ignores it.
  if (auto roundingOp =
          mlir::dyn_cast<mlir::arith::ArithRoundingModeInterface>(op)) {
    if (roundingOp.getRoundingModeAttr()) {
      op->emitWarning("[hfusion-optimize-scalar-transfers]: cant change device "
                      "for operation with explicit rounding mode");
      return false;
    }
  }

  if (isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
          arith::MulFOp, arith::MulIOp, arith::DivUIOp, arith::MaxSIOp,
          arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp, arith::NegFOp,
          arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MinNumFOp,
          arith::MinimumFOp, arith::MaxNumFOp, arith::MaximumFOp, arith::ShLIOp,
          arith::ShRSIOp, arith::ShRUIOp, arith::FloorDivSIOp,
          arith::CeilDivSIOp, arith::CeilDivUIOp, math::ExpOp, math::LogOp,
          math::AbsFOp, arith::ExtFOp, arith::FPToSIOp, arith::FPToUIOp,
          arith::SIToFPOp, arith::UIToFPOp, arith::ExtSIOp, arith::ExtUIOp,
          arith::TruncIOp, arith::TruncFOp>(op))
    return true;

  return false;
}

void ConsumedMark::print(raw_ostream &os) const {
  os << "<consumed-mark>: " << isMarked;
}

void ProducedMark::print(raw_ostream &os) const {
  os << "<produced-mark>: " << isMarked;
}

/// The monotonic Gen/Kill transfer function
/// The "consumed" property is generated iff
///   operation is scalar to vector transfer.
///
/// The "consumed" property is killed iff
///   Operation is not simple math operation
///   or scalar to vector transfer
///
/// Operation is considered vector to scalar transfer iff:
///   (a) Operation is `tensor.insert` with type
///       `tensor<ElemT>` or `tensor<1xElemT>`
///   (b) Operation is `linalg::fillOp` with
///       first rank tensor type

LogicalResult
ConsumedValuesAnalysis::visitOperation(Operation *op,
                                       ArrayRef<ConsumedMark *> operands,
                                       ArrayRef<const ConsumedMark *> results) {

  auto propagateOperands = [&](ConsumedMark *mark) {
    propagateIfChanged(mark, mark->mark());
  };

  // Generate "consumed" property and propagate it
  // upwards in the DataFlow graph
  if (isScalarToVectorTransfer(op)) {
    propagateOperands(operands[0]);
    return success();
  }

  // Kill "consumed" property if the operation is a vector-to-scalar transfer,
  // an SCF operation, or should be placed directly on the scalar unit.
  if (!isAllowedOp(op)) {
    return success();
  }

  // We assume that any allowedOperation have
  // only one returned result
  assert(results.size() == 1);
  const auto *result = results.front();

  if (result->isMarked) {
    llvm::for_each(operands, propagateOperands);
  }

  return success();
}

/// The monotonic Gen/Kill transfer function
/// The "produced" (by vector context)
/// property is generated iff
/// operation is vector transfer.
///
/// The "produced" property is killed iff
///   Operation is not simple math operation
///   or vector to scalar transfer
///
/// Operation is considered vector to scalar transfer iff:
///   (a) Operation is `tensor.extract` with type
///       `tensor<ElemT>` or `tensor<1xElemT>`

LogicalResult
ProducedValuesAnalysis::visitOperation(Operation *op,
                                       ArrayRef<const ProducedMark *> operands,
                                       ArrayRef<ProducedMark *> results) {

  auto propagateResults = [&](ProducedMark *mark) {
    propagateIfChanged(mark, mark->mark());
  };

  // Generate "produced" property
  if (isVectorToScalarTransfer(op)) {
    llvm::for_each(results, propagateResults);
    return success();
  }

  // Kill "produced" property iff the operation
  // is scalarToVectorTransfer, scf operation or
  // if this operations should be placed on scalar unit
  if (!isAllowedOp(op)) {
    return success();
  }

  // The input of the operation
  // input = join(operands, false)
  auto input =
      llvm::any_of(operands, [](const ProducedMark *m) { return m->isMarked; });

  // Iff input is not killed, propagate result of join
  if (input) {
    llvm::for_each(results, propagateResults);
  }

  return success();
}

// IntermediateValuesAnalyzer

IntermediateValuesAnalyzer::IntermediateValuesAnalyzer(Operation *root) {
  SymbolTableCollection symbolTable;

  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<ProducedValuesAnalysis>();
  solver.load<ConsumedValuesAnalysis>(symbolTable);
  (void)solver.initializeAndRun(root);
}

bool IntermediateValuesAnalyzer::isIntermediateTransfer(Value val) {
  auto *producedMark = solver.lookupState<ProducedMark>(val);
  auto *consumedMark = solver.lookupState<ConsumedMark>(val);

  return producedMark && consumedMark && producedMark->isMarked &&
         consumedMark->isMarked;
}

bool IntermediateValuesAnalyzer::isIntermediateTransfer(Operation *op) {
  return !op->getResults().empty() &&
         llvm::any_of(op->getResults(), [&](Value result) {
           return isIntermediateTransfer(result);
         });
}

void IntermediateValuesAnalyzer::registerNewValue(Value val, bool isMarked) {
  assert(!wasVisited(val) &&
         "value was previously visited by analysis, can't re-register it");

  solver.getOrCreateState<ProducedMark>(val)->isMarked = isMarked;
  solver.getOrCreateState<ConsumedMark>(val)->isMarked = isMarked;
}

bool IntermediateValuesAnalyzer::wasVisited(Value val) {
  auto *producedMark = solver.lookupState<ProducedMark>(val);
  auto *consumedMark = solver.lookupState<ConsumedMark>(val);

  // Value was already visited by dataflow analysis
  if (producedMark || consumedMark) {
    return true;
  }

  return false;
}

void IntermediateValuesAnalyzer::invalidate(Operation *op) {
  solver.eraseState(solver.getProgramPointAfter(op));
  for (Value result : op->getResults())
    solver.eraseState(result);
}

void IntermediateValuesAnalyzer::invalidate(Block *block) {
  for (BlockArgument argument : block->getArguments())
    solver.eraseState(argument);
}

} // namespace detail
} // namespace hfusion
} // namespace mlir
