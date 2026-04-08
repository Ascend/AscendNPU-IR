//===- EliminateSingleIterationScfFor.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Transforms/Passes.h"
#include "bishengir/Transforms/AffineMinMaxValueBounds.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace bishengir {

using namespace mlir;

#define GEN_PASS_DEF_ELIMINATESINGLEITERATIONSCFFOR
#include "bishengir/Transforms/Passes.h.inc"

namespace {

static std::string getContainingFuncName(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp) {
    if (auto barg = mlir::dyn_cast<BlockArgument>(v)) {
      defOp = barg.getOwner()->getParentOp();
    }
  }
  if (!defOp)
    return "unknown";
  if (auto func = defOp->getParentOfType<func::FuncOp>())
    return func.getName().str();
  return "unknown";
}

static void dumpAffineMinmaxBoundsToStderr(
    const AffineMinMaxValueBoundsCollector &bounds) {
  struct Entry {
    std::string funcName;
    std::string valueStr;
    int64_t lb;
    int64_t ub;
  };

  SmallVector<Entry> entries;
  entries.reserve(bounds.getBoundsMap().size());
  for (auto &it : bounds.getBoundsMap()) {
    Value v = it.first;
    auto [lb, ub] = it.second;

    std::string valueStr;
    llvm::raw_string_ostream os(valueStr);
    v.print(os);
    os.flush();

    entries.push_back(Entry{getContainingFuncName(v), valueStr, lb, ub});
  }

  // DenseMap iteration order is non-deterministic; sort for stable tests.
  llvm::sort(entries, [](const Entry &a, const Entry &b) {
    if (a.funcName != b.funcName)
      return a.funcName < b.funcName;
    return a.valueStr < b.valueStr;
  });

  llvm::outs() << "=== AFFINE_MINMAX_BOUNDS DUMP START ===\n";
  for (auto &e : entries) {
    llvm::outs() << "AFFINE_MINMAX_BOUNDS func=" << e.funcName
                 << " value=" << e.valueStr << " LB=" << e.lb << " UB="
                 << e.ub << "\n";
  }
  llvm::outs() << "=== AFFINE_MINMAX_BOUNDS DUMP END ===\n";
}

static void eliminateOneIterationScfFor(scf::ForOp forOp, IRRewriter &rewriter) {
  Block *bodyPtr = forOp.getBody();
  Block &body = *bodyPtr;
  auto yieldOp = cast<scf::YieldOp>(body.getTerminator());

  IRMapping mapping;
  mapping.map(body.getArgument(0), forOp.getLowerBound());

  ValueRange initArgs = forOp.getInitArgs();
  for (auto it : llvm::enumerate(initArgs))
    mapping.map(body.getArgument(1 + it.index()), it.value());

  rewriter.setInsertionPoint(forOp);
  for (Operation &op : body.without_terminator())
    rewriter.clone(op, mapping);

  SmallVector<Value> newResults;
  newResults.reserve(yieldOp.getNumOperands());
  for (Value operand : yieldOp.getOperands())
    newResults.push_back(mapping.lookup(operand));

  if (forOp.getNumResults() == 0) {
    rewriter.eraseOp(forOp);
    return;
  }
  rewriter.replaceOp(forOp, newResults);
}

struct EliminateSingleIterationScfForPass
    : public impl::EliminateSingleIterationScfForBase<
          EliminateSingleIterationScfForPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    AffineMinMaxValueBoundsCollector bounds;
    bounds.populate(module);

    if (dumpAffineMinmaxBounds)
      dumpAffineMinmaxBoundsToStderr(bounds);

    IRRewriter rewriter(module.getContext());
    SmallVector<scf::ForOp> singleIter;

    module.walk([&](scf::ForOp forOp) {
      if (bounds.provesSingleIterationScfFor(forOp))
        singleIter.push_back(forOp);
    });

    for (scf::ForOp forOp : singleIter)
      eliminateOneIterationScfFor(forOp, rewriter);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createEliminateSingleIterationScfForPass() {
  return std::make_unique<EliminateSingleIterationScfForPass>();
}

} // namespace bishengir
