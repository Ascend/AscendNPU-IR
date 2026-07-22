#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_REMOVEVECTORLAYOUTATTR
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "remove-vector-layout-attr"

using namespace mlir;

namespace {

void eliminateForOp(scf::ForOp op, IRRewriter &rewriter) {
  auto forOp = cast<scf::ForOp>(op);
  Block *oldLoopBody = forOp.getBody();
  rewriter.setInsertionPointAfter(op);

  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), forOp.getInitArgs());
  auto &body = newForOp.getBody()->getOperations();
  if (!body.empty())
    newForOp.getBody()->getOperations().pop_back();
  newForOp->setAttrs(forOp->getAttrs());
  Block *loopBody = newForOp.getBody();
  rewriter.setInsertionPointToStart(loopBody);
  SmallVector<Value> iterArgs;
  iterArgs.push_back(newForOp.getInductionVar());
  for (auto [initArg, iterArg] :
       llvm::zip(newForOp.getInitArgs(), newForOp.getRegionIterArgs())) {
    iterArgs.push_back(iterArg);
  }
  rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);
  SmallVector<Value> replacements;
  for (auto [opResult, newOpResult] :
       llvm::zip(forOp->getResults(), newForOp.getResults())) {
    replacements.push_back(newOpResult);
  }
  rewriter.replaceOp(op, replacements);
}

void eliminateNormalOp(Operation *op, IRRewriter &rewriter) {
  bool isTarget = false;
  SmallVector<Type> results;
  for (auto res : op->getResults()) {
    if (auto vecRes = dyn_cast<VectorType>(res.getType())) {
      isTarget = true;
      results.push_back(vecRes.cloneWith({}));
    } else {
      results.push_back(res.getType());
    }
  }
  if (!isTarget)
    return;

  rewriter.setInsertionPoint(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addAttributes(op->getAttrs());
  state.addTypes(results);
  auto replacer = rewriter.create(state);
  rewriter.replaceOp(op, replacer);
}

void eliminateVectorLayoutCastOp(hivmave::VectorLayoutCastOp op,
                                 IRRewriter &rewriter) {
  rewriter.replaceOp(op, op->getOperand(0));
}

struct RemoveVectorLayoutAttrPass
    : public impl::RemoveVectorLayoutAttrBase<RemoveVectorLayoutAttrPass> {
  using RemoveVectorLayoutAttrBase<
      RemoveVectorLayoutAttrPass>::RemoveVectorLayoutAttrBase;

public:
  void runOnOperation() override;
};

void RemoveVectorLayoutAttrPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<Operation *> ops;
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) { ops.push_back(op); });

  for (auto op : ops) {
    if (isa<func::FuncOp>(op) || isa<func::CallOp>(op))
      continue;
    else if (auto castOp = dyn_cast<hivmave::VectorLayoutCastOp>(op))
      eliminateVectorLayoutCastOp(castOp, rewriter);
    else if (auto forOp = dyn_cast<scf::ForOp>(op))
      eliminateForOp(forOp, rewriter);
    else
      eliminateNormalOp(op, rewriter);
  }
}

} // namespace
std::unique_ptr<Pass> hivmave::createRemoveVectorLayoutAttrPass() {
  return std::make_unique<RemoveVectorLayoutAttrPass>();
}