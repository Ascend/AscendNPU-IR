#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/LiftScalarProducer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

using namespace mlir;

namespace mlir::hfusion {
namespace {

/// Lift one producer per match; retain scalar definitions until this driver
/// ends. Keeping the cache keys alive avoids rewrite-listener bookkeeping.
using LiftedScalarMap = DenseMap<std::pair<Value, Type>, Value>;

struct LiftBroadcastScalarProducer final
    : OpRewritePattern<vector::BroadcastOp> {
  LiftBroadcastScalarProducer(MLIRContext *context, LiftedScalarMap &lifted)
      : OpRewritePattern(context), lifted(lifted) {}
  LiftedScalarMap &lifted;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcast,
                                PatternRewriter &rewriter) const override {
    Operation *scalar = broadcast.getSource().getDefiningOp();
    if (!scalar || scalar->getBlock() != broadcast->getBlock() ||
        scalar->getNumRegions() != 0 || scalar->getNumResults() != 1 ||
        scalar->getNumOperands() == 0 || !isMemoryEffectFree(scalar) ||
        !OpTrait::hasElementwiseMappableTraits(scalar) ||
        !scalar->getResult(0).getType().isIntOrFloat() ||
        !llvm::all_of(scalar->getOperandTypes(),
                      [](Type type) { return type.isIntOrFloat(); }))
      return failure();

    // Vector division flushes subnormal operands on Ascend950PR, so lifting
    // division-like scalar operations is not value preserving.
    if (isa<arith::DivFOp, arith::RemFOp>(scalar))
      return failure();

    bool inVectorRegion = false;
    for (Operation *parent = scalar->getParentOp(); parent;
         parent = parent->getParentOp())
      for (NamedAttribute attr : parent->getAttrs())
        inVectorRegion |=
            attr.getName().strref().starts_with("outlined-loop-target-");
    if (!inVectorRegion)
      return failure();

    auto resultType = cast<VectorType>(broadcast.getType());
    auto key = std::make_pair(broadcast.getSource(), Type(resultType));
    auto found = lifted.find(key);
    if (found != lifted.end()) {
      rewriter.replaceOp(broadcast, found->second);
      return success();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(scalar);
    IRMapping mapping;
    for (Value operand : scalar->getOperands()) {
      if (mapping.contains(operand))
        continue;
      auto operandType = resultType.cloneWith(std::nullopt, operand.getType());
      mapping.map(operand, rewriter.create<vector::BroadcastOp>(
                               broadcast.getLoc(), operandType, operand));
    }
    Operation *vectorOp = rewriter.clone(*scalar, mapping);
    vectorOp->getResult(0).setType(resultType);
    lifted[key] = vectorOp->getResult(0);
    rewriter.replaceOp(broadcast, vectorOp->getResults());
    return success();
  }
};

} // namespace

LogicalResult liftScalarProducers(func::FuncOp func) {
  LiftedScalarMap lifted;
  RewritePatternSet patterns(func.getContext());
  patterns.add<LiftBroadcastScalarProducer>(func.getContext(), lifted);
  GreedyRewriteConfig config;
  config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  config.fold = false;
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  size_t remainingRounds = 1;
  func.walk([&](Operation *) { ++remainingRounds; });
  while (remainingRounds--) {
    SmallVector<Operation *> broadcasts;
    func.walk([&](vector::BroadcastOp op) { broadcasts.push_back(op); });
    bool changed = false;
    if (failed(applyOpPatternsGreedily(broadcasts, frozenPatterns, config,
                                       &changed)))
      return failure();
    if (!changed)
      break;
    if (remainingRounds == 0)
      return failure();
  }
  return success();
}

} // namespace mlir::hfusion
