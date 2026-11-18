#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/LiftScalarProducer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include <utility>

using namespace mlir;

namespace mlir::hfusion {
namespace {

/// Lift one producer per match; retain scalar definitions until this driver
/// ends. Keeping the cache keys alive avoids rewrite-listener bookkeeping.
using LiftedScalarMap = DenseMap<std::pair<Value, Type>, Value>;

/// fptosi/fptoui cast toward zero (IEEE-754 truncation). After lifting, the
/// vector cast is lowered by ArithToHIVMAVE, which currently defaults a missing
/// `round_mode` to ROUND; that would change the value (e.g. 1.75 -> 2 instead
/// of 1). Only treat an op as safe to lift when it carries an explicit
/// truncating round mode. A missing attribute is conservatively unsafe here
/// because fixing the lowering default lives in a separate change.
static bool hasTruncatingRounding(Operation *op) {
  Attribute roundAttr = op->getAttr("round_mode");
  if (!roundAttr)
    return false;
  return llvm::TypeSwitch<Attribute, bool>(roundAttr)
      .Case<hfusion::RoundModeAttr>([](hfusion::RoundModeAttr attr) {
        return attr.getValue() == hfusion::RoundMode::TRUNC ||
               attr.getValue() == hfusion::RoundMode::TRUNCWITHOVERFLOW;
      })
      .Case<hivm::RoundModeAttr>([](hivm::RoundModeAttr attr) {
        return attr.getValue() == hivm::RoundMode::TRUNC ||
               attr.getValue() == hivm::RoundMode::TRUNCWITHOVERFLOW;
      })
      .Default([](Attribute) { return false; });
}

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

    // fptosi/fptoui truncate toward zero. Lifting them to a vector cast whose
    // lowering applies a non-truncating round mode would change the value
    // (e.g. 1.75 -> 2 instead of 1), so only lift when the op carries an
    // explicit truncating round mode.
    if (isa<arith::FPToSIOp, arith::FPToUIOp>(scalar) &&
        !hasTruncatingRounding(scalar))
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
  // Let the driver's own worklist carry the broadcasts this pattern creates
  // for each lifted operand, instead of re-walking the whole function every
  // round; otherwise lifting a K-deep chain costs O(K * numOps) walks.
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  config.fold = false;
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  SmallVector<Operation *> broadcasts;
  func.walk([&](vector::BroadcastOp op) { broadcasts.push_back(op); });
  return applyOpPatternsGreedily(broadcasts, frozenPatterns, config);
}

} // namespace mlir::hfusion
