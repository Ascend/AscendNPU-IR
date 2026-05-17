#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
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
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <list>

namespace mlir {
#define GEN_PASS_DEF_APPENDVECTORLAYOUT
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "append-vector-layout"

using namespace mlir;

namespace {

bool isPredicateOp(Operation *op) {
  return op && isa<hivmave::VFPgeOp, hivmave::VFPltOp, hivmave::VFPltMOp>(op);
}

hivmave::VectorLayoutAttr
createVecLayoutAttr(MLIRContext *ctx, hivmave::VecMemTypeAttr memTypeAttr,
                    hivmave::VecRegTypeAttr regTypeAttr) {
  return hivmave::VectorLayoutAttr::get(ctx, memTypeAttr, regTypeAttr);
}

hivmave::VectorLayoutAttr createVecLayoutAttr(MLIRContext *ctx,
                                              hivmave::VecMemType memType,
                                              hivmave::VecRegType regType) {
  return createVecLayoutAttr(ctx, hivmave::VecMemTypeAttr::get(ctx, memType),
                             hivmave::VecRegTypeAttr::get(ctx, regType));
}

LogicalResult setResultVecTypeAttr(
    Operation *op, RewriterBase &rewriter,
    SmallVector<std::pair<hivmave::VecMemType, hivmave::VecRegType>>
        &layoutTypes) {
  rewriter.setInsertionPoint(op);
  SmallVector<Value> newOperands;
  // fit predicate vector operand
  for (auto operand : op->getOperands()) {
    if (isPredicateOp(operand.getDefiningOp())) {
      auto vecType = dyn_cast<VectorType>(operand.getType());
      auto newAttr = createVecLayoutAttr(op->getContext(), layoutTypes[0].first,
                                         layoutTypes[0].second);
      auto newType = vecType.cloneWith(newAttr);
      auto newOperand = rewriter.create<hivmave::VectorLayoutCastOp>(
          op->getLoc(), newType, operand);
      newOperands.push_back(newOperand);
    } else {
      newOperands.push_back(operand);
    }
  }

  // new results
  SmallVector<Type> newResultTypes;
  for (auto [idx, resType] : llvm::enumerate(op->getResultTypes())) {
    if (auto vecType = dyn_cast<VectorType>(resType)) {
      auto layoutType = layoutTypes[idx];
      auto attr = createVecLayoutAttr(op->getContext(), layoutType.first,
                                      layoutType.second);
      newResultTypes.push_back(vecType.cloneWith(attr));
    } else {
      newResultTypes.push_back(resType);
    }
  }
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(newOperands);
  state.addAttributes(op->getAttrs());
  state.addTypes(newResultTypes);

  auto replacer = rewriter.create(state);
  rewriter.replaceOp(op, replacer);
  return success();
}

// set same layout type for all result vector types
LogicalResult setResultVecTypeAttr(
    Operation *op, RewriterBase &rewriter,
    std::pair<hivmave::VecMemType, hivmave::VecRegType> layoutType) {
  SmallVector<std::pair<hivmave::VecMemType, hivmave::VecRegType>> layoutTypes;
  for (auto resType : op->getResultTypes()) {
    if (isa<VectorType>(resType)) {
      layoutTypes.push_back(layoutType);
    }
  }
  return setResultVecTypeAttr(op, rewriter, layoutTypes);
}

std::pair<hivmave::VecMemType, hivmave::VecRegType>
caculateLayoutByVecL(VectorType vecType) {

  unsigned int sparse =
      2048U / static_cast<unsigned int>(vecType.getNumElements()) / vecType.getElementTypeBitWidth();
  LLVM_DEBUG(llvm::dbgs() << "[caculateLayoutByResVecL]: sparse = " << sparse
                          << "\n";);
  if (sparse == 4) {
    return {hivmave::VecMemType::EVEN_4, hivmave::VecRegType::N};
  } else if (sparse == 2) {
    return {hivmave::VecMemType::EVEN, hivmave::VecRegType::N};
  }
  return {hivmave::VecMemType::DENSE, hivmave::VecRegType::N};
}

std::pair<hivmave::VecMemType, hivmave::VecRegType>
caculateLayoutByResVecL(Operation *op) {
  auto res = op->getResults()[0];
  if (auto vecType = dyn_cast<VectorType>(res.getType())) {
    return caculateLayoutByVecL(vecType);
  }
  return {hivmave::VecMemType::DENSE, hivmave::VecRegType::N};
}

bool isDenseOnlyOp(Operation *op) {
  return isa<hivmave::VFStoreWithStrideOp>(op);
}

bool isLayoutFreeOp(Operation *op) {
  return isa<UnrealizedConversionCastOp, hivmave::VectorLayoutCastOp>(op);
}

bool isDirectlyUsedByDenseOnlyOp(Operation *op) {
  DenseSet<Operation *> candidateOps = {op};
  while (!candidateOps.empty()) {
    auto candidateOp = *candidateOps.begin();
    LLVM_DEBUG(llvm::dbgs() << "[candidate op]: " << candidateOp->getName() << "\n";);
    candidateOps.erase(candidateOp);
    if (isDenseOnlyOp(candidateOp)) {
      LLVM_DEBUG(llvm::dbgs() << "[candidate op]: " << candidateOp->getName() << "is dense only op\n";);
      return true;
    }
    if (candidateOp == op || isLayoutFreeOp(candidateOp)) {
      for (auto user : candidateOp->getUsers()) {
        candidateOps.insert(user);
      }
    }
  }
  return false;
}

struct AnalyzeLayoutForVFLoadOp
    : public OpRewritePattern<hivmave::VFLoadOp> {
public:
  using OpRewritePattern<hivmave::VFLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivmave::VFLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (isDirectlyUsedByDenseOnlyOp(loadOp)) {
      // if user only receive dense layout vector, vfload must be dense layout.
      return setResultVecTypeAttr(loadOp, rewriter, {hivmave::VecMemType::DENSE, hivmave::VecRegType::N});
    } else {
      auto vecType = loadOp.getVectorType();
      auto layoutTypes = caculateLayoutByVecL(vecType);
      return setResultVecTypeAttr(loadOp, rewriter, layoutTypes);
    }
  }
};

template <typename VecAllocOp>
struct AnalyzeLayoutForVecAllocaterOp : public OpRewritePattern<VecAllocOp> {
public:
  using OpRewritePattern<VecAllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(VecAllocOp op,
                                PatternRewriter &rewriter) const override {
    auto layoutTypes = caculateLayoutByResVecL(op);
    return setResultVecTypeAttr(op, rewriter, layoutTypes);
  }
};

template <typename VLConvertionOp>
struct AnalyzeVLConvertionOp : public OpRewritePattern<VLConvertionOp> {
public:
  using OpRewritePattern<VLConvertionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(VLConvertionOp vlcOp,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(vlcOp.getSrc().getType());
    assert(srcType && "src operand is not vector type");
    auto resType = dyn_cast<VectorType>(vlcOp.getRes().getType());
    assert(resType && "res is not vector type");

    auto layoutType = caculateLayoutByVecL(resType);
    auto res = setResultVecTypeAttr(vlcOp, rewriter, layoutType);
    return res;
  }
};

struct PassThroughForLoop : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Block *oldLoopBody = forOp.getBody();
    rewriter.setInsertionPointAfter(forOp);

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
    rewriter.replaceOp(forOp, replacements);
    return success();
  }
};

struct LeagalizeCallOp : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(callOp);
    SmallVector<Value> newOperands;
    for (auto operand : callOp->getOperands()) {
      if (auto vecType = dyn_cast<VectorType>(operand.getType())) {
        auto newOperand = rewriter.create<hivmave::VectorLayoutCastOp>(
            callOp->getLoc(), vecType.cloneWith({}), operand);
        newOperands.push_back(newOperand);
      } else {
        newOperands.push_back(operand);
      }
    }
    auto newCallOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        callOp, callOp.getCallee(), callOp->getResultTypes(), newOperands);

    // copy custom attribtues
    for (auto attr : callOp->getAttrs()) {
      if (attr.getName() != "callee") {
        newCallOp->setAttr(attr.getName(), attr.getValue());
      }
    }

    rewriter.setInsertionPointAfter(newCallOp);
    for (auto res : newCallOp->getResults()) {
      if (auto vecType = dyn_cast<VectorType>(res.getType())) {
        auto newRes = rewriter.create<hivmave::VectorLayoutCastOp>(
            newCallOp.getLoc(),
            vecType.cloneWith(createVecLayoutAttr(newCallOp->getContext(),
                                                  hivmave::VecMemType::DENSE,
                                                  hivmave::VecRegType::N)),
            res);
        rewriter.replaceUsesWithIf(res, newRes, [&](OpOperand &use) {
          return use.getOwner() != newRes;
        });
      }
    }
    return success();
  }
};

void addVectorDefiningOpPatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  patterns.add<AnalyzeLayoutForVFLoadOp>(ctx);
}

void addVectorAllocaterOpPatterns(RewritePatternSet &patterns,
                                  MLIRContext *ctx) {
  patterns.add<AnalyzeLayoutForVecAllocaterOp<hivmave::VFBroadcastScalarMaskOp>,
               AnalyzeLayoutForVecAllocaterOp<hivmave::VFBroadcastScalarOp>,
               AnalyzeLayoutForVecAllocaterOp<UnrealizedConversionCastOp>,
               AnalyzeLayoutForVecAllocaterOp<hivmave::VFVCIOp>>(ctx);
}

void addVLConvertionOpPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<AnalyzeVLConvertionOp<hivmave::VFTruncIOp>,
               AnalyzeVLConvertionOp<hivmave::VFTruncFOp>,
               AnalyzeVLConvertionOp<hivmave::VFExtSIOp>,
               AnalyzeVLConvertionOp<hivmave::VFExtUIOp>,
               AnalyzeVLConvertionOp<hivmave::VFFpToSIntOp>,
               AnalyzeVLConvertionOp<hivmave::VFFpToUIntOp>,
               AnalyzeVLConvertionOp<hivmave::VFSIntToFpOp>,
               AnalyzeVLConvertionOp<hivmave::VFUIntToFpOp>,
               AnalyzeVLConvertionOp<hivmave::VFExtFOp>>(ctx);
}

void addConsistencyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<PassThroughForLoop, LeagalizeCallOp>(ctx);
}

void analyzeNormalOp(Operation *op, PatternRewriter &rewriter) {
  if (isPredicateOp(op)) {
    LLVM_DEBUG(llvm::dbgs()
                   << "[SKIP predicated Op] " << op->getName() << "\n";);
    return;
  }
  SmallVector<std::pair<hivmave::VecMemType, hivmave::VecRegType>> layoutTypes;
  for (auto resType : op->getResultTypes()) {
    if (auto vecType = dyn_cast<VectorType>(resType)) {
      layoutTypes.push_back(caculateLayoutByVecL(vecType));
    }
  }
  if (!layoutTypes.empty()) {
    (void)setResultVecTypeAttr(op, rewriter, layoutTypes);
  }
}

// like applyPatternsGreedily, this function will match ops and rewrite it,
// however, this function will not verify after all. And no repeated apply to
// the same op.
LogicalResult applyPatternsByPreOrder(Operation *rootOp,
                                      RewritePatternSet &patterns) {
  PatternRewriter rewriter(rootOp->getContext());
  SmallVector<Operation *> ops;
  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<func::FuncOp>(op))
      ops.push_back(op);
  });
  for (auto op : ops) {
    LLVM_DEBUG(llvm::dbgs() << "analyzing op: " << op->getName() << "\n";);
    bool matched = false;
    for (const auto &pattern : patterns.getNativePatterns()) {
      if (pattern->getRootKind() == op->getName()) {
        matched = true;
        if (failed(pattern->matchAndRewrite(op, rewriter))) {
          LLVM_DEBUG(llvm::dbgs() << "Applied pattern to " << op->getName()
                                  << " failed\n";);
          return failure();
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Applied pattern to " << op->getName()
                                  << " succeed\n";);
        }
      }
    }
    if (!matched) {
      LLVM_DEBUG(llvm::dbgs() << "Analyze normal op " << op->getName() << "\n");
      analyzeNormalOp(op, rewriter);
    }
  }
  return success();
}

struct AppendVectorLayoutPass
    : public impl::AppendVectorLayoutBase<AppendVectorLayoutPass> {
  using AppendVectorLayoutBase<AppendVectorLayoutPass>::AppendVectorLayoutBase;

public:
  void runOnOperation() override;
};
} // namespace

void AppendVectorLayoutPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "==== START Appending vector layout ====\n";);
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp)) {
    // Only apply to VF
    return;
  }
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  addVectorDefiningOpPatterns(patterns, ctx);
  addVectorAllocaterOpPatterns(patterns, ctx);
  addVLConvertionOpPatterns(patterns, ctx);
  addConsistencyPatterns(patterns, ctx);
  if (failed(applyPatternsByPreOrder(funcOp, patterns))) {
    llvm::errs() << "\n[AppendVectorLayoutAttr failed]:\n";
    funcOp->print(llvm::errs(), mlir::OpPrintingFlags().assumeVerified());
    signalPassFailure();
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "==== END of Appending vector layout ====\n";
             funcOp.print(llvm::dbgs()); llvm::dbgs() << "\n");
}

std::unique_ptr<Pass> hivmave::createAppendVectorLayoutPass() {
  return std::make_unique<AppendVectorLayoutPass>();
}