#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
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
#define GEN_PASS_DEF_ANNOTATEDISTOPLAYOUT
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "annotate-dist-op-layout"

using namespace mlir;

namespace {

unsigned int legalizeElemBitWidth(unsigned int elemBitWidth) {
  return elemBitWidth >= 8 ? elemBitWidth : 8;
}

unsigned int calculateAlignmentBitWidth(VectorType vecType) {
  Type elementType = vecType.getElementType();
  auto layout = dyn_cast<hivmave::VectorLayoutAttr>(vecType.getLayout());
  auto memType = dyn_cast<hivmave::VecMemTypeAttr>(layout.getMem()).getValue();
  auto dataWidth = elementType.getIntOrFloatBitWidth();
  LLVM_DEBUG(llvm::dbgs() << "Vector Elem BitWidth: " << dataWidth << "\n");
  auto alignmentBitWidth = legalizeElemBitWidth(dataWidth);
  LLVM_DEBUG(llvm::dbgs() << "legalized alignmentBitWidth: "
                          << alignmentBitWidth << "\n");
  if (memType == hivmave::VecMemType::EVEN) {
    alignmentBitWidth *= 2;
  } else if (memType == hivmave::VecMemType::EVEN_4) {
    alignmentBitWidth *= 4;
  }
  alignmentBitWidth = alignmentBitWidth > 32 ? 32 : alignmentBitWidth;
  LLVM_DEBUG(llvm::dbgs() << "calculated alignmentBitWidth: "
                          << alignmentBitWidth << "\n");
  return alignmentBitWidth;
}

int getAlignmentBitWidth(Operation *op) {
  if (auto elementAlignmentAttr =
          op->getAttr(utils::elementAlignmentBitWidth)) {
    return dyn_cast<mlir::IntegerAttr>(elementAlignmentAttr).getInt();
  }
  return -1;
}

LogicalResult setAlignmentBitWidthAttr(Operation *op, RewriterBase &rewriter,
                                       unsigned int alignmentBitWidth) {
  if (op->hasAttr(utils::elementAlignmentBitWidth)) {
    return failure();
  }
  auto bitWidthAttr =
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), alignmentBitWidth);
  op->setAttr(utils::elementAlignmentBitWidth, bitWidthAttr);
  return success();
}

int maxAlignmentBitWidthContextFreeAnalysis(Operation *op) {
  int alignmentBitWidth = -1;
  for (auto operand : op->getOperands()) {
    if (auto vecType = dyn_cast<VectorType>(operand.getType())) {
      if (vecType.getLayout()) {
        alignmentBitWidth = std::max(alignmentBitWidth,
                                     (int)calculateAlignmentBitWidth(vecType));
      }
    }
  }
  for (auto resType : op->getResultTypes()) {
    if (auto vecType = dyn_cast<VectorType>(resType)) {
      if (vecType.getLayout()) {
        alignmentBitWidth = std::max(alignmentBitWidth,
                                     (int)calculateAlignmentBitWidth(vecType));
      }
    }
  }
  return alignmentBitWidth;
}

bool isDenseOnlyOp(Operation *op) {
  return isa<hivmave::VFStoreWithStrideOp>(op);
}

int contextFreeAnalysis(Operation *op) {
  if (isDenseOnlyOp(op)) {
    if (auto sts = dyn_cast<hivmave::VFStoreWithStrideOp>(op)) {
      return sts.getVectorType().getElementTypeBitWidth();
    } else {
      assert(0 && "invalid op");
    }
  } else {
    return maxAlignmentBitWidthContextFreeAnalysis(op);
  }
}

int contextFreeBitwidthCollect(Operation *op) {
  int alignmentBitWidth = -1;
  for (auto operand : op->getOperands()) {
    if (auto vecType = dyn_cast<VectorType>(operand.getType())) {
      alignmentBitWidth = std::max(
          alignmentBitWidth, getAlignmentBitWidth(operand.getDefiningOp()));
    }
  }
  return alignmentBitWidth;
}

bool isPredicateArithOp(Operation *op) {
  return isa<hivmave::PregAndOp, hivmave::PregOrOp, hivmave::PregXorOp>(op);
}

int pregDefinerAnalysis(Operation *op) {
  auto operand = op->getOperand(0);
  auto defOp = operand.getDefiningOp();
  return getAlignmentBitWidth(defOp);
}

int pregSelfAnalysis(Operation *op) {
  return hivm::util::PREDICATE_BITS / dyn_cast<VectorType>(op->getResultTypes()[0]).getNumElements() * 8;
}

int pregParentOpAnalysis(Operation *op, DenseMap<Operation *, int> &parOpAlign) {
  if (parOpAlign.find(op->getParentOp()) != parOpAlign.end()) {
    if (parOpAlign[op->getParentOp()] == -1) {
      return contextFreeBitwidthCollect(op);
    }
  }
  return -1;
}

int pgeOpAnalysis(hivmave::VFPgeOp pgeOp) {
  for (auto user : pgeOp->getUsers()) {
    if (auto vgatherOp = dyn_cast<hivmave::VFGatherOp>(user)) {
      auto v = vgatherOp.getIndexVec();
      return v.getType().getElementTypeBitWidth();
    }
    // This will be convert to template lib call, which is not properly
    // set mask bitwidth.
    if (auto maskStoreOp = dyn_cast<hivmave::VFMaskedStoreOp>(user)) {
      auto v = maskStoreOp.getVectorType();
      if (v.getElementTypeBitWidth() == 64) {
        return 32;
      } 
    }
  }
  return -1;
}

void analyzeAlignmentBitWidth(Operation *op,
                              DenseMap<Operation *, int> &parOpAlign,
                              IRRewriter &rewriter) {
  int alignmentBitWidth = -1;
  if (isPredicateArithOp(op)) {
    alignmentBitWidth = std::max({pregDefinerAnalysis(op), pregSelfAnalysis(op), pregParentOpAnalysis(op, parOpAlign)});
  } else {
    alignmentBitWidth = contextFreeAnalysis(op);
  }
  (void)setAlignmentBitWidthAttr(op, rewriter, alignmentBitWidth);
  parOpAlign[op] = alignmentBitWidth;
}

void analyzeSpecialAlignmentBitWidth(Operation *op, IRRewriter &rewriter) {
  int alignmentBitWidth = -1;
  if (auto pgeOp = dyn_cast<hivmave::VFPgeOp>(op)) {
    alignmentBitWidth = pgeOpAnalysis(pgeOp);
  }
  (void)setAlignmentBitWidthAttr(op, rewriter, alignmentBitWidth);
}

// General op is the op need to be annotated by alignmentbitwith
// Or op with vector value.
bool isGeneralOp(Operation *op) {
  if (isa<func::FuncOp, func::CallOp, scf::ForOp, scf::YieldOp,
          hivmave::VectorLayoutCastOp, UnrealizedConversionCastOp,
          hivmave::VFPgeOp, hivmave::VFPltOp>(op)) {
    return false;
  }
  for (auto operand : op->getOperands()) {
    if (isa<VectorType>(operand.getType())) {
      return true;
    }
  }
  for (auto resType : op->getResultTypes()) {
    if (isa<VectorType>(resType)) {
      return true;
    }
  }
  return false;
}

// Specify Ops that need to be analyzed 
// for special cases.
bool isSpecifyOp(Operation *op) {
  return isa<hivmave::VFPgeOp>(op);
}

struct AnnotateDistOpLayoutPass
    : public impl::AnnotateDistOpLayoutBase<AnnotateDistOpLayoutPass> {
  using AnnotateDistOpLayoutBase<
      AnnotateDistOpLayoutPass>::AnnotateDistOpLayoutBase;

public:
  void runOnOperation() override;
};

void AnnotateDistOpLayoutPass::runOnOperation() {
  // Annotate dist op pattern depending on Vecotor Layout Attr.
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp)) {
    // Apply to VF only.
    return;
  }
  IRRewriter rewriter(ctx);
  LLVM_DEBUG(llvm::dbgs() << "START of AnnotateDistOpLayout\n";);
  DenseMap<Operation *, int> parOpAlign;
  funcOp.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
    parOpAlign[forOp] = getAlignmentBitWidth(forOp);
  });
  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isGeneralOp(op))
      analyzeAlignmentBitWidth(op, parOpAlign, rewriter);
    else if (isSpecifyOp(op))
      analyzeSpecialAlignmentBitWidth(op, rewriter);
  });
  LLVM_DEBUG(llvm::dbgs() << "END of AnnotateDistOpLayout\n";
             funcOp->print(llvm::dbgs()); llvm::dbgs() << "\n";);
}

} // namespace

std::unique_ptr<Pass> hivmave::createAnnotateDistOpLayoutPass() {
  return std::make_unique<AnnotateDistOpLayoutPass>();
}
