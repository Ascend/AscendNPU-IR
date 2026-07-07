#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
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
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include <list>
#include <stack>
#include <utility>
namespace mlir {
#define GEN_PASS_DEF_ANALYZEALIGNMENTBITWIDTH
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "analyze-alignment-bitwidth"
#define DBG(msg)                                                               \
  LLVM_DEBUG(llvm::dbgs() << "[" << __func__ << "]: "; msg;                    \
             llvm::dbgs() << "\n";)

using namespace mlir;

namespace {

hivmave::VecMemType getVecType(Type vector) {
  auto vecType = dyn_cast<VectorType>(vector);
  auto layout = dyn_cast<hivmave::VectorLayoutAttr>(vecType.getLayout());
  auto memAttr = dyn_cast<hivmave::VecMemTypeAttr>(layout.getMem());
  return memAttr.getValue();
}

hivmave::VecMemType getVecType(Value vector) {
  return getVecType(vector.getType());
}

LogicalResult setAlignmentBitWidthAttr(Operation *op, RewriterBase &rewriter,
                                       int alignmentBitWidth) {
  if (op->hasAttr(utils::elementAlignmentBitWidth)) {
    return failure();
  }
  auto bitWidthAttr =
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), alignmentBitWidth);
  op->setAttr(utils::elementAlignmentBitWidth, bitWidthAttr);
  return success();
}

template <typename Vector>
int inferenceAlignmentBitwidthByVectorLayout(Vector vector) {
  auto layoutType = getVecType(vector);
  int alignment = -1;
  switch (layoutType) {
  case hivmave::VecMemType::B8:
    alignment = 8;
    break;
  case hivmave::VecMemType::B16:
  case hivmave::VecMemType::B8_2VL:
    alignment = 16;
    break;
  case hivmave::VecMemType::B32:
  case hivmave::VecMemType::B16_2VL:
  case hivmave::VecMemType::B8_4VL:
    alignment = 32;
    break;
  default:
    alignment = -1;
    break;
  }
  return alignment;
}

int analyzeByRes0(Operation *op) {
  if (op->getNumResults() == 0) {
    return -1;
  }
  if (!isa<VectorType>(op->getResult(0).getType())) {
    return -1;
  }
  return inferenceAlignmentBitwidthByVectorLayout(op->getResult(0));
}

int analyzeSpecialOp(Operation *op) {
  if (auto storeOp = dyn_cast<hivmave::VFMaskedStoreOp>(op)) {
    return inferenceAlignmentBitwidthByVectorLayout(storeOp.getVectorType());
  } else if (auto storeOp = dyn_cast<hivmave::VFStoreWithStrideOp>(op)) {
    return inferenceAlignmentBitwidthByVectorLayout(storeOp.getVectorType());
  } else if (auto vgatherOp = dyn_cast<hivmave::VFGatherOp>(op)) {
    return inferenceAlignmentBitwidthByVectorLayout(vgatherOp.getMask());
  } else if (auto interleaveOp = dyn_cast<hivmave::VFInterleaveOp>(op)) {
    // Has functionType means the interleaveOp try to
    // convert dense layout to sparse layout without type convertion
    // So it should use result's layout as elementAlignment.
    if (interleaveOp->getAttrOfType<hivmave::FunctionDistTypeAttr>("functionType"))
      return inferenceAlignmentBitwidthByVectorLayout(interleaveOp.getRes1());
    else
      return inferenceAlignmentBitwidthByVectorLayout(interleaveOp.getSrc1());
  } else if (auto deinterleaveOp = dyn_cast<hivmave::VFDeInterleaveOp>(op)) {
    // Has functionType means the deinterleaveOp try to
    // convert dense layout to sparse layout without type convertion
    // So it should use result's layout as elementAlignment.
    if (deinterleaveOp->getAttrOfType<hivmave::FunctionDistTypeAttr>("functionType"))
      return inferenceAlignmentBitwidthByVectorLayout(deinterleaveOp.getRes1());
    else
      return inferenceAlignmentBitwidthByVectorLayout(deinterleaveOp.getSrc1());
  } else if (auto truncOp = dyn_cast<hivmave::VFTruncFOp>(op)) {
    // special case of ave-process-vsstb optimization
    return inferenceAlignmentBitwidthByVectorLayout(truncOp.getSrc());
  }
  return -1;
}

bool skipOp(Operation *op) {
  bool isVectorOp = llvm::any_of(op->getOperandTypes(),
                                 [](Type tp) { return isa<VectorType>(tp); }) ||
                    llvm::any_of(op->getResultTypes(),
                                 [](Type tp) { return isa<VectorType>(tp); });
  if (!isVectorOp)
    return true;
  if (isa<UnrealizedConversionCastOp, scf::ForOp, scf::YieldOp, func::CallOp,
          hivmave::VectorLayoutCastOp>(op))
    return true;
  return false;
}

void analyzeVectorOp(Operation *op, IRRewriter &rewriter) {
  DBG(llvm::dbgs() << "analyze Op: " << *op;);
  if (skipOp(op))
    return;
  int alignment = analyzeSpecialOp(op);
  if (alignment == -1)
    alignment = analyzeByRes0(op);
  DBG(llvm::dbgs() << "alignmentBitwidth: " << alignment;);
  (void)setAlignmentBitWidthAttr(op, rewriter, alignment);
}

struct AnalyzeAlignmentBitwidthPass
    : public impl::AnalyzeAlignmentBitwidthBase<AnalyzeAlignmentBitwidthPass> {
  using AnalyzeAlignmentBitwidthBase<
      AnalyzeAlignmentBitwidthPass>::AnalyzeAlignmentBitwidthBase;

public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!hivm::isVF(funcOp)) {
      return;
    }

    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    funcOp->walk([&](Operation *op) { analyzeVectorOp(op, rewriter); });
  }
};

} // namespace

std::unique_ptr<Pass> hivmave::createAnalyzeAlignmentBitwidthPass() {
  return std::make_unique<AnalyzeAlignmentBitwidthPass>();
}
