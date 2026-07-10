#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <utility>
namespace mlir {
#define GEN_PASS_DEF_ANALYZEVECTORLAYOUT
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "analyze-vector-layout"
#define DBG(msg)                                                               \
  LLVM_DEBUG(llvm::dbgs() << "[" << __func__ << "]: "; msg;                    \
             llvm::dbgs() << "\n";)

using namespace mlir;

namespace {

using State = hivmave::VecMemType;
using FunctionType = hivmave::FunctionDistType;

template <typename S1, typename S2> constexpr uint32_t COMB_CASE(S1 s1, S2 s2) {
  return (static_cast<uint32_t>(s1) << 8) | static_cast<uint32_t>(s2);
}

template <typename S1, typename S2, typename S3>
constexpr uint32_t COMB_CASE(S1 s1, S2 s2, S3 s3) {
  return (static_cast<uint32_t>(s1) << 16) | (static_cast<uint32_t>(s2) << 8) |
         static_cast<uint32_t>(s3);
}

template <typename S1, typename S2, typename S3, typename S4>
constexpr uint32_t COMB_CASE(S1 s1, S2 s2, S3 s3, S4 s4) {
  return (static_cast<uint32_t>(s1) << 24) | (static_cast<uint32_t>(s2) << 16) |
         (static_cast<uint32_t>(s3) << 8) << static_cast<uint32_t>(s4);
}

hivmave::FunctionDistTypeAttr createFunctionDistTypeAttr(MLIRContext *ctx,
                                                         FunctionType ftype) {
  return hivmave::FunctionDistTypeAttr::get(ctx, ftype);
}

hivmave::VectorLayoutAttr
createVecLayoutAttr(MLIRContext *ctx, hivmave::VecMemTypeAttr memTypeAttr) {
  return hivmave::VectorLayoutAttr::get(ctx, memTypeAttr);
}

hivmave::VectorLayoutAttr createVecLayoutAttr(MLIRContext *ctx,
                                              hivmave::VecMemType memType) {
  return createVecLayoutAttr(ctx, hivmave::VecMemTypeAttr::get(ctx, memType));
}

void setResultVecTypeAttr(Operation *op, RewriterBase &rewriter,
                          DenseMap<Value, State> &states) {
  rewriter.setInsertionPoint(op);
  DBG(llvm::dbgs() << "set op: " << op->getName(););
  SmallVector<Type> newResultTypes;
  bool hasVecRes = false;
  std::optional<State> fallbackState;
  for (auto res : op->getResults()) {
    if (auto vecType = dyn_cast<VectorType>(res.getType())) {
      if (states.contains(res)) {
        fallbackState = states[res];
        break;
      }
    }
  }
  for (auto res : op->getResults()) {
    if (auto vecType = dyn_cast<VectorType>(res.getType())) {
      hasVecRes = true;
      if (states.contains(res)) {
        auto attr = createVecLayoutAttr(op->getContext(), states[res]);
        DBG(llvm::dbgs() << "set result state: " << states[res];);
        newResultTypes.push_back(vecType.cloneWith(attr));
        continue;
      }
      if (fallbackState.has_value()) {
        auto attr = createVecLayoutAttr(op->getContext(), *fallbackState);
        DBG(llvm::dbgs() << "set result fallback state: " << *fallbackState;);
        newResultTypes.push_back(vecType.cloneWith(attr));
        continue;
      }
    }
    newResultTypes.push_back(res.getType());
  }
  if (!hasVecRes) {
    DBG(llvm::dbgs() << "Skip ucc";);
    return;
  }
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addAttributes(op->getAttrs());
  state.addTypes(newResultTypes);

  auto replacer = rewriter.create(state);
  rewriter.replaceOp(op, replacer);
}

void setForOpLayoutAttr(scf::ForOp forOp, IRRewriter &rewriter) {
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
}

void legalizeCallOp(func::CallOp callOp, IRRewriter &rewriter) {
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
      State s;
      switch (vecType.getElementTypeBitWidth()) {
      case 1:
      case 8:
        s = State::B8;
        break;
      case 16:
        s = State::B16;
        break;
      case 32:
        s = State::B32;
        break;
      default:
        assert(0);
      }
      auto newRes = rewriter.create<hivmave::VectorLayoutCastOp>(
          newCallOp.getLoc(),
          vecType.cloneWith(createVecLayoutAttr(newCallOp->getContext(), s)),
          res);
      rewriter.replaceUsesWithIf(res, newRes, [&](OpOperand &use) {
        return use.getOwner() != newRes;
      });
    }
  }
}

auto getFunctionTypeAttr(IRRewriter &rewriter, FunctionType ftype) {
  return createFunctionDistTypeAttr(rewriter.getContext(), ftype);
}

bool setFunctionAttr(IRRewriter &rewriter, Operation *op, FunctionType ftype) {
  if (ftype == FunctionType::INVALID) {
    return false;
  }
  // There is no need to set op to NONE attribute.
  if (ftype != FunctionType::NONE)
    op->setAttr("functionType", getFunctionTypeAttr(rewriter, ftype));
  return true;
}

struct TreeSolve;
using TreeSolvePtr = std::shared_ptr<TreeSolve>;
using TreeSolves = std::vector<TreeSolvePtr>;

struct SolveStep;
using SolveStepPtr = std::shared_ptr<SolveStep>;

struct SolveStep {
  SolveStepPtr subSolveStep;
  Operation *problem;
  FunctionType problemSolve;

  SolveStep(SolveStepPtr sub, Operation *prob, FunctionType probS)
      : subSolveStep(sub), problem(prob), problemSolve(probS) {}
};

struct TreeSolve : public std::enable_shared_from_this<TreeSolve> {
  DenseMap<Value, State> inputs;
  SolveStepPtr solveStep;
  Operation *problem;

  TreeSolve() : inputs({}), solveStep(nullptr), problem(nullptr) {}
  TreeSolve(DenseMap<Value, State> ins, SolveStepPtr step)
      : inputs(std::move(ins)), solveStep(std::move(step)), problem(nullptr) {}

  SmallVector<Value> getVectorOperand(Operation *op) {
    SmallVector<Value> res;
    for (auto operand : op->getOperands()) {
      if (isa<VectorType>(operand.getType())) {
        res.push_back(operand);
      }
    }
    return res;
  }

  SmallVector<Value> getVectorResults(Operation *op) {
    SmallVector<Value> results;
    for (auto res : op->getResults()) {
      if (isa<VectorType>(res.getType())) {
        results.push_back(res);
      }
    }
    return results;
  }

  int elemBitwidthOf(Value vector) {
    // assume value is vector type
    auto vecType = dyn_cast<VectorType>(vector.getType());
    auto res = vecType.getElementTypeBitWidth();
    return res == 64 ? 32 : static_cast<int>(res);
  }

  State defaultStateForBitwidth(int bitwidth) {
    switch (bitwidth) {
    case 1:
    case 8:
      return State::B8;
    case 16:
      return State::B16;
    case 32:
      return State::B32;
    default:
      assert(0 && "unsupported bitwidth for default state");
      return State::B8;
    }
  }

  State getState(Value v) {
    assert(inputs.contains(v) && "no state for value");
    return inputs[v];
  }

  TreeSolvePtr wrapThis(FunctionType probS,
                        SmallVector<std::pair<Value, State>> newIns) {
    DBG(
        if (problem->getNumResults() > 0) {
          auto res = problem->getResult(0);
          llvm::dbgs() << "====Find solve: " << probS << " -> "
                       << getState(res);
        } else { llvm::dbgs() << "====Find solve: " << probS; });
    DBG(llvm::dbgs() << "========Inputs: [\n";
        for (auto [v, sv]
             : newIns) llvm::dbgs()
        << "========\t" << sv << "->" << v << "\n";
        llvm::dbgs() << "]";);
    DenseMap<Value, State> ins;
    ins.copyFrom(inputs);
    for (auto newIn : newIns) {
      if (ins.contains(newIn.first) && ins[newIn.first] != newIn.second) {
        DBG(llvm::dbgs() << "========State Conflict: " << newIn.first << "["
                         << ins[newIn.first] << " <=> " << newIn.second
                         << "]";);
        return nullptr;
      }
      ins.insert(newIn);
    }
    auto newStep = std::make_shared<SolveStep>(solveStep, problem, probS);
    return std::make_shared<TreeSolve>(std::move(ins), newStep);
  }

  TreeSolvePtr wrapThis(FunctionType probS,
                        SmallVector<std::pair<Value, State>> baseIns,
                        SmallVector<std::pair<Value, State>> diffIns) {
    SmallVector<std::pair<Value, State>> newIns(baseIns.begin(), baseIns.end());
    for (auto in : diffIns)
      newIns.push_back(in);
    return wrapThis(probS, newIns);
  }

  TreeSolves normalizeTreeSolves(TreeSolves solves) {
    TreeSolves res;
    for (auto solve : solves) {
      if (solve != nullptr)
        res.push_back(solve);
    }
    return res;
  }

  TreeSolves solveProblem(Operation *op) {
    assert(problem == nullptr);
    problem = op;
    TreeSolves solves =
        TypeSwitch<Operation *, TreeSolves>(problem)
            .Case<hivmave::VFPgeOp>(
                [&](auto op) { return solvePregGenProblem(op); })
            .Case<hivmave::VFBroadcastScalarMaskOp>(
                [&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFLoadOp>([&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFMaskedStoreOp>(
                [&](auto op) { return solveProblem(op); })
            .Case<UnrealizedConversionCastOp>(
                [&](auto op) { return solveProblem(op); })
            .Case<hivmave::VectorLayoutCastOp>(
                [&](auto op) { return solveVectorLayoutCastProblem(op); })
            .Case<scf::ForOp>([&](auto op) { return solveProblem(op); })
            .Case<scf::YieldOp>([&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFBroadcastScalarOp>(
                [&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFPltOp>(
                [&](auto op) { return solvePregGenProblem(op); })
            .Case<hivmave::VFSelectOp, hivmave::VFAddsOp, hivmave::VFExpOp,
                  hivmave::VFMulsOp, hivmave::PregXorOp, hivmave::PregOrOp,
                  hivmave::PregAndOp, hivmave::PregNotOp, hivmave::VFMulOp,
                  hivmave::VFAddOp, hivmave::VFSubOp, hivmave::VFDivOp,
                  hivmave::VFBroadcastVectorOp, hivmave::VFXorOp,
                  hivmave::VFAndOp, hivmave::VFOrOp, hivmave::ReductionOp,
                  hivmave::VFModOp, hivmave::VFModUIOp, hivmave::VFExpOp,
                  hivmave::VFShlOp, hivmave::VFShrOp, hivmave::VFShrsOp,
                  hivmave::VFShlsOp, hivmave::VFMinOp, hivmave::VFMaxOp,
                  hivmave::VFMinsOp, hivmave::VFMaxsOp, hivmave::VMinSIOp,
                  hivmave::VMaxSIOp, hivmave::VFAbsOp, hivmave::VFNegOp,
                  hivmave::VFNotOp, hivmave::VFSqrtOp, hivmave::VFRsqrtOp,
                  hivmave::VFLnOp, hivmave::VFReluOp, hivmave::VFDivfOp,
                  hivmave::VFDivFHPOp, hivmave::VFAbsDiffOp, hivmave::VFSaddOp,
                  hivmave::VFSsubOp, hivmave::VFRndOp, hivmave::VFPReluOp,
                  hivmave::VFExpdifOp, hivmave::VFVMULLOp, hivmave::VFMulaOp,
                  hivmave::VMaxUIOp, hivmave::VMinUIOp, hivmave::VFLRelusOp,
                  hivmave::VMaxsSIOp, hivmave::VMinsSIOp, hivmave::VMaxsUIOp,
                  hivmave::VMinsUIOp, hivmave::VFCmpS, hivmave::VFVtrcOp>(
                [&](auto op) { return solveMaskProblem(op); })
            .Case<hivmave::VFCmpOp>([&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFStoreWithStrideOp>(
                [&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFTruncFOp, hivmave::VFTruncIOp>(
                [&](auto op) { return solveTruncProblem(op); })
            .Case<hivmave::VFExtFOp, hivmave::VFExtUIOp, hivmave::VFExtSIOp>(
                [&](auto op) { return solveExtProblem(op); })
            .Case<hivmave::VFFpToUIntOp, hivmave::VFSIntToFpOp,
                  hivmave::VFUIntToFpOp, hivmave::VFFpToSIntOp>(
                [&](auto op) { return solveTypeConvertProblem(op); })
            .Case<func::CallOp>([&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFGatherOp>(
                [&](auto op) { return solveProblem(op); })
            .Case<hivmave::VFDeInterleaveOp, hivmave::VFInterleaveOp>(
                [&](auto op) { return solveLayoutChangeProblem(op); })
            .Case<hivmave::VFVCIOp>(
                [&](auto op) { return solveProblem(op); })
            // Any op that falls through to Default is not an explicitly handled
            // operation in the TypeSwitch. It could be a non-HIVMAVE dialect op
            // (e.g. arith.addi), a HIVMAVE op not yet registered here, or a
            // standard vector dialect op (e.g. vector.bitcast).
            .Default([&](Operation *op) { return solveProblemDefault(op); });
    return normalizeTreeSolves(solves);
  }

  TreeSolves solveProblem(hivmave::VFMaskedStoreOp op) {
    auto vector = op.getVal();
    auto mask = op.getMask();
    int elemBitwidth = elemBitwidthOf(vector);
    switch (elemBitwidth) {
    case 1:
      return {
          wrapThis(FunctionType::PB8, {{vector, State::B8}, {mask, State::B8}}),
          wrapThis(FunctionType::PB16,
                   {{vector, State::B16}, {mask, State::B16}}),
          wrapThis(FunctionType::PB32,
                   {{vector, State::B32}, {mask, State::B32}})};
    case 8:
      return {wrapThis(FunctionType::NORM,
                       {{vector, State::B8}, {mask, State::B8}}),
              wrapThis(FunctionType::PK,
                       {{vector, State::B8_2VL}, {mask, State::B16}}),
              wrapThis(FunctionType::PK4,
                       {{vector, State::B8_4VL}, {mask, State::B32}})};
    case 16:
      return {wrapThis(FunctionType::NORM,
                       {{vector, State::B16}, {mask, State::B16}}),
              wrapThis(FunctionType::PK,
                       {{vector, State::B16_2VL}, {mask, State::B32}})};
    case 32:
      return {wrapThis(FunctionType::NORM,
                       {{vector, State::B32}, {mask, State::B32}})};
    default:
      return {};
    }
  }

  TreeSolves solveProblem(hivmave::VFStoreWithStrideOp op) {
    auto vector = op.getVal();
    auto mask = op.getMask();
    auto vecType = op.getVectorType();
    auto elemBitwidth = vecType.getElementTypeBitWidth();
    switch (elemBitwidth) {
    case 8:
      return {wrapThis(FunctionType::NORM,
                       {{vector, State::B8}, {mask, State::B8}}),
              wrapThis(FunctionType::DINTLV2,
                       {{vector, State::B8_2VL}, {mask, State::B8}}),
              wrapThis(FunctionType::DINTLV4,
                       {{vector, State::B8_4VL}, {mask, State::B8}})};
    case 16:
      return {wrapThis(FunctionType::NORM,
                       {{vector, State::B16}, {mask, State::B16}}),
              wrapThis(FunctionType::DINTLV2,
                       {{vector, State::B16_2VL}, {mask, State::B16}})};
    case 32:
      return {wrapThis(FunctionType::NORM,
                       {{vector, State::B32}, {mask, State::B32}})};
    default:
      return {};
    }
  }

  TreeSolves solveProblem(UnrealizedConversionCastOp op) {
    // su == sv and treat all value as vector value
    // if src or res is vector value
    auto src = op->getOperand(0);
    auto res = op->getResult(0);
    return {wrapThis(FunctionType::NONE, {{src, getState(res)}})};
  }

  template <typename PregGenOp> TreeSolves solvePregGenProblem(PregGenOp op) {
    auto res = op.getRes();
    switch (getState(res)) {
    case State::B8:
      return {wrapThis(FunctionType::PB8, {})};
    case State::B16:
      return {wrapThis(FunctionType::PB16, {})};
    case State::B32:
      return {wrapThis(FunctionType::PB32, {})};
    default:
      return {};
    }
  }

  TreeSolves solveProblem(hivmave::VFBroadcastScalarMaskOp op) {
    auto mask = op.getMask();
    auto res = op.getRes();
    switch (COMB_CASE(elemBitwidthOf(res), getState(res))) {
    case COMB_CASE(8, State::B8):
      return {wrapThis(FunctionType::NONE, {{mask, State::B8}})};
    case COMB_CASE(8, State::B8_2VL):
      return {wrapThis(FunctionType::NONE, {{mask, State::B16}})};
    case COMB_CASE(8, State::B8_4VL):
      return {wrapThis(FunctionType::NONE, {{mask, State::B32}})};
    case COMB_CASE(16, State::B16):
      return {wrapThis(FunctionType::NONE, {{mask, State::B16}})};
    case COMB_CASE(16, State::B16_2VL):
      return {wrapThis(FunctionType::NONE, {{mask, State::B32}})};
    case COMB_CASE(32, State::B32):
      return {wrapThis(FunctionType::NONE, {{mask, State::B32}})};
    default:
      return {};
    }
  }

  TreeSolves solveProblem(hivmave::VFLoadOp op) {
    auto res = op.getRes();
    switch (COMB_CASE(elemBitwidthOf(res), getState(res))) {
    case COMB_CASE(1, State::B8): // preg load
      return {wrapThis(FunctionType::PB8, {})};
    case COMB_CASE(1, State::B16): // preg load, unpack by intlv
      return {wrapThis(FunctionType::PB16, {})};
    case COMB_CASE(1, State::B32): // preg load, unpack by intlv
      return {wrapThis(FunctionType::PB32, {})};
    case COMB_CASE(8, State::B8):
      return {wrapThis(FunctionType::NORM, {})};
    case COMB_CASE(8, State::B8_2VL):
      return {wrapThis(FunctionType::PK, {})};
    case COMB_CASE(8, State::B8_4VL):
      return {wrapThis(FunctionType::PK4, {})};
    case COMB_CASE(16, State::B16):
      return {wrapThis(FunctionType::NORM, {})};
    case COMB_CASE(16, State::B16_2VL):
      return {wrapThis(FunctionType::PK, {})};
    case COMB_CASE(32, State::B32):
      return {wrapThis(FunctionType::NORM, {})};
    default:
      return {};
    }
  }

  TreeSolves solveProblem(scf::ForOp forOp) {
    SmallVector<std::pair<Value, State>> newIns;
    for (auto [iterArg, res] :
         llvm::zip(forOp.getRegionIterArgs(), forOp->getResults())) {
      if (inputs.contains(iterArg) && inputs.contains(res)) {
        if (getState(iterArg) != getState(res))
          return {};
      }
    }
    for (auto [initArg, iterArg, res] :
         llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs(),
                   forOp->getResults())) {
      if (inputs.contains(iterArg)) {
        newIns.push_back({initArg, getState(iterArg)});
      } else if (inputs.contains(res)) {
        newIns.push_back({initArg, getState(res)});
      } else {
        newIns.push_back({initArg, getState(iterArg)});
      }
    }

    return {wrapThis(FunctionType::NONE, newIns)};
  }

  TreeSolves solveProblem(scf::YieldOp yieldOp) {
    SmallVector<std::pair<Value, State>> newIns;
    auto forOp = yieldOp->getParentOfType<scf::ForOp>();
    for (auto [res, operand] :
         llvm::zip(forOp->getResults(), yieldOp->getOperands())) {
      if (isa<VectorType>(res.getType())) {
        newIns.push_back({operand, getState(res)});
      }
    }
    return {wrapThis(FunctionType::NONE, newIns)};
  }

  TreeSolves solveProblem(hivmave::VFBroadcastScalarOp brs) {
    auto res = brs.getRes();
    switch (COMB_CASE(elemBitwidthOf(res), getState(res))) {
    case COMB_CASE(1, State::B8):
    case COMB_CASE(1, State::B16):
    case COMB_CASE(1, State::B32):
    case COMB_CASE(8, State::B8):
    case COMB_CASE(8, State::B8_2VL):
    case COMB_CASE(8, State::B8_4VL):
    case COMB_CASE(16, State::B16):
    case COMB_CASE(16, State::B16_2VL):
    case COMB_CASE(32, State::B32):
      return {wrapThis(FunctionType::NONE, {})};
    default:
      return {};
    }
  }

  template <typename MaskOp> TreeSolves solveMaskProblem(MaskOp op) {
    auto res = op->getResult(0);
    auto mask = op.getMask();
    SmallVector<std::pair<Value, State>> newIns;
    for (auto operand : getVectorOperand(op)) {
      if (mask != operand) {
        switch (COMB_CASE(elemBitwidthOf(operand), getState(res))) {
        case COMB_CASE(1, State::B8):
        case COMB_CASE(1, State::B16):
        case COMB_CASE(1, State::B32):
        case COMB_CASE(8, State::B8):
        case COMB_CASE(8, State::B8_2VL):
        case COMB_CASE(8, State::B8_4VL):
        case COMB_CASE(16, State::B16):
        case COMB_CASE(16, State::B16_2VL):
        case COMB_CASE(32, State::B32):
          newIns.push_back({operand, getState(res)});
          break;
        default:
          return {};
        }
      }
    }
    switch (getState(res)) {
    case State::B8:
      return {wrapThis(FunctionType::NONE, newIns, {{mask, State::B8}})};
    case State::B8_2VL:
      return {wrapThis(FunctionType::NONE, newIns, {{mask, State::B16}})};
    case State::B8_4VL:
      return {wrapThis(FunctionType::NONE, newIns, {{mask, State::B32}})};
    case State::B16:
      return {wrapThis(FunctionType::NONE, newIns, {{mask, State::B16}})};
    case State::B16_2VL:
      return {wrapThis(FunctionType::NONE, newIns, {{mask, State::B32}})};
    case State::B32:
      return {wrapThis(FunctionType::NONE, newIns, {{mask, State::B32}})};
    default:
      return {};
    };
  }

  // The difference between vcmp and other mask op is
  // that the result of vcmp is vector of i1. other
  // mask ops are vectors of the same type of operands.
  // more precisly is that the result of vcmp is a mask,
  // others not.
  TreeSolves solveProblem(hivmave::VFCmpOp op) {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto mask = op.getMask();
    auto res = op.getRes();
    switch (COMB_CASE(elemBitwidthOf(lhs), getState(res))) {
    case COMB_CASE(32, State::B32):
      return {
          wrapThis(FunctionType::NONE,
                   {{lhs, State::B32}, {rhs, State::B32}, {mask, State::B32}})};
    case COMB_CASE(16, State::B32):
      return {wrapThis(
          FunctionType::NONE,
          {{lhs, State::B16_2VL}, {rhs, State::B16_2VL}, {mask, State::B32}})};
    case COMB_CASE(8, State::B32):
      return {wrapThis(
          FunctionType::NONE,
          {{lhs, State::B8_4VL}, {rhs, State::B8_4VL}, {mask, State::B32}})};
    case COMB_CASE(16, State::B16):
      return {
          wrapThis(FunctionType::NONE,
                   {{lhs, State::B16}, {rhs, State::B16}, {mask, State::B16}})};
    case COMB_CASE(8, State::B16):
      return {wrapThis(
          FunctionType::NONE,
          {{lhs, State::B8_2VL}, {rhs, State::B8_2VL}, {mask, State::B16}})};
    case COMB_CASE(8, State::B8):
      return {
          wrapThis(FunctionType::NONE,
                   {{lhs, State::B8}, {rhs, State::B8}, {mask, State::B8}})};
    default:
      return {};
    }
  }

  template <typename TruncOp> TreeSolves solveTruncProblem(TruncOp op) {
    auto src = op.getSrc();
    auto mask = op.getMask();
    auto res = op.getRes();
    auto srcElemWidth = elemBitwidthOf(src);
    // Special optimization in ave-process-vsstb(trunc[even] + trunc[odd] + vor)
    if (op->getAttr(hivmave::Layout_ChangeAttr::getMnemonic())) {
      switch (getState(res)) {
      case State::B16:
        return {wrapThis(FunctionType::C2C,
                         {{src, State::B32}, {mask, State::B32}})};
      case State::B8:
        return {wrapThis(FunctionType::C2C,
                         {{src, State::B16}, {mask, State::B16}})};
      default:
        return {};
      }
    }
    switch (COMB_CASE(srcElemWidth, getState(res))) {
    case COMB_CASE(32, State::B32): // i64 case
    case COMB_CASE(32, State::B16_2VL):
    case COMB_CASE(32, State::B8_4VL):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B32}, {mask, State::B32}})};
    case COMB_CASE(16, State::B8_2VL):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B16}, {mask, State::B16}})};
    case COMB_CASE(16, State::B8_4VL):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B16_2VL}, {mask, State::B32}})};
    default:
      return {};
    }
  }

  template <typename ExtOp> TreeSolves solveExtProblem(ExtOp op) {
    auto src = op.getSrc();
    auto mask = op.getMask();
    auto res = op.getRes();
    auto srcElemWidth = elemBitwidthOf(src);
    switch (COMB_CASE(srcElemWidth, getState(res))) {
    case COMB_CASE(8, State::B16):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B8_2VL}, {mask, State::B16}})};
    case COMB_CASE(8, State::B16_2VL):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B8_4VL}, {mask, State::B32}})};
    case COMB_CASE(8, State::B32):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B8_4VL}, {mask, State::B32}})};
    case COMB_CASE(16, State::B32):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B16_2VL}, {mask, State::B32}})};
    case COMB_CASE(32, State::B32):
      return {wrapThis(FunctionType::EVEN,
                       {{src, State::B32}, {mask, State::B32}})};
    default:
      return {};
    }
  }

  template <typename TypeConvertOp>
  TreeSolves solveTypeConvertProblem(TypeConvertOp op) {
    auto src = op.getSrc();
    auto res = op.getRes();
    int srcElemBitwidth = elemBitwidthOf(src);
    int resElemBitwidth = elemBitwidthOf(res);
    if (srcElemBitwidth > resElemBitwidth)
      return solveTruncProblem(op);
    else if (srcElemBitwidth < resElemBitwidth)
      return solveExtProblem(op);
    else
      return solveMaskProblem(op);
  }

  TreeSolves solveProblem(func::CallOp op) {
    if (llvm::any_of(op->getResultTypes(),
                     [](Type tp) { return isa<VectorType>(tp); })) {
      return solveProblemDefault(op);
    }
    // All possible combinations of vector operands.
    std::vector<SmallVector<std::pair<Value, State>>> allInsCombine;
    for (auto vector : getVectorOperand(op)) {
      SmallVector<std::pair<Value, State>> tmpIns;
      switch (elemBitwidthOf(vector)) {
      case 1: {
        tmpIns.push_back({vector, State::B8});
        tmpIns.push_back({vector, State::B16});
        tmpIns.push_back({vector, State::B32});
      }; break;
      case 8: {
        tmpIns.push_back({vector, State::B8});
        tmpIns.push_back({vector, State::B8_2VL});
        tmpIns.push_back({vector, State::B8_4VL});
      }; break;
      case 16: {
        tmpIns.push_back({vector, State::B16});
        tmpIns.push_back({vector, State::B16_2VL});
      }; break;
      case 32:
        tmpIns.push_back({vector, State::B32});
        break;
      default:
        break;
      }
      if (allInsCombine.empty()) {
        for (auto tmpIn : tmpIns) {
          allInsCombine.push_back({tmpIn});
        }
      } else {
        std::vector<SmallVector<std::pair<Value, State>>> newAllInsCombine;
        for (auto tmpIn : tmpIns) {
          for (auto insCombine : allInsCombine) {
            SmallVector<std::pair<Value, State>> newInsCombine(
                insCombine.begin(), insCombine.end());
            newInsCombine.push_back(tmpIn);
            newAllInsCombine.push_back(newInsCombine);
          }
        }
        allInsCombine = newAllInsCombine;
      }
    }
    TreeSolves solves;
    for (auto insCombine : allInsCombine) {
      solves.push_back(wrapThis(FunctionType::NONE, insCombine));
    }
    return solves;
  }

  TreeSolves solveProblem(hivmave::VFGatherOp op) {
    auto index = op.getIndexVec();
    auto mask = op.getMask();
    auto res = op.getRes();
    SmallVector<std::pair<Value, State>> newIns;
    switch (COMB_CASE(elemBitwidthOf(index), getState(res))) {
    case COMB_CASE(16, State::B8_2VL):
      return {wrapThis(FunctionType::NONE,
                       {{mask, State::B8}, {index, State::B16}}),
              wrapThis(FunctionType::NONE,
                       {{mask, State::B16}, {index, State::B16}})};
    case COMB_CASE(16, State::B16):
      return {wrapThis(FunctionType::NONE,
                       {{mask, State::B16}, {index, State::B16}})};
    case COMB_CASE(16, State::B16_2VL):
      return {
          wrapThis(FunctionType::INTLV2,
                   {{mask, State::B16}, {index, State::B16}}),
          wrapThis(FunctionType::NONE,
                   {{mask, State::B32},
                    {index, State::B16_2VL}})};
    case COMB_CASE(32, State::B16_2VL):
      return {wrapThis(FunctionType::NONE,
                       {{mask, State::B32}, {index, State::B32}})};
    case COMB_CASE(32, State::B32):
      return {wrapThis(FunctionType::NONE,
                       {{mask, State::B32}, {index, State::B32}})};
    default:
      return {};
    }
  }

  TreeSolves solveProblem(hivmave::VFVCIOp op) {
    auto res = op.getRes();
    switch (COMB_CASE(elemBitwidthOf(res), getState(res))) {
    case COMB_CASE(8, State::B8):
    case COMB_CASE(16, State::B16):
    case COMB_CASE(32, State::B32):
      return {wrapThis(FunctionType::NONE, {})};
    case COMB_CASE(8, State::B8_2VL):
    case COMB_CASE(16, State::B16_2VL):
      return {wrapThis(FunctionType::INTLV2, {})};
    case COMB_CASE(8, State::B8_4VL):
      return {wrapThis(FunctionType::INTLV4, {})};
    default:
      return {};
    }
  }

  template <typename LayoutChangeOp>
  TreeSolves solveLayoutChangeProblem(LayoutChangeOp op) {
    auto src1 = op.getSrc1();
    auto src2 = op.getSrc2();
    auto res1 = op.getRes1();
    auto res2 = op.getRes2();
    auto layoutChange = op.getLayoutChange();
    if (getState(res1) != getState(res2))
      return {};
    // Treat as original op lowered from triton
    if (!layoutChange) {
      switch (COMB_CASE(elemBitwidthOf(src1), getState(res1))) {
      case (COMB_CASE(8, State::B8)):
        return {wrapThis(FunctionType::NONE,
                         {{src1, State::B8}, {src2, State::B8}})};
      case (COMB_CASE(16, State::B16)):
        return {wrapThis(FunctionType::NONE,
                         {{src1, State::B16}, {src2, State::B16}})};
      case (COMB_CASE(32, State::B32)):
        return {wrapThis(FunctionType::NONE,
                         {{src1, State::B32}, {src2, State::B32}})};
      case (COMB_CASE(8, State::B8_2VL)):
        return {wrapThis(FunctionType::INTLV2,
                         {{src1, State::B8}, {src2, State::B8}})};
      case (COMB_CASE(8, State::B8_4VL)):
        return {wrapThis(FunctionType::INTLV4,
                         {{src1, State::B8}, {src2, State::B8}})};
      case (COMB_CASE(16, State::B16_2VL)):
        return {wrapThis(FunctionType::INTLV2,
                         {{src1, State::B16}, {src2, State::B16}})};
      default:
        return {};
      }
    }
    
    if (layoutChange.value() == hivmave::Layout_Change::UNCHANGED)
      return solveProblemDefault(op);

    switch (COMB_CASE(layoutChange.value(), getState(res1))) {
    case COMB_CASE(hivmave::Layout_Change::DENSE, State::B16):
      return {wrapThis(FunctionType::NONE,
                       {{src1, State::B16_2VL}, {src2, State::B16_2VL}})};
    case COMB_CASE(hivmave::Layout_Change::DENSE, State::B8_2VL):
      return {wrapThis(FunctionType::NONE,
                       {{src1, State::B8_4VL}, {src2, State::B8_4VL}})};
    case COMB_CASE(hivmave::Layout_Change::DENSE, State::B8):
      return {wrapThis(FunctionType::NONE,
                       {{src1, State::B8_2VL}, {src2, State::B8_2VL}})};
    case COMB_CASE(hivmave::Layout_Change::SPARSE, State::B16_2VL):
      return {wrapThis(FunctionType::NONE,
                       {{src1, State::B16}, {src2, State::B16}})};
    case COMB_CASE(hivmave::Layout_Change::SPARSE, State::B8_2VL):
      return {
          wrapThis(FunctionType::NONE, {{src1, State::B8}, {src2, State::B8}})};
    case COMB_CASE(hivmave::Layout_Change::SPARSE, State::B8_4VL):
      return {wrapThis(FunctionType::NONE,
                       {{src1, State::B8_2VL}, {src2, State::B8_2VL}})};
    default:
      return {};
    }
  }

  TreeSolves solveProblemDefault(Operation *op) {
    SmallVector<Value> vecResults = getVectorResults(op);
    // suppose that su == sv
    if (vecResults.empty()) {
      return {};
    }
    SmallVector<std::pair<Value, State>> newInputs;
    for (auto operand : getVectorOperand(op)) {
      newInputs.push_back({operand, getState(vecResults[0])});
    }
    return {wrapThis(FunctionType::NONE, newInputs)};
  }

  TreeSolves solveVectorLayoutCastProblem(hivmave::VectorLayoutCastOp op) {
    auto src = op.getSrc();
    auto res = op.getRes();

    // VectorLayoutCastOp is a convergence point: the layout encoded in
    // the result type is the definitive constraint.
    auto resVecType = cast<VectorType>(res.getType());
    // Check for null layout attribute first. dyn_cast<void(VectorLayoutAttr)>
    // on a null/absent Attribute triggers an assertion in llvm::dyn_cast,
    // so we must verify the layout is present before casting.
    if (auto layout = resVecType.getLayout()) {
      if (auto layoutAttr =
              dyn_cast<hivmave::VectorLayoutAttr>(layout)) {
        auto memTypeAttr =
            dyn_cast<hivmave::VecMemTypeAttr>(layoutAttr.getMem());
        if (memTypeAttr) {
          State resState = memTypeAttr.getValue();
          // Converge: both result and source are constrained to the
          // type-encoded layout. If downstream resolved to a different
          // state, wrapThis will detect the conflict and prune.
          return {wrapThis(FunctionType::NONE,
                          {{res, resState}, {src, resState}})};
        }
      }
    }

    // No layout attribute on result -> this is the "exit" side of a
    // constrainVectorLayout pair. The src has its layout encoded in its type.
    // Instead of propagating downstream constraints upstream (which causes
    // conflicts with src's type-encoded layout), enumerate all (src_state,
    // res_state) combinations to determine the appropriate INTLV/DINTLV
    // FunctionType. No constraint propagation to src.
    auto srcVecType = cast<VectorType>(src.getType());
    if (auto srcLayout = srcVecType.getLayout()) {
      if (auto srcLayoutAttr =
              dyn_cast<hivmave::VectorLayoutAttr>(srcLayout)) {
        auto srcMemTypeAttr =
            dyn_cast<hivmave::VecMemTypeAttr>(srcLayoutAttr.getMem());
        if (srcMemTypeAttr) {
          State srcState = srcMemTypeAttr.getValue();
          State resState = getState(res);
          switch (COMB_CASE(srcState, resState)) {
          // === Same layout -> NONE ===
          case COMB_CASE(State::B8, State::B8):
          case COMB_CASE(State::B16, State::B16):
          case COMB_CASE(State::B32, State::B32):
          case COMB_CASE(State::B8_2VL, State::B8_2VL):
          case COMB_CASE(State::B16_2VL, State::B16_2VL):
          case COMB_CASE(State::B8_4VL, State::B8_4VL):
            return {wrapThis(FunctionType::NONE, {{src, srcState}})};

          // === src denser -> res sparser (src_eff < res_eff) -> DINTLV ===
          case COMB_CASE(State::B8, State::B16):
          case COMB_CASE(State::B8, State::B8_2VL):
            return {wrapThis(FunctionType::DINTLV2, {{src, srcState}})};
          case COMB_CASE(State::B8, State::B32):
          case COMB_CASE(State::B8, State::B8_4VL):
          case COMB_CASE(State::B8, State::B16_2VL):
            return {wrapThis(FunctionType::DINTLV4, {{src, srcState}})};
          case COMB_CASE(State::B16, State::B32):
          case COMB_CASE(State::B16, State::B8_4VL):
          case COMB_CASE(State::B16, State::B16_2VL):
            return {wrapThis(FunctionType::DINTLV2, {{src, srcState}})};

          // === src sparser -> res denser (src_eff > res_eff) -> INTLV ===
          case COMB_CASE(State::B8_2VL, State::B8):
          case COMB_CASE(State::B16, State::B8):
            return {wrapThis(FunctionType::INTLV2, {{src, srcState}})};
          case COMB_CASE(State::B32, State::B8):
          case COMB_CASE(State::B8_4VL, State::B8):
          case COMB_CASE(State::B16_2VL, State::B8):
            return {wrapThis(FunctionType::INTLV4, {{src, srcState}})};
          case COMB_CASE(State::B32, State::B16):
          case COMB_CASE(State::B32, State::B8_2VL):
          case COMB_CASE(State::B8_4VL, State::B16):
          case COMB_CASE(State::B8_4VL, State::B8_2VL):
          case COMB_CASE(State::B16_2VL, State::B16):
          case COMB_CASE(State::B16_2VL, State::B8_2VL):
            return {wrapThis(FunctionType::INTLV2, {{src, srcState}})};

          // === Same eff width, src sparse -> res dense -> INTLV ===
          case COMB_CASE(State::B8_2VL, State::B16):
          case COMB_CASE(State::B16_2VL, State::B32):
            return {wrapThis(FunctionType::INTLV2, {{src, srcState}})};
          case COMB_CASE(State::B8_4VL, State::B32):
            return {wrapThis(FunctionType::INTLV4, {{src, srcState}})};

          // === Same eff width, src dense -> res sparse -> DINTLV ===
          case COMB_CASE(State::B16, State::B8_2VL):
          case COMB_CASE(State::B32, State::B16_2VL):
            return {wrapThis(FunctionType::DINTLV2, {{src, srcState}})};
          case COMB_CASE(State::B32, State::B8_4VL):
            return {wrapThis(FunctionType::DINTLV4, {{src, srcState}})};

          // Unrecognized combination -> fallback
          default:
            return {wrapThis(FunctionType::NONE, {{src, srcState}})};
          }
        }
      }
    }
    // Fallback: no constraint propagation
    return {wrapThis(FunctionType::NONE, {})};
  }
};

bool isVectorOp(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](Type tp) { return isa<VectorType>(tp); }) ||
         llvm::any_of(op->getResultTypes(),
                      [](Type tp) { return isa<VectorType>(tp); });
}

struct AnalyzeVectorLayoutPass
    : public impl::AnalyzeVectorLayoutBase<AnalyzeVectorLayoutPass> {
  using AnalyzeVectorLayoutBase<
      AnalyzeVectorLayoutPass>::AnalyzeVectorLayoutBase;

  Operation *failedOp = nullptr;
  TreeSolves failedSolves;

public:
  void runOnOperation() override {
    auto solves = solveIt();
    if (solves.empty()) {
      signalPassFailure();
      auto &os = llvm::errs();
      os << "No Solve\n";
      if (failedOp) {
        os << "========== Vector Layout Analysis Failure ==========\n";
        os << "Location: ";
        failedOp->getLoc().print(os);
        os << "\n";
        os << "Operation: " << *failedOp << "\n";
        os << "Opcode: " << failedOp->getName() << "\n";
        os << "Operand types:\n";
        for (auto operand : failedOp->getOperands()) {
          os << "  " << operand.getType() << "\n";
        }
        os << "Result types:\n";
        for (auto res : failedOp->getResults()) {
          os << "  " << res.getType() << "\n";
        }
        os << "Candidates in the solution space: " << failedSolves.size()
            << "\n";
        if (!failedSolves.empty()) {
          os << "Input states from last valid candidates:\n";
          DenseSet<Value> visited;
          for (auto &solve : failedSolves) {
            for (auto &[val, state] : solve->inputs) {
              if (visited.insert(val).second) {
                os << "  Value: " << val << "  State: " << state << "\n";
              }
            }
          }
        }
        os << "\nPossible causes and solutions:\n";
        os << "  1. The operation may not have been lowered to the HIVMAVE "
               "dialect before VectorLayout analysis.\n";
        os << "     -> If using -analyze-vector-layout standalone: lower all "
               "vector ops to HIVMAVE first.\n";
        os << "     -> If running end-to-end: an upstream pass failed to "
               "lower this op. Op: " << failedOp->getName() << "\n";
        os << "  2. The operation type may not be handled in solveProblem "
               "TypeSwitch.\n";
        os << "     -> Add a new Case for " << failedOp->getName()
            << " in solveProblem().\n";
        os << "  3. The specific VecMemType combination is not supported by "
               "this op.\n";
        os << "     -> Check COMB_CASE branches in the corresponding "
               "solveProblem function.\n";
        os << "  4. Conflicting layout requirements from multiple consumers.\n";
        os << "     -> Check if the operands/results have incompatible layout "
               "constraints.\n";
        os << "  5. Unsupported element bitwidth (only 1/8/16/32 are "
               "supported).\n";
        os << "     -> Verify all vector types have supported bitwidths.\n";
        os << "======================================================\n";
      }
      return;
    }
    if (failed(applySolve(solves[0]))) {
      signalPassFailure();
      llvm::errs() << "Failed to apply solve\n";
      return;
    }
  }

  LogicalResult applySolve(TreeSolvePtr solve) {
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);
    auto inputs = solve->inputs;
    auto curStep = solve->solveStep;
    while (curStep != nullptr) {
      auto op = curStep->problem;
      if (!setFunctionAttr(rewriter, op, curStep->problemSolve)) {
        llvm::errs() << "Solve for " << op->getName() << "is not valid\n";
        return failure();
      }
      if (isa<hivmave::VectorLayoutCastOp>(op)) {
        // Entry side: result already has type-encoded layout from IR. Skip
        // to avoid overwriting it. Constraint already enforced during solve.
        // Exit side: result has no layout. Set it to the solver-resolved
        // state so downstream ops (e.g. preg.xor) see consistent types.
        auto castVecType = cast<VectorType>(op->getResult(0).getType());
        if (!castVecType.getLayout())
          setResultVecTypeAttr(op, rewriter, inputs);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op))
        setForOpLayoutAttr(forOp, rewriter);
      else if (auto callOp = dyn_cast<func::CallOp>(op))
        legalizeCallOp(callOp, rewriter);
      else
        setResultVecTypeAttr(op, rewriter, inputs);
      curStep = curStep->subSolveStep;
    }
    return success();
  }

  TreeSolves initializeSolutions() {
    auto funcOp = getOperation();
    SmallVector<Value> noUserVecValues;
    funcOp->walk([&](Operation *op) {
      for (auto res : op->getResults()) {
        if (isa<VectorType>(res.getType()) && res.use_empty()) {
          DBG(llvm::dbgs() << "No user vector: " << res);
          noUserVecValues.push_back(res);
        }
      }
    });

    if (noUserVecValues.empty())
      return {std::make_shared<TreeSolve>()};

    // For each dangling vector, enumerate all possible states by element
    // bitwidth and compute Cartesian product to form all possible input
    // combinations.
    std::vector<SmallVector<std::pair<Value, State>>> allInsCombine;
    for (auto vec : noUserVecValues) {
      auto bitwidth = cast<VectorType>(vec.getType()).getElementTypeBitWidth();
      if (bitwidth == 64)
        bitwidth = 32;
      SmallVector<std::pair<Value, State>> tmpIns;
      switch (bitwidth) {
      case 1: {
        tmpIns.push_back({vec, State::B8});
        tmpIns.push_back({vec, State::B16});
        tmpIns.push_back({vec, State::B32});
      }; break;
      case 8: {
        tmpIns.push_back({vec, State::B8});
        tmpIns.push_back({vec, State::B8_2VL});
        tmpIns.push_back({vec, State::B8_4VL});
      }; break;
      case 16: {
        tmpIns.push_back({vec, State::B16});
        tmpIns.push_back({vec, State::B16_2VL});
      }; break;
      case 32:
        tmpIns.push_back({vec, State::B32});
        break;
      default:
        break;
      }

      if (allInsCombine.empty()) {
        for (auto tmpIn : tmpIns)
          allInsCombine.push_back({tmpIn});
      } else {
        std::vector<SmallVector<std::pair<Value, State>>> newAllInsCombine;
        for (auto tmpIn : tmpIns) {
          for (auto &insCombine : allInsCombine) {
            SmallVector<std::pair<Value, State>> newInsCombine(
                insCombine.begin(), insCombine.end());
            newInsCombine.push_back(tmpIn);
            newAllInsCombine.push_back(std::move(newInsCombine));
          }
        }
        allInsCombine = std::move(newAllInsCombine);
      }
    }

    TreeSolves solves;
    for (auto &insCombine : allInsCombine) {
      DenseMap<Value, State> inputs;
      for (auto [v, s] : insCombine)
        inputs.insert({v, s});
      solves.push_back(std::make_shared<TreeSolve>(std::move(inputs), nullptr));
    }
    return solves;
  }

  TreeSolves solveIt() {
    auto funcOp = getOperation();
    TreeSolves solves = initializeSolutions();
    SmallVector<Operation *> ops;
    funcOp->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { ops.push_back(op); });

    for (auto op : llvm::reverse(ops)) {
      if (isVectorOp(op)) {
        solves = getSolveSpace(op, solves);
        if (solves.empty())
          return {};
      }
    }
    return solves;
  }

  TreeSolves getSolveSpace(Operation *op, TreeSolves &solves) {
    DBG(llvm::dbgs() << "Solving problem of: " << *op;);
    TreeSolves newSolves;
    for (auto solve : solves) {
      for (auto s : solve->solveProblem(op)) {
        newSolves.push_back(s);
      }
    }
    DBG(llvm::dbgs() << "solve num from " << solves.size() << " To "
                     << newSolves.size() << "\n";);
    if (newSolves.empty()) {
      failedOp = op;
      failedSolves = solves;
      DBG(llvm::dbgs() << "Failed to solve problem";);
    } else {
      DBG(llvm::dbgs() << "Solve problem succeed.";);
    }
    return newSolves;
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createAnalyzeVectorLayoutPass() {
  return std::make_unique<AnalyzeVectorLayoutPass>();
}
