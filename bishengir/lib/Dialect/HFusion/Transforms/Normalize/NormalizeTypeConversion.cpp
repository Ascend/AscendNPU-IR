//===- NormalizeTypeConversion.cpp ------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Transforms/Normalize/NormalizeTypeConversionTemplate.h"

namespace mlir::hfusion {

template <typename ElemType, typename OpType>
struct HFusionNormalizeToTargetTypeTraits;

template <typename OpType>
struct HFusionNormalizeF16ToF32Traits;

struct HFusionNormalizeReduceBoolToLogicalTraits;

struct HFusionReduceI1AddToSelectMaxCompareTraits;

struct HFusionReduceI1AndOrToI16Traits;

struct HFusionReduceNormalize910_95Traits;

template <typename CumOpType>
struct HFusionNormalizeCumOpTraitsBase;

template <typename CumOpType>
struct HFusionNormalizeCumOpF16ToF32Traits;

template <typename CumOpType>
struct HFusionNormalizeCumOpI8ToTargetTraits;

struct HFusionNormalizeGatherIndexToI32Traits;

template <typename FuncType, typename FuncAttrType, typename OpType>
NamedAttribute getOpFunAttr(OpType op, PatternRewriter &rewriter) {
  FuncType func = op.getFunAttr().getValue();
  auto attr = rewriter.getAttr<FuncAttrType>(func);
  auto funAttr = rewriter.getNamedAttr("fun", attr);
  return funAttr;
}

template <typename OpType,
          typename = std::enable_if<
              std::is_same_v<OpType, linalg::ElemwiseBinaryOp> ||
              std::is_same_v<OpType, linalg::ElemwiseUnaryOp> ||
              std::is_same_v<OpType, hfusion::ElemwiseBinaryOp> ||
              std::is_same_v<OpType, hfusion::ElemwiseUnaryOp> ||
              std::is_same_v<OpType, hfusion::SelectOp> >>
SmallVector<NamedAttribute> getOpAttr(OpType op,
                                      PatternRewriter &rewriter) {
  if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
    return {getOpFunAttr<linalg::BinaryFn, linalg::BinaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseUnaryOp>) {
    return {getOpFunAttr<linalg::UnaryFn, linalg::UnaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
    return {
        getOpFunAttr<hfusion::BinaryFn, hfusion::BinaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
    return {getOpFunAttr<hfusion::UnaryFn, hfusion::UnaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, hfusion::SelectOp>) {
    // no extra attrs
    return {};
  } else
    llvm_unreachable("Unsupport Normalize OpType.");
}

static void replaceI1ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                           const SmallVector<Value> &newResults,
                                           PatternRewriter &rewriter,
                                           bool enableOverflow = true) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isI1ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value castResult =
        castTo(rewriter, newResult, rewriter.getI1Type(),
               hfusion::RoundMode::TRUNC, std::nullopt, enableOverflow);
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

SmallVector<Value> normalizeF16ToF32(PatternRewriter &rewriter,
                                     const SmallVector<Value> &values) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isF16ElemType(v.getType())) {
      result.push_back(v);
      continue;
    }
    Value castResult = castTo(rewriter, v, rewriter.getF32Type());
    result.push_back(castResult);
  }
  return result;
}

template <typename... Patterns>
void addTypeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<Patterns...>(patterns.getContext());
}

template <typename ElemType, typename... Ops>
void addNormalizeToTargetPatterns(RewritePatternSet &patterns) {
  addTypeConversionPatterns<
      mlir::NormalizeToTargetTemplate<
          ElemType, Ops, HFusionNormalizeToTargetTypeTraits<ElemType, Ops>>...>(
      patterns);
}

template <typename ElemType, typename OpType>
struct HFusionNormalizeToTargetTypeTraits : public NormalizeTraitsBase {
  static bool shouldNormalize(OpType op) {
    if (!op.hasPureTensorSemantics())
      return false;

    if constexpr (std::is_same_v<ElemType, int8_t>) {
      if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
        // only part of hfusion elemwise binary op support both i8
        auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
        hfusion::BinaryFn func = binOp.getFun();
        // bit operation can support b8 operand
        static DenseSet<hfusion::BinaryFn> binarySet = {
            hfusion::BinaryFn::vor, hfusion::BinaryFn::vand,
            hfusion::BinaryFn::vxor};
        if (binarySet.contains(func))
          return false;
      }

      if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
        // only part of hfusion elemwise unary op support i8
        auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op);
        hfusion::UnaryFn func = unaryOp.getFun();
        static DenseSet<hfusion::UnaryFn> unarySet = {hfusion::UnaryFn::vnot};
        if (unarySet.contains(func))
          return false;
      }

      if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
        // The following ops support i8 type in c310
        // Could compute on i8 directly and no need cast to f16 nor f32
        if (archisAscend950) {
          auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
          linalg::BinaryFn func = binOp.getFun();
          static DenseSet<linalg::BinaryFn> binarySet950 = {
              linalg::BinaryFn::add, linalg::BinaryFn::sub,
              linalg::BinaryFn::max_signed, linalg::BinaryFn::min_signed,
              linalg::BinaryFn::max_unsigned, linalg::BinaryFn::min_unsigned};
          if (binarySet950.contains(func))
            return false;
        }
      }
    }

    // no linalg elemwise unary/binary op support i8
    bool computeByF16 = shoudComputeByF16(op);
    bool computeByF32 = shoudComputeByF32(op);
    if (!computeByF16 && !computeByF32)
      return false;

    return true;
  }

  static CastSignKind getCastSignKind(OpType op) {
    if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      bool useChainUnsigned = false;
      if (op->getNumResults() > 0)
        useChainUnsigned = analyzeUnsignedNeeds(op->getResult(0));
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn linalgFn = binOp.getFunAttr().getValue();
bool isInputUnsigned = linalgFn == linalg::BinaryFn::max_unsigned ||
                       linalgFn == linalg::BinaryFn::min_unsigned ||
                       useChainUnsigned;
      return isInputUnsigned ? CastSignKind::Unsigned : CastSignKind::Signed;
    }
    return CastSignKind::Signed;
  }

  static SmallVector<Value> getInputs(OpType op) {
    return SmallVector<Value>(op.getInputs().begin(), op.getInputs().end());
  }

  static SmallVector<Value> getOutputs(OpType op) {
    return SmallVector<Value>(op.getOutputs().begin(), op.getOutputs().end());
  }

  static Type getTargetType(PatternRewriter &rewriter, OpType op) {
    if (shoudComputeByF16(op))
      return rewriter.getF16Type();
    if (shoudComputeByF32(op))
      return rewriter.getF32Type();
    return nullptr;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc, OpType op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    SmallVector<NamedAttribute> attrs = getOpAttr(op, rewriter);
    if constexpr (std::is_same_v<OpType, linalg::ElemwiseUnaryOp> ||
                  std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      // no attr needs to be changed
      return rewriter.create<OpType>(loc, ValueRange{newInputs},
                                     ValueRange{newOutputs}, attrs);
    }

    if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      // convert linalg binary op to hfusion
      static DenseMap<linalg::BinaryFn, hfusion::BinaryFn> binAttrMap = {
          {linalg::BinaryFn::max_unsigned, hfusion::BinaryFn::maxf},
          {linalg::BinaryFn::max_signed, hfusion::BinaryFn::maxf},
          {linalg::BinaryFn::min_unsigned, hfusion::BinaryFn::minf},
          {linalg::BinaryFn::min_signed, hfusion::BinaryFn::minf},
      };
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn linalgFn = binOp.getFunAttr().getValue();
      if (binAttrMap.contains(linalgFn)) {
        hfusion::BinaryFn hfusionFn = binAttrMap[linalgFn];
        return hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp,
                                       hfusion::BinaryFn,
                                       hfusion::BinaryFnAttr>(
            rewriter, loc, hfusionFn, ValueRange{newInputs},
            ValueRange{newOutputs});
      }
      // other linalg elemwise binary op can be created using origin attr
      return rewriter.create<linalg::ElemwiseBinaryOp>(
          loc, ValueRange{newInputs}, ValueRange{newOutputs}, attrs);
    }

    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
      auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op);
      hfusion::UnaryFn unaryFn = unaryOp.getFun();
      if (unaryFn == hfusion::UnaryFn::absi) {
        // convert hfusion absi to linalg abs op
        return hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                      linalg::UnaryFnAttr>(
            rewriter, loc, linalg::UnaryFn::abs, ValueRange{newInputs},
            ValueRange(newOutputs));
      }
      // other hfusion elemwise unary op can be created using origin attr
      return rewriter.create<hfusion::ElemwiseUnaryOp>(
          loc, ValueRange{newInputs}, ValueRange{newOutputs}, attrs);
    }

    return rewriter.create<OpType>(loc, ValueRange{newInputs},
                                   ValueRange{newOutputs}, attrs);
  }

  static void replaceResults(PatternRewriter &rewriter, OpType op,
                             ValueRange newResults, CastSignKind intKind) {
    SmallVector<Value> copiedResults(newResults.begin(), newResults.end());
    if constexpr (std::is_same_v<OpType, hfusion::SelectOp>) {
      replaceI8ResultsWithTargetType(op->getResults(), copiedResults, rewriter,
                                     false);
      return;
    }
    if constexpr (std::is_same_v<ElemType, bool>) {
      replaceI1ResultsWithTargetType(op->getResults(), copiedResults, rewriter);
    }
    // TODO: set argument enableOverflow = false inside for all non-arithmatic
    // op type
    if constexpr (std::is_same_v<ElemType, int8_t>) {
      bool isUnsigned = intKind == CastSignKind::Unsigned;
      replaceI8ResultsWithTargetType(op->getResults(), copiedResults, rewriter,
                                     true, isUnsigned);
    }
  }

private:
  static bool analyzeUnsignedNeeds(Value value) {
    // Since the `add` and `sub` operations in the Linalg dialect have the same
    // representation for uint and int scenarios, it is not possible to directly
    // determine the sign nature through the Linalg dialect itself. Instead, the
    // sign nature needs to be analyzed through the use-chain of the Value.
    // In cdiv, only a single layer of BFS on the use-chain is required.
    // Other maybe need more levels?
    for (Operation *user : value.getUsers()) {
      // In cdiv, the op type that needs to be checked is hfusion::Cast.
      if (auto castOp = dyn_cast<hfusion::CastOp>(user)) {
        if (castOp.getCast() == hfusion::TypeFn::cast_unsigned) {
          return true;
        }
      }
    }
    return false;
  }

  static bool shoudComputeByF16(OpType op) {
    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
      hfusion::BinaryFn func = binOp.getFun();
      // can compute on i8 directly and no need cast to f16
      static DenseSet<hfusion::BinaryFn> binarySet = {
          // can compute on i8 directly and no need cast to f16
          hfusion::BinaryFn::shli, hfusion::BinaryFn::shrsi,
          hfusion::BinaryFn::shrui,
          // should compute on f32 for high precision and change to use float
          // ops to compute f32 data
          hfusion::BinaryFn::ceildivsi, hfusion::BinaryFn::floordivsi,
          hfusion::BinaryFn::ceildivui, hfusion::BinaryFn::mod};
      return !binarySet.contains(func);
    } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn func = binOp.getFun();
      // I8 type for add and sub op need cast to f16 and no need cast to f32
      auto firstInputType = binOp.getInputs()[0].getType();
      auto secondInputType = binOp.getInputs()[1].getType();
      const bool isI8Type =
          (isI8ElemType(firstInputType) && isI8ElemType(secondInputType));
      const bool isAddOrSubOp =
          (func == linalg::BinaryFn::add || func == linalg::BinaryFn::sub);
      if (isI8Type && isAddOrSubOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << " I8 type for add and sub op need cast to f16 \n");
        return true;
      }
      // should compute on f32 for high precision
      static DenseSet<linalg::BinaryFn> binarySet = {
          linalg::BinaryFn::mul, linalg::BinaryFn::div_unsigned,
          linalg::BinaryFn::div, linalg::BinaryFn::add,
          linalg::BinaryFn::sub,
      };
      return !binarySet.contains(func);
    }
    return true;
  }

  static bool shoudComputeByF32(OpType op) {
    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
      hfusion::BinaryFn func = binOp.getFun();
      static DenseSet<hfusion::BinaryFn> binarySet = {hfusion::BinaryFn::mod};
      return binarySet.contains(func);
    } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn func = binOp.getFun();
      // 910B: should compute on f32
      static DenseSet<linalg::BinaryFn> binarySet = {
          linalg::BinaryFn::mul,
          linalg::BinaryFn::add,
          linalg::BinaryFn::sub,
      };
      return binarySet.contains(func);
    }
    return false;
  }
};

template <typename OpType>
struct HFusionNormalizeF16ToF32Traits : public NormalizeTraitsBase {
  static bool shouldNormalize(OpType op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if constexpr (std::is_same_v<OpType, linalg::ElemwiseUnaryOp>) {
      static DenseSet<linalg::UnaryFn> linalgUnarySet = {linalg::UnaryFn::log};
      linalg::UnaryFn unaryFn = op.getFun();
      if (linalgUnarySet.contains(unaryFn))
        return true;
    }
    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      static DenseSet<hfusion::BinaryFn> hfusionBinarySet = {
          hfusion::BinaryFn::powf};
      hfusion::BinaryFn binaryFn = op.getFun();
      if (hfusionBinarySet.contains(binaryFn))
        return true;
    }
    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
      static DenseSet<hfusion::UnaryFn> hfusionUnarySet = {
          hfusion::UnaryFn::rsqrt};
      hfusion::UnaryFn unaryFn = op.getFun();
      if (hfusionUnarySet.contains(unaryFn))
        return true;
    }
    return false;
  }

  static Value rebuildOpInF32(PatternRewriter &rewriter, Location loc, OpType op,
                              ArrayRef<Value> newInputs,
                              ArrayRef<Value> newInits) {
    SmallVector<NamedAttribute> attrs = getOpAttr(op, rewriter);
    auto newOp = rewriter.create<OpType>(loc, ValueRange{newInputs},
                                         ValueRange{newInits}, attrs);
    return newOp->getResult(0);
  }
};

template <typename CumOpType>
struct HFusionNormalizeCumOpTraitsBase : public NormalizeTraitsBase {
  static bool shouldNormalize(CumOpType op) {
    return std::is_same_v<CumOpType, hfusion::CumsumOp> ||
           std::is_same_v<CumOpType, hfusion::CumprodOp>;
  }

  static Value getInput(CumOpType op) { return op.getInput(); }
  static Value getOutput(CumOpType op) { return op.getOutput(); }

  static Value rebuildOpInF32(PatternRewriter &rewriter, Location loc,
                              CumOpType op, Value newInput, Value newOutput) {
    auto newOp = rewriter.create<CumOpType>(loc, TypeRange{newOutput},
                                             newInput, op.getCumDims(),
                                             op.getReverse());
    return newOp->getResult(0);
  }
};

template <typename CumOpType>
struct HFusionNormalizeCumOpF16ToF32Traits
    : public HFusionNormalizeCumOpTraitsBase<CumOpType> {};

template <typename CumOpType>
struct HFusionNormalizeCumOpI8ToTargetTraits
    : public HFusionNormalizeCumOpTraitsBase<CumOpType> {};

struct HFusionNormalizeReduceBoolToLogicalTraits : public NormalizeTraitsBase {
  static bool shouldNormalize(linalg::ReduceOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return false;
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (isa<arith::AddIOp, arith::MaxUIOp, arith::MaxSIOp,
            arith::MulIOp, arith::MinUIOp, arith::MinSIOp>(bodyOp))
      return true;
    return false;
  }

  static std::optional<mlir::ReduceBoolLogicalKind>
  getLogicalReduceKind(linalg::ReduceOp op) {
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (isa<arith::AddIOp, arith::MaxUIOp, arith::MaxSIOp>(bodyOp))
      return mlir::ReduceBoolLogicalKind::Or;
    if (isa<arith::MulIOp, arith::MinUIOp, arith::MinSIOp>(bodyOp))
      return mlir::ReduceBoolLogicalKind::And;
    return std::nullopt;
  }

  static LogicalResult rewriteToLogicalReduce(PatternRewriter &rewriter,
                                              linalg::ReduceOp op,
                                              mlir::ReduceBoolLogicalKind kind) {
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (bodyOp == nullptr)
      return failure();
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(bodyOp->getBlock());
    if (kind == mlir::ReduceBoolLogicalKind::Or) {
      auto newOp = rewriter.create<arith::OrIOp>(bodyOp->getLoc(),
                                                  bodyOp->getOperand(0),
                                                  bodyOp->getOperand(1));
      rewriter.modifyOpInPlace(bodyOp, [&]() { bodyOp->replaceAllUsesWith(newOp); });
    } else {
      auto newOp = rewriter.create<arith::AndIOp>(bodyOp->getLoc(),
                                                   bodyOp->getOperand(0),
                                                   bodyOp->getOperand(1));
      rewriter.modifyOpInPlace(bodyOp, [&]() { bodyOp->replaceAllUsesWith(newOp); });
    }
    return success();
  }
};

/// Before conversion:
/// ```mlir
///    %26 = bufferization.alloc_tensor() : tensor<i1>
///    %27 = linalg.fill ins(%false : i1) outs(%26 : tensor<i1>) -> tensor<i1>
///      %reduced = linalg.reduce ins(%25 : tensor<8xi1>) outs(%27 : tensor<i1>)
///      dimensions = [0]
///       (%in: i1, %init: i1) {
///          %30 = arith.addi %in, %init : i1
///          linalg.yield %30 : i1
///        }
/// ```
/// After conversion:
/// ```mlir
///        %27 = tensor.empty() : tensor<8xi16>
///        %28 = hfusion.select ins(%25, %cst_0, %cst : tensor<8xi1>,
///        tensor<8xi16>, tensor<8xi16>) outs(%27 : tensor<8xi16>) ->
///        tensor<8xi16> %29 = bufferization.alloc_tensor() : tensor<i16> %30 =
///        linalg.fill ins(%c0_i16 : i16) outs(%29 : tensor<i16>) -> tensor<i16>
///        %reduced = linalg.reduce ins(%28 : tensor<8xi16>) outs(%30 :
///        tensor<i16>) dimensions = [0]
///          (%in: i16, %init: i16) {
///            %35 = arith.maxsi %in, %init : i16
///            linalg.yield %35 : i16
///          }
///        %31 = tensor.empty() : tensor<1xi1>
///        %32 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
///        ins(%reduced, %c0_i16 : tensor<i16>, i16) outs(%31 : tensor<1xi1>) ->
///        tensor<1xi1>
/// ```
struct HFusionReduceI1AddToSelectMaxCompareTraits : public NormalizeTraitsBase {
  static bool shouldNormalize(linalg::ReduceOp reduceOp) {
    if (!reduceOp.hasPureTensorSemantics())
      return false;
    if (!hacc::utils::isAscend950(reduceOp->getParentOfType<ModuleOp>()))
      return false;
    SmallVector<Value> inputs = reduceOp.getInputs();
    SmallVector<Value> inits = reduceOp.getInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return false;
    Block &body = reduceOp.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (!isa<arith::AddIOp>(bodyOp))
      return false;
    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1 || dimensions[0] != 0)
      return false;
    auto inputType = dyn_cast<RankedTensorType>(inputs[0].getType());
    auto initType = dyn_cast<RankedTensorType>(inits[0].getType());
    if (!inputType || !initType)
      return false;
    if (!inputType.getElementType().isInteger(1) ||
        !initType.getElementType().isInteger(1))
      return false;
    if (initType.getRank() != 0)
      return false;
    return true;
  }

  static Value getInput(linalg::ReduceOp op) { return op.getInputs()[0]; }

  static Value createPredicateTensor(PatternRewriter &rewriter, Location loc,
                                     Value likeValue, Type elemType,
                                     int64_t fillValue) {
    auto shapedType = dyn_cast<RankedTensorType>(likeValue.getType());
    auto targetType = RankedTensorType::get(shapedType.getShape(), elemType);
    auto attr = DenseElementsAttr::get(targetType,
                                       rewriter.getI16IntegerAttr(fillValue));
    return rewriter.create<arith::ConstantOp>(loc, targetType, attr);
  }

  static Value createSelectOp(PatternRewriter &rewriter, Location loc,
                              Value cond, Value trueVal, Value falseVal,
                              Value output) {
    auto selectResult = rewriter.create<hfusion::SelectOp>(
        loc, output.getType(), ValueRange{cond, trueVal, falseVal},
        ValueRange{output});
    return selectResult.getResult(0);
  }

  static Value createReduceInit(PatternRewriter &rewriter, Location loc,
                                linalg::ReduceOp op, Type elemType) {
    Value cst0 = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getIntegerAttr(elemType, 0));
    auto scalarTensorType = RankedTensorType::get({}, elemType);
    auto allocOp = rewriter.create<bufferization::AllocTensorOp>(
        loc, scalarTensorType, ValueRange{});
    auto fillOp = rewriter.create<linalg::FillOp>(loc, cst0, allocOp.getResult());
    return fillOp.getResult(0);
  }

  static Value createMaxReduce(PatternRewriter &rewriter, Location loc,
                               linalg::ReduceOp op, Value input, Value init) {
    auto dimensions = op.getDimensions();
    auto newReduce = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{input}, ValueRange{init}, dimensions,
        [&](OpBuilder &builder, Location loc, ValueRange operands) {
          Value max = builder.create<arith::MaxSIOp>(loc, operands[0], operands[1]);
          builder.create<linalg::YieldOp>(loc, ValueRange{max});
        });
    return newReduce.getResult(0);
  }

  static Value createCmpNeZero(PatternRewriter &rewriter, Location loc,
                               linalg::ReduceOp, Value reduced, Type elemType) {
    MLIRContext *ctx = rewriter.getContext();
    auto scalarI16Tensor = RankedTensorType::get({}, elemType);
    auto zeroScalarAttr =
        DenseElementsAttr::get(scalarI16Tensor, rewriter.getI16IntegerAttr(0));
    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, scalarI16Tensor,
                                                          zeroScalarAttr);
    Type scalarI1Tensor = RankedTensorType::get({1}, rewriter.getI1Type());
    Value compareOut = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{1}, rewriter.getI1Type());
    auto cmpFnAttr = hfusion::CompareFnAttr::get(ctx, hfusion::CompareFn::vne);
    auto compareResult = rewriter.create<hfusion::CompareOp>(
        loc, scalarI1Tensor, ValueRange{reduced, zeroScalar},
        ValueRange{compareOut}, cmpFnAttr);

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value extracted = rewriter.create<tensor::ExtractOp>(
        loc, compareResult.getResult(0), zeroIdx);
    Type extractedScalarI1Tensor =
        RankedTensorType::get({}, rewriter.getI1Type());
    return rewriter.create<tensor::FromElementsOp>(
        loc, extractedScalarI1Tensor, extracted);
  }

  static void replaceResults(PatternRewriter &rewriter, linalg::ReduceOp op,
                             Value result) {
    rewriter.replaceOp(op, result);
  }
};

static Value createScalarFillOp(PatternRewriter &rewriter, Location loc,
                                Type elemType, int64_t initValue) {
  Value initConstant = rewriter.create<arith::ConstantOp>(
      loc, elemType, rewriter.getIntegerAttr(elemType, initValue));
  auto scalarTensorType = RankedTensorType::get({}, elemType);
  auto allocOp = rewriter.create<bufferization::AllocTensorOp>(
      loc, scalarTensorType, ValueRange{});
  auto fillOp =
      rewriter.create<linalg::FillOp>(loc, initConstant, allocOp.getResult());
  return fillOp.getResult(0);
}

/// Convert reduce i1 and/or operations to i16 reduce with min/max operations.
///
/// The conversion leverages the following equivalence:
///   - i1 and operation: 0/1 values -> min over i16 (0/1 values)
///   - i1 or operation: 0/1 values -> max over i16 (0/1 values)
///
/// Before conversion (and):
/// ```mlir
///   %1 = linalg.fill ins(%true : i1) outs(%0 : tensor<i1>) -> tensor<i1>
///   %reduced = reduce <andi> ins(%arg0 : tensor<Nxi1>) outs(%1 : tensor<i1>)
/// ```
///
/// After conversion (and):
/// ```mlir
///   %input_i16 = hfusion.cast ins(%arg0 : tensor<Nxi1>) outs(...) : tensor<Nxi16>
///   %init = linalg.fill ins(%c1_i16 : i16) outs(...) : tensor<i16>
///   %reduced = reduce <minsi> ins(%input_i16 : tensor<Nxi16>) outs(%init : tensor<i16>)
///   %final = hfusion.cast ins(%reduced : tensor<i16>) outs(...) : tensor<i1>
/// ```
///
/// Note: The i1<->i16 cast operations are not directly supported by hardware,
/// but other normalize patterns will optimize them into hardware-compliant
/// instructions (e.g., NormalizeCastLoweringOp pattern).
struct HFusionReduceI1AndOrToI16Traits : public NormalizeTraitsBase {
  static bool shouldNormalize(linalg::ReduceOp reduceOp) {
    if (!reduceOp.hasPureTensorSemantics())
      return false;
    if (!hacc::utils::isRegBasedArch(reduceOp->getParentOfType<ModuleOp>()))
      return false;
    SmallVector<Value> inputs = reduceOp.getInputs();
    SmallVector<Value> inits = reduceOp.getInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return false;
    Block &body = reduceOp.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    return isa<arith::AndIOp>(bodyOp) || isa<arith::OrIOp>(bodyOp);
  }

  static Value getInput(linalg::ReduceOp op) { return op.getInputs()[0]; }

  static Value castInputToI16(PatternRewriter &rewriter, Location loc,
                              Value input, Type elemType) {
    return castTo(rewriter, input, elemType, hfusion::RoundMode::RINT);
  }

  static bool isAndReduce(linalg::ReduceOp reduceOp) {
    Block &body = reduceOp.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    return isa<arith::AndIOp>(bodyOp);
  }

  static Value createReduceInit(PatternRewriter &rewriter, Location loc,
                                linalg::ReduceOp op, Type elemType,
                                int64_t initValue) {
    return createScalarFillOp(rewriter, loc, elemType, initValue);
  }

  static Value createReducedOp(PatternRewriter &rewriter, Location loc,
                               linalg::ReduceOp op, Value input, Value init,
                               bool isAndReduce) {
    auto dimensions = op.getDimensions();
    auto newReduce = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{input}, ValueRange{init}, dimensions,
        [&](OpBuilder &builder, Location loc, ValueRange operands) {
          Value result;
          if (isAndReduce) {
            result = builder.create<arith::MinSIOp>(loc, operands[0], operands[1]);
          } else {
            result = builder.create<arith::MaxSIOp>(loc, operands[0], operands[1]);
          }
          builder.create<linalg::YieldOp>(loc, ValueRange{result});
        });
    return newReduce.getResult(0);
  }

  static Value castResultToI1(PatternRewriter &rewriter, Location loc,
                              Value result) {
    return castTo(rewriter, result, rewriter.getI1Type(), hfusion::RoundMode::RINT);
  }

  static void replaceResults(PatternRewriter &rewriter, linalg::ReduceOp op,
                             Value result) {
    rewriter.replaceAllUsesWith(op.getResult(0), result);
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<bool, tensor::ConcatOp>
    : public NormalizeTraitsBase {
  static bool shouldNormalize(tensor::ConcatOp op) {
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> results = op->getResults();
    return hasI1ElemType(inputs) || hasI1ElemType(results);
  }

  static SmallVector<Value> getInputs(tensor::ConcatOp op) {
    return op.getInputs();
  }

  static SmallVector<Value> getOutputs(tensor::ConcatOp) { return {}; }

  static Type getTargetType(PatternRewriter &rewriter, tensor::ConcatOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(tensor::ConcatOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          tensor::ConcatOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &) {
    return rewriter.create<tensor::ConcatOp>(loc, op.getDim(),
                                             ValueRange(newInputs));
  }

  static void replaceResults(PatternRewriter &rewriter, tensor::ConcatOp op,
                             ValueRange newResults, CastSignKind) {
    replaceI1ResultsWithTargetType(op->getResults(),
                                   SmallVector<Value>(newResults.begin(),
                                                      newResults.end()),
                                   rewriter,
                                   false);
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<bool, CompareOp>
    : public NormalizeTraitsBase {
  static bool shouldNormalize(CompareOp op) { return true; }

  static SmallVector<Value> getInputs(CompareOp op) {
    return SmallVector<Value>(op.getInputs().begin(), op.getInputs().end());
  }

  static SmallVector<Value> getOutputs(CompareOp op) { return {}; }

  static Type getTargetType(PatternRewriter &rewriter, CompareOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(CompareOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc, CompareOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return hfusion::createCmpOp(rewriter, loc, newInputs[0], newInputs[1],
                                op.getCompareFn());
  }

  static void replaceResults(PatternRewriter &rewriter, CompareOp op,
                             ValueRange newResults, CastSignKind) {
    rewriter.replaceOp(op, newResults);
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<bool, linalg::TransposeOp>
    : public NormalizeTraitsBase {
  static bool shouldNormalize(linalg::TransposeOp op) { return true; }

  static SmallVector<Value> getInputs(linalg::TransposeOp op) {
    return {op.getInput()};
  }

  static SmallVector<Value> getOutputs(linalg::TransposeOp op) {
    return SmallVector<Value>(op.getDpsInits().begin(),
                              op.getDpsInits().end());
  }

  static Type getTargetType(PatternRewriter &rewriter,
                            linalg::TransposeOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(linalg::TransposeOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          linalg::TransposeOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return rewriter.create<linalg::TransposeOp>(
        loc, newInputs.front(), newOutputs.front(), op.getPermutation());
  }

  static void replaceResults(PatternRewriter &rewriter,
                             linalg::TransposeOp op,
                             ValueRange newResults, CastSignKind) {
    SmallVector<Value> oldResults(op->getResults().begin(),
                                  op->getResults().end());
    SmallVector<Value> newResultsVec(newResults.begin(), newResults.end());
    replaceI1ResultsWithTargetType(oldResults, newResultsVec, rewriter, false);
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<int8_t, linalg::ReduceOp>
    : public NormalizeTraitsBase {
  static bool shouldNormalize(linalg::ReduceOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if (!shouldComputeByFloat(op))
      return false;
    return true;
  }

  static SmallVector<Value> getInputs(linalg::ReduceOp op) {
    return SmallVector<Value>(op.getInputs().begin(), op.getInputs().end());
  }

  static SmallVector<Value> getOutputs(linalg::ReduceOp op) {
    return SmallVector<Value>(op.getInits().begin(), op.getInits().end());
  }

  static Type getTargetType(PatternRewriter &rewriter, linalg::ReduceOp op) {
    if (shoudComputeI8ByF32(op))
      return rewriter.getF32Type();
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(linalg::ReduceOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          linalg::ReduceOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    Type srcType = rewriter.getI8Type();
    Type targetType = getTargetType(rewriter, op);
    return createNewReduceOp(op, rewriter, srcType, targetType, newInputs,
                             newOutputs);
  }

  static void replaceResults(PatternRewriter &rewriter, linalg::ReduceOp op,
                             ValueRange newResults, CastSignKind) {
    SmallVector<Value> oldResults(op->getResults().begin(),
                                  op->getResults().end());
    SmallVector<Value> newResultsVec(newResults.begin(), newResults.end());
    replaceI8ResultsWithTargetType(oldResults, newResultsVec, rewriter);
  }

private:
  static bool shouldComputeByFloat(linalg::ReduceOp reduceOp) {
    Block &body = reduceOp.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto bodyOp = yieldOp.getValues()[0].getDefiningOp();
    // can compute on i8 directly and no need cast to float.
    if (isa<arith::XOrIOp>(bodyOp) || isa<arith::OrIOp>(bodyOp) ||
        isa<arith::AndIOp>(bodyOp)) {
      return false;
    }
    return true;
  }

  static bool shoudComputeI8ByF32(linalg::ReduceOp op) {
    Block *block = &op.getRegion().front();
    for (Operation &bodyOp : *block) {
      if (dyn_cast_or_null<arith::AddIOp>(bodyOp)) {
        return true;
      }
    }
    return false;
  }
};

struct HFusionReduceNormalize910_95Traits : public NormalizeTraitsBase {
  static bool shouldNormalize(linalg::ReduceOp op) {
    if (!hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>()) ||
        !hacc::utils::isAscend950(op->getParentOfType<ModuleOp>()))
      return false;
    if (!op.hasPureTensorSemantics())
      return false;
    auto &region = op.getRegion();
    if (llvm::any_of(region.getOps(),
                     [](Operation &innerOp) { return isa<arith::XOrIOp>(&innerOp); }))
      return false;
    if (op->hasAttr("reduce_mode"))
      return false;
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits))
      return false;
    return true;
  }

  static CastSignKind getCastSignKind(linalg::ReduceOp op) {
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (isa<arith::MaxUIOp, arith::MinUIOp>(bodyOp))
      return CastSignKind::Unsigned;
    return CastSignKind::Signed;
  }

  static SmallVector<Value> getInputs(linalg::ReduceOp op) {
    return SmallVector<Value>(op.getInputs().begin(), op.getInputs().end());
  }

  static SmallVector<Value> getOutputs(linalg::ReduceOp op) {
    return SmallVector<Value>(op.getInits().begin(), op.getInits().end());
  }

  static Value createI8ToI16Cast(PatternRewriter &rewriter, Location loc,
                                 Value value, CastSignKind intKind) {
    Type dstType = rewriter.getI16Type();
    if (intKind == CastSignKind::Unsigned)
      return castTo(rewriter, value, dstType, TypeFn::cast_unsigned);
    return castTo(rewriter, value, dstType);
  }

  static Operation *rebuildOpInI16(PatternRewriter &rewriter, Location loc,
                                   linalg::ReduceOp op,
                                   SmallVector<Value> &newInputs,
                                   SmallVector<Value> &newOutputs) {
    return createNewReduceI16Op(op, rewriter, rewriter.getI8Type(),
                                rewriter.getI16Type(), newInputs, newOutputs);
  }

  static void replaceResults(PatternRewriter &rewriter, linalg::ReduceOp op,
                             ValueRange newResults, CastSignKind) {
    replaceI8ResultsWithTargetType(op->getResults(),
                                   SmallVector<Value>(newResults.begin(),
                                                      newResults.end()),
                                   rewriter);
  }

private:
  static Operation *mapReduceBodyOpToI16(PatternRewriter &rewriter,
                                         Location loc, Operation *bodyOp,
                                         Type srcType, IRMapping &mapper) {
    if (isa<linalg::YieldOp>(bodyOp))
      return rewriter.clone(*bodyOp, mapper);
    // only support binary arith ops here
      assert(bodyOp->getNumOperands() == 2 && "only support binary arith op");
    Value newLhs = mapper.lookupOrNull(bodyOp->getOperand(0));
    Value newRhs = mapper.lookupOrNull(bodyOp->getOperand(1));
    if (!newLhs || !newRhs)
      return nullptr;
    Type lhsType = newLhs.getType();
    Type rhsType = newRhs.getType();
    if (isa<IntegerType>(lhsType) && isa<IntegerType>(rhsType) &&
        lhsType.getIntOrFloatBitWidth() == 16 &&
        rhsType.getIntOrFloatBitWidth() == 16) {
      if (isa<arith::AddIOp>(bodyOp))
        return rewriter.create<arith::AddIOp>(loc, lhsType, newLhs, newRhs);
      // max/min signed/unsigned
      if (isa<arith::MaxSIOp>(bodyOp))
        return rewriter.create<arith::MaxSIOp>(loc, lhsType, newLhs, newRhs);
      if (isa<arith::MaxUIOp>(bodyOp))
        return rewriter.create<arith::MaxUIOp>(loc, lhsType, newLhs, newRhs);
      if (isa<arith::MinSIOp>(bodyOp))
        return rewriter.create<arith::MinSIOp>(loc, lhsType, newLhs, newRhs);
      if (isa<arith::MinUIOp>(bodyOp))
        return rewriter.create<arith::MinUIOp>(loc, lhsType, newLhs, newRhs);
      // if it is some other integer op, fall back to cloning (with mapping).
      return rewriter.clone(*bodyOp, mapper);
    }

    // for all other cases just clone with the mapper.
    return rewriter.clone(*bodyOp, mapper);
  }
  // create the new reduction op with i16 inputs
  static Operation *createNewReduceI16Op(linalg::ReduceOp op,
                                          PatternRewriter &rewriter,
                                          Type srcType, Type targetType,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newInits) {
    IRMapping mapper;
    for (const auto &[idx, operand] : llvm::enumerate(op.getInputs()))
      mapper.map(operand, newInputs[idx]);
    for (const auto &[idx, operand] : llvm::enumerate(op.getInits()))
      mapper.map(operand, newInits[idx]);

    Operation *newOp = rewriter.cloneWithoutRegions(*op, mapper);
    for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
      ShapedType shapedType = dyn_cast_or_null<ShapedType>(res.getType());
      bool isSrcType = shapedType && isI8ElemType(shapedType);
      if (!shapedType || !isSrcType)
        continue;
      auto newShapedType = shapedType.clone(targetType);
      newOp->getResult(idx).setType(newShapedType);
    }
    Region &newRegion = newOp->getRegions().front();
    Block *newBlock = rewriter.createBlock(&newRegion);
    rewriter.setInsertionPointToStart(newBlock);

    Block *block = &op.getRegion().front();
    for (BlockArgument bbArg : block->getArguments()) {
      Type argType = bbArg.getType();
      bool isSrcType = argType.isInteger(8);
      Type newArgType = (isSrcType ? targetType : argType);
      mapper.map(bbArg, newBlock->addArgument(newArgType, bbArg.getLoc()));
    }

    Location loc = newRegion.getLoc();
    for (Operation &bodyOp : *block) {
      Operation *newBodyOp =
          mapReduceBodyOpToI16(rewriter, loc, &bodyOp, srcType, mapper);
      if (newBodyOp) {
        mapper.map(bodyOp.getResults(), newBodyOp->getResults());
      } else {
        Operation *cloned = rewriter.clone(bodyOp, mapper);
        mapper.map(bodyOp.getResults(), cloned->getResults());
      }
    }
    rewriter.setInsertionPointAfter(newOp);
    return newOp;
  }
};

template <typename ElemType>
struct HFusionInterleaveNormalizeToTargetTypeTraits
    : public NormalizeTraitsBase {
  static bool shouldNormalize(hfusion::InterleaveOp op) { return true; }

  static SmallVector<Value> getInputs(hfusion::InterleaveOp op) {
    return op.getInput();
  }

  static SmallVector<Value> getOutputs(hfusion::InterleaveOp op) {
    return op.getODSResults(0);
  }

  static Type getTargetType(PatternRewriter &rewriter, hfusion::InterleaveOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(hfusion::InterleaveOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          hfusion::InterleaveOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return rewriter.create<hfusion::InterleaveOp>(loc,
                                                  ValueRange(newOutputs),
                                                  ValueRange(newInputs));
  }

  static void replaceResults(PatternRewriter &rewriter,
                             hfusion::InterleaveOp op,
                             ValueRange newResults, CastSignKind) {
    SmallVector<Value> oldResults(op->getResults().begin(),
                                  op->getResults().end());
    SmallVector<Value> newResultsVec(newResults.begin(), newResults.end());
    if constexpr (std::is_same_v<ElemType, bool>) {
      replaceI1ResultsWithTargetType(oldResults, newResultsVec, rewriter, false);
    }
    if constexpr (std::is_same_v<ElemType, int8_t>) {
      replaceI8ResultsWithTargetType(oldResults, newResultsVec, rewriter, false);
    }
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<bool, hfusion::InterleaveOp>
    : public HFusionInterleaveNormalizeToTargetTypeTraits<bool> {};

template <>
struct HFusionNormalizeToTargetTypeTraits<int8_t, hfusion::InterleaveOp>
    : public HFusionInterleaveNormalizeToTargetTypeTraits<int8_t> {};

template <>
struct HFusionNormalizeToTargetTypeTraits<int8_t, hfusion::DeinterleaveOp>
    : public NormalizeTraitsBase {
  static bool shouldNormalize(hfusion::DeinterleaveOp op) { return true; }

  static SmallVector<Value> getInputs(hfusion::DeinterleaveOp op) {
    return op.getODSOperands(0);
  }

  static SmallVector<Value> getOutputs(hfusion::DeinterleaveOp op) {
    return op.getODSResults(0);
  }

  static Type getTargetType(PatternRewriter &rewriter,
                            hfusion::DeinterleaveOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(hfusion::DeinterleaveOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          hfusion::DeinterleaveOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return rewriter.create<hfusion::DeinterleaveOp>(
        loc, TypeRange(newOutputs), newInputs[0],
        op.getDeInterLeaveChannelIdx());
  }

  static void replaceResults(PatternRewriter &rewriter,
                             hfusion::DeinterleaveOp op,
                             ValueRange newResults, CastSignKind) {
    SmallVector<Value> oldResults(op->getResults().begin(),
                                  op->getResults().end());
    SmallVector<Value> newResultsVec(newResults.begin(), newResults.end());
    replaceI8ResultsWithTargetType(oldResults, newResultsVec, rewriter, false);
  }
};

template <typename ElemType>
struct HFusionReduceWithIndexNormalizeToTargetTypeTraits
    : public NormalizeTraitsBase {
  static bool shouldNormalize(hfusion::ReduceWithIndexOp op) {
    return op.hasPureTensorSemantics();
  }

  static SmallVector<Value> getInputs(hfusion::ReduceWithIndexOp op) {
    return SmallVector<Value>(op.getInputs().begin(), op.getInputs().end());
  }

  static SmallVector<Value> getOutputs(hfusion::ReduceWithIndexOp op) {
    return SmallVector<Value>(op.getInits().begin(), op.getInits().end());
  }

  static Type getTargetType(PatternRewriter &rewriter,
                            hfusion::ReduceWithIndexOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(hfusion::ReduceWithIndexOp op) {
    if constexpr (std::is_same_v<ElemType, int8_t>) {
      return op.getUnsignedSrc() ? CastSignKind::Unsigned
                                 : CastSignKind::Signed;
    }
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          hfusion::ReduceWithIndexOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, TypeRange{newOutputs[0].getType(), newOutputs[1].getType()},
        newInputs, newOutputs, op.getReduceKindAttr(),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());
  }

  static void replaceResults(PatternRewriter &rewriter,
                             hfusion::ReduceWithIndexOp op,
                             ValueRange newResults, CastSignKind intKind) {
    SmallVector<Value> oldResults(op->getResults().begin(),
                                  op->getResults().end());
    SmallVector<Value> newResultsVec(newResults.begin(), newResults.end());
    if constexpr (std::is_same_v<ElemType, bool>) {
      replaceI1ResultsWithTargetType(oldResults, newResultsVec, rewriter);
    }
    if constexpr (std::is_same_v<ElemType, int8_t>) {
      replaceI8ResultsWithTargetType(oldResults, newResultsVec, rewriter);
    }
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<bool, hfusion::ReduceWithIndexOp>
    : public HFusionReduceWithIndexNormalizeToTargetTypeTraits<bool> {};

template <>
struct HFusionNormalizeToTargetTypeTraits<int8_t, hfusion::ReduceWithIndexOp>
    : public HFusionReduceWithIndexNormalizeToTargetTypeTraits<int8_t> {};

struct HFusionNormalizeGatherIndexToI32Traits : public NormalizeTraitsBase {
  static bool shouldNormalize(hfusion::GatherOp) { return true; }

  static Value getIndex(hfusion::GatherOp op) { return op.getIndex(); }

  static Value castIndexToI32(PatternRewriter &rewriter, Location loc,
                              Value index) {
    return hfusion::castTo(rewriter, index, rewriter.getI32Type());
  }

  static void rebuildOp(PatternRewriter &rewriter, hfusion::GatherOp op,
                        Value newIndex) {
    rewriter.replaceOpWithNewOp<hfusion::GatherOp>(
        op, op.getSrc(), newIndex, op.getInit(), op.getAxis());
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<int8_t, hfusion::GatherOp>
    : public NormalizeTraitsBase {
  static bool shouldNormalize(hfusion::GatherOp op) {
    SmallVector<Value> source = op.getODSOperands(0);
    SmallVector<Value> inits = op.getODSOperands(2);
    return hasI8ElemType(source) || hasI8ElemType(inits);
  }

  static SmallVector<Value> getInputs(hfusion::GatherOp op) {
    return {op.getODSOperands(0)[0], op.getODSOperands(1)[0]};
  }

  static SmallVector<Value> getOutputs(hfusion::GatherOp op) {
    return {op.getODSOperands(2)[0]};
  }

  static Type getTargetType(PatternRewriter &rewriter, hfusion::GatherOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(hfusion::GatherOp) {
    return CastSignKind::Signed;
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          hfusion::GatherOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    return rewriter.create<hfusion::GatherOp>(loc, newInputs[0], newInputs[1],
                                              newOutputs[0], op.getAxis());
  }

  static void replaceResults(PatternRewriter &rewriter, hfusion::GatherOp op,
                             ValueRange newResults, CastSignKind) {
    replaceI8ResultsWithTargetType(op->getResults(),
                                   SmallVector<Value>(newResults.begin(),
                                                      newResults.end()),
                                   rewriter,
                                   false);
  }
};

template <typename ElemType>
struct HFusionBroadcastNormalizeToTargetTypeTraits
    : public NormalizeTraitsBase {
  static bool shouldNormalize(linalg::BroadcastOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    if constexpr (std::is_same_v<ElemType, bool>) {
      // @TODO: Need a more general solution for skip normalize in simt scope.
      if (isInSimtScope(op))
        return false;
    }
    Value input = op.getInput();
    Value init = op.getInit();
    if constexpr (std::is_same_v<ElemType, bool>)
      return isI1ElemType(input.getType()) || isI1ElemType(init.getType());
    else
      return isI8ElemType(input.getType()) || isI8ElemType(init.getType());
  }

  static SmallVector<Value> getInputs(linalg::BroadcastOp op) {
    return {op.getInput()};
  }

  static SmallVector<Value> getOutputs(linalg::BroadcastOp op) {
    return {};
  }

  static Type getTargetType(PatternRewriter &rewriter,
                            linalg::BroadcastOp) {
    return rewriter.getF16Type();
  }

  static CastSignKind getCastSignKind(linalg::BroadcastOp) {
    return CastSignKind::Signed;
  }

  static Value createCastOp(PatternRewriter &rewriter, Location loc,
                             Value input, Type targetElemType,
                             CastRoundKind kind, Value output,
                             CastSignKind intKind) {
    return NormalizeTraitsBase::createCastOp(rewriter, loc, input,
                                             targetElemType,
                                             CastRoundKind::Trunc, output,
                                             intKind);
  }

  static Operation *rebuildOpInTargetType(PatternRewriter &rewriter,
                                          Location loc,
                                          linalg::BroadcastOp op,
                                          SmallVector<Value> &newInputs,
                                          SmallVector<Value> &newOutputs) {
    Value newInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, op.getInit(), rewriter.getF16Type());
    return rewriter.create<linalg::BroadcastOp>(loc, newInputs[0],
                                                newInit,
                                                op.getDimensionsAttr());
  }

  static void replaceResults(PatternRewriter &rewriter,
                             linalg::BroadcastOp op,
                             ValueRange newResults, CastSignKind) {
    Value init = op.getInit();
    Type castBackType;
    if constexpr (std::is_same_v<ElemType, bool>)
      castBackType = rewriter.getI1Type();
    else
      castBackType = rewriter.getI8Type();
    Value newResult = hfusion::castTo(rewriter, newResults[0], castBackType,
                                      hfusion::RoundMode::TRUNC, init,
                                      false);
    rewriter.replaceAllUsesWith(op->getResult(0), newResult);
    rewriter.eraseOp(op);
  }

private:
  static bool isInSimtScope(Operation *op) {
    auto scopeOp = op->getParentOfType<scope::ScopeOp>();
    if (!scopeOp)
      return false;
    if (auto vectorType = scopeOp->getAttrOfType<StringAttr>("vector_type"))
      return vectorType.getValue() == "simt";
    return false;
  }
};

template <>
struct HFusionNormalizeToTargetTypeTraits<bool, linalg::BroadcastOp>
    : public HFusionBroadcastNormalizeToTargetTypeTraits<bool> {};

template <>
struct HFusionNormalizeToTargetTypeTraits<int8_t, linalg::BroadcastOp>
    : public HFusionBroadcastNormalizeToTargetTypeTraits<int8_t> {};

template <>
struct HFusionNormalizeToTargetTypeTraits<bool, hfusion::SelectOp>
    : public NormalizeTraitsBase {
  static bool shouldNormalize(hfusion::SelectOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();
    return allI1ElemType(inputs) || hasI1ElemType(outputs);
  }

  static SmallVector<Value> getInputs(hfusion::SelectOp op) {
    return SmallVector<Value>(op.getInputs().begin(), op.getInputs().end());
  }

  static SmallVector<Value> getOutputs(hfusion::SelectOp op) {
    return SmallVector<Value>(op.getOutputs().begin(), op.getOutputs().end());
  }

  static Value createSelectOp(PatternRewriter &rewriter, Location loc,
                              Value cond, Value trueVal, Value falseVal,
                              Value output) {
    auto selectOp = rewriter.create<hfusion::SelectOp>(
        loc, output.getType(), ValueRange{cond, trueVal, falseVal},
        ValueRange{output});
    return selectOp.getResult(0);
  }

  static Value createCmpOp(PatternRewriter &rewriter, Location loc,
                           Value lhs, Value rhs, CompareKind kind) {
#ifndef NDEBUG
    if (kind != CompareKind::NE)
      llvm_unreachable("createCmpOp only supports CompareKind::NE");
#endif
    MLIRContext *ctx = rewriter.getContext();
    auto resultType = RankedTensorType::get(
        dyn_cast<RankedTensorType>(lhs.getType()).getShape(),
        rewriter.getI1Type());
    Value compareOut = rewriter.create<tensor::EmptyOp>(
        loc, dyn_cast<RankedTensorType>(lhs.getType()).getShape(),
        rewriter.getI1Type());
    auto cmpFnAttr = hfusion::CompareFnAttr::get(ctx, hfusion::CompareFn::vne);
    auto compareOp = rewriter.create<hfusion::CompareOp>(
        loc, resultType, ValueRange{lhs, rhs}, ValueRange{compareOut},
        cmpFnAttr);
    return compareOp.getResult(0);
  }
};

void populateNormalizeI1ToTargetPatterns(RewritePatternSet &patterns) {
  if (archisAscend950)
    addTypeConversionPatterns<mlir::ReduceI1AddToSelectMaxCompareTemplate<
        linalg::ReduceOp, HFusionReduceI1AddToSelectMaxCompareTraits>>(patterns);
  addNormalizeToTargetPatterns<bool, hfusion::InterleaveOp,
                               linalg::BroadcastOp>(patterns);
  addTypeConversionPatterns<mlir::NormalizeReduceBoolToLogicalTemplate<
      linalg::ReduceOp, HFusionNormalizeReduceBoolToLogicalTraits>>(patterns);
  if (archIsRegbased) {
    addTypeConversionPatterns<
        mlir::NormalizeToTargetTypeI1SelectTemplate<
            hfusion::SelectOp, HFusionNormalizeToTargetTypeTraits<bool, hfusion::SelectOp>>,
        mlir::ReduceI1AndOrToI16Template<linalg::ReduceOp,
                                          HFusionReduceI1AndOrToI16Traits>>(
        patterns);
  }
  addNormalizeToTargetPatterns<bool, CompareOp, linalg::TransposeOp,
                               tensor::ConcatOp,
                               hfusion::ReduceWithIndexOp>(patterns);
}

void populateNormalizeI8ToTargetPatterns(RewritePatternSet &patterns) {
  if (archIsRegbased)
    addTypeConversionPatterns<mlir::ReduceNormalize910_95Template<
        linalg::ReduceOp, HFusionReduceNormalize910_95Traits>>(patterns);
  if (!archIsRegbased) {
    addNormalizeToTargetPatterns<int8_t, hfusion::ElemwiseBinaryOp,
                                 hfusion::ElemwiseUnaryOp,
                                 linalg::ElemwiseBinaryOp,
                                 linalg::ElemwiseUnaryOp,
                                 hfusion::SelectOp, linalg::ReduceOp,
                                 hfusion::InterleaveOp,
                                 hfusion::DeinterleaveOp, hfusion::GatherOp,
                                 linalg::BroadcastOp>(patterns);
    addTypeConversionPatterns<mlir::NormalizeCumOpI8ToTargetType<
        hfusion::CumsumOp, HFusionNormalizeCumOpI8ToTargetTraits<hfusion::CumsumOp>>,
        mlir::NormalizeCumOpI8ToTargetType<
            hfusion::CumprodOp, HFusionNormalizeCumOpI8ToTargetTraits<hfusion::CumprodOp>>>(patterns);
  }
  if (archisAscend950) {
    addNormalizeToTargetPatterns<int8_t, linalg::ElemwiseBinaryOp>(patterns);
  }
  // TODO: support regbase i8 template function implementation
  addNormalizeToTargetPatterns<int8_t, hfusion::ReduceWithIndexOp>(patterns);
}

void populateNormalizeGatherIndexPatterns(RewritePatternSet &patterns) {
  addTypeConversionPatterns<mlir::NormalizeGatherIndexToI32Template<
      hfusion::GatherOp, HFusionNormalizeGatherIndexToI32Traits>>(patterns);
}

void populateNormalizeF16ToF32Patterns(RewritePatternSet &patterns) {
  addTypeConversionPatterns<
      mlir::NormalizeF16ToF32Type<linalg::ElemwiseUnaryOp, HFusionNormalizeF16ToF32Traits<linalg::ElemwiseUnaryOp>>,
      mlir::NormalizeF16ToF32Type<hfusion::ElemwiseBinaryOp, HFusionNormalizeF16ToF32Traits<hfusion::ElemwiseBinaryOp>>,
      mlir::NormalizeF16ToF32Type<hfusion::ElemwiseUnaryOp, HFusionNormalizeF16ToF32Traits<hfusion::ElemwiseUnaryOp>>>(patterns);
  addTypeConversionPatterns<mlir::NormalizeCumOpF16ToF32Type<
      hfusion::CumsumOp, HFusionNormalizeCumOpF16ToF32Traits<hfusion::CumsumOp>>,
      mlir::NormalizeCumOpF16ToF32Type<
          hfusion::CumprodOp, HFusionNormalizeCumOpF16ToF32Traits<hfusion::CumprodOp>>>(patterns);
}

} // namespace mlir::hfusion