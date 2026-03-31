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

namespace mlir::hfusion {

template <typename srcType>
static bool hasElemType(const SmallVector<Value> &values) {
  if constexpr (std::is_same_v<bool, srcType>) {
    return hasI1ElemType(values);
  }
  if constexpr (std::is_same_v<int8_t, srcType>) {
    return hasI8ElemType(values);
  }
  if constexpr (std::is_same_v<int16_t, srcType>) {
    return hasI16ElemType(values);
  }
  if constexpr (std::is_same_v<float, srcType>) {
    return hasF16ElemType(values);
  }
  return false;
}

template <typename FuncType, typename FuncAttrType, typename OpType>
static NamedAttribute getOpFunAttr(OpType op, PatternRewriter &rewriter) {
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
              std::is_same_v<OpType, hfusion::SelectOp>>>
static SmallVector<NamedAttribute> getOpAttr(OpType op,
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

template <typename targetType,
          typename = std::enable_if<(std::is_same_v<bool, targetType> ||
                                     std::is_same_v<int8_t, targetType>)>>
static void replaceResultsWithTargetType(const SmallVector<Value> &oldResults,
                                         const SmallVector<Value> &newResults,
                                         PatternRewriter &rewriter,
                                         bool isUnsigned) {
  if constexpr (std::is_same_v<bool, targetType>) {
    replaceI1ResultsWithTargetType(oldResults, newResults, rewriter);
  }
  if constexpr (std::is_same_v<int8_t, targetType>) {
    replaceI8ResultsWithTargetType(oldResults, newResults, rewriter, true,
                                   isUnsigned);
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

bool analyzeUnsignedNeeds(Value value) {
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

template <typename ElemType>
SmallVector<Value> normalizeToTargetType(PatternRewriter &rewriter,
                                         const SmallVector<Value> &values,
                                         Type targetType,
                                         bool isInputUnsigned) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isElemType<ElemType>(v.getType())) {
      result.push_back(v);
      continue;
    }
    Value castResult =
        isInputUnsigned ? castTo(rewriter, v, targetType, TypeFn::cast_unsigned)
                        : castTo(rewriter, v, targetType);
    result.push_back(castResult);
  }
  return result;
}

template <typename ElemType, typename OpType>
struct NormalizeToTargetType : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (!hasElemType<ElemType>(op.getInputs()) &&
        !hasElemType<ElemType>(op.getOutputs())) {
      return failure();
    }

    if (isSupportOperand<ElemType>(op)) {
      return failure();
    }

    bool computeByF16 = shoudComputeByF16(op);
    bool computeByF32 = shoudComputeByF32(op);
    if (!computeByF16 && !computeByF32) {
      return failure();
    }

    Type targetType;
    if (computeByF16) {
      targetType = rewriter.getF16Type();
    } else if (computeByF32) {
      targetType = rewriter.getF32Type();
    } else {
      llvm_unreachable("Unsupported Op.");
    }

    bool isInputUnsigned = false;
    bool UseChainUnsigned = false;
    if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn linalgFn = binOp.getFunAttr().getValue();
      if (op->getNumResults() > 0) {
        UseChainUnsigned = analyzeUnsignedNeeds(op->getResult(0));
      }
      isInputUnsigned = linalgFn == linalg::BinaryFn::max_unsigned ||
                        linalgFn == linalg::BinaryFn::min_unsigned ||
                        UseChainUnsigned;
    }

    SmallVector<Value> newInputs = normalizeToTargetType<ElemType>(
        rewriter, op.getInputs(), targetType, isInputUnsigned);
    SmallVector<Value> newOutputs = normalizeToTargetType<ElemType>(
        rewriter, op.getOutputs(), targetType, isInputUnsigned);
    Operation *newOp = createBodyOp(op, newInputs, newOutputs, rewriter);
    if (std::is_same_v<OpType, hfusion::SelectOp>) {
      replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                     rewriter, false);
    } else {
      // TODO: set argument enableOverflow = false inside for all non-arithmatic
      // op type
      replaceResultsWithTargetType<ElemType>(
          op->getResults(), newOp->getResults(), rewriter, isInputUnsigned);
    }
    return success();
  }

private:
  template <typename OpElemType>
  bool isSupportOperand(OpType op) const = delete;

  template <>
  bool isSupportOperand<bool>(OpType op) const {
    return false;
  }

  template <>
  bool isSupportOperand<int8_t>(OpType op) const {
    if constexpr (std::is_same_v<OpType, linalg::FillOp> ||
                  std::is_same_v<OpType, linalg::BroadcastOp> ||
                  std::is_same_v<OpType, linalg::CopyOp> ||
                  std::is_same_v<OpType, hfusion::CastOp>) {
      return true;
    }

    if constexpr (std::is_same_v<OpType, hfusion::SelectOp>) {
      return false;
    }

    if constexpr (std::is_same_v<OpType, linalg::ElemwiseUnaryOp> ||
                  std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      // The following ops support i8 type in c310
      // Could compute on i8 directly and no need cast to f16 nor f32
      if (archisAscend950) {
        if constexpr (std::is_same_v<OpType, linalg::AddOp> ||
                      std::is_same_v<OpType, linalg::SubOp> ||
                      std::is_same_v<OpType, linalg::MaxOp> ||
                      std::is_same_v<OpType, linalg::MinOp> ||
                      std::is_same_v<OpType, linalg::SelectOp>) {
          return true;
        }
        if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>)
          if (
              op.getFun() == linalg::BinaryFn::add ||
              op.getFun() == linalg::BinaryFn::sub ||
              op.getFun() == linalg::BinaryFn::max_signed ||
              op.getFun() == linalg::BinaryFn::min_signed ||
              op.getFun() == linalg::BinaryFn::max_unsigned ||
              op.getFun() == linalg::BinaryFn::min_unsigned)
            return true;
      }
      // no linalg elemwise unary/binary op support i8
      return false;
    }

    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
      // only part of hfusion elemwise unary op support i8
      auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op);
      hfusion::UnaryFn func = unaryOp.getFun();
      static DenseSet<hfusion::UnaryFn> unarySet = {hfusion::UnaryFn::vnot};
      return unarySet.contains(func);
    }

    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      // only part of hfusion elemwise binary op support both i8
      auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
      hfusion::BinaryFn func = binOp.getFun();
      // bit operation can support b8 operand
      static DenseSet<hfusion::BinaryFn> binarySet = {hfusion::BinaryFn::vor,
                                                      hfusion::BinaryFn::vand,
                                                      hfusion::BinaryFn::vxor};
      return binarySet.contains(func);
    }
    return false;
  }

  bool shoudComputeByF16(OpType op) const {
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
          linalg::BinaryFn::mul,
          linalg::BinaryFn::div_unsigned,
          linalg::BinaryFn::div,
          linalg::BinaryFn::add,
          linalg::BinaryFn::sub,
      };
      return !binarySet.contains(func);
    }
    return true;
  }

  bool shoudComputeByF32(OpType op) const {
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

  Operation *createBodyOp(OpType op, SmallVector<Value> &newInputs,
                          SmallVector<Value> &newOutputs,
                          PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    SmallVector<NamedAttribute> attrs = getOpAttr(op, rewriter);
    if constexpr (std::is_same_v<OpType, hfusion::SelectOp> ||
                  std::is_same_v<OpType, linalg::ElemwiseUnaryOp> ||
                  std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      // no attr needs to be changed
      return rewriter.create<OpType>(loc, ValueRange{newInputs},
                                     ValueRange{newOutputs}, attrs);
    }

    if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      static DenseMap<linalg::BinaryFn, hfusion::BinaryFn> binAttrMap = {
          {linalg::BinaryFn::max_unsigned, hfusion::BinaryFn::maxf},
          {linalg::BinaryFn::max_signed, hfusion::BinaryFn::maxf},
          {linalg::BinaryFn::min_unsigned, hfusion::BinaryFn::minf},
          {linalg::BinaryFn::min_signed, hfusion::BinaryFn::minf},
      };
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn linalgFn = binOp.getFunAttr().getValue();
      if (binAttrMap.contains(linalgFn)) {
        // convert linalg binary op to hfusion
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
      // other hfusion elemwise binary op can be created using origin attr
      return rewriter.create<hfusion::ElemwiseUnaryOp>(
          loc, ValueRange{newInputs}, ValueRange{newOutputs}, attrs);
    }
    llvm_unreachable("Unsupport OpType to create with F16 operand.");
  }
};

template <typename OpType>
struct NormalizeF16ToF32Type : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    if (!hasF16ElemType(inputs) || !shouldComputeByF32(op)) {
      return failure();
    }

    normalizeOpF16ToF32(rewriter, op);
    return success();
  }

private:
  void normalizeOpF16ToF32(PatternRewriter &rewriter, OpType op) const {
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();

    SmallVector<Value> newInputs = normalizeF16ToF32(rewriter, inputs);
    SmallVector<Value> newOutputs = normalizeF16ToF32(rewriter, outputs);

    SmallVector<NamedAttribute> attrs = getOpAttr(op, rewriter);
    Operation *newOp = rewriter.create<OpType>(
        op.getLoc(), ValueRange{newInputs}, ValueRange{newOutputs}, attrs);
    Value castResult =
        castTo(rewriter, newOp->getResults()[0], rewriter.getF16Type());
    rewriter.replaceAllUsesWith(op->getResults()[0], castResult);
  }

  bool shouldComputeByF32(OpType op) const {
    // cast f32 to compute for high precision
    // linalg unaryFn op set
    if (std::is_same_v<OpType, linalg::ElemwiseUnaryOp>) {
      static DenseSet<linalg::UnaryFn> linalgUnarySet = {linalg::UnaryFn::log};
      if (auto unaryOp = cast<linalg::ElemwiseUnaryOp>(op)) {
        linalg::UnaryFn unaryFn = unaryOp.getFun();
        if (linalgUnarySet.contains(unaryFn)) {
          return true;
        }
      }
    }

    // hfusion binaryFn op set
    if (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      static DenseSet<hfusion::BinaryFn> hfusionBinarySet = {
          hfusion::BinaryFn::powf};
      if (auto binaryOp = cast<hfusion::ElemwiseBinaryOp>(op)) {
        hfusion::BinaryFn binaryFn = binaryOp.getFun();
        if (hfusionBinarySet.contains(binaryFn)) {
          return true;
        }
      }
    }

    // hfusion unaryFn op set
    if (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
      static DenseSet<hfusion::UnaryFn> hfusionUnarySet = {
          hfusion::UnaryFn::rsqrt};
      if (auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op)) {
        hfusion::UnaryFn unaryFn = unaryOp.getFun();
        if (hfusionUnarySet.contains(unaryFn)) {
          return true;
        }
      }
    }
    return false;
  }
};

template <typename CumOpType>
struct NormalizeCumOpF16ToF32Type : public OpRewritePattern<CumOpType> {
public:
  using OpRewritePattern<CumOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = {op.getInput()};
    SmallVector<Value> outputs = {op.getOutput()};
    if ((!hasF16ElemType(inputs) && !hasF16ElemType(outputs)) ||
        !(std::is_same_v<CumOpType, hfusion::CumsumOp> ||
          std::is_same_v<CumOpType, hfusion::CumprodOp>)) {
      return failure();
    }
    auto newInputs = normalizeF16ToF32(rewriter, inputs);
    auto newOutputs = normalizeF16ToF32(rewriter, outputs);
    Operation *newOp = rewriter.create<CumOpType>(
        op.getLoc(), TypeRange{newOutputs}, newInputs[0], op.getCumDims(),
        op.getReverse());
    Value castResult =
        castTo(rewriter, newOp->getResults()[0], rewriter.getF16Type());
    rewriter.replaceAllUsesWith(op->getResults()[0], castResult);
    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, linalg::ReduceOp>
    : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (isa<arith::AddIOp, arith::MaxUIOp, arith::MaxSIOp>(bodyOp)) {
      // As it is a bool, `add` and `max` can be converted into `or`.
      replaceBinary<arith::OrIOp>(bodyOp, rewriter);
      return success();
    }
    if (isa<arith::MulIOp, arith::MinUIOp, arith::MinSIOp>(bodyOp)) {
      // As it is a bool, `mul` and `min` can be converted into `and`.
      replaceBinary<arith::AndIOp>(bodyOp, rewriter);
      return success();
    }
    return failure();
  }

private:
  template <typename targetType>
  void replaceBinary(Operation *op, PatternRewriter &rewriter) const {
    if (op == nullptr) {
      return;
    }
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(op->getBlock());
    auto targetOp = rewriter.create<targetType>(op->getLoc(), op->getOperand(0),
                                                op->getOperand(1));
    rewriter.modifyOpInPlace(op, [&]() { op->replaceAllUsesWith(targetOp); });
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
struct ReduceI1AddToSelectMaxCompare
    : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (!reduceOp.hasPureTensorSemantics())
      return failure();

    SmallVector<Value> inputs = reduceOp.getInputs();
    SmallVector<Value> inits = reduceOp.getInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();
    Block &body = reduceOp.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (!isa<arith::AddIOp>(bodyOp))
      return failure();
    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1 || dimensions[0] != 0)
      return failure();
    Value input = reduceOp.getInputs()[0];
    Value init = reduceOp.getInits()[0];

    auto inputType = input.getType().mlir::dyn_cast<RankedTensorType>();
    auto initType = init.getType().mlir::dyn_cast<RankedTensorType>();
    if (!inputType || !initType)
      return failure();
    if (!inputType.getElementType().isInteger(1) ||
        !initType.getElementType().isInteger(1))
      return failure();

    if (initType.getRank() != 0)
      return failure();
    Location loc = reduceOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto shape = inputType.getShape();
    Type i16 = rewriter.getI16Type();
    auto tensorI16Type = RankedTensorType::get(shape, i16);
    auto oneAttr =
        DenseElementsAttr::get(tensorI16Type, rewriter.getI16IntegerAttr(1));
    auto zeroAttr =
        DenseElementsAttr::get(tensorI16Type, rewriter.getI16IntegerAttr(0));
    Value cstOneTensor =
        rewriter.create<arith::ConstantOp>(loc, tensorI16Type, oneAttr);
    Value cstZeroTensor =
        rewriter.create<arith::ConstantOp>(loc, tensorI16Type, zeroAttr);
    Value selectOut = rewriter.create<tensor::EmptyOp>(loc, shape, i16);
    auto selectResult = rewriter.create<hfusion::SelectOp>(
        loc, tensorI16Type, ValueRange{input, cstOneTensor, cstZeroTensor},
        ValueRange{selectOut});
    auto scalarI16Tensor = RankedTensorType::get({}, i16);
    auto zeroScalarAttr =
        DenseElementsAttr::get(scalarI16Tensor, rewriter.getI16IntegerAttr(0));
    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, scalarI16Tensor,
                                                          zeroScalarAttr);

    Value cst0 = rewriter.create<arith::ConstantOp>(
        loc, i16, rewriter.getI16IntegerAttr(0));
    auto allocOp = rewriter.create<bufferization::AllocTensorOp>(
        loc, scalarI16Tensor, ValueRange{});
    Value allocTensor = allocOp.getResult();
    auto fillOp = rewriter.create<linalg::FillOp>(loc, cst0, allocTensor);
    Value fillResult = fillOp.getResult(0);

    auto newReduce = rewriter.create<linalg::ReduceOp>(
        loc, ValueRange{selectResult.getResult(0)}, ValueRange{fillResult},
        dimensions, [&](OpBuilder &builder, Location loc, ValueRange operands) {
          Value max = {
              builder.create<arith::MaxSIOp>(loc, operands[0], operands[1])};
          builder.create<linalg::YieldOp>(loc, ValueRange{max});
        });
    Value reducedTensor = newReduce.getResult(0);
    Type scalarI1Tensor = RankedTensorType::get({1}, rewriter.getI1Type());
    Value compareOut = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{1}, rewriter.getI1Type());
    auto cmpFnAttr = hfusion::CompareFnAttr::get(ctx, hfusion::CompareFn::vne);
    auto compareResult = rewriter.create<hfusion::CompareOp>(
        loc, scalarI1Tensor, ValueRange{reducedTensor, zeroScalar},
        ValueRange{compareOut}, cmpFnAttr);

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value extracted = rewriter.create<tensor::ExtractOp>(
        loc, compareResult.getResult(0), zeroIdx);
    Type extractedScalarI1Tensor =
        RankedTensorType::get({}, rewriter.getI1Type());
    Value scalarResult = rewriter.create<tensor::FromElementsOp>(
        loc, extractedScalarI1Tensor, extracted);
    rewriter.replaceOp(reduceOp, scalarResult);
    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, tensor::ConcatOp>
    : public OpRewritePattern<tensor::ConcatOp> {
public:
  using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op->getResults();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newOp = rewriter.create<tensor::ConcatOp>(op.getLoc(), op.getDim(),
                                                   ValueRange(newInputs));
    replaceI1ResultsWithTargetType({op.getResult()}, {newOp.getResult()},
                                   rewriter,
                                   /*enableOverflow*/ false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, CompareOp>
    : public OpRewritePattern<CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    if (!hasI1ElemType(inputs))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    Value newLhs = newInputs[0];
    Value newRhs = newInputs[1];
    auto *newOp =
        createCmpOp(rewriter, op->getLoc(), newLhs, newRhs, op.getCompareFn());
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, linalg::TransposeOp>
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = {op.getInput()};
    SmallVector<Value> inits = op.getDpsInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inits);
    auto newOp = rewriter.create<linalg::TransposeOp>(
        op.getLoc(), newInputs.front(), newInits.front(), op.getPermutation());
    replaceI1ResultsWithTargetType(op.getResult(), newOp->getResults(),
                                   rewriter,
                                   /*enableOverflow*/ false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, linalg::ReduceOp>
    : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (!shouldComputeByFloat(op)) {
      return failure();
    }
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    FloatType targetType = nullptr;
    SmallVector<Value> newInputs;
    SmallVector<Value> newInits;
    if (shoudComputeI8ByF32(op)) {
      targetType = rewriter.getF32Type();
      newInputs =
          normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inputs);
      newInits = normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inits);
    } else {
      targetType = rewriter.getF16Type();
      newInputs =
          normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inputs);
      newInits = normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    }

    Operation *newOp = createNewReduceOp(op, rewriter, rewriter.getI8Type(),
                                         targetType, newInputs, newInits);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);
    return success();
  }

private:
  bool shouldComputeByFloat(linalg::ReduceOp reduceOp) const {
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

  bool shoudComputeI8ByF32(linalg::ReduceOp op) const {
    Block *block = &op.getRegion().front();
    for (Operation &bodyOp : *block) {
      if (dyn_cast_or_null<arith::AddIOp>(bodyOp)) {
        return true;
      }
    }
    return false;
  }
};

struct ReduceNormalize910_95 : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    bool archIs950 = hacc::utils::isAscend950(moduleOp);
    if (!archIs950) {
      return failure();
    }
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    // do not cast xor example
    auto &region = op.getRegion();
    if (llvm::any_of(region.getOps(),
                     [](Operation &op) { return isa<arith::XOrIOp>(&op); })) {
      return failure();
    }
    // do not cast reduce_with_index
    if (op->hasAttr("reduce_mode")) {
      return failure();
    }
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }
    MLIRContext *ctx = rewriter.getContext();
    Type i16Type = IntegerType::get(ctx, 16);

    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();

    bool isInputUnsigned = false;
    if (isa<arith::MaxUIOp, arith::MinUIOp>(bodyOp)) {
      isInputUnsigned = true;
    }

    SmallVector<Value> newInputs =
        normalizeSrcToI16Type<int8_t>(rewriter, inputs, isInputUnsigned);
    SmallVector<Value> newInits =
        normalizeSrcToI16Type<int8_t>(rewriter, inits, isInputUnsigned);
    Operation *newOp = createNewReduceI16Op(op, rewriter, rewriter.getI8Type(),
                                            i16Type, newInputs, newInits);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);

    return success();
  }

private:
  template <typename srcType>
  SmallVector<Value> normalizeSrcToI16Type(PatternRewriter &rewriter,
                                           const SmallVector<Value> &values,
                                           bool isInputUnsigned) const {
    SmallVector<Value> result;
    result.reserve(values.size());
    for (Value v : values) {
      if (!isElemType<srcType>(v.getType())) {
        result.push_back(v);
        continue;
      }
      Type dstType = rewriter.getI16Type();
      Value castResult =
          isInputUnsigned ? castTo(rewriter, v, dstType, TypeFn::cast_unsigned)
                          : castTo(rewriter, v, dstType);
      result.push_back(castResult);
    }
    return result;
  }

  static Operation *mapReduceBodyOpToI16(PatternRewriter &rewriter,
                                         Location loc, Operation *bodyOp,
                                         Type srcType, IRMapping &mapper) {
    if (isa<linalg::YieldOp>(bodyOp)) {
      return rewriter.clone(*bodyOp, mapper);
    }
    // only support binary arith ops here
    assert(bodyOp->getNumOperands() == 2 && "only support binary arith op");
    Value oldLhs = bodyOp->getOperand(0);
    Value oldRhs = bodyOp->getOperand(1);
    Value newLhs = mapper.lookupOrNull(oldLhs);
    Value newRhs = mapper.lookupOrNull(oldRhs);

    if (!newLhs || !newRhs) {
      return nullptr;
    }

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
    for (const auto &[idx, operand] : llvm::enumerate(op.getInputs())) {
      mapper.map(operand, newInputs[idx]);
    }
    for (const auto &[idx, operand] : llvm::enumerate(op.getInits())) {
      mapper.map(operand, newInits[idx]);
    }

    Operation *newOp = rewriter.cloneWithoutRegions(*op, mapper);
    for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
      ShapedType shapedType = dyn_cast_or_null<ShapedType>(res.getType());
      bool isSrcType = shapedType && isI8ElemType(shapedType);
      if (!shapedType || !isSrcType) {
        continue;
      }
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

template <typename OpType>
Operation *createInterleaveLikeOp(OpType op, SmallVector<Value> &newInputs,
                                  SmallVector<Value> &newOutputs,
                                  PatternRewriter &rewriter) {
  Location loc = op.getLoc();

  if constexpr (std::is_same_v<OpType, hfusion::InterleaveOp>) {
    return rewriter.create<hfusion::InterleaveOp>(loc, ValueRange(newOutputs),
                                                  ValueRange(newInputs));
  }
  if constexpr (std::is_same_v<OpType, hfusion::DeinterleaveOp>) {
    return rewriter.create<hfusion::DeinterleaveOp>(
        loc, TypeRange(newOutputs), newInputs[0],
        op.getDeInterLeaveChannelIdx());
  }
  llvm_unreachable(
      "Unsupport interleaveLike OpType to create with F16 Operand.");
}

template <>
struct NormalizeToTargetType<int8_t, hfusion::InterleaveOp>
    : public OpRewritePattern<hfusion::InterleaveOp> {
public:
  using OpRewritePattern<hfusion::InterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInput();
    SmallVector<Value> inits = op.getODSResults(0);
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    Operation *newOp =
        createInterleaveLikeOp(op, newInputs, newInits, rewriter);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, hfusion::DeinterleaveOp>
    : public OpRewritePattern<hfusion::DeinterleaveOp> {
public:
  using OpRewritePattern<hfusion::DeinterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::DeinterleaveOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getODSOperands(0);
    SmallVector<Value> inits = op.getODSResults(0);
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    Operation *newOp =
        createInterleaveLikeOp(op, newInputs, newInits, rewriter);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, false);
    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, hfusion::ReduceWithIndexOp>
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
public:
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    assert(inputs.size() == 2);
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inits);
    Operation *newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        op.getLoc(), TypeRange{newInits[0].getType(), newInits[1].getType()},
        newInputs, newInits, op.getReduceKindAttr(), op.getUnsignedSrcAttr(),
        op.getTieBreakLeftAttr(), op.getDimensionsAttr());
    replaceI1ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, hfusion::ReduceWithIndexOp>
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
public:
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    bool unsignedSrc = op.getUnsignedSrc();
    auto newInputs = normalizeSrcToTargetType<int8_t, Float16Type>(
        rewriter, inputs, unsignedSrc);
    auto newInits = normalizeSrcToTargetType<int8_t, Float16Type>(
        rewriter, inits, unsignedSrc);
    Operation *newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        op.getLoc(), TypeRange{newInits[0].getType(), newInits[1].getType()},
        newInputs, newInits, op.getReduceKindAttr(), op.getUnsignedSrcAttr(),
        op.getTieBreakLeftAttr(), op.getDimensionsAttr());
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);
    return success();
  }
};

template <typename CumOpType>
struct NormalizeCumOpI8ToTargetType : public OpRewritePattern<CumOpType> {
public:
  using OpRewritePattern<CumOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getODSOperands(0);
    SmallVector<Value> outputs = op.getODSResults(0);
    if (!hasI8ElemType(inputs) && !hasI8ElemType(outputs)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inputs);
    auto newOutputs =
        normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, outputs);
    Operation *newOp = rewriter.create<CumOpType>(
        op.getLoc(), TypeRange{newOutputs}, newInputs[0], op.getCumDims(),
        op.getReverse());
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);
    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, hfusion::GatherOp>
    : public OpRewritePattern<hfusion::GatherOp> {
public:
  using OpRewritePattern<hfusion::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::GatherOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> source = op.getODSOperands(0);
    SmallVector<Value> indices = op.getODSOperands(1);
    SmallVector<Value> inits = op.getODSOperands(2);
    if (!hasI8ElemType(source) && !hasI8ElemType(inits)) {
      return failure();
    }

    auto newSource =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, source);
    auto newInits =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    Operation *newOp = rewriter.create<hfusion::GatherOp>(
        op.getLoc(), newSource[0], indices[0], newInits[0], op.getAxis());
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, /*enableOverflow*/ false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, linalg::BroadcastOp>
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value input = op.getInput();
    Value init = op.getInit();
    Location loc = op.getLoc();

    if (!isI8ElemType(input.getType()) && !isI8ElemType(init.getType())) {
      return failure();
    }

    Value newInput = hfusion::castTo(rewriter, input, rewriter.getF16Type(),
                                     hfusion::RoundMode::TRUNC);
    Value newInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, init, rewriter.getF16Type());
    Value newBrcOp = rewriter
                         .create<linalg::BroadcastOp>(loc, newInput, newInit,
                                                      op.getDimensionsAttr())
                         ->getResult(0);
    Value newResult = hfusion::castTo(rewriter, newBrcOp, rewriter.getI8Type(),
                                      hfusion::RoundMode::TRUNC, init,
                                      /* enableOverflow = */ false);

    rewriter.replaceAllUsesWith(op->getResult(0), newResult);
    rewriter.eraseOp(op);

    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, hfusion::InterleaveOp>
    : public OpRewritePattern<hfusion::InterleaveOp> {
public:
  using OpRewritePattern<hfusion::InterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInput();
    SmallVector<Value> inits = op.getODSResults(0);
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inits);
    Operation *newOp =
        createInterleaveLikeOp(op, newInputs, newInits, rewriter);
    replaceI1ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, linalg::BroadcastOp>
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value input = op.getInput();
    Value init = op.getInit();
    Location loc = op.getLoc();

    if (!isI1ElemType(input.getType()) && !isI1ElemType(init.getType())) {
      return failure();
    }

    Value newInput = hfusion::castTo(rewriter, input, rewriter.getF16Type(),
                                     hfusion::RoundMode::TRUNC);
    Value newInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, init, rewriter.getF16Type());
    Value newBrcOp = rewriter
                         .create<linalg::BroadcastOp>(loc, newInput, newInit,
                                                      op.getDimensionsAttr())
                         ->getResult(0);
    Value newResult = hfusion::castTo(rewriter, newBrcOp, rewriter.getI1Type(),
                                      hfusion::RoundMode::TRUNC, init,
                                      /* enableOverflow = */ false);

    rewriter.replaceAllUsesWith(op->getResult(0), newResult);
    rewriter.eraseOp(op);
    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, hfusion::SelectOp>
    : public OpRewritePattern<hfusion::SelectOp> {
public:
  using OpRewritePattern<hfusion::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::SelectOp op,
                                PatternRewriter &rewriter) const override {

    if (!op.hasPureTensorSemantics())
      return failure();

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();
    assert(inputs.size() == 3);

    if (!allI1ElemType(inputs) && !hasI1ElemType(outputs))
      return failure();

    Location loc = op.getLoc();

    Value cond = inputs[0];
    Value trueVal = inputs[1];
    Value falseVal = inputs[2];

    auto trueType = trueVal.getType().mlir::dyn_cast<RankedTensorType>();
    auto shape = trueType.getShape();
    Type i16 = rewriter.getI16Type();
    auto newType = RankedTensorType::get(shape, i16);

    auto oneAttr =
        DenseElementsAttr::get(newType, rewriter.getI16IntegerAttr(1));
    auto zeroAttr =
        DenseElementsAttr::get(newType, rewriter.getI16IntegerAttr(0));

    Value oneTensor = rewriter.create<arith::ConstantOp>(loc, newType, oneAttr);
    Value zeroTensor =
        rewriter.create<arith::ConstantOp>(loc, newType, zeroAttr);

    Value empty = rewriter.create<tensor::EmptyOp>(loc, shape, i16);

    auto convertI1TensorToI16 = [&trueType, &newType, &oneTensor, &zeroTensor,
                                 &empty, &loc, &rewriter](Value v) -> Value {
      if (!trueType || !trueType.getElementType().isInteger(1))
        return v;

      auto select = rewriter.create<hfusion::SelectOp>(
          loc, newType, ValueRange{v, oneTensor, zeroTensor},
          ValueRange{empty});

      return select.getResult(0);
    };

    Value newTrue = convertI1TensorToI16(trueVal);
    Value newFalse = convertI1TensorToI16(falseVal);
    Operation *newOp = rewriter.create<hfusion::SelectOp>(
        loc, empty.getType(), ValueRange{cond, newTrue, newFalse},
        ValueRange{empty});

    Value selectResult = newOp->getResult(0);

    auto i1Type = RankedTensorType::get(shape, rewriter.getI1Type());
    Value compareOut =
        rewriter.create<tensor::EmptyOp>(loc, shape, rewriter.getI1Type());

    auto cmpFnAttr = hfusion::CompareFnAttr::get(rewriter.getContext(),
                                                 hfusion::CompareFn::vne);

    auto compareOp = rewriter.create<hfusion::CompareOp>(
        loc, i1Type, ValueRange{selectResult, zeroTensor},
        ValueRange{compareOut}, cmpFnAttr);

    rewriter.replaceOp(op, compareOp.getResult(0));

    return success();
  }
};

void populateNormalizeI1ToTargetPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (archisAscend950)
    patterns.add<ReduceI1AddToSelectMaxCompare>(ctx);
  if (archIsRegbased){
    patterns.add<NormalizeToTargetType<bool, hfusion::SelectOp>>(ctx);
  }
  patterns.add<NormalizeToTargetType<bool, hfusion::InterleaveOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, linalg::BroadcastOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, linalg::ReduceOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, CompareOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, linalg::TransposeOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, tensor::ConcatOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, hfusion::ReduceWithIndexOp>>(ctx);
}

void populateNormalizeI8ToTargetPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (archIsRegbased)
    patterns.add<ReduceNormalize910_95>(ctx);
  if (!archIsRegbased) {
    patterns.add<NormalizeToTargetType<int8_t, hfusion::ElemwiseBinaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::ElemwiseUnaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseBinaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseUnaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::SelectOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::ReduceOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::InterleaveOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::DeinterleaveOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::GatherOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::BroadcastOp>>(ctx);
    patterns.add<NormalizeCumOpI8ToTargetType<hfusion::CumsumOp>>(ctx);
    patterns.add<NormalizeCumOpI8ToTargetType<hfusion::CumprodOp>>(ctx);
  }
  if (archisAscend950) {
    patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseBinaryOp>>(ctx);
  }
  // TODO: support regbase i8 template function implementation
  patterns.add<NormalizeToTargetType<int8_t, hfusion::ReduceWithIndexOp>>(ctx);
}

void populateNormalizeF16ToF32Patterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeF16ToF32Type<linalg::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeF16ToF32Type<hfusion::ElemwiseBinaryOp>>(ctx);
  patterns.add<NormalizeF16ToF32Type<hfusion::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeCumOpF16ToF32Type<hfusion::CumsumOp>>(ctx);
  patterns.add<NormalizeCumOpF16ToF32Type<hfusion::CumprodOp>>(ctx);
}

} // namespace mlir::hfusion