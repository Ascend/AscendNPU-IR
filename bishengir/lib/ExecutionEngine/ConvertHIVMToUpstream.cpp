//===- LowerHIVMToLLVM.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements converting HIVM operations to upstream dialect's
// equivalent
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/ExecutionEngine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ScopedPrinter.h"
#include <type_traits>

#define DEBUG_TYPE "execution-engine-convert-hivm-to-upstream"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_EXECUTIONENGINEHIVMTOUPSTREAMCONVERSION
#include "bishengir/ExecutionEngine/Passes.h.inc"
} // namespace mlir

namespace {

using namespace mlir;
using ShapedValue = TypedValue<ShapedType>;

template <typename T>
static TypedAttr getConstantTypedAttr(Type type, T &&value) {
  return TypeSwitch<Type, TypedAttr>(type)
      .Case<IntegerType, IndexType>(
          [&](Type type) { return IntegerAttr::get(type, value); })
      .Case([&](FloatType type) { return FloatAttr::get(type, value); })
      .Default([](Type type) {
        llvm::report_fatal_error(StringRef("Unsupported constant type: ") +
                                 llvm::to_string(type));
        return nullptr;
      });
}

static ShapedValue
reallocShapedValue(PatternRewriter &rewriter, ShapedValue value,
                   const Location loc,
                   llvm::function_ref<OpFoldResult(int64_t)> dimGetter,
                   Type newElementType = nullptr) {
  auto dimMaker = [value, dimGetter,
                   &rewriter](llvm::function_ref<Value(int64_t)> defaultMaker) {
    const auto type = value.getType();
    SmallVector<OpFoldResult> newDims;
    for (auto dimIdx : llvm::seq(type.getRank())) {
      auto dim = dimGetter(dimIdx);
      if (!dim) {
        if (type.isDynamicDim(dimIdx))
          newDims.push_back(defaultMaker(dimIdx));
        else
          newDims.push_back(rewriter.getIndexAttr(type.getDimSize(dimIdx)));
        continue;
      }
      if (dim.is<Attribute>() &&
          cast<IntegerAttr>(dim.get<Attribute>()).getValue().getZExtValue() ==
              0)
        continue;
      newDims.push_back(dim);
    }
    return newDims;
  };

  if (newElementType == nullptr)
    newElementType = getElementTypeOrSelf(value.getType());

  return TypeSwitch<ShapedType, ShapedValue>(value.getType())
      .Case([&](RankedTensorType type) {
        auto dims = dimMaker([&](int64_t dimIdx) -> Value {
          return rewriter.create<tensor::DimOp>(loc, value, dimIdx).getResult();
        });

        auto emptyTensor =
            rewriter.create<tensor::EmptyOp>(loc, dims, newElementType);

        return cast<ShapedValue>(emptyTensor.getResult());
      })
      .Case([&](MemRefType type) {
        auto dims = dimMaker([&](int64_t dimIdx) -> Value {
          return rewriter.create<memref::DimOp>(loc, value, dimIdx).getResult();
        });

        auto emptyBuffer =
            rewriter.create<memref::AllocOp>(loc, dims, newElementType);

        return cast<ShapedValue>(emptyBuffer.getResult());
      })
      .Default([&](Type type) {
        llvm::report_fatal_error(StringRef("Unsupported result type: ") +
                                 llvm::to_string(type));
        return nullptr;
      });
}

static ShapedValue reallocShapedValue(PatternRewriter &rewriter,
                                      ShapedValue value, const Location loc,
                                      const Type newElementType = nullptr) {
  return reallocShapedValue(
      rewriter, value, loc, [](auto) { return nullptr; }, newElementType);
}

template <typename Op> struct EraseOpPattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename From>
struct GenericPreprocessAndRewrite : public OpRewritePattern<From> {
  using Base = GenericPreprocessAndRewrite<From>;
  using FromOp = From;

  using OpRewritePattern<From>::OpRewritePattern;

  virtual ~GenericPreprocessAndRewrite() = default;

private:
  template <unsigned start = 0>
  static constexpr void
  computeNonBroadcastableScalarOnlyOperands(SmallVector<bool, 3> &isScalar) {
    if constexpr (From::template hasTrait<OpTrait::BroadcastableOTF>()) {

      isScalar.push_back(From::template hasTrait<
                         OpTrait::ScalarOnlyHWTrait<start>::template Impl>());

      if constexpr (!From::template hasTrait<OpTrait::ElementwiseNaryOpTrait<
                        start + 1>::template Impl>())
        computeNonBroadcastableScalarOnlyOperands<start + 1>(isScalar);
    }
  }

public:
  SmallVector<Value> inlineBroadcast(PatternRewriter &rewriter,
                                     hivm::HIVMStructuredOp op) const {
    if (!op.isInlineBroadcastable()) {
      LDBG("Not inline-broadcastable!");
      return {};
    }

    SmallVector<int64_t> brcDims;
    op.getBroadcastLoopDims(brcDims);

    const auto loc = op.getLoc();
    assert(op.getNumDpsInits() == 1 &&
           "Can't broadcast to zero/multiple tensors/buffers");
    Value result = op.getDpsInitOperand(0)->get();

    SmallVector<bool, 3> isScalarOnly;
    computeNonBroadcastableScalarOnlyOperands(isScalarOnly);

    bool isBroadcastNeeded = false;
    auto newValues = llvm::map_to_vector(
        op.getHIVMInputOperands(false), [&](OpOperand *operand) {
          Value input = operand->get();

          // Ignore inputs that have to be scalars
          if (isScalarOnly[operand->getOperandNumber()])
            return input;

          // Ignore inputs that don't require broadcast
          if (const auto inputType = dyn_cast<ShapedType>(input.getType());
              inputType &&
              (brcDims.empty() || inputType.getShape()[brcDims[0]] != 1))
            return input;

          isBroadcastNeeded = true;
          auto newInput =
              reallocShapedValue(rewriter, cast<ShapedValue>(result), loc,
                                 getElementTypeOrSelf(input.getType()));
          LDBG("Operand " << input.getType() << " is broadcast!");
          const bool isTensor = isa<TensorType>(newInput.getType());

          auto brc = rewriter.create<hivm::VBrcOp>(
              loc, isTensor ? TypeRange(newInput.getType()) : TypeRange(),
              input, newInput, rewriter.getDenseI64ArrayAttr(brcDims));

          return isTensor ? brc.getResult()[0] : Value(newInput);
        });
#ifndef NDEBUG
    if (!isBroadcastNeeded) {
      LDBG("No broadcast needed!");
    }
#endif
    return isBroadcastNeeded ? newValues : SmallVector<Value>();
  }

  SmallVector<Value> inlineTranspose(PatternRewriter &rewriter,
                                     hivm::HIVMStructuredOp op) const {
    if (!op.isInlineTransposable()) {
      LDBG("Not inline-transposable!");
      return {};
    }

    auto trnDims = op.getPermutationArray();
    if (trnDims.empty()) {
      LDBG("No transpose needed!");
      return {};
    }

    const auto loc = op.getLoc();
    assert(op.getNumDpsInits() == 1 &&
           "Can't transpose with zero/multiple tensors/buffers");
    Value result = op.getDpsInitOperand(0)->get();

    return llvm::map_to_vector(
        op.getHIVMInputOperands(false), [&](OpOperand *operand) {
          Value input = operand->get();
          auto newInput =
              reallocShapedValue(rewriter, cast<ShapedValue>(result), loc);
          const bool isTensor = isa<TensorType>(newInput.getType());

          auto trn = rewriter.create<hivm::VTransposeOp>(
              loc, isTensor ? TypeRange(newInput.getType()) : TypeRange(),
              input, newInput, nullptr, rewriter.getDenseI64ArrayAttr(trnDims));

          return isTensor ? trn.getResult()[0] : Value(newInput);
        });
  }

  FailureOr<SmallVector<Value>>
  preprocessOperands(PatternRewriter &rewriter,
                     hivm::HIVMStructuredOp op) const {
    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics())
      return op.emitError(
          "has to be composed of either pure tensors or pure memrefs");

    auto broadcastOperands = inlineBroadcast(rewriter, op);
    if (!broadcastOperands.empty())
      return broadcastOperands;

    auto transposedOperands = inlineTranspose(rewriter, op);
    if (!transposedOperands.empty())
      return transposedOperands;

    return llvm::map_to_vector(
        op.getHIVMInputOperands(false),
        [&](OpOperand *operand) { return operand->get(); });
  }

  LogicalResult matchAndRewrite(From op,
                                PatternRewriter &rewriter) const final {
    auto preprocessedOperands = preprocessOperands(rewriter, op);
    if (failed(preprocessedOperands))
      return failure();

    return rewriteFromGeneric(op, std::move(preprocessedOperands.value()),
                              rewriter);
  }

  virtual LogicalResult
  rewriteFromGeneric(FromOp op, SmallVector<Value> &&preprocessedOperands,
                     PatternRewriter &rewriter) const = 0;
};

template <typename From, typename To>
struct RewriteFromGenericToGeneric final
    : public GenericPreprocessAndRewrite<From> {
  using Base = GenericPreprocessAndRewrite<From>;
  using Base::Base;

  LogicalResult rewriteFromGeneric(From op,
                                   SmallVector<Value> &&preprocessedOperands,
                                   PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<To>(op, op.getResultTypes(),
                                    preprocessedOperands, op.getDpsInits());
    return success();
  }
};

template <typename From>
struct RewriteUsingMapOp : public GenericPreprocessAndRewrite<From> {
  using Base = RewriteUsingMapOp<From>;
  using FromOp = From;

  using GenericPreprocessAndRewrite<From>::GenericPreprocessAndRewrite;

  virtual ~RewriteUsingMapOp() = default;

  LogicalResult rewriteFromGeneric(FromOp op,
                                   SmallVector<Value> &&preprocessedOperands,
                                   PatternRewriter &rewriter) const final {
    assert(op.getDst().size() == 1);
    rewriter.replaceOpWithNewOp<linalg::MapOp>(
        op, preprocessedOperands, op.getDst()[0],
        [this](OpBuilder &rewriter, const Location loc, ValueRange operands) {
          rewriter.create<linalg::YieldOp>(
              loc, ValueRange(rewriteFromMap(rewriter, loc, operands)));
        });
    return success();
  }

  virtual Value rewriteFromMap(OpBuilder &rewriter, Location loc,
                               ValueRange operands) const = 0;
};

template <typename FromOp, typename ToOp>
struct RewriteVBitwiseOp final : public RewriteUsingMapOp<FromOp> {
  using RewriteUsingMapOp<FromOp>::RewriteUsingMapOp;

  Value rewriteFromMap(OpBuilder &rewriter, const Location loc,
                       ValueRange operands) const final {
    assert(operands.size() == 2);
    Value lhs = operands[0], rhs = operands[1];

    if (auto floatType = dyn_cast<FloatType>(lhs.getType())) {
      lhs = rewriter.create<arith::BitcastOp>(
          loc, rewriter.getIntegerType(floatType.getWidth()), lhs);
      rhs = rewriter.create<arith::BitcastOp>(
          loc, rewriter.getIntegerType(floatType.getWidth()), rhs);
    }

    Value result = rewriter.create<ToOp>(loc, lhs, rhs);

    if (const auto type = dyn_cast<FloatType>(operands[0].getType()))
      result = rewriter.create<arith::BitcastOp>(loc, type, result);
    return result;
  }
};

template <typename Input, typename... Pairs> struct SwitchFinder;

template <typename Input> struct SwitchFinder<Input> {
  using type = void; // No match found
};

template <typename Input, typename T, typename U, typename... Rest>
struct SwitchFinder<Input, T, U, Rest...> {
  using type = typename SwitchFinder<Input, Rest...>::type; // Match others
};

template <typename Input, typename U, typename... Rest>
struct SwitchFinder<Input, Input, U, Rest...> {
  using type = U; // Match found
};

template <typename Input, typename... Pairs>
using SwitchFinder_t = typename SwitchFinder<Input, Pairs...>::type;

template <typename From, typename To, auto equivalentFn,
          typename EquivalentFnAttr =
              SwitchFinder_t<To, hfusion::ElemwiseUnaryOp, hfusion::UnaryFnAttr,
                             hfusion::ElemwiseBinaryOp, hfusion::BinaryFnAttr,
                             linalg::ElemwiseUnaryOp, linalg::UnaryFnAttr,
                             linalg::ElemwiseBinaryOp, linalg::BinaryFnAttr>,
          typename = std::enable_if_t<!std::is_same_v<EquivalentFnAttr, void>>,
          typename = std::enable_if_t<std::is_same_v<
              decltype(equivalentFn),
              decltype(std::declval<EquivalentFnAttr>().getValue())>>>
struct RewriteElemwiseOp final : public GenericPreprocessAndRewrite<From> {
  using GenericPreprocessAndRewrite<From>::GenericPreprocessAndRewrite;

  LogicalResult rewriteFromGeneric(From op,
                                   SmallVector<Value> &&preprocessedOperands,
                                   PatternRewriter &rewriter) const final {
    [[maybe_unused]] auto elemwiseOp = rewriter.replaceOpWithNewOp<To>(
        op, op.getResultTypes(), preprocessedOperands, op.getDst(),
        ArrayRef{rewriter.getNamedAttr(
            "fun", rewriter.getAttr<EquivalentFnAttr>(equivalentFn))});
    LDBG("New Elemwise equivalent: " << elemwiseOp);
    return success();
  }
};

struct RewriteVReduceOp : public OpRewritePattern<hivm::VReduceOp> {
  using OpRewritePattern<hivm::VReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto hasPureBufferSemantics = op.hasPureBufferSemantics();
    const auto hasPureTensorSemantics = op.hasPureTensorSemantics();
    if (!hasPureBufferSemantics && !hasPureTensorSemantics)
      return op.emitError(
          "Should have either pure tensor or buffer semantics!");

    SmallVector<Value> srcs(1, op.getSrc());
    if (auto indices = op.getIndices())
      srcs.push_back(indices);
    SmallVector<Value> dsts = op.getDst();
    if (hasPureBufferSemantics) {
      auto toTensor = [&](Value value) -> Value {
        return rewriter.create<bufferization::ToTensorOp>(
            loc, value, rewriter.getUnitAttr(), nullptr);
      };
      srcs = llvm::map_to_vector(srcs, toTensor);
      dsts = llvm::map_to_vector(dsts, toTensor);
    }

    const auto dstsWithoutDims =
        llvm::map_to_vector(dsts, [&](Value dst) -> Value {
          const auto reassociation = reshape_utils::getReAssociation(
              op.getReduceDims(), cast<ShapedType>(dst.getType()).getRank());
          return rewriter.create<tensor::CollapseShapeOp>(loc, dst,
                                                          reassociation);
        });

    const auto reduceOperation = op.getArith().getReduceOp();
    Operation *reduceOp;
    auto createReduceWithIndexOp = [&](hfusion::ReduceWithIndexKind reduceKind,
                                       bool isTieBreakLeft) -> Operation * {
      return rewriter.create<hfusion::ReduceWithIndexOp>(
          loc, ValueRange(dstsWithoutDims).getTypes(), srcs, dstsWithoutDims,
          rewriter.getAttr<hfusion::ReduceWithIndexKindAttr>(reduceKind),
          rewriter.getBoolAttr(isTieBreakLeft), op.getReduceDimsAttr());
    };
    switch (reduceOperation) {
    case hivm::ReduceOperation::min_with_index_left:
      reduceOp =
          createReduceWithIndexOp(hfusion::ReduceWithIndexKind::MIN, true);
      break;
    case hivm::ReduceOperation::min_with_index_right:
      reduceOp =
          createReduceWithIndexOp(hfusion::ReduceWithIndexKind::MIN, false);
      break;
    case hivm::ReduceOperation::max_with_index_left:
      reduceOp =
          createReduceWithIndexOp(hfusion::ReduceWithIndexKind::MAX, true);
      break;
    case hivm::ReduceOperation::max_with_index_right:
      reduceOp =
          createReduceWithIndexOp(hfusion::ReduceWithIndexKind::MAX, false);
      break;
    default:
      auto status = success();
      reduceOp = rewriter.create<linalg::ReduceOp>(
          loc, srcs, dstsWithoutDims, op.getReduceDims(),
          [&](OpBuilder &builder, Location loc, ValueRange operands) {
            const auto elementType = operands[0].getType();

            SmallVector<Value> results;
            switch (reduceOperation) {
            case hivm::ReduceOperation::sum:
              if (isa<FloatType>(elementType))
                results = {builder.create<arith::AddFOp>(loc, operands[0],
                                                         operands[1])};
              else
                results = {builder.create<arith::AddIOp>(loc, operands[0],
                                                         operands[1])};
              break;
            case hivm::ReduceOperation::prod:
              if (isa<FloatType>(elementType))
                results = {builder.create<arith::MulFOp>(loc, operands[0],
                                                         operands[1])};
              else
                results = {builder.create<arith::MulIOp>(loc, operands[0],
                                                         operands[1])};
              break;
            case hivm::ReduceOperation::any:
            case hivm::ReduceOperation::max:
              if (isa<FloatType>(elementType))
                results = {builder.create<arith::MaxNumFOp>(loc, operands[0],
                                                            operands[1])};
              else if (const auto intType = dyn_cast<IntegerType>(elementType);
                       intType && !intType.isUnsigned())
                results = {builder.create<arith::MaxSIOp>(loc, operands[0],
                                                          operands[1])};
              else
                results = {builder.create<arith::MaxUIOp>(loc, operands[0],
                                                          operands[1])};
              break;
            case hivm::ReduceOperation::all:
            case hivm::ReduceOperation::min:
              if (isa<FloatType>(elementType))
                results = {builder.create<arith::MinNumFOp>(loc, operands[0],
                                                            operands[1])};
              else if (const auto intType = dyn_cast<IntegerType>(elementType);
                       intType && !intType.isUnsigned())
                results = {builder.create<arith::MinSIOp>(loc, operands[0],
                                                          operands[1])};
              else
                results = {builder.create<arith::MinUIOp>(loc, operands[0],
                                                          operands[1])};
              break;
            case hivm::ReduceOperation::xori:
              results = {
                  builder.create<arith::XOrIOp>(loc, operands[0], operands[1])};
              break;
            case hivm::ReduceOperation::ori:
              results = {
                  builder.create<arith::OrIOp>(loc, operands[0], operands[1])};
              break;
            case hivm::ReduceOperation::andi:
              results = {
                  builder.create<arith::AndIOp>(loc, operands[0], operands[1])};
              break;
            default:
              status = rewriter.notifyMatchFailure(
                  loc, ("Unhandled reduce operation: " +
                        hivm::stringifyReduceOperation(reduceOperation).str())
                           .c_str());
              return;
            }
            builder.create<linalg::YieldOp>(loc, results);
          });
      if (failed(status))
        return status;
    }

    auto expandedReduceResults = llvm::map_to_vector(
        llvm::zip_equal(ValueRange(dsts).getTypes(), reduceOp->getResults()),
        [&](auto zippedResult) -> Value {
          auto [originalResType, newReduceRes] = zippedResult;
          const auto reassociation = reshape_utils::getReAssociation(
              op.getReduceDims(), cast<ShapedType>(originalResType).getRank());
          return rewriter.create<tensor::ExpandShapeOp>(
              loc, originalResType, newReduceRes, reassociation);
        });

    if (hasPureTensorSemantics)
      rewriter.replaceOp(op, expandedReduceResults);
    else {
      for (auto [src, dst] : llvm::zip(expandedReduceResults, op.getDst()))
        rewriter.create<bufferization::MaterializeInDestinationOp>(
            loc, Type(), src, dst, UnitAttr(), rewriter.getUnitAttr());
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct RewriteVTransposeOp : public OpRewritePattern<hivm::VTransposeOp> {

  using OpRewritePattern<hivm::VTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VTransposeOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, op.getSrc(), op.getDst(), op.getPermutation());
    return success();
  }
};

struct RewriteVBrcOp : public OpRewritePattern<hivm::VBrcOp> {

  using OpRewritePattern<hivm::VBrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VBrcOp op,
                                PatternRewriter &rewriter) const final {
    const auto resultType = cast<ShapedType>(op.getDst().getType());

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op.getResultTypes(), op.getSrc(), op.getDst(),
        op.getIndexingMapsArray(),
        SmallVector<utils::IteratorType>(resultType.getRank(),
                                         utils::IteratorType::parallel),
        [](OpBuilder &rewriter, Location loc, ValueRange operands) {
          rewriter.create<linalg::YieldOp>(loc, operands[0]);
        });
    return success();
  }
};

template <typename FromOp, typename ToIOp, typename ToFOp, int64_t identity>
struct RewriteVCumOp : public OpRewritePattern<FromOp> {

  using OpRewritePattern<FromOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromOp op,
                                PatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto cumDim = op.getCumDims()[0];
    const auto shapedResult = cast<ShapedValue>(op.getDst());
    const auto shapedType = shapedResult.getType();
    const auto rank = shapedType.getRank();

    auto tempBuffer =
        reallocShapedValue(rewriter, shapedResult, loc, [&](int64_t dimIdx) {
          return dimIdx > cumDim ? rewriter.getIndexAttr(1) : nullptr;
        });

    auto identityAttr =
        getConstantTypedAttr(shapedType.getElementType(), identity);
    auto identityValue = rewriter.create<arith::ConstantOp>(loc, identityAttr);
    auto filler = rewriter.create<linalg::FillOp>(
        loc, ValueRange(identityValue), ValueRange(tempBuffer));
    if (isa<TensorType>(tempBuffer.getType()))
      tempBuffer = cast<ShapedValue>(filler.getResult(0));

    SmallVector<utils::IteratorType> iteratorTypes(
        cumDim + 1, utils::IteratorType::parallel);
    iteratorTypes.append(rank - cumDim - 1, utils::IteratorType::reduction);

    SmallVector<AffineMap> indexingMaps(2,
                                        rewriter.getMultiDimIdentityMap(rank));
    auto affineDims =
        llvm::map_to_vector(llvm::seq(cumDim + 1), [&](auto dimIdx) {
          return rewriter.getAffineDimExpr(dimIdx);
        });
    affineDims.append(rank - cumDim - 1, rewriter.getAffineConstantExpr(0));
    indexingMaps.push_back(
        AffineMap::get(rank, 0, affineDims, op.getContext()));

    const auto isTensor = !op.getResultTypes().empty();

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        isTensor ? TypeRange({shapedType, tempBuffer.getType()}) : TypeRange(),
        op.getSrc(), ValueRange({op.getDst(), tempBuffer}), indexingMaps,
        iteratorTypes,
        [](OpBuilder &rewriter, const Location loc, ValueRange args) {
          Value result;
          if (isa<FloatType>(args[0].getType()))
            result = rewriter.create<ToFOp>(loc, args[0], args[2]);
          else
            result = rewriter.create<ToIOp>(loc, args[0], args[2]);

          rewriter.create<linalg::YieldOp>(loc, ValueRange({result, result}));
        });

    if (isTensor)
      rewriter.replaceOp(op, genericOp.getResult(0));
    else
      rewriter.eraseOp(op);
    return success();
  }
};

struct RewriteVConcatOp : public OpRewritePattern<hivm::VConcatOp> {

  using OpRewritePattern<hivm::VConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VConcatOp op,
                                PatternRewriter &rewriter) const final {
    const auto hasPureBufferSemantics = op.hasPureBufferSemantics();
    const auto hasPureTensorSemantics = op.hasPureTensorSemantics();
    if (!hasPureBufferSemantics && !hasPureTensorSemantics)
      return op.emitError(
          "has to be composed of either pure tensors or pure memrefs");

    SmallVector<Value> srcs = op.getSrc();

    if (hasPureBufferSemantics)
      srcs = llvm::map_to_vector(srcs, [&](Value value) -> Value {
        return rewriter.create<bufferization::ToTensorOp>(
            op.getLoc(), value, rewriter.getUnitAttr(), nullptr);
      });

    auto concatOp =
        rewriter.create<tensor::ConcatOp>(op.getLoc(), op.getDim(), srcs);

    if (hasPureTensorSemantics) {
      rewriter.replaceOp(op, concatOp);
      return success();
    }

    rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
        op, Type(), concatOp, op.getDst(), UnitAttr(), rewriter.getUnitAttr());
    return success();
  }
};

struct RewriteVArangeOp : public OpRewritePattern<hivm::VArangeOp> {

  using OpRewritePattern<hivm::VArangeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VArangeOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<hfusion::ArangeOp>(
        op, op.getOffset(), op.getStrides(), op.getDst());
    return success();
  }
};

struct RewriteLoadOp : public OpRewritePattern<hivm::LoadOp> {

  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::LoadOp op,
                                PatternRewriter &rewriter) const final {
    if (const auto padMode = op.getPadModeAttr();
        !padMode || padMode.getPadmode() == hivm::PadMode::PadNull) {
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, op.getResultTypes(),
                                                  ValueRange(op.getSrc()),
                                                  ValueRange(op.getDst()));
      return success();
    }

    const auto loc = op.getLoc();
    auto src = cast<ShapedValue>(op.getSrc());
    auto dst = cast<ShapedValue>(op.getDst());
    const auto rank = src.getType().getRank();
    const auto srcShape = src.getType().getShape();
    const auto dstShape = dst.getType().getShape();
    const auto isTensor = isa<TensorType>(src.getType());

    if (isTensor) {
      src = cast<ShapedValue>(
          rewriter
              .create<bufferization::ToMemrefOp>(
                  loc,
                  MemRefType::get(srcShape, src.getType().getElementType()),
                  src, rewriter.getUnitAttr())
              .getResult());

      auto originalDst = rewriter.create<bufferization::ToMemrefOp>(
          loc, MemRefType::get(dstShape, dst.getType().getElementType()), dst,
          nullptr);
      dst = cast<ShapedValue>(
          rewriter.create<bufferization::CloneOp>(loc, originalDst)
              .getResult());
    }

    // 1. Collect some metadata about paddings
    Value leftPadNum = op.getLeftPaddingNum();
    Value rightPadNum = op.getRightPaddingNum();
    Value sourceLastDimSize =
        rewriter.create<memref::DimOp>(loc, src, rank - 1);
    Value totalLastDimSize = sourceLastDimSize;
    if (leftPadNum)
      totalLastDimSize =
          rewriter.create<arith::AddIOp>(loc, totalLastDimSize, leftPadNum);
    Value rightPadOffset = totalLastDimSize;
    if (rightPadNum)
      totalLastDimSize =
          rewriter.create<arith::AddIOp>(loc, totalLastDimSize, rightPadNum);
    Value totalDstLastDimSize =
        rewriter.create<memref::DimOp>(loc, dst, rank - 1);
    const Value sizeDiff = rewriter.create<arith::SubIOp>(
        loc, totalDstLastDimSize, totalLastDimSize);

    // right padding always exists but might be zero
    if (!rightPadNum)
      rightPadNum = sizeDiff;
    // left padding exists only if the user specifies it or specifies the right
    // padding
    else if (!leftPadNum) {
      leftPadNum = sizeDiff;
      rightPadOffset =
          rewriter.create<arith::AddIOp>(loc, leftPadNum, sourceLastDimSize);
    }

    // TODO: Enable runtime verification op interface to verify hfusion and hivm
    // dynamic parameters

    const auto padMode = op.getPadModeAttr().getPadmode();
    Value padValue;
    switch (padMode) {
    case hivm::PadMode::PadValue:
      padValue = op.getPadValue();
      break;
    case hivm::PadMode::PadFirstElem: {
      SmallVector<OpFoldResult> sizes;
      sizes.reserve(rank);
      for (auto [index, dim] : llvm::enumerate(srcShape.drop_back()))
        if (ShapedType::isDynamic(dim))
          sizes.push_back(
              rewriter.create<memref::DimOp>(loc, src, index).getResult());
        else
          sizes.push_back(rewriter.getIndexAttr(dim));
      sizes.push_back(rewriter.getIndexAttr(1));

      padValue = rewriter.create<memref::SubViewOp>(
          loc, src, SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(0)),
          sizes, SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)));
      break;
    }
    default:
      // At this point, there has to be a padding
      llvm::llvm_unreachable_internal(
          ("Unhandled padding mode: " + hivm::stringifyPadMode(padMode).str())
              .c_str());
    }

    // 2. Insert left paddings
    if (leftPadNum)
      loadPartiallyFromSrc(rewriter, loc, padValue, dst,
                           rewriter.getIndexAttr(0), leftPadNum, true);

    // 3. Insert data
    auto srcLastDimSizeAsFoldResult =
        ShapedType::isDynamic(srcShape.back())
            ? OpFoldResult(sourceLastDimSize)
            : rewriter.getIndexAttr(srcShape.back());
    loadPartiallyFromSrc(rewriter, loc, src, dst,
                         leftPadNum ? OpFoldResult(leftPadNum)
                                    : rewriter.getIndexAttr(0),
                         srcLastDimSizeAsFoldResult);

    // 4. Insert right paddings
    if (rightPadNum)
      loadPartiallyFromSrc(rewriter, loc, padValue, dst,
                           rightPadOffset == sourceLastDimSize
                               ? srcLastDimSizeAsFoldResult
                               : rightPadOffset,
                           rightPadNum, true);

    if (isTensor)
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
          op, dst, rewriter.getUnitAttr(), rewriter.getUnitAttr());
    else
      rewriter.eraseOp(op);

    return success();
  }

  void loadPartiallyFromSrc(PatternRewriter &rewriter, const Location opLoc,
                            Value src, ShapedValue dst,
                            OpFoldResult lastDimOffset,
                            OpFoldResult lastDimSize,
                            const bool shouldBrc = false) const {
    const auto rank = dst.getType().getRank();

    SmallVector<OpFoldResult> offsets(rank - 1, rewriter.getIndexAttr(0));
    offsets.push_back(lastDimOffset);

    SmallVector<OpFoldResult> sizes;
    sizes.reserve(rank);
    for (auto [index, dim] :
         llvm::enumerate(dst.getType().getShape().drop_back())) {
      if (ShapedType::isDynamic(dim))
        sizes.push_back(rewriter.create<memref::DimOp>(dst.getLoc(), dst, index)
                            .getResult());
      else
        sizes.push_back(rewriter.getIndexAttr(dim));
    }
    sizes.push_back(lastDimSize);

    auto subview = rewriter.create<memref::SubViewOp>(
        dst.getLoc(), dst, offsets, sizes,
        SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)));

    if (shouldBrc)
      rewriter.create<hivm::VBrcOp>(
          opLoc, TypeRange(), src, subview,
          isa<MemRefType>(src.getType())
              ? rewriter.getDenseI64ArrayAttr({rank - 1})
              : rewriter.getDenseI64ArrayAttr({}));
    else
      rewriter.create<memref::CopyOp>(opLoc, src, subview);
  }
};

struct RewriteUsingTypeConverter {

  explicit RewriteUsingTypeConverter(const TypeConverter &typeConverter)
      : typeConverter(typeConverter) {}

  FailureOr<Attribute> legalizeAttributeTypes(Attribute attr) const {
    return TypeSwitch<Attribute, FailureOr<Attribute>>(attr)
        .Case([this](ArrayAttr arrayAttr) -> FailureOr<Attribute> {
          LDBG("Found ArrayAttr: " << arrayAttr);
          SmallVector<Attribute> newArrayAttr;
          for (auto attr : arrayAttr) {
            auto legalizedAttr = legalizeAttributeTypes(attr);
            if (failed(legalizedAttr))
              return failure();
            newArrayAttr.push_back(legalizedAttr.value());
          }
          return ArrayAttr::get(arrayAttr.getContext(), newArrayAttr);
        })
        .Case([this](DictionaryAttr dictionaryAttr) -> FailureOr<Attribute> {
          LDBG("Found DictionaryAttr: " << dictionaryAttr);
          SmallVector<NamedAttribute> newDictionary;
          for (auto namedAttr : dictionaryAttr) {
            auto newAttr = legalizeAttributeTypes(namedAttr.getValue());
            if (failed(newAttr))
              return failure();
            namedAttr.setValue(newAttr.value());
            newDictionary.push_back(namedAttr);
          }
          return DictionaryAttr::get(dictionaryAttr.getContext(),
                                     newDictionary);
        })
        .Case([this](TypeAttr typeAttr) -> FailureOr<Attribute> {
          LDBG("Found TypeAttr: " << typeAttr);
          auto legalizedType = legalize(typeAttr.getValue());
          if (failed(legalizedType))
            return failure();
          return TypeAttr::get(legalizedType.value());
        })
        .Default([](Attribute attr) -> FailureOr<Attribute> {
          LDBG("Found Default: " << attr);
          return attr;
        });
  }

  FailureOr<Type> legalize(Type type) const {
    const auto newType = typeConverter.convertType(type);
    LDBG("Type '" << type << "' is converted into '" << newType << "'");
    if (!newType)
      return failure();
    return newType;
  }

  FailureOr<bool> convertAttributes(Operation *op, IRRewriter &rewriter) const {
    bool isChanged = false;
    LDBG("Converting Attributes");
    for (auto attr : op->getAttrs()) {
      LDBG("Convert NamedAttribute: Name = ["
           << attr.getName() << "], Value = [" << attr.getValue() << "]");
      const auto legalizedAttr = legalizeAttributeTypes(attr.getValue());
      if (failed(legalizedAttr))
        return rewriter.notifyMatchFailure(op->getLoc(),
                                           "Attrs should be convertible!");

      if (attr.getValue() != *legalizedAttr) {
        LDBG("Value changed to " << *legalizedAttr);
        op->setAttr(attr.getName(), *legalizedAttr);
        isChanged = true;
      }
    }
    return isChanged;
  }

  FailureOr<bool> convertResults(Operation *op, IRRewriter &rewriter) const {
    bool isChanged = false;
    LDBG("Converting Results");
    for (auto result : op->getResults()) {
      const auto newType = legalize(result.getType());
      if (failed(newType))
        return rewriter.notifyMatchFailure(result.getLoc(),
                                           "Result should be convertible!");

      if (result.getType() != *newType) {
        LDBG("Result " << result.getResultNumber() << " changed from "
                       << result.getType() << " to " << *newType);
        result.setType(*newType);
        isChanged = true;
      }
    }
    return isChanged;
  }

  FailureOr<bool> convertRegions(Operation *op, IRRewriter &rewriter) const {
    bool isChanged = false;
    LDBG("Converting Regions");
    for (auto &region : op->getRegions()) {
      LDBG("Converting region " << region.getRegionNumber());
      for (auto &block : llvm::make_early_inc_range(region.getBlocks())) {
        LDBG("Converting block with types (" << block.getArgumentTypes()
                                             << ")");
        const auto signatureConversion =
            typeConverter.convertBlockSignature(&block);
        if (!signatureConversion)
          return rewriter.notifyMatchFailure(region.getLoc(),
                                             "Region should be convertible!");

        for (auto [arg, newType] :
             llvm::zip_equal(block.getArguments(),
                             signatureConversion->getConvertedTypes())) {
          if (arg.getType() == newType)
            continue;
          LDBG("Argument type " << arg.getType() << " changed to " << newType);
          arg.setType(newType);
          isChanged = true;
        }
      }
    }
    return isChanged;
  }

  LogicalResult matchAndRewrite(Operation *op, IRRewriter &rewriter) const {
    bool isChanged = false;
    auto *newOp = rewriter.cloneWithoutRegions(*op);

    auto status = convertAttributes(newOp, rewriter);
    if (failed(status))
      return failure();
    isChanged = isChanged || *status;

    status = convertResults(newOp, rewriter);
    if (failed(status))
      return failure();
    isChanged = isChanged || *status;

    status = convertRegions(op, rewriter);
    if (failed(status))
      return failure();
    isChanged = isChanged || *status;

    if (isChanged) {
      for (auto [region, newRegion] :
           llvm::zip_equal(op->getRegions(), newOp->getRegions()))
        rewriter.inlineRegionBefore(region, newRegion, newRegion.end());
      rewriter.replaceOp(op, newOp);
    } else
      rewriter.eraseOp(newOp);
    return success();
  }

  const TypeConverter &typeConverter;
};

struct ConvertHIVMToUpstream
    : public impl::ExecutionEngineHIVMToUpstreamConversionBase<
          ConvertHIVMToUpstream> {

  using Base::Base;

  template <typename T> static T getIfNotHIVM(T &&thing) {
    return thing && isa<hivm::HIVMDialect>(thing.getDialect()) ? T{} : thing;
  }

  LogicalResult applyTypeConversion() {
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    auto convertTypes = [&converter](ArrayRef<Type> types) {
      return llvm::map_to_vector(types, [&converter](Type type) {
        return converter.convertType(type);
      });
    };
    converter.addConversion([&convertTypes](FunctionType type) {
      const auto inputs = convertTypes(type.getInputs());
      const auto results = convertTypes(type.getResults());
      return FunctionType::get(type.getContext(), inputs, results);
    });
    converter.addConversion([&convertTypes](TupleType type) {
      return TupleType::get(type.getContext(), convertTypes(type.getTypes()));
    });
    converter.addConversion([](MemRefType type) {
      return MemRefType::get(type.getShape(), type.getElementType(),
                             getIfNotHIVM(type.getLayout()),
                             getIfNotHIVM(type.getMemorySpace()));
    });
    converter.addConversion([](UnrankedMemRefType type) {
      return UnrankedMemRefType::get(type.getElementType(),
                                     getIfNotHIVM(type.getMemorySpace()));
    });

    const auto status = getOperation()->walk(
        [pattern = RewriteUsingTypeConverter(converter)](Operation *op) {
          IRRewriter rewriter(op);
          if (failed(pattern.matchAndRewrite(op, rewriter)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    LDBG("Module after type conversion:\n" << *getOperation());
    if (status.wasInterrupted())
      return failure();
    if (failed(verify(getOperation())))
      return failure();
    return success();
  }

  void runOnOperation() override {
    auto &ctx = getContext();

    if (failed(applyTypeConversion())) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&ctx);
    patterns
        .add<EraseOpPattern<hivm::SetMaskNormOp>,
             EraseOpPattern<hivm::SetFlagOp>, EraseOpPattern<hivm::WaitFlagOp>,
             EraseOpPattern<hivm::PipeBarrierOp>>(&ctx);
    patterns
        .add<RewriteFromGenericToGeneric<hivm::VAbsOp, linalg::AbsOp>,
             RewriteFromGenericToGeneric<hivm::VAddOp, linalg::AddOp>,
             RewriteFromGenericToGeneric<hivm::VSubOp, linalg::SubOp>,
             RewriteFromGenericToGeneric<hivm::VMulOp, linalg::MulOp>,
             RewriteFromGenericToGeneric<hivm::VDivOp, linalg::DivOp>,
             RewriteFromGenericToGeneric<hivm::VMaxOp, linalg::MaxOp>,
             RewriteFromGenericToGeneric<hivm::VMinOp, linalg::MinOp>,
             RewriteFromGenericToGeneric<hivm::VExpOp, linalg::ExpOp>,
             RewriteFromGenericToGeneric<hivm::VLnOp, linalg::LogOp>,
             RewriteFromGenericToGeneric<hivm::VRsqrtOp, linalg::RsqrtOp>,
             RewriteFromGenericToGeneric<hivm::VSqrtOp, linalg::SqrtOp>,
             RewriteFromGenericToGeneric<hivm::VTanhOp, linalg::TanhOp>,
             RewriteFromGenericToGeneric<hivm::VRecOp, linalg::ReciprocalOp>,
             RewriteFromGenericToGeneric<hivm::VSelOp, linalg::SelectOp>,
             RewriteFromGenericToGeneric<hivm::VErfOp, linalg::ErfOp>,
             RewriteFromGenericToGeneric<hivm::StoreOp, linalg::CopyOp>>(&ctx);
    patterns.add<RewriteElemwiseOp<hivm::VReluOp, hfusion::ElemwiseUnaryOp,
                                   hfusion::UnaryFn::relu>,
                 RewriteElemwiseOp<hivm::VNotOp, hfusion::ElemwiseUnaryOp,
                                   hfusion::UnaryFn::vnot>>(&ctx);
    patterns.add<RewriteVBitwiseOp<hivm::VAndOp, arith::AndIOp>,
                 RewriteVBitwiseOp<hivm::VOrOp, arith::OrIOp>,
                 RewriteVBitwiseOp<hivm::VXorOp, arith::XOrIOp>>(&ctx);
    patterns
        .add<RewriteVCumOp<hivm::VCumprodOp, arith::MulIOp, arith::MulFOp, 1>,
             RewriteVCumOp<hivm::VCumsumOp, arith::AddIOp, arith::AddFOp, 0>>(
            &ctx);
    patterns.add<RewriteVBrcOp, RewriteVTransposeOp, RewriteVArangeOp,
                 RewriteVConcatOp, RewriteVReduceOp, RewriteLoadOp>(&ctx);

    ConversionTarget target(ctx);
    target.addIllegalDialect<hivm::HIVMDialect>();
    target.addLegalDialect<
        linalg::LinalgDialect, bufferization::BufferizationDialect,
        tensor::TensorDialect, memref::MemRefDialect, arith::ArithDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::execution_engine::createConvertHIVMToUpstreamPass() {
  return std::make_unique<ConvertHIVMToUpstream>();
}
