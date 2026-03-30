//===- LinalgToHFusion.cpp - conversion from Linalg to HFusion dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

struct LinalgMapToHFusionPattern : public OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MapOp op,
                                PatternRewriter &rewriter) const final {
    Region &mapper = op.getMapper();
    if (!mapper.hasOneBlock())
      return failure();
    Block &block = mapper.front();
    if (block.getOperations().size() !=
        2) // only process maximum operations inside linalg map of 2
      return failure();
    auto &mapped = *block.getOperations().begin();
    auto callOp = dyn_cast<func::CallOp>(mapped);
    if (callOp == nullptr)
      return failure();
    StringRef funcName = callOp.getCallee();
    if (funcName.starts_with("__hmf_relu")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::relu);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_log1p")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::log1p);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_sqrt")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::sqrt);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_fabs")) {
      auto unaryAttr =
          rewriter.getAttr<linalg::UnaryFnAttr>(linalg::UnaryFn::abs);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_exp")) {
      auto unaryAttr =
          rewriter.getAttr<linalg::UnaryFnAttr>(linalg::UnaryFn::exp);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_rsqrt")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::rsqrt);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_log")) {
      auto unaryAttr =
          rewriter.getAttr<linalg::UnaryFnAttr>(linalg::UnaryFn::log);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_isinf")) {
      rewriter.replaceOpWithNewOp<hfusion::IsInfOp>(
          op, TypeRange(op.getResult()), ValueRange{op.getInputs()[0]});
      return success();
    }
    // TODO: funcName of is_nan op need to confirm
    if (funcName.starts_with("__hmf_isnan")) {
      rewriter.replaceOpWithNewOp<hfusion::IsNanOp>(
          op, TypeRange(op.getResult()), ValueRange{op.getInputs()[0]});
      return success();
    }
    if (funcName.starts_with("__hmf_recipf") ||
        funcName.starts_with("__hmf_recipDh")) {
      Type resultType = mlir::getElementTypeOrSelf(op.getInit().getType());
      auto constOne = rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getFloatAttr(resultType, 1));
      auto emptyTensor = mlir::tensor::createTensorEmptyOp(
          rewriter, op->getLoc(), op.getInputs()[0]);
      auto fillOp = rewriter.create<linalg::FillOp>(
          op->getLoc(), ValueRange{constOne.getResult()},
          ValueRange{emptyTensor});
      auto binaryAttr =
          rewriter.getAttr<linalg::BinaryFnAttr>(linalg::BinaryFn::div);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseBinaryOp>(
          op, ValueRange{fillOp->getResults()[0], op.getInputs()[0]},
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_rint") ||
        funcName.starts_with("__hmf_roundf")) {
      auto arg = op.getInputs().front();
      auto roundMode = funcName.starts_with("__hmf_rint")
                           ? hfusion::RoundMode::RINT
                           : hfusion::RoundMode::ROUND;
      auto rounding = rewriter.getAttr<hfusion::RoundModeAttr>(roundMode);
      auto unsignedAttr = rewriter.getAttr<hfusion::UnsignedModeAttr>(
          hfusion::UnsignedMode::SI2SI);
      auto enableOverflow = rewriter.getBoolAttr(true);
      auto enableSaturate = rewriter.getBoolAttr(true);

      hfusion::TypeFn castIntegerType = hfusion::TypeFn::cast_signed;
      auto castAttr = rewriter.getAttr<hfusion::TypeFnAttr>(castIntegerType);
      rewriter.replaceOpWithNewOp<hfusion::CastOp>(
          op, TypeRange{arg.getType()}, ValueRange{arg}, ValueRange{arg},
          rounding, enableOverflow, enableSaturate, castAttr, unsignedAttr);
      return success();
    }
    if (funcName.starts_with("__hmf_tanf") ||
        funcName.starts_with("__hmf_tanDh")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::tan);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_tanhf") ||
        funcName.starts_with("__hmf_tanhDh")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::tanh);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_atan")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::atan);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_ilogb")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::ilogb);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_ldexp")) {
      auto binaryAttr =
          rewriter.getAttr<hfusion::BinaryFnAttr>(hfusion::BinaryFn::ldexp);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseBinaryOp>(
          op, ValueRange({op.getInputs()[0], op.getInputs()[1]}),
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_flip")) {
      // There is only one input which becomes the last dimension.
      // So, only need to do flip on the first input in the vector.
      rewriter.replaceOpWithNewOp<hfusion::FlipOp>(
          op, ValueRange{op.getInit()},
          ValueRange{op.getInputs()[0], op.getInputs()[1]});
      return success();
    }
    if (funcName.starts_with("__hmf_powf")) {
      auto binaryAttr =
          rewriter.getAttr<hfusion::BinaryFnAttr>(hfusion::BinaryFn::powf);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseBinaryOp>(
          op, ValueRange({op.getInputs()[0], op.getInputs()[1]}),
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_powi")) {
      auto binaryAttr =
          rewriter.getAttr<hfusion::BinaryFnAttr>(hfusion::BinaryFn::powi);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseBinaryOp>(
          op, ValueRange({op.getInputs()[0], op.getInputs()[1]}),
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    return failure();
  }
};

struct LinalgGenericToHFusionArangePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getOutputs().size() != 1 || op.getInputs().size() != 0)
      return failure();
    // Should iterate and store over the whole tensor/memref
    if (!llvm::all_of(op.getIteratorTypesArray(), linalg::isParallelIterator) ||
        !op.getIndexingMapsArray()[0].isIdentity())
      return failure();
    // Should only yield value from index
    if (!op.hasIndexSemantics() || !op.getBody()->getArgument(0).use_empty())
      return failure();

    Value target = op.getOutputs()[0];
    auto type = dyn_cast<ShapedType>(target.getType());
    if (type == nullptr || !type.getElementType().isIntOrFloat())
      return failure();

    // Note: currently, only 1-D arange is supported
    if (!type.hasRank() || type.getRank() != 1)
      return failure();
    auto yieldOp = *(op.getBody()->getOps<linalg::YieldOp>().begin());
    Value yieldVal = yieldOp.getValues()[0];
    auto castOp = yieldVal.getDefiningOp<arith::IndexCastOp>();
    if (castOp == nullptr)
      return failure();
    auto indexOp = castOp.getIn().getDefiningOp<linalg::IndexOp>();
    if (indexOp == nullptr)
      return failure();

#ifndef NDEBUG
    // Get the strides necessary from the dps init
    Value init = op.getDpsInitOperand(0)->get();
    auto shapedTy = dyn_cast<ShapedType>(init.getType());
    assert(shapedTy && "Expecting shaped type as output of arange");
#endif
    rewriter.replaceOpWithNewOp<hfusion::ArangeOp>(op, target);
    return success();
  }
};

struct LinalgGenericToHFusionGatherPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // hfusion.gather (DPS style):
    // Inputs: [SourceData, Indices]
    // Inits/Outs: [Output]
    if (op.getOutputs().size() != 1 || op.getInputs().size() != 2)
      return failure();

    // Should iteratorrTypes be [parallel, ..., gather, *]
    SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
    int64_t gatherLoopDim = -1;
    for (size_t i = 0; i < iterators.size(); ++i) {
      if (iterators[i] == utils::IteratorType::gather) {
        if (gatherLoopDim != -1)
          return failure();
        gatherLoopDim = (int64_t)i;
      }
    }
    if (gatherLoopDim == -1)
      return failure();

    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    AffineMap sourceMap = maps[0];  // Source
    AffineMap indicesMap = maps[1]; // Indices
    AffineMap outputMap = maps[2];  // Output
    if (indicesMap.isFunctionOfDim(gatherLoopDim) ||
        outputMap.isFunctionOfDim(gatherLoopDim)) {
      return failure();
    }
    if (!sourceMap.isFunctionOfDim(gatherLoopDim))
      return failure();

    int64_t axis = -1;
    for (unsigned i = 0; i < sourceMap.getNumResults(); ++i) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(sourceMap.getResult(i))) {
        if (dimExpr.getPosition() == gatherLoopDim) {
          axis = i;
          break;
        }
      }
    }
    if (axis == -1)
      return failure();

    Block &block = op.getRegion().front();
    if (block.getNumArguments() != 3)
      return failure();
    Value argSrc = block.getArgument(0);
    Value argIndices = block.getArgument(1);
    Value argOut = block.getArgument(2);

    auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return failure();
    auto selectOp = yieldOp.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!selectOp)
      return failure();
    if (selectOp.getTrueValue() != argSrc || selectOp.getFalseValue() != argOut)
      return failure();
    auto cmpOp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::eq)
      return failure();
    Value cmpLhs = cmpOp.getLhs();
    Value cmpRhs = cmpOp.getRhs();
    Value loopIndexVal = nullptr;
    if (cmpLhs == argIndices) {
      loopIndexVal = cmpRhs;
    } else if (cmpRhs == argIndices) {
      loopIndexVal = cmpLhs;
    } else {
      return failure();
    }

    Operation *defOp = loopIndexVal.getDefiningOp();
    if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
      defOp = castOp.getIn().getDefiningOp();
    }
    auto indexOp = dyn_cast<linalg::IndexOp>(defOp);
    if (!indexOp)
      return failure();

    if ((int)indexOp.getDim() != gatherLoopDim)
      return failure();

    Value source = op.getDpsInputs()[0];
    Value indices = op.getDpsInputs()[1];
    Value output = op.getDpsInits()[0];
    rewriter.replaceOpWithNewOp<hfusion::GatherOp>(
        op,
        source,  // Operand 0: Source
        indices, // Operand 1: Indices
        output,  // Operand 2: Outs (Init)
        axis     // Attribute: Axis
    );
    return success();
  }
};

// handle the atomic op in the form of linalg.generic
// use hfusion.store to represent the atomic op
// Input:
//  linalg.generic
//    ins(%subview_2, %extracted_slice : memref<?xf32>, tensor<?xf32>)
//    outs(%subview_2 : memref<?xf32>)
//      attrs = {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel",
//      MemSyncScope = "gpu"} {
//    ^bb0(%in: f32, %in_3: f32, %out: f32):
//      %output = arith.addf %in, %in_3 : f32
//      linalg.yield %output : f32
//    }
//
// Output:
//  %memref = bufferization.to_memref %extracted_slice : memref<?xf32>
//  hfusion.store {atomic_kind = #hfusion.atomic_kind<add>}
//    ins(%16 : memref<?xf32>)
//    outs(%subview_2 : memref<?xf32>)
struct AtomicLinalgGenericToHFusionStorePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  std::optional<StringRef> getAtomicAttrRef(linalg::GenericOp op) const {
    StringAttr linalgAtomicRmwAttr =
        op->getAttrOfType<StringAttr>(StringRef("GenericAtomicRMW"));
    if (!linalgAtomicRmwAttr) {
      return std::nullopt;
    }

    return linalgAtomicRmwAttr.getValue();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    auto linalgAtomicRmwStr = getAtomicAttrRef(op);
    if (!linalgAtomicRmwStr.has_value()) {
      return failure();
    }

    auto atomicRef = *linalgAtomicRmwStr;
    hfusion::AtomicKindAttr atomicKind;
    auto *context = rewriter.getContext();
    if (atomicRef.ends_with("add")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::ADD);
    } else if (atomicRef.ends_with("max")) {
      atomicKind = AtomicKindAttr::get(
          context, atomicRef == "umax" ? AtomicKind::UMAX : AtomicKind::MAX);
    } else if (atomicRef.ends_with("min")) {
      atomicKind = AtomicKindAttr::get(
          context, atomicRef == "umin" ? AtomicKind::UMIN : AtomicKind::MIN);
    } else if (atomicRef.ends_with("and")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::AND);
    } else if (atomicRef.ends_with("xor")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::XOR);
    } else if (atomicRef.ends_with("or")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::OR);
    } else if (atomicRef.ends_with("cas")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::CAS);
    } else if (atomicRef.ends_with("exch")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::XCHG);
    } else {
      op.emitOpError("unsupported atomic operation: ");
      llvm_unreachable("Not implemented");
    }
    if (atomicKind == AtomicKindAttr::get(context, AtomicKind::CAS)) {
      rewriter.create<hfusion::AtomicCasOp>(
          op.getLoc(), TypeRange(),
          ValueRange{op.getInputs()[1], op.getInputs()[2]}, op.getInputs()[0]);
      rewriter.eraseOp(op);
      return success();
    }
    if (atomicKind == AtomicKindAttr::get(context, AtomicKind::XCHG)) {
      rewriter.create<hfusion::AtomicXchgOp>(op.getLoc(), TypeRange(),
                                             ValueRange{op.getInputs()[1]},
                                             op.getInputs()[0]);
      rewriter.eraseOp(op);
      return success();
    }
    // hivm.copy only accept tensor/tensor or memref/memref as input/output
    // and the atomicRMW Op might be masked
    // need to turn the input tensor into the same type the dst memref has
    auto hfusionStoreOp = rewriter.create<hfusion::StoreOp>(
        op.getLoc(), ValueRange(op.getInputs()[1]),
        ValueRange(op.getInputs()[0]));
    hfusionStoreOp.setAtomicKindAttr(atomicKind);
    rewriter.eraseOp(op);
    return success();
  }
};

// To replace the linalg::reduceOp with attr of reduce_with_index
// with hfusion.reduce_with_index Op
// Input:
// %reduced:2 = linalg.reduce
//    ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>)
//    outs(%0, %1 : tensor<256xf32>, tensor<256xi32>)
//    dimensions = [1]  {reduce_mode = "max_with_index", tie_break_left =
//    "true"}
//
// Output:
// %2:2 = hfusion.reduce_with_index {tie_break_left = true} <max>
//    ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>)
//    outs(%0, %1 : tensor<256xf32>, tensor<256xi32>)
//    dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
struct LinalgToHFusionReduceWithIndex
    : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const final {
    StringAttr linalgReduceAttr = op->getAttrOfType<StringAttr>("reduce_mode");
    if (!linalgReduceAttr) {
      return failure();
    }

    hfusion::ReduceWithIndexKind reduceKind;
    if (linalgReduceAttr == "max_with_index") {
      reduceKind = hfusion::ReduceWithIndexKind::MAX;
    } else if (linalgReduceAttr == "min_with_index") {
      reduceKind = hfusion::ReduceWithIndexKind::MIN;
    } else {
      return failure();
    }

    StringAttr linalgSrcUnsignedAttr =
        op->getAttrOfType<StringAttr>("unsigned_src");
    if (!linalgSrcUnsignedAttr)
      return failure();

    const bool sourceUnsigned = (linalgSrcUnsignedAttr == "true");

    bool tieBreakLeft = true;
    if (auto tieBreakLeftAttr =
            op->getAttrOfType<StringAttr>("tie_break_left")) {
      tieBreakLeft = (tieBreakLeftAttr == "true");
    }

    ValueRange inits = op.getInits();
    auto reduceKindAttr =
        ReduceWithIndexKindAttr::get(rewriter.getContext(), reduceKind);
    auto sourceUnsignedAttr =
        BoolAttr::get(rewriter.getContext(), sourceUnsigned);
    auto tieBreakLeftAttr = BoolAttr::get(rewriter.getContext(), tieBreakLeft);

    assert(inits.size() == 2);
    rewriter.replaceOpWithNewOp<hfusion::ReduceWithIndexOp>(
        op, TypeRange{inits[0].getType(), inits[1].getType()},
        /*input*/ op.getInputs(), /*outputValue&Index*/ inits, reduceKindAttr,
        sourceUnsignedAttr, tieBreakLeftAttr, op.getDimensionsAttr());
    return success();
  }
};

void mlir::hfusion::populateLinalgToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<LinalgMapToHFusionPattern, LinalgGenericToHFusionArangePattern,
           AtomicLinalgGenericToHFusionStorePattern,
           LinalgGenericToHFusionGatherPattern, LinalgToHFusionReduceWithIndex>(
          patterns.getContext());
}

namespace {
struct LinalgToHFusionConversionPass
    : public impl::ConvertLinalgToHFusionBase<LinalgToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void LinalgToHFusionConversionPass::runOnOperation() {
  auto *module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<memref::MemRefDialect, linalg::LinalgDialect,
                         bufferization::BufferizationDialect,
                         tensor::TensorDialect, hfusion::HFusionDialect>();
  // also add dialects that maybe created by hfusion dialect ops
  target.addLegalDialect<arith::ArithDialect, math::MathDialect>();
  target.addDynamicallyLegalOp<linalg::ReduceOp>([](Operation *op) {
    StringAttr linalgReduceAttr =
        op->getAttrOfType<StringAttr>(StringRef("reduce_mode"));
    return !linalgReduceAttr;
  });
  // Mark linalg.map and libclc func decls as illegal
  target.addIllegalOp<linalg::MapOp>();
  target.addIllegalOp<linalg::GenericOp>();

  RewritePatternSet patterns(&getContext());
  populateLinalgToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createLinalgToHFusionConversionPass() {
  return std::make_unique<LinalgToHFusionConversionPass>();
}
