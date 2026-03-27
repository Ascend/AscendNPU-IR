//===--------- HandleReduce.cpp - Handle non-vectorizeable reduce pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_GENERICUNROLLER
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

static Value buildNest(OpBuilder& builder, Location loc, Value init,
                       ArrayRef<Value> dims,
                       function_ref<Value(OpBuilder&, Value, ValueRange)> leaf) {
  auto rank = static_cast<int64_t>(dims.size());
  SmallVector<Value, 4> ivs;

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  std::function<Value(OpBuilder&, int64_t, Value)> rec =
    [&leaf, &rank, &ivs, &loc, &c0, &dims, &c1, &rec]
      (OpBuilder& builder, int64_t i, Value acc)-> Value {
        if (i == rank) {
          return leaf(builder, acc, ivs);
        }

        auto forOp = builder.create<scf::ForOp>(loc, c0, dims[i], c1, ValueRange{acc},
          [&ivs, &rec, &i]
            (OpBuilder& builder, Location loc, Value iv, ValueRange iterArgs) {
              ivs.push_back(iv);
              Value inner = rec(builder, i + 1, iterArgs.front());
              ivs.pop_back();
              builder.create<scf::YieldOp>(loc, inner);
            });

        return forOp.getResult(0);
      };

  return rec(builder, 0, init);
}

static Value evalCombiner(OpBuilder& builder, Location loc,
                          Region& region, ValueRange argValues) {
  auto& body = region.front();

  assert(body.getNumArguments() == argValues.size());
  assert(body.getNumArguments() == 2);

  auto cloneOriginalBody = [&]() -> Value {
    IRMapping map;
    for (auto [arg, val] : llvm::zip(body.getArguments(), argValues)) {
      map.map(arg, val);
    }

    for (auto& op : body.without_terminator()) {
      builder.clone(op, map);
    }

    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    return map.lookup(yield.getValues().front());
  };

  auto getF32Value = [&](Value v) -> Value {
    auto ty = dyn_cast<FloatType>(v.getType());
    if (!ty || ty.isF32()) {
      return v;
    }
    assert(!ty.isF64() && "f64 never expected here");
    return builder.create<arith::ExtFOp>(loc, builder.getF32Type(), v);
  };

  auto canLegalizeToF32 = [&](Operation& op) -> bool {
    return isa<arith::AddFOp,
               arith::SubFOp,
               arith::MulFOp,
               arith::DivFOp,
               arith::RemFOp,
               arith::NegFOp,
               arith::MaximumFOp,
               arith::MinimumFOp,
               arith::MaxNumFOp,
               arith::MinNumFOp,
               arith::CmpFOp,
               arith::SelectOp>(op);
  };

  auto regionArgTy = dyn_cast<FloatType>(body.getArgument(0).getType());
  bool useF32FloatPath = regionArgTy && !regionArgTy.isF32();
  if (regionArgTy) {
    assert(!regionArgTy.isF64() && "f64 never expected here");
  }

  if (!useF32FloatPath) {
    return cloneOriginalBody();
  }

  for (auto& op : body.without_terminator()) {
    if (!canLegalizeToF32(op)) {
      op.emitWarning("failed to legalize float reduce combiner to f32, cloning body as-is");
      return cloneOriginalBody();
    }
  }

  IRMapping map;
  for (auto [arg, val] : llvm::zip(body.getArguments(), argValues)) {
    map.map(arg, getF32Value(val));
  }

  for (auto& op : body.without_terminator()) {
    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      Value lhs = map.lookup(add.getLhs());
      Value rhs = map.lookup(add.getRhs());
      map.map(add.getResult(), builder.create<arith::AddFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto sub = dyn_cast<arith::SubFOp>(op)) {
      Value lhs = map.lookup(sub.getLhs());
      Value rhs = map.lookup(sub.getRhs());
      map.map(sub.getResult(), builder.create<arith::SubFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto mul = dyn_cast<arith::MulFOp>(op)) {
      Value lhs = map.lookup(mul.getLhs());
      Value rhs = map.lookup(mul.getRhs());
      map.map(mul.getResult(), builder.create<arith::MulFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto div = dyn_cast<arith::DivFOp>(op)) {
      Value lhs = map.lookup(div.getLhs());
      Value rhs = map.lookup(div.getRhs());
      map.map(div.getResult(), builder.create<arith::DivFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto rem = dyn_cast<arith::RemFOp>(op)) {
      Value lhs = map.lookup(rem.getLhs());
      Value rhs = map.lookup(rem.getRhs());
      map.map(rem.getResult(), builder.create<arith::RemFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto neg = dyn_cast<arith::NegFOp>(op)) {
      Value operand = map.lookup(neg.getOperand());
      map.map(neg.getResult(), builder.create<arith::NegFOp>(loc, operand));
      continue;
    }

    if (auto maximum = dyn_cast<arith::MaximumFOp>(op)) {
      Value lhs = map.lookup(maximum.getLhs());
      Value rhs = map.lookup(maximum.getRhs());
      map.map(maximum.getResult(),
              builder.create<arith::MaximumFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto minimum = dyn_cast<arith::MinimumFOp>(op)) {
      Value lhs = map.lookup(minimum.getLhs());
      Value rhs = map.lookup(minimum.getRhs());
      map.map(minimum.getResult(),
              builder.create<arith::MinimumFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto maxnum = dyn_cast<arith::MaxNumFOp>(op)) {
      Value lhs = map.lookup(maxnum.getLhs());
      Value rhs = map.lookup(maxnum.getRhs());
      map.map(maxnum.getResult(),
              builder.create<arith::MaxNumFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto minnum = dyn_cast<arith::MinNumFOp>(op)) {
      Value lhs = map.lookup(minnum.getLhs());
      Value rhs = map.lookup(minnum.getRhs());
      map.map(minnum.getResult(),
              builder.create<arith::MinNumFOp>(loc, lhs, rhs));
      continue;
    }

    if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
      Value lhs = map.lookup(cmp.getLhs());
      Value rhs = map.lookup(cmp.getRhs());
      map.map(cmp.getResult(),
              builder.create<arith::CmpFOp>(loc, cmp.getPredicate(), lhs, rhs));
      continue;
    }

    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      Value cond = map.lookup(select.getCondition());
      Value trueValue = map.lookup(select.getTrueValue());
      Value falseValue = map.lookup(select.getFalseValue());
      map.map(select.getResult(),
              builder.create<arith::SelectOp>(loc, cond, trueValue, falseValue));
      continue;
    }

    op.emitWarning("unexpected op after float reduce combiner legalization check, cloning body as-is");
    return cloneOriginalBody();
  }

  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  return map.lookup(yield.getValues().front());
}

static LogicalResult lowerReduce(PatternRewriter& rewriter, Operation* op,
                                 Value input, Value outInit,
                                 Region& combiner, ArrayRef<int64_t> reduceDims) {
  auto loc = op->getLoc();
  auto inputType = cast<ShapedType>(input.getType());
  auto outType = cast<ShapedType>(outInit.getType());

  auto inputRank = inputType.getRank();
  auto outRank = outType.getRank();

  auto inputElemTy = dyn_cast<FloatType>(inputType.getElementType());
  bool useF32Accumulation = inputElemTy && !inputElemTy.isF32();
  if (inputElemTy) {
    assert(!inputElemTy.isF64() && "f64 never expected here");
  }

  auto promoteToF32IfNeeded = [&](OpBuilder& builder, Value v) -> Value {
    if (!useF32Accumulation) {
      return v;
    }
    auto ty = dyn_cast<FloatType>(v.getType());
    if (!ty || ty.isF32()) {
      return v;
    }
    return builder.create<arith::ExtFOp>(loc, builder.getF32Type(), v);
  };

  auto convertFromF32IfNeeded = [&](OpBuilder& builder, Value v, Type dstTy) -> Value {
    if (!useF32Accumulation || v.getType() == dstTy) {
      return v;
    }
    auto dstFloatTy = dyn_cast<FloatType>(dstTy);
    if (!dstFloatTy) {
      return v;
    }
    return builder.create<arith::TruncFOp>(loc, dstTy, v);
  };

  assert(llvm::is_sorted(reduceDims));
  auto isUnique = [](ArrayRef<int64_t> a) {
    SetVector<int64_t> unique;
    for (auto v : a) {
      unique.insert(v);
    }
    return unique.size() == a.size();
  };
  assert(isUnique(reduceDims));
  auto isValidReduceDims = [&inputRank](ArrayRef<int64_t> a) {
    return std::any_of(a.begin(), a.end(), [inputRank](int64_t d) {
      return d >= 0 && d < inputRank;
    });
  };
  assert(isValidReduceDims(reduceDims));

  SmallVector<Value, 4> outDims;
  outDims.reserve(outRank);
  for (int64_t i = 0; i < outRank; i++) {
    outDims.push_back(rewriter.createOrFold<tensor::DimOp>(loc, outInit, i));
  }

  SmallVector<Value, 4> reduceSizes;
  reduceSizes.reserve(reduceDims.size());
  for (auto d : reduceDims) {
    reduceSizes.push_back(rewriter.createOrFold<tensor::DimOp>(loc, input, d));
  }

  SmallVector<int64_t, 4> inDimToOutPos(inputRank, -1);
  SmallVector<int64_t, 4> inDimToReducePos(inputRank, -1);

  auto isReduceDim = [&reduceDims](int64_t d) {
    return llvm::binary_search(reduceDims, d);
  };

  int64_t outPos = 0;
  int64_t reducePos = 0;
  for (int64_t d = 0; d < inputRank; d++) {
    if (isReduceDim(d)) {
      inDimToReducePos[d] = reducePos++;
    } else {
      inDimToOutPos[d] = outPos++;
    }
  }

  if (!reduceDims.empty()) {
    auto expectedOutRank = inputRank - static_cast<int64_t>(reduceDims.size());
    assert(outRank == expectedOutRank);
  }

  auto makeInputIdxs = [&inputRank, &isReduceDim, &inDimToReducePos, &inDimToOutPos]
    (ValueRange outIvs, ValueRange reduceIvs) {
      SmallVector<Value, 4> idxs;
      idxs.resize(inputRank);
      for (int64_t d = 0; d < inputRank; d++) {
        if (isReduceDim(d)) {
          idxs[d] = reduceIvs[inDimToReducePos[d]];
        } else {
          idxs[d] = outIvs[inDimToOutPos[d]];
        }
      }
      return idxs;
    };

  Value allReduceDimsSizeOne = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  for (auto sz : reduceSizes) {
    Value is1 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, sz, c1);
    allReduceDimsSizeOne = rewriter.create<arith::AndIOp>(loc, allReduceDimsSizeOne, is1);
  }

  auto leafOuter = [&reduceDims, &c0, &loc, &input, &makeInputIdxs,
                              &allReduceDimsSizeOne, &combiner, &reduceSizes,
                              &promoteToF32IfNeeded, &convertFromF32IfNeeded, &outType]
    (OpBuilder& builder, Value acc, ValueRange outIvs) -> Value {
      SmallVector<Value, 4> reduceZeros(reduceDims.size(), c0);
      Value first = builder.create<tensor::ExtractOp>(loc, input, makeInputIdxs(outIvs, reduceZeros));
      Value firstAcc = promoteToF32IfNeeded(builder, first);

      auto ifOp = builder.create<scf::IfOp>(
        loc, allReduceDimsSizeOne,
        [&first, &acc, &outIvs]
          (OpBuilder& builder, Location loc) {
            Value res = builder.create<tensor::InsertOp>(loc, first, acc, outIvs);
            builder.create<scf::YieldOp>(loc, res);
          },
        [&c0, &input, &makeInputIdxs, &outIvs, &combiner, &firstAcc, &reduceSizes, &acc,
         &promoteToF32IfNeeded, &convertFromF32IfNeeded, &outType]
          (OpBuilder& builder, Location loc) {
            auto reduceLeaf = [&loc, &c0, &input, &makeInputIdxs, &outIvs, &combiner,
                               &promoteToF32IfNeeded]
              (OpBuilder& builder, Value acc, ValueRange reduceIvs) {
                Value isFirstElement = builder.create<arith::ConstantIntOp>(loc, 1, 1);
                for (auto iv : reduceIvs) {
                  Value eq0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, iv, c0);
                  isFirstElement = builder.create<arith::AndIOp>(loc, isFirstElement, eq0);
                }

                auto stepIf = builder.create<scf::IfOp>(
                  loc, isFirstElement,
                  [&acc]
                    (OpBuilder& builder, Location loc) {
                      builder.create<scf::YieldOp>(loc, acc);
                    },
                  [&input, &makeInputIdxs, &outIvs, &reduceIvs, &combiner, &acc,
                   &promoteToF32IfNeeded]
                    (OpBuilder& builder, Location loc) {
                      Value in = builder.create<tensor::ExtractOp>(loc, input, makeInputIdxs(outIvs, reduceIvs));
                      in = promoteToF32IfNeeded(builder, in);
                      Value next = evalCombiner(builder, loc, combiner, ValueRange{in, acc});
                      builder.create<scf::YieldOp>(loc, next);
                    });

                return stepIf.getResult(0);
              };

              Value accFinal = buildNest(builder, loc, firstAcc, reduceSizes, reduceLeaf);
              accFinal = convertFromF32IfNeeded(builder, accFinal, outType.getElementType());
              Value res = builder.create<tensor::InsertOp>(loc, accFinal, acc, outIvs);
              builder.create<scf::YieldOp>(loc, res);
          });

      return ifOp.getResult(0);
    };

    Value result = buildNest(rewriter, loc, outInit, outDims, leafOuter);
    rewriter.replaceOp(op, result);
    return success();
}

static Value createMinMaxCmp(OpBuilder& builder, Location loc, Value lhs, Value rhs, bool max, bool unsignedCmp) {
  auto elementType = lhs.getType();
  if (isa<FloatType>(elementType)) {
    auto pred = max ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT;
    return builder.create<arith::CmpFOp>(loc, pred, lhs, rhs);
  }
  arith::CmpIPredicate pred;
  if (unsignedCmp) {
    pred = max ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::ult;
  } else {
    pred = max ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::slt;
  }
  return builder.create<arith::CmpIOp>(loc, pred, lhs, rhs);
}

static SmallVector<Value, 2> buildNest2(OpBuilder& builder, Location loc,
                                        ValueRange inits, ArrayRef<Value> dims,
                                        function_ref<SmallVector<Value, 2>(OpBuilder&, ValueRange, ValueRange)> leaf) {
  auto rank = static_cast<int64_t>(dims.size());
  SmallVector<Value, 4> ivs;

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  std::function<SmallVector<Value, 2>(OpBuilder&, int64_t, ValueRange)> rec =
    [&ivs, &rec, rank, leaf, loc, c0, dims, c1]
      (OpBuilder& builder, int64_t i, ValueRange accs) {
        if (i == rank) {
          return leaf(builder, accs, ivs);
        }

        auto forOp = builder.create<scf::ForOp>(loc, c0, dims[i], c1, accs,
          [&ivs, &rec, i]
            (OpBuilder& builder, Location loc, Value iv, ValueRange iterArgs) {
              ivs.push_back(iv);
              SmallVector<Value, 2> inner = rec(builder, i + 1, iterArgs);
              ivs.pop_back();
              builder.create<scf::YieldOp>(loc, inner);
            });

        SmallVector<Value, 2> res;
        res.append(forOp.getResults().begin(), forOp.getResults().end());
        return res;
      };

  return rec(builder, 0, inits);
}

static LogicalResult lowerReduceWithIndex(PatternRewriter& rewriter,
                                          hfusion::ReduceWithIndexOp op,
                                          unsigned dim,
                                          bool max,
                                          bool unsignedCmp) {
  Value values = op.getDpsInputOperand(0)->get();
  Value idxs = op.getDpsInputOperand(1)->get();
  Value initValues = op.getDpsInitOperand(0)->get();
  Value initIdxs = op.getDpsInitOperand(1)->get();
  auto inputRank = cast<RankedTensorType>(values.getType()).getRank();

  assert(inputRank >= 1);
  Location loc = op.getLoc();

  auto outValType = cast<RankedTensorType>(initValues.getType());
  assert(dim < static_cast<unsigned>(inputRank));
  auto outRank = outValType.getRank();

  SmallVector<Value, 4> outDims;
  outDims.reserve(outRank);
  for (int64_t i = 0; i < outRank; i++) {
    outDims.push_back(rewriter.createOrFold<tensor::DimOp>(loc, initValues, i));
  }

  Value reduceSize = rewriter.createOrFold<tensor::DimOp>(loc, values, dim);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto makeInputIdxs = [inputRank, dim]
    (ValueRange outIvs, Value reduceIv) {
      SmallVector<Value, 4> idxs;
      idxs.reserve(inputRank);
      for (int64_t d = 0; d < inputRank; d++) {
        if (static_cast<unsigned>(d) == dim) {
          idxs.push_back(reduceIv);
        } else {
          auto outPos = (d < static_cast<int64_t>(dim)) ? d : (d - 1);
          idxs.push_back(outIvs[outPos]);
        }
      }
      return idxs;
    };

  auto leafOuter = [makeInputIdxs, c0, loc, values, idxs, c1, reduceSize, max, unsignedCmp]
    (OpBuilder& builder, ValueRange accs, ValueRange outIvs) -> SmallVector<Value, 2> {
      assert(accs.size() == 2);
      Value acc = accs[0];
      Value accIdxs = accs[1];

      auto in0 = makeInputIdxs(outIvs, c0);
      Value first = builder.create<tensor::ExtractOp>(loc, values, in0);
      Value firstIdx = builder.create<tensor::ExtractOp>(loc, idxs, in0);

      auto leaf = builder.create<scf::ForOp>(loc, c1, reduceSize, c1,
        ValueRange{first, firstIdx},
        [makeInputIdxs, outIvs, values, max, idxs, unsignedCmp]
          (OpBuilder& builder, Location loc, Value iv, ValueRange iterArgs) {
            Value cur = iterArgs[0];
            Value curIdx = iterArgs[1];

            SmallVector<Value, 4> inIdxs = makeInputIdxs(outIvs, iv);
            Value in = builder.create<tensor::ExtractOp>(loc, values, inIdxs);
            Value inIdx = builder.create<tensor::ExtractOp>(loc, idxs, inIdxs);

            Value join = createMinMaxCmp(builder, loc, in, cur, max, unsignedCmp);

            Value next = builder.create<arith::SelectOp>(loc, join, in, cur);
            Value nextIdx = builder.create<arith::SelectOp>(loc, join, inIdx, curIdx);

            builder.create<scf::YieldOp>(loc, ValueRange{next, nextIdx});
          });

      Value val = leaf.getResult(0);
      Value idx = leaf.getResult(1);

      Value out = builder.create<tensor::InsertOp>(loc, val, acc, outIvs);
      Value outIdxs = builder.create<tensor::InsertOp>(loc, idx, accIdxs, outIvs);
      return {out, outIdxs};
    };

  SmallVector<Value, 2> res = buildNest2(rewriter, loc, ValueRange{initValues, initIdxs}, outDims, leafOuter);
  rewriter.replaceOp(op, ValueRange{res[0], res[1]});
  return success();
}

struct HandleReduceOpPattern : OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter& rewriter) const override {
    if (hacc::isSkippable(op)) {
      return failure();
    }
    if (hacc::isLegalToAutoVectorizeReduce(op)) {
      // will be processed by AutoVectorize/v2
      return failure();
    }

    SmallVector<unsigned, 2> reduceDims;
    op.getReductionDims(reduceDims);
    return lowerReduce(rewriter, op,
      op.getDpsInputOperand(0)->get(),
      op.getDpsInitOperand(0)->get(),
      op.getRegion(),
      llvm::to_vector_of<int64_t>(reduceDims));
  }
};

struct HandleReduceWithIndexOpPattern : OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter& rewriter) const override {
    if (!op->hasAttrOfType<BoolAttr>("unsigned_src")) {
      return failure();
    }
    if (auto uSrc = op->getAttrOfType<BoolAttr>("unsigned_src"); !uSrc.getValue()) {
      return failure();
    }

    SmallVector<unsigned> reduceDims;
    op.getReductionDims(reduceDims);
    assert(reduceDims.size() == 1 && "according to .td spec of ReduceWithIndexOp, only one reduce dim is supported");
    auto dim = reduceDims.front();

    auto kind = op->getAttrOfType<hfusion::ReduceWithIndexKindAttr>("reduce_kind").getReduceWithIndexKind();
    return lowerReduceWithIndex(
      rewriter,
      op,
      dim,
      kind == hfusion::ReduceWithIndexKind::MAX,
      /*unsignedCmp=*/ true);
  }
};

struct GenericUnroller final : public impl::GenericUnrollerBase<GenericUnroller> {
  void runOnOperation() override {
    auto* ctx = &getContext();
    RewritePatternSet patterns{&getContext()};
    patterns.add<HandleReduceOpPattern>(ctx);
    patterns.add<HandleReduceWithIndexOpPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::hfusion::createGenericUnrollerPass() {
  return std::make_unique<GenericUnroller>();
}

} // namespace mlir