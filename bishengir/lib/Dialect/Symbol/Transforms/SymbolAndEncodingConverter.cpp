//===- SymbolAndEncodingConverter.cpp -------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass converting between bind_symbolic_shape
// and tensor encoding
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "symbol-and-encoding-converter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace symbol {
#define GEN_PASS_DEF_SYMBOLTOENCODING
#define GEN_PASS_DEF_ENCODINGTOSYMBOL
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"

namespace {

// TODO: replace the code implementation here after merging the full
// implementation of SymbolManager
class SymbolManager {
public:
  explicit SymbolManager(Operation *op) : op_(op) {}
  void initialize() {
    op_->walk([this](symbol::SymbolicIntOp op) {
      symbolMap_[op.getSymbolName()] = op;
    });
  }

  FailureOr<symbol::SymbolicIntOp>
  getSymbolicInt(FlatSymbolRefAttr symbolName) {
    auto it = symbolMap_.find(symbolName.getValue());
    if (it == symbolMap_.end())
      return failure();
    return it->second;
  }

private:
  Operation *op_;
  DenseMap<StringRef, symbol::SymbolicIntOp> symbolMap_;
};

struct InsertCastForEncoding : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    LogicalResult result = failure();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto funcOp = cast<func::FuncOp>(
        SymbolTable::lookupSymbolIn(moduleOp, op.getCallee()));
    auto funcInputTypes = funcOp.getFunctionType().getInputs();
    SmallVector<Value> operands(op.getOperands());

    rewriter.setInsertionPoint(op);
    for (size_t i = 0; i < operands.size(); i++) {
      if (operands[i].getType() != funcInputTypes[i]) {
        operands[i] = rewriter.create<tensor::CastOp>(
            operands[i].getLoc(), funcInputTypes[i], operands[i]);
        result = success();
      }
    }

    SmallVector<Type> originalResultTypes(op->getResultTypes());
    if (succeeded(result) ||
        originalResultTypes != funcOp.getFunctionType().getResults()) {
      op = rewriter.replaceOpWithNewOp<func::CallOp>(op, funcOp, operands);
    }

    SmallVector<Value> results(op.getResults());
    rewriter.setInsertionPointAfter(op);
    for (size_t i = 0; i < originalResultTypes.size(); i++) {
      if (results[i].getType() != originalResultTypes[i]) {
        auto newResult = rewriter.create<tensor::CastOp>(
            results[i].getLoc(), originalResultTypes[i], results[i]);
        rewriter.replaceAllUsesExcept(results[i], newResult, newResult);
        result = success();
      }
    }
    return result;
  }
};

struct ConvertBindSymbolicShapeToTensorEncode
    : public OpRewritePattern<symbol::BindSymbolicShapeOp> {
  using OpRewritePattern<symbol::BindSymbolicShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(symbol::BindSymbolicShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();
    auto symbols = op.getShapeSymbols();
    auto exprs = op.getShapeExpressions().getAffineMap().getResults();
    auto oprType = cast<RankedTensorType>(operand.getType());
    SmallVector<Attribute> encoding;

    encoding.reserve(encoding.size());

    for (auto expr : exprs) {
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        encoding.push_back(rewriter.getI64IntegerAttr(constExpr.getValue()));
      } else if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr)) {
        auto symbolicIntOp = cast<symbol::SymbolicIntOp>(
            symbols[symbolExpr.getPosition()].getDefiningOp());
        encoding.push_back(symbolicIntOp.getSymbolNameAttr());
      } else {
        llvm_unreachable("Propagate symbol pass must be called first to make "
                         "sure no binary expression appears");
      }
    }

    auto encodedType =
        RankedTensorType::get(oprType.getShape(), oprType.getElementType(),
                              rewriter.getArrayAttr(encoding));
    if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
      rewriter.modifyOpInPlace(blockArgument.getOwner()->getParentOp(), [&]() {
        auto parentOp = operand.getParentBlock()->getParentOp();
        operand.setType(encodedType);
        if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
          auto funcType = funcOp.getFunctionType();
          auto argNum = blockArgument.getArgNumber();
          SmallVector<Type> newInputs(funcType.getInputs());
          newInputs[argNum] = encodedType;
          funcOp.setFunctionType(
              rewriter.getFunctionType(newInputs, funcType.getResults()));
        } else {
          llvm_unreachable(
              "Unhandled block argument for symbol to encoding conversion.");
        }
      });
    } else {
      rewriter.modifyOpInPlace(operand.getDefiningOp(),
                               [&]() { operand.setType(encodedType); });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct HandleReturnOp : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = cast<func::FuncOp>(op->getParentOp());
    auto funcType = funcOp.getFunctionType();
    SmallVector<Type> resultTypes(funcType.getResults());
    LogicalResult result = failure();
    for (const auto &[idx, returnVal] : llvm::enumerate(op.getOperands())) {
      if (returnVal.getType() != resultTypes[idx]) {
        resultTypes[idx] = returnVal.getType();
        result = success();
      }
    }
    if (succeeded(result)) {
      rewriter.modifyOpInPlace(funcOp, [&]() {
        funcOp.setFunctionType(
            rewriter.getFunctionType(funcType.getInputs(), resultTypes));
      });
    }
    return result;
  }
};

bool addBindSymbolicShapeOp(Value bindingValue, ArrayAttr encoding,
                            Operation *insertAfter, SymbolManager *manager,
                            OpBuilder &builder) {
  SmallVector<Value> shapeSymbols;
  SmallVector<AffineExpr, 4> exprs;
  int64_t dynSymbolNum = 0;
  MLIRContext *ctx = builder.getContext();
  LDBG("Binding symbolic shape for " << bindingValue);
  for (auto attr : encoding) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(attr)) {
      exprs.push_back(getAffineConstantExpr(integerAttr.getInt(), ctx));
    } else if (auto symbolAttr = dyn_cast<FlatSymbolRefAttr>(attr)) {
      exprs.push_back(getAffineSymbolExpr(dynSymbolNum, ctx));
      dynSymbolNum++;
      auto symbolicIntOp = manager->getSymbolicInt(symbolAttr);
      if (failed(symbolicIntOp))
        return false;
      shapeSymbols.push_back((*symbolicIntOp).getResult());
      if ((*symbolicIntOp)->getBlock() != insertAfter->getBlock())
        return false;
      if (insertAfter->isBeforeInBlock(*symbolicIntOp))
        insertAfter = *symbolicIntOp;

      LDBG("SymbolicIntOp: " << shapeSymbols.back());
    }
  }
  auto shapeExprsMap = AffineMap::get(0, dynSymbolNum, exprs, ctx);
  LDBG("Created shapeExprs: " << shapeExprsMap);
  LDBG("Inserting after: " << *insertAfter);
  builder.setInsertionPointAfter(insertAfter);
  builder.create<symbol::BindSymbolicShapeOp>(
      bindingValue.getLoc(), bindingValue, shapeSymbols,
      AffineMapAttr::get(shapeExprsMap));
  return true;
}

struct HandleFuncOp : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  explicit HandleFuncOp(MLIRContext *ctx, SymbolManager *manager)
      : OpRewritePattern<func::FuncOp>(ctx, /*benefit=*/2), manager(manager) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    auto funcType = op.getFunctionType();
    SmallVector<Type> newInputs(funcType.getInputs());
    auto arguments = op.getFunctionBody().getArguments();
    LogicalResult result = failure();

    for (size_t i = 0; i < newInputs.size(); i++) {
      if (auto tensorType = dyn_cast<RankedTensorType>(newInputs[i]);
          tensorType && tensorType.getEncoding()) {
        auto encoding = cast<ArrayAttr>(tensorType.getEncoding());
        addBindSymbolicShapeOp(arguments[i], encoding,
                               &op.getBody().front().front(), manager,
                               rewriter);
        newInputs[i] = RankedTensorType::get(tensorType.getShape(),
                                             tensorType.getElementType());
        rewriter.modifyOpInPlace(op,
                                 [&]() { arguments[i].setType(newInputs[i]); });
        result = success();
      }
    }

    if (succeeded(result)) {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setFunctionType(FunctionType::get(op.getContext(), newInputs,
                                             funcType.getResults()));
      });
    }
    return result;
  }

private:
  SymbolManager *manager;
};

struct ConvertTensorEncodeToBindSymbolicShape : public RewritePattern {
  ConvertTensorEncodeToBindSymbolicShape(MLIRContext *ctx,
                                         SymbolManager *manager)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        manager(manager) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    LogicalResult result = failure();
    for (size_t i = 0; i < op->getNumResults(); i++) {
      if (auto rankedTensorType =
              dyn_cast<RankedTensorType>(op->getResult(i).getType());
          rankedTensorType && rankedTensorType.getEncoding()) {
        auto encoding = cast<ArrayAttr>(rankedTensorType.getEncoding());
        if (!addBindSymbolicShapeOp(op->getResult(i), encoding, op, manager,
                                    rewriter))
          return failure();
        rewriter.modifyOpInPlace(op, [&]() {
          op->getResult(i).setType(RankedTensorType::get(
              rankedTensorType.getShape(), rankedTensorType.getElementType()));
        });
        result = success();
      }
    }
    return result;
  }

private:
  SymbolManager *manager;
};

class SymbolToEncodingPass
    : public impl::SymbolToEncodingBase<SymbolToEncodingPass> {
public:
  void runOnOperation() override;
};

void SymbolToEncodingPass::runOnOperation() {
  Operation *moduleOp = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<ConvertBindSymbolicShapeToTensorEncode, HandleReturnOp,
               InsertCastForEncoding>(ctx);

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

class EncodingToSymbolPass
    : public impl::EncodingToSymbolBase<EncodingToSymbolPass> {
public:
  void runOnOperation() override;
};

void EncodingToSymbolPass::runOnOperation() {
  Operation *moduleOp = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  std::unique_ptr<SymbolManager> manager =
      std::make_unique<SymbolManager>(moduleOp);

  manager->initialize();
  patterns.add<HandleReturnOp, InsertCastForEncoding>(ctx);
  patterns.add<HandleFuncOp, ConvertTensorEncodeToBindSymbolicShape>(
      ctx, manager.get());

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> createSymbolToEncodingPass() {
  return std::make_unique<SymbolToEncodingPass>();
}

std::unique_ptr<Pass> createEncodingToSymbolPass() {
  return std::make_unique<EncodingToSymbolPass>();
}

} // namespace symbol
} // namespace mlir