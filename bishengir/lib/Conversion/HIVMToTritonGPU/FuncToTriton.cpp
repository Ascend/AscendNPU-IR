//===---- FuncToTriton.cpp - conversion from Func to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Conversion/HIVMToTritonGPU/MemRefDescriptor.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::triton;

namespace {
/// Collects all discardable attributes from a function op into the output
/// vector. These are attributes that are not part of any registered dialect
/// interface (e.g. FunctionOpInterface) and can be freely transferred to
/// the converted triton function.
static void filterFuncAttributes(
    FunctionOpInterface func, SmallVectorImpl<NamedAttribute> &result) {
  for (const NamedAttribute &attr : func->getDiscardableAttrs()) {
    result.push_back(attr);
  }
}

static Value narrowABIIndexArg(ConversionPatternRewriter &rewriter,
                               Location loc, Value abiArg, Type originalType) {
  auto i32Ty = rewriter.getI32Type();
  Value narrowed = abiArg;
  if (!abiArg.getType().isInteger(32))
    narrowed = rewriter.create<arith::TruncIOp>(loc, i32Ty, abiArg);
  if (isa<IndexType>(originalType))
    return rewriter.create<arith::IndexCastUIOp>(loc, originalType, narrowed);
  return narrowed;
}

// Stage 1 leaves a placeholder cast for every MemRef block argument it could
// not see through:
//     %p, %ap, %off, %sz.., %st.. = unrealized_conversion_cast %memrefArg
// Its results stand for the argument's descriptor fields. Map them onto the
// real ones and report that the cast itself must not be cloned.
static bool resolveDescriptorPlaceholder(
    UnrealizedConversionCastOp cast,
    const DenseMap<Value, SmallVector<Value>> &argDescriptors,
    IRMapping &argMapper) {
  if (cast.getInputs().size() != 1)
    return false;
  auto it = argDescriptors.find(cast.getInputs().front());
  if (it == argDescriptors.end())
    return false;
  if (cast.getNumResults() != it->second.size())
    return false;

  for (auto [res, field] : llvm::zip_equal(cast.getResults(), it->second))
    argMapper.map(res, field);
  return true;
}

class FuncOpPattern : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter::SignatureConversion result(op.getNumArguments());
    auto ttConverter = getTypeConverter<TritonTypeConverter>();
    // Perserve index of memref argument with shared attribute
    SmallVector<std::optional<int>> sharedIds;
    // Perserve index of memref argument with fractal_layout attribute
    SmallVector<std::pair<int, Attribute>> fractalAttrArgIds;
    FunctionType ttFuncType = ttConverter->convertTTFunctionSignature(op,
        ttConverter->getOptions().useBarePtrCallConv,
        result,
        sharedIds,
        fractalAttrArgIds);
    if (!ttFuncType)
      return rewriter.notifyMatchFailure(op, "Could not convert funcop");

    SmallVector<NamedAttribute, 8> attributes;
    filterFuncAttributes(op, attributes);

    auto newTTFunc = rewriter.create<triton::FuncOp>(
        op.getLoc(), op.getName(), ttFuncType, attributes);

    cast<FunctionOpInterface>(newTTFunc.getOperation())
        .setVisibility(op.getVisibility());
    newTTFunc->setAttr(hivm::TFuncCoreTypeAttr::name,
        hivm::TFuncCoreTypeAttr::get(
            newTTFunc->getContext(), hivm::TFuncCoreType::AIV));

    // Reset shared and fractal_layout attribute for converged tt function argument
    for (auto idx : sharedIds) {
      if (idx)
        newTTFunc.setArgAttr(result.getInputMapping(*idx)->inputNo,
            SharedMemoryAttr::name,
            rewriter.getUnitAttr());
    }

    for (auto idx_attr : fractalAttrArgIds) {
      newTTFunc.setArgAttr(result.getInputMapping(idx_attr.first)->inputNo,
          "hivm.fractal_layout",
          idx_attr.second);
    }

    auto *newEntryBlock = newTTFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(newEntryBlock);
    IRMapping argMapper;

    // Update block argument types in new tt.func and build the map from old
    // block argument to new block argument.
    auto newArgs = newEntryBlock->getArguments();
    auto &oldEntryBlock = op.getBody().front();
    DenseMap<Value, SmallVector<Value>> argDescriptors;
    for (auto [idx, oldArg] : llvm::enumerate(oldEntryBlock.getArguments())) {
      auto mapping = result.getInputMapping(idx);
      if (!mapping)
        continue;
      if (auto memrefTy = mlir::dyn_cast<MemRefType>(oldArg.getType())) {
        SmallVector<Value> fields;
        if (mapping->size == getDescriptorSize(memrefTy.getRank())) {
          // Descriptor calling convention: the fields are the arguments.
          for (unsigned i = 0; i < mapping->size; ++i)
            fields.push_back(newArgs[mapping->inputNo + i]);
        } else {
          // Bare-pointer calling convention: rebuild the descriptor from the type
          SmallVector<int64_t> staticStrides;
          int64_t staticOffset = 0;
          if (failed(getStridesAndOffset(memrefTy, staticStrides, staticOffset)))
            return rewriter.notifyMatchFailure(
                op, "bare-pointer MemRef argument has no strided layout");
          auto isDyn = [](int64_t v) { return ShapedType::isDynamic(v); };
          if (!memrefTy.hasStaticShape() || isDyn(staticOffset) ||
              llvm::any_of(staticStrides, isDyn))
            return rewriter.notifyMatchFailure(
                op, "bare-pointer MemRef argument has a dynamic layout field");
          Value ptr = newArgs[mapping->inputNo];
          auto i64 = [&](int64_t v) -> Value {
            return rewriter.create<arith::ConstantIntOp>(op.getLoc(), v, 64);
          };
          fields.push_back(ptr);
          fields.push_back(ptr);
          fields.push_back(i64(staticOffset));
          for (int64_t dim : memrefTy.getShape())
            fields.push_back(i64(dim));
          for (int64_t stride : staticStrides)
            fields.push_back(i64(stride));
        }
        argDescriptors[oldArg] = fields;
        argMapper.map(oldArg, fields[1]);
      } else if (isa<IndexType>(oldArg.getType())) {
        auto narrowedArg = narrowABIIndexArg(rewriter, op.getLoc(),
            newArgs[mapping->inputNo], oldArg.getType());
        argMapper.map(oldArg, narrowedArg);
      } else {
        argMapper.map(oldArg, newArgs[mapping->inputNo]);
      }
    }

    // Clone all of operators in entry block recursively.
    // Note: There is only one top block named entry block in ttir
    assert(op.getBody().getBlocks().size() == 1 &&
           "Multi blocks are not supported");
    for (auto &oldOp : oldEntryBlock.getOperations()) {
      // Replace the func.return with tt.return
      if (isa<func::ReturnOp>(oldOp)) {
        rewriter.create<triton::ReturnOp>(op.getLoc());
        continue;
      }
      // Wire the descriptor placeholder's results onto the real fields instead
      // of cloning the cast itself.
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(oldOp)) {
        if (resolveDescriptorPlaceholder(cast, argDescriptors, argMapper))
          continue;
      }
      rewriter.clone(oldOp, argMapper);
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::hivm::populateFuncToTritonPatterns(
    TritonTypeConverter &converter, RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<FuncOpPattern>(converter, context);
}
