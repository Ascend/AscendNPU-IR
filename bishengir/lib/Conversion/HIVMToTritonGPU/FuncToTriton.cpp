//===---- FuncToTriton.cpp - conversion from Func to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
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

class FuncOpPattern : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get memref_attr from func attributes
    auto memrefAttr =
        dyn_cast_or_null<DenseI32ArrayAttr>(op->getAttr("memref_attr"));
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
    // block argument to new block argument
    auto newArgs = newEntryBlock->getArguments();
    auto &oldEntryBlock = op.getBody().front();
    for (auto [idx, oldArg] : llvm::enumerate(oldEntryBlock.getArguments())) {
      if (auto memrefTy = mlir::dyn_cast<MemRefType>(oldArg.getType())) {
        // In SignatureConversion, inputNo is the index of memref argument
        // in tt.func signature.
        auto dataPtr1 = newArgs[result.getInputMapping(idx)->inputNo];
        Value offset;
        if (result.getInputMapping(idx)->size > 1)
          offset = newArgs[result.getInputMapping(idx)->inputNo + 2];
        bool hasOffset = memrefAttr && (int64_t)idx < memrefAttr.size()
                         && memrefAttr[idx] != 0;
        Value newPtr = dataPtr1;
        // Map the old memref data pointer to new tt.ptr
        if (hasOffset) {
          // Try to get static offset from memref type
          int64_t staticOffset;
          SmallVector<int64_t> strides;
          if (succeeded(getStridesAndOffset(memrefTy, strides, staticOffset)) &&
              staticOffset != ShapedType::kDynamic && staticOffset != 0) {
            // Static non-zero offset: create a constant and add to pointer
            auto constOffset = rewriter.create<arith::ConstantIntOp>(op.getLoc(), staticOffset, 64);
            newPtr = rewriter.create<triton::AddPtrOp>(
              op.getLoc(), dataPtr1.getType(), newPtr, constOffset);
          } else {
            // Dynamic offset: use the runtime offset argument
            newPtr = rewriter.create<triton::AddPtrOp>(
              op.getLoc(), dataPtr1.getType(), newPtr, offset);
          }
        }
        // Skip over the full rank-aware descriptor emitted above; the cloned
        // body still models the original memref value through its data pointer.
        argMapper.map(oldArg, newPtr);
      } else if (isa<IndexType>(oldArg.getType())) {
        auto narrowedArg = narrowABIIndexArg(rewriter,
            op.getLoc(),
            newArgs[result.getInputMapping(idx)->inputNo],
            oldArg.getType());
        argMapper.map(oldArg, narrowedArg);
      } else {
        argMapper.map(oldArg, newArgs[result.getInputMapping(idx)->inputNo]);
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
