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
static Type getTritonABIType(Type type) {
  if (isa<IndexType>(type))
    return IntegerType::get(type.getContext(), 64);
  return HIVMToTritonTypeConvert(type);
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
    auto oldFuncTy = op.getFunctionType();
    // Convert function type which contains parameter types
    // Note: The function can only be simt vf, and no return value.
    SmallVector<Type> newInputTypes;
    std::optional<int> sharedIdx = std::nullopt;
    // Collect fractal layout attributes to propagate to expanded args.
    SmallVector<std::pair<int, Attribute>> fractalArgAttrs;
    int newArgCounter = 0;
    static constexpr int FixCount = 3;
    for (auto [idx, inputTy] : llvm::enumerate(oldFuncTy.getInputs())) {
      if (auto memrefTy = dyn_cast<MemRefType>(inputTy)) {
        // The following conversion logic should be consistent with
        // 'LLVMTypeConverter::convertMemRefType'
        auto ptrTy = HIVMToTritonTypeConvert(inputTy);
        auto indexTy = rewriter.getI64Type();

        // Expand the memref descriptor into TTIR-friendly scalars:
        //   base ptr, aligned ptr, offset, sizes[rank], strides[rank].
        // Multi-dimensional memrefs need per-dimension metadata here because
        // TTIR cannot carry the original LLVM descriptor struct directly.
        newInputTypes.push_back(ptrTy);
        newInputTypes.push_back(ptrTy);
        newInputTypes.push_back(indexTy);

        auto rank = memrefTy.getRank();
        for (int i = 0; i < rank; ++i)
          newInputTypes.push_back(indexTy); // sizes
        for (int i = 0; i < rank; ++i)
          newInputTypes.push_back(indexTy); // strides

        // Record the index of the shared memory base pointer.
        if (op.getArgAttr(idx, SharedMemoryAttr::name)) {
          if (sharedIdx) {
            llvm::report_fatal_error("Duplicate shared memory base pointer");
          }
          sharedIdx = newArgCounter;
        }
        // Record fractal layout attribute for propagation.
        if (auto fractalAttr = op.getArgAttr(idx, "hivm.fractal_layout")) {
          fractalArgAttrs.push_back({newArgCounter, fractalAttr});
        }
        newArgCounter += FixCount;
        newArgCounter += rank;
        newArgCounter += rank;
      } else {
        newInputTypes.push_back(getTritonABIType(inputTy));
        newArgCounter++;
      }
    }

    auto funcType = FunctionType::get(op.getContext(), newInputTypes,
                                      oldFuncTy.getResults());
    auto newFunc =
        rewriter.create<triton::FuncOp>(op.getLoc(), op.getName(), funcType);
    if (sharedIdx) {
      newFunc.setArgAttr(*sharedIdx, SharedMemoryAttr::name,
                         rewriter.getUnitAttr());
    }
    for (auto &[argIdx, attr] : fractalArgAttrs) {
      newFunc.setArgAttr(argIdx, "hivm.fractal_layout", attr);
    }
    auto *newEntryBlock = newFunc.addEntryBlock();

    rewriter.setInsertionPointToStart(newEntryBlock);
    IRMapping argMapper;

    // Update block argument types in new tt.func and build the map from old
    // block argument to new block argument
    int argIdx = 0;
    auto newArgs = newEntryBlock->getArguments();
    auto &oldEntryBlock = op.getBody().front();
    for (auto [idx, oldArg] : llvm::enumerate(oldEntryBlock.getArguments())) {
      if (auto memrefTy = mlir::dyn_cast<MemRefType>(oldArg.getType())) {
        auto dataPtr1 = newArgs[argIdx];
        auto rank = memrefTy.getRank();
        // Skip over the full rank-aware descriptor emitted above; the cloned
        // body still models the original memref value through its data pointer.
        argIdx += FixCount;
        argIdx += rank;
        argIdx += rank;
        argMapper.map(oldArg, dataPtr1);
      } else if (isa<IndexType>(oldArg.getType())) {
        auto narrowedArg =
            narrowABIIndexArg(rewriter, op.getLoc(), newArgs[argIdx++],
                              oldArg.getType());
        argMapper.map(oldArg, narrowedArg);
      } else {
        argMapper.map(oldArg, newArgs[argIdx++]);
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

void mlir::hivm::populateFuncToTritonPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<FuncOpPattern>(context);
}
