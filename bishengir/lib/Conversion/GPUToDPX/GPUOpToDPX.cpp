//===--GPUOpToDPX.cpp - GPU Op to DPX Conversion ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/GPUToDPX/GPUOpToDPX.h"
#include "bishengir/Conversion/GPUToDPX/IndexIntrinsicsOpLowering.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

static Value castValueToType(OpBuilder &builder, Location loc, Value val,
                             Type targetType) {
  OpBuilder::InsertionGuard g(builder);
  if (!val)
    return nullptr;
  Type srcType = val.getType();
  if (srcType == targetType)
    return val;
  // int
  if (auto srcInt = dyn_cast<IntegerType>(srcType)) {
    if (auto dstInt = dyn_cast<IntegerType>(targetType)) {
      unsigned s = srcInt.getWidth();
      unsigned d = dstInt.getWidth();
      if (s == d)
        return builder.create<LLVM::BitcastOp>(loc, targetType, val);
      if (s > d)
        return builder.create<LLVM::TruncOp>(loc, targetType, val);
      return builder.create<LLVM::ZExtOp>(loc, targetType, val);
    }
  }
  // vector types: cast elementwise
  if (auto srcVec = dyn_cast<VectorType>(srcType)) {
    if (auto dstVec = dyn_cast<VectorType>(targetType)) {
      int ne = srcVec.getNumElements();
      Value out = builder.create<LLVM::UndefOp>(loc, targetType);
      for (int i = 0; i < ne; ++i) {
        Value idx = builder.create<LLVM::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(i));
        Value elt = builder.create<LLVM::ExtractElementOp>(
            loc, srcVec.getElementType(), val, idx);
        Value castedElt =
            castValueToType(builder, loc, elt, dstVec.getElementType());
        out = builder.create<LLVM::InsertElementOp>(loc, targetType, out,
                                                    castedElt, idx);
      }
      return out;
    }
  }
  return builder.create<LLVM::BitcastOp>(loc, targetType, val);
}

Value getGridDims(LLVM::LLVMFuncOp funcOp, OpBuilder builder, Location loc, gpu::Dimension Dim) {
  OpBuilder::InsertionGuard g(builder);
  Type int64Ty = builder.getI64Type();
  for (auto [idx, value] : llvm::enumerate(funcOp.getArguments())) {
    auto dictAttr = funcOp.getArgAttrDict(idx);
    if (!dictAttr)
      continue;

    Attribute attr = dictAttr.get(gpu::GPUBlockMappingAttr::name);
    auto blockMappingAttr = dyn_cast_or_null<gpu::GPUBlockMappingAttr>(attr);
    if (!blockMappingAttr)
      continue;

    gpu::MappingId mapping = blockMappingAttr.getBlock();
    Value gridDim = value;
    // Target type is i64, cast if needed
    if (value.getType() != int64Ty) {
      gridDim = castValueToType(builder, loc, value, int64Ty);
    }

    if ([&]() {
        switch (Dim) {
          case gpu::Dimension::x: return mapping == gpu::MappingId::DimX;
          case gpu::Dimension::y: return mapping == gpu::MappingId::DimY;
          case gpu::Dimension::z: return mapping == gpu::MappingId::DimZ;
        }
        return false;
      }()) {
        return gridDim;
    }
  }
}

struct GPUGridDimOpOpLowering : public ConvertOpToLLVMPattern<gpu::GridDimOp> {
  using ConvertOpToLLVMPattern<gpu::GridDimOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::GridDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      llvm::dbgs() << op << "\n" << "not in LLVMFunc\n";
      return success();
    }

    Value newVal = getGridDims(funcOp, rewriter, op.getLoc(), op.getDimension());

    rewriter.replaceOp(op, newVal);
    return success();
  }
};

struct LinearBlockIdOPConversion : public ConvertOpToLLVMPattern<gpu::LinearBlockIdOp> {
  using ConvertOpToLLVMPattern<gpu::LinearBlockIdOp>::ConvertOpToLLVMPattern;
private:
  unsigned indexBitwidth;

public:
  explicit LinearBlockIdOPConversion(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::LinearBlockIdOp>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()) {}

  LogicalResult
  matchAndRewrite(gpu::LinearBlockIdOp op, OpAdaptor v,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Operation *newOp = rewriter.create<ascend_dpx::BlockIdxOp>(loc, IntegerType::get(context, 32));
    
    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct BarrierOPConversion : public OpConversionPattern<gpu::BarrierOp> {
  using OpConversionPattern<gpu::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ascend_dpx::SyncThreadsOp>(op);
    return success();
  }
};

namespace mlir::triton::ascend {
void populateGPUOpToDPXPatterns(LLVMTypeConverter &converter,
                                RewritePatternSet &patterns,
                                PatternBenefit benefit) {
  patterns.add<
      gpu::index_lowering::OpLowering<gpu::ThreadIdOp, ascend_dpx::ThreadIdXOp,
                                      ascend_dpx::ThreadIdYOp,
                                      ascend_dpx::ThreadIdZOp>,
      gpu::index_lowering::OpLowering<gpu::BlockDimOp, ascend_dpx::BlockDimXOp,
                                      ascend_dpx::BlockDimYOp,
                                      ascend_dpx::BlockDimZOp>,
      gpu::index_lowering::OpLowering<gpu::BlockIdOp, ascend_dpx::BlockIdxXOp,
                                      ascend_dpx::BlockIdxYOp,
                                      ascend_dpx::BlockIdxZOp>,
      GPUGridDimOpOpLowering, LinearBlockIdOPConversion>(converter);

  patterns.add<BarrierOPConversion>(converter, patterns.getContext(), benefit);
}
} // namespace mlir::triton::ascend