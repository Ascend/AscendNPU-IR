//===- TensorConstructors.cpp - Conversion impl. for Constructor Ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===-----------------------------------------------------------------------===//

#include "bishengir/Conversion/TorchToHFusion/PopulatePatterns.h"
#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToHFusion/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APInt.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertAtenArangeStartStepOp
    : public OpConversionPattern<AtenArangeStartStepOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenArangeStartStepOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    bool pinMemory;
    if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return op.emitError(
          "unimplemented: pin_memory must be either None or false");
    }

    auto getConstantIntegerFromDefiningOp = [](Value operand,
                                               int &extractedInt) {
      auto castOp =
          dyn_cast<mlir::UnrealizedConversionCastOp>(operand.getDefiningOp());
      if (!castOp) {
        return failure();
      }

      auto constOp = castOp.getOperand(0).getDefiningOp<Torch::ConstantIntOp>();
      if (!constOp) {
        return failure();
      }
      extractedInt = constOp.getValue();
      return success();
    };

    // With static shape, dynamo will decompose arange.start_step with
    // arange.start_step(step=1), mul and sub op when step is not equal to 1.
    // Besides, npu_inductor_mlir will make it fallback with dynamic shape.
    // So here we just need to ensure the step is int(1) and raise error in
    // other cases.
    int stepAttr;
    if (failed(getConstantIntegerFromDefiningOp(adaptor.getStep(), stepAttr))) {
      return op->emitError("Step should be constant");
    }

    if (stepAttr != 1) {
      auto err_msg = "Step should be 1, bug got " + std::to_string(stepAttr);
      return op->emitError(err_msg);
    }

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = this->getTypeConverter();
    RankedTensorType resultType = cast<RankedTensorType>(
        typeConverter->convertType(op->getResult(0).getType()));
    Type dtype = resultType.getElementType();
    Value start =
        convertScalarToDtype(rewriter, loc, adaptor.getStart(), dtype);
    Value end = convertScalarToDtype(rewriter, loc, adaptor.getEnd(), dtype);
    Value step = convertScalarToDtype(rewriter, loc, adaptor.getStep(), dtype);

    Value resultShape;
    if (isa<mlir::IntegerType>(dtype)) {
      Value subOut = rewriter.create<arith::SubIOp>(loc, end, start);
      resultShape = rewriter.create<arith::CeilDivSIOp>(loc, subOut, step);
    } else {
      Value subOut = rewriter.create<arith::SubFOp>(loc, end, start);
      Value divOut = rewriter.create<arith::DivFOp>(loc, subOut, step);
      Value ceilOut = rewriter.create<math::CeilOp>(loc, divOut);
      resultShape =
          rewriter.create<arith::FPToUIOp>(loc, rewriter.getI64Type(), ceilOut);
    }
    resultShape = castIntToIndex(rewriter, loc, resultShape);

    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultShape), dtype);

    auto arange =
        rewriter.create<hfusion::ArangeOp>(loc, resultTensor).getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, arange);
    return success();
  }
};

} // namespace

void mlir::populateTensorConstructorsPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  // Add some legal ops for torch-torch lowering.
  target.addLegalOp<ConstantIntOp>();
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenArangeStartStepOp>();
  patterns.add<ConvertAtenArangeStartStepOp>(typeConverter, context);
}
