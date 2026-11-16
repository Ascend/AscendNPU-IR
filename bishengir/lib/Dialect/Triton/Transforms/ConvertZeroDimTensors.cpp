//===- ConvertZeroDimTensors.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// As the TritonToTritonGPU Conversion Pass is not able to handle zero dim
// tensors: ie, tensor<f32>, this pass converts all instances of 0D tensors
// (tensor<T>) to 1D tensors of shape tensor<1xT>
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace bishengir::triton {
#define GEN_PASS_DEF_CONVERTZERODIMTENSORS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

bool isSupportedZeroDimTensorOp(Operation *op) {
  if (isa<tensor::EmptyOp, tensor::ExtractOp, tensor::InsertOp>(op)) {
    return true;
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    DenseElementsAttr constValue =
        dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!constValue || !constValue.isSplat()) {
      return false;
    }
    return true;
  }

  if (isa<scf::ForOp, scf::IfOp, scf::YieldOp>(op)) {
    return true;
  }

  if (isa<triton::SplatOp, triton::StoreOp, triton::LoadOp, triton::BroadcastOp,
          triton::ReshapeOp>(op)) {
    return true;
  }

  return op->hasTrait<OpTrait::Elementwise>();
}

bool hasZeroDimTensor(Operation *op) {
  for (Value result : op->getResults()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      if (tensorType.getRank() == 0) {
        return true;
      }
    }
  }

  for (Value result : op->getOperands()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      if (tensorType.getRank() == 0) {
        return true;
      }
    }
  }

  if (auto scfForOp = dyn_cast<scf::ForOp>(op)) {
    for (BlockArgument regionArg : scfForOp.getRegionIterArgs()) {
      if (auto tensorType = dyn_cast<RankedTensorType>(regionArg.getType())) {
        if (tensorType.getRank() == 0) {
          return true;
        }
      }
    }
  }

  return false;
}

void updateValueWithElementType(IRRewriter &rewriter, Value value,
                                Type elementType) {
  RankedTensorType newType = RankedTensorType::get({1}, elementType);
  Operation *parentOp;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    parentOp = blockArg.getOwner()->getParentOp();
  } else {
    parentOp = value.getDefiningOp();
  }

  rewriter.modifyOpInPlace(parentOp, [&]() { value.setType(newType); });
}

void updateValue(IRRewriter &rewriter, Value value) {
  Type curType = value.getType();
  RankedTensorType tensorType = dyn_cast<RankedTensorType>(curType);
  if (!tensorType || tensorType.getRank() > 0) {
    return;
  }
  Type elementType = tensorType.getElementType();

  updateValueWithElementType(rewriter, value, elementType);
}

void updateValues(IRRewriter &rewriter, ValueRange values) {
  for (Value value : values) {
    updateValue(rewriter, value);
  }
}

void updateOp(IRRewriter &rewriter, Operation *op) {
  // If an op is elementwise, update its return shape to operand shape
  if (op->hasTrait<OpTrait::Elementwise>()) {
    Type curRetType = op->getResult(0).getType();

    Type retElementType;
    if (auto retTensorType = dyn_cast<RankedTensorType>(curRetType)) {
      retElementType = retTensorType.getElementType();
    } else {
      retElementType = curRetType;
    }

    Type retType;

    if (auto operandType =
            dyn_cast<RankedTensorType>(op->getOperand(0).getType())) {
      retType = RankedTensorType::get(operandType.getShape(), retElementType);
    } else {
      retType = retElementType;
    }

    rewriter.modifyOpInPlace(op, [&]() {
      for (Value result : op->getResults()) {
        result.setType(retType);
      }
    });
    return;
  }
  rewriter.setInsertionPoint(op);
  Location loc = op->getLoc();

  TypeSwitch<Operation *>(op)
      .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
        updateValue(rewriter, constOp.getResult());
        RankedTensorType newType = cast<RankedTensorType>(constOp.getType());
        DenseElementsAttr oldValue =
            cast<DenseElementsAttr>(constOp.getValue());
        Attribute scalarAttr = oldValue.getSplatValue<Attribute>();
        DenseElementsAttr newValue =
            DenseElementsAttr::get(newType, scalarAttr);
        constOp.setValueAttr(newValue);
      })
      .Case<tensor::InsertOp>([&](tensor::InsertOp insertOp) {
        Value constZero =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
        insertOp.getIndicesMutable().append(constZero);
        updateValue(rewriter, insertOp.getResult());
      })
      .Case<tensor::ExtractOp>([&](tensor::ExtractOp extractOp) {
        Value constZero =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
        extractOp.getIndicesMutable().append(constZero);
      })
      .Case<scf::ForOp>([&](scf::ForOp forOp) {
        updateValues(rewriter, forOp.getRegionIterArgs());
        updateValues(rewriter, forOp->getResults());
      })
      .Default([&](Operation *) {
        // Default to updating the type of the results to a 1D tensor if they
        // are 0D tensors
        updateValues(rewriter, op->getResults());
      });
}

void rewriteOps(IRRewriter &rewriter, SmallVector<Operation *> zeroDimOps) {
  for (Operation *op : zeroDimOps) {
    updateOp(rewriter, op);
  }
}

class ConvertZeroDimTensorsPass
    : public impl::ConvertZeroDimTensorsBase<ConvertZeroDimTensorsPass> {
public:
  using ConvertZeroDimTensorsBase::ConvertZeroDimTensorsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<Operation *> zeroDimProducers;

    bool failed = false;

    mod.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (hasZeroDimTensor(op)) {
        if (isSupportedZeroDimTensorOp(op)) {
          zeroDimProducers.push_back(op);
        } else {
          failed = true;
          op->emitError("Operation " + op->getName().getStringRef() +
                        " with a 0D tensor is not supported");
        }
      }
    });

    if (failed) {
      mod.emitError("Unsupported operations with 0D tensors found in the "
                    "SIMT kernel!");
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(mod);
    rewriteOps(rewriter, zeroDimProducers);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertZeroDimTensorsPass() {
  return std::make_unique<ConvertZeroDimTensorsPass>();
}

} // namespace bishengir::triton
