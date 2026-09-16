//===- AutoInferBufferSize.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
#define GEN_PASS_DEF_AUTOINFERBUFFERSIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct AutoInferBufferSizePass
    : public impl::AutoInferBufferSizeBase<AutoInferBufferSizePass> {
  void runOnOperation() override;
};

} // namespace

static bool isAnnotatedWithSize(Value val) {
  for (auto *userOp : val.getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(userOp)) {
      if (markOp->hasAttrOfType<IntegerAttr>(kBufferSizeInByteAttr)) {
        return true;
      }
    }
  }
  return false;
}

static void insertAnnotation(Operation *allocOp, Value val,
                             int64_t bufferSizeInByte) {
  OpBuilder b(allocOp);
  b.setInsertionPointAfter(allocOp);
  auto newMarkOp = b.create<annotation::MarkOp>(allocOp->getLoc(), val);
  newMarkOp->setAttr(kBufferSizeInByteAttr,
                     b.getI64IntegerAttr(bufferSizeInByte));
}

void AutoInferBufferSizePass::runOnOperation() {
  auto funcOp = getOperation();
  if (!funcOp->hasAttr(utils::kEnableAutoMarkBufferSize)) {
    return;
  }

  int64_t numOfElements = -1;
  funcOp->walk([&](annotation::MarkOp markOp) {
    // given that the number of elements in the buffer is the same,
    // bail out if numOfElements has been calculated
    if (numOfElements != -1 ||
        !markOp->hasAttrOfType<IntegerAttr>(kBufferSizeInByteAttr)) {
      return;
    }
    int64_t bufferSizeInBit =
        markOp->getAttrOfType<IntegerAttr>(kBufferSizeInByteAttr).getInt() *
        mlir::utils::kBitsToByte;
    int64_t elementWidthInBit =
        getElementTypeOrSelf(markOp.getSrc().getType()).getIntOrFloatBitWidth();
    numOfElements = bufferSizeInBit / elementWidthInBit;
  });
  // no annotation.markOp with buffer size found
  if (numOfElements == 1) {
    return;
  }
  // infer memref.alloc Ops that are not annotated
  funcOp->walk([&](memref::AllocOp allocOp) {
    auto memrefVal = allocOp->getResults()[0];
    auto memrefTy = cast<MemRefType>(memrefVal.getType());
    if (memrefTy.hasStaticShape()) {
      return;
    }
    // if there is an annotation.mark with buffer_size_in_byte attr, bail out
    if (isAnnotatedWithSize(memrefVal)) {
      return;
    }
    int64_t elementWidthInBit =
        getElementTypeOrSelf(memrefTy).getIntOrFloatBitWidth();
    int64_t bufferSizeInByte =
        numOfElements * elementWidthInBit / mlir::utils::kBitsToByte;
    insertAnnotation(allocOp, memrefVal, bufferSizeInByte);
  });
}

std::unique_ptr<Pass> mlir::hivm::createAutoInferBufferSizePass() {
  return std::make_unique<AutoInferBufferSizePass>();
}
