//===- AutoInferBufferSize.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
