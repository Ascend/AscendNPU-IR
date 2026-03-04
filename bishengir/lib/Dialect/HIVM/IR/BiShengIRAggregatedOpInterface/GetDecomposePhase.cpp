//===- GetDecomposePhase.cpp - GetDecomposePhase implementations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

using namespace bishengir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// VBrcOp
//===----------------------------------------------------------------------===//

DecomposePhase VBrcOp::getDecomposePhase() {
  return DecomposePhase::AFTER_RECOGNIZE_BROADCAST;
}

//===----------------------------------------------------------------------===//
// VConcatOp
//===----------------------------------------------------------------------===//

DecomposePhase VConcatOp::getDecomposePhase() {
  return DecomposePhase::AFTER_HIVM_STRIDE_ALIGNMENT;
}

//===----------------------------------------------------------------------===//
// VDeinterleaveOp
//===----------------------------------------------------------------------===//

DecomposePhase VDeinterleaveOp::getDecomposePhase() {
  return DecomposePhase::AFTER_RECOGNIZE_DEINTERLEAVE;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

DecomposePhase LoadOp::getDecomposePhase() {
   return DecomposePhase::NO_CONSTRAINT;
}

//===----------------------------------------------------------------------===//
// ND2NZOp
//===----------------------------------------------------------------------===//

DecomposePhase ND2NZOp::getDecomposePhase() {
  return DecomposePhase::AFTER_INFER_HIVM_DATA_LAYOUT;
}

//===----------------------------------------------------------------------===//
// VPadOp
//===----------------------------------------------------------------------===//

DecomposePhase VPadOp::getDecomposePhase() {
  return DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

DecomposePhase VReduceOp::getDecomposePhase() {
  return DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
}
