//===- GetDecomposePhase.cpp - GetDecomposePhase implementations ----------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

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
  return DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
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

//===----------------------------------------------------------------------===//
// VTransposeOp
//===----------------------------------------------------------------------===//

DecomposePhase VTransposeOp::getDecomposePhase() {
  return DecomposePhase::AFTER_HIVM_STRIDE_ALIGNMENT;
}

//===----------------------------------------------------------------------===//
// Binary VectorOp
//===----------------------------------------------------------------------===//
#define GET_DECOMPOSE_PHASE_BINARY_VEC_OP(opName)                                           \
  DecomposePhase opName::getDecomposePhase() {                                              \
    return DecomposePhase::AFTER_LIFT_LOWEST_STRIDE;                                        \
}

GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VAndOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VOrOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VMulOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VAddOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VSubOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VDivOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VMaxOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VMinOp)
GET_DECOMPOSE_PHASE_BINARY_VEC_OP(VXorOp)
