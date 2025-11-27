//===- GetPipe.cpp - Get pipe implementation ------------------------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

PIPE CopyOp::getPipe() {
  assert(hasPureBufferSemantics() && "Operating on tensor, please bufferize.");
  MemRefType srcMemrefType = dyn_cast<MemRefType>(getSrcOperandType());
  MemRefType dstMemrefType = dyn_cast<MemRefType>(getDstOperandType());
  auto srcMemSpaceAttr = srcMemrefType.getMemorySpace();
  auto dstMemSpaceAttr = dstMemrefType.getMemorySpace();
  assert(srcMemSpaceAttr && "Source should have memory space by now.");
  assert(dstMemSpaceAttr && "dst should have memory space by now.");

  const DenseMap<std::pair<AddressSpace, AddressSpace>, PIPE> kSrcDstSpace2Pipe{
      {std::make_pair(AddressSpace::UB, AddressSpace::UB), PIPE::PIPE_V},
      {std::make_pair(AddressSpace::L0C, AddressSpace::GM), PIPE::PIPE_FIX},
      {std::make_pair(AddressSpace::GM, AddressSpace::L1), PIPE::PIPE_MTE2},
      {std::make_pair(AddressSpace::UB, AddressSpace::L1), PIPE::PIPE_MTE3},
  };

  auto nowSrcDstSpace =
      std::make_pair(cast<AddressSpaceAttr>(srcMemSpaceAttr).getAddressSpace(),
                     cast<AddressSpaceAttr>(dstMemSpaceAttr).getAddressSpace());
  auto iter = kSrcDstSpace2Pipe.find(nowSrcDstSpace);
  if (iter != kSrcDstSpace2Pipe.end()) {
    return iter->second;
  }
  llvm_unreachable("Unknown PIPE!");
}

//===----------------------------------------------------------------------===//
// VBrcOp
//===----------------------------------------------------------------------===//

PIPE VBrcOp::getPipe() {
  Type dstType = this->getDst().getType();
  if (getHIVMAddressSpace(dstType) == hivm::AddressSpace::L1) {
    return PIPE::PIPE_MTE2;
  }
  if (getHIVMAddressSpace(dstType) == hivm::AddressSpace::UB) {
    return PIPE::PIPE_V;
  }
  llvm_unreachable("Unknown PIPE!");
}
