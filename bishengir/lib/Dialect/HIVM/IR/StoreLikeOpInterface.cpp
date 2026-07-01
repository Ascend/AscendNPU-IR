//===- InferCoreType.cpp - InferCoreType Interface Impl. ------------------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// FixpipeOp
//===----------------------------------------------------------------------===//

bool FixpipeOp::isAtomic() {
  // FixpipeOp unsupports atomic in c220
  return false;
}

bool FixpipeOp::isHWAtomic() {
  // FixpipeOp unsupports atomic implemented by hardware in c220
  return false;
}

bool FixpipeOp::isSWAtomic() {
  // FixpipeOp unsupports atomic implemented by software in c220
  return false;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

bool StoreOp::isAtomic() {
  auto atomicKind = getAtomicKind();
  return atomicKind.has_value() && atomicKind.value() != hivm::AtomicKind::NONE;
}

bool StoreOp::isHWAtomic() {
  if (getAtomicKind().has_value()) {
    auto atomicKind = getAtomicKind().value();
    return (atomicKind == hivm::AtomicKind::ADD) ||
           (atomicKind == hivm::AtomicKind::MAX) ||
           (atomicKind == hivm::AtomicKind::MIN);
  }

  return false;
}

bool StoreOp::isSWAtomic() { return isAtomic() && (!isHWAtomic()); }

//===----------------------------------------------------------------------===//
// ScatterStoreOp
//===----------------------------------------------------------------------===//

bool ScatterStoreOp::isAtomic() {
  // ScatterStoreOp unsupports atomic in c220
  return false;
}

bool ScatterStoreOp::isHWAtomic() {
  // ScatterStoreOp unsupports atomic implemented by hardware in c220
  return false;
}

bool ScatterStoreOp::isSWAtomic() {
  // ScatterStoreOp unsupports atomic implemented by software in c220
  return false;
}