//===- RegbaseUtils.cpp - Utilities to support the HIVM dialect -----------===//
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
//
// This file implements RegBase utilities for the HIVM dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

namespace mlir {
namespace hivm {

bool isVFCall(Operation *op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    if (callOp->hasAttr(hivm::VectorFunctionAttr::name))
      return true;
  }
  return false;
}

bool isVF(func::FuncOp funcOp) {
  return funcOp->hasAttr(hivm::VectorFunctionAttr::name);
}

} // namespace hivm
} // namespace mlir
