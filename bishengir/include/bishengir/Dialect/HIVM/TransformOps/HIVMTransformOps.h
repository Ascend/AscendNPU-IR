//===- HIVMTransformOps.h - HIVM transform ops -------------------*- C++-*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// HIVM Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h.inc"

namespace mlir {
namespace hivm {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS_H
