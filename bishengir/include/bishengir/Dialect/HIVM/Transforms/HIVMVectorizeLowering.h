//===- HIVMVectorizeLowering.h - HIVM vectorization lowerings ---*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMVECTORIZELOWERING_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMVECTORIZELOWERING_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hivm::vectorize {

FailureOr<SmallVector<OperationName>> getExpectedHIVMLowering(Operation *op);
bool isVectorizableTarget(OperationName name);
bool hasVectorizableLowering(Operation *op);

} // namespace mlir::hivm::vectorize

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMVECTORIZELOWERING_H
