//===- VectorizableOpInterface.h ------------------------------------------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::hivm {
/// Default `VectorizableOpInterface::vectorize` implementation for HIVM
/// elementwise ops. Returns failure for ops without an elementwise lowering.
LogicalResult vectorizeElementwise(Operation *op, RewriterBase &rewriter,
                                   ArrayRef<int64_t> vectorSizes);

/// Returns true for HIVM ops that currently lower through
/// `VectorizableOpInterface::vectorize`.
bool canVectorizeHIVMOp(Operation *op);
} // namespace mlir::hivm

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_H
