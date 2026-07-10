//===- LocalMatmulLikeOpInterface.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hivm {
class DataLayoutAttr;
enum class MatmulBiasMode : uint32_t;

namespace detail {

FailureOr<DataLayoutAttr> getLocalMatmulOperandALayoutImpl(Operation *op);
FailureOr<DataLayoutAttr> getLocalMatmulOperandBLayoutImpl(Operation *op);
FailureOr<DataLayoutAttr> getLocalMatmulOperandCLayoutImpl(Operation *op);
llvm::SmallDenseMap<Value, DataLayoutAttr>
getLocalMatmulOperandsCurrentLayoutImpl(Operation *op);
llvm::SmallDenseMap<Value, DataLayoutAttr>
getLocalMatmulOperandsTargetLayoutImpl(Operation *op);

} // namespace detail
} // namespace hivm
} // namespace mlir

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/LocalMatmulLikeOpInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_H
