//===- HIVMVectorize.h - HIVM dialect trait definitions ---------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMVECTORIZE_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMVECTORIZE_H

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

namespace mlir::hivm {
enum class VectorArithKind {
  ADD,
  SUB,
  MUL,
  DIV,
  MAX,
  MIN,
};

/// @brief Creates the identity element for a given arithmetic operation and element type.
///
/// The identity element is a neutral value that, when used with the specified
/// operation, does not change the result (e.g., 0 for addition, 1 for multiplication).
/// This is used as the padding value in vector.transfer_read operations to ensure
/// that masked-off lanes do not affect the computation result.
///
/// @param builder OpBuilder used to create the constant operation
/// @param loc Location information for the created constant
/// @param elemType Element type (must be FloatType or IntegerType)
/// @param kind The arithmetic operation kind that determines which identity element to use
///
/// @return A Value representing the identity element constant for the given operation:
///         - ADD/SUB: Returns 0 (additive identity)
///         - MUL/DIV: Returns 1 (multiplicative identity)
///         - MAX: Returns $-\infty$ for floats, signed minimum for integers
///         - MIN: Returns $+\infty$ for floats, signed maximum for integers
///
/// @throws llvm_unreachable if elemType is neither FloatType nor IntegerType
Value getIdentityElement(OpBuilder &builder, Location loc, Type elemType,
                         VectorArithKind kind);


Value createVectorArithOp(OpBuilder &builder, Location loc,
                          VectorArithKind kind, Value lhs, Value rhs);
}

#endif