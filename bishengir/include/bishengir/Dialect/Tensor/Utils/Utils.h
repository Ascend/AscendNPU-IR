//===- Utils.h - Utilities to support the Tensor dialect ----------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

#ifndef MLIR_DIALECT_TENSOR_UTILS_UTILS_H
#define MLIR_DIALECT_TENSOR_UTILS_UTILS_H

namespace mlir {
namespace tensor {
namespace reshape_utils {

enum class ElementKind { HasMutation, NoMutation, Unit };

// helper func: stringtify ElementKind
std::string stringtifyElementKind(ElementKind kind);

// overloaded << operator to support cout << ElementKind
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ElementKind kind);

using namespace llvm;
// The structure to hold the mapping for each dimension.
struct HyperrectangularSlice {
  int64_t dimension; // which dimension (0-based)
  int64_t offset;    // the starting coordinate in that dimension
  int64_t size;      // number of indices in that dimension
  int64_t stride;    // the mapping’s stride in that dimension
  explicit HyperrectangularSlice(int d, int o, int s, int st)
      : dimension(d), offset(o), size(s), stride(st){};
};

using Hyperrectangle = SmallVector<HyperrectangularSlice>;

std::optional<Hyperrectangle>
getExtendHyperrectangleFromArray(int64_t oldShape, int64_t offset, int64_t size,
                           int64_t stride, ArrayRef<int64_t> staticNewShape);

/// Creates an inverse ExpandShapeOp for a given CollapseShapeOp.
/// This allows undoing the effect of a collapse operation.
///
/// @param builder The builder to use for creating operations
/// @param collapseOp The CollapseShapeOp to invert
/// @return A new ExpandShapeOp that inverts the collapse
tensor::ExpandShapeOp createCollapseInverse(OpBuilder &builder,
                                            tensor::CollapseShapeOp collapseOp);

/// Creates an inverse CollapseShapeOp for a given ExpandShapeOp.
/// This allows undoing the effect of an expand operation.
///
/// @param builder The builder to use for creating operations
/// @param expandOp The ExpandShapeOp to invert
/// @return A new CollapseShapeOp that inverts the expand
tensor::CollapseShapeOp createExpandInverse(OpBuilder &builder,
                                            tensor::ExpandShapeOp expandOp);

/// Given a list of reshape ops, construct a reverse reshaped value of
/// the initial value.
///
/// For example, trace sequence is
/// ```mlir
///    %tmp = reshape0 %src,
///    %res = reshape1 %tmp
/// ```
///
/// Then, then the output is the reverse of `initialValue`:
/// ```mlir
///   %tmp' = reverse_reshape1 %initialValue
///   %output = reverse_reshape0 %tmp'
/// ```
/// Note: the trace sequence is store in reverse order.
Value getReverseReshapedValue(OpBuilder &builder, Value initialValue,
                              const SmallVector<Operation *> &trace);
} // namespace reshape_utils

/// @brief Reifies shape values by tracing them back to block arguments and
/// updating operand chains.
///
/// This function recursively traces a value back through its defining
/// operations to establish connections to block arguments. It handles tensor
/// dimension operations specially and maintains a cache of settled values to
/// avoid redundant processing. The function also updates operand chains to use
/// the reified values.
///
/// @param initialVal The initial value to reify and trace back to arguments
/// @param opOpr Optional operand pointer that should be updated with the
/// reified value
/// @param builder OpBuilder used for creating new operations during reification
/// @param settled Cache map storing already processed value mappings to avoid
/// recomputation
/// @return Value The reified value that traces back to a block argument
Value reifyShapeToArg(Value initialVal, std::optional<OpOperand *> opOpr,
                      OpBuilder &builder, DenseMap<Value, Value> &settled);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_UTILS_UTILS_H
