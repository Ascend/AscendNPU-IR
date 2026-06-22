//===- HIVMTraits.td - HIVM dialect trait definitions -------*- tablegen-*-===//
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
// This is the definition file for HIVM dialect traits.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMTRAITS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMTRAITS_TD

include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// HIVM Dialect operation traits.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HIVM Dialect operation traits.
//===----------------------------------------------------------------------===//

/// Op Trait for verification that Op has the same ranks for all operands and
/// results types except temp buffer, if known.
def HIVMOpSameOperandsAndResultRank :
  NativeOpTrait<"HIVMOpSameOperandsAndResultRank", [HIVMStructuredOpInterface]>;

/// Op Trait for Elementwise N-ary Vector Ops.
/// - Semantic: Operation performs elementwise operation on one or more input
///             operands and produces a single result.
/// - Constraints:
///   1. Op implements DestinationStyleOpInterface; support both tensor and
///      buffer semantic.
///   2. The number of input operands is N.
///   3. Rank is equal for shaped operands and result (enforced by
///      HIVMOpSameOperandsAndResultRank).
///   4. If op has buffer semantic, the ultimate dimension's stride is equal for
///      every operand.
/// - Note:
///   1. Verification on shape and stride will only be performed for fully
///      static-shaped operands.
///   2. Constraints on element data type or whether the op support scalar
///      operands are not modeled by this trait.
/// - Dependency:
///   1. HIVMStructuredOpInterface
///   2. HIVMOpSameOperandsAndResultRank
class ElementwiseNaryOpTrait<int N>
    : ParamNativeOpTrait<"ElementwiseNaryOpTrait", !cast<string>(N),
                         [HIVMStructuredOpInterface,
                          HIVMOpSameOperandsAndResultRank]>;

/// Specify the maximum rank that library supports for current op.
/// This trait must be used for every op with HIVMStructuredOpInterface
class OpLibraryMaxRankTrait<int MaxRank>
    : ParamNativeOpTrait<"OpLibraryMaxRankTrait", !cast<string>(MaxRank)>;

/// For op with statically known max rank trait that library supports.
class StaticMaxRankTrait<int MaxRank> : OpLibraryMaxRankTrait<MaxRank>;

/// For op with statically unknown max rank trait that library supports.
def InferMaxRankTrait : OpLibraryMaxRankTrait<0>;

/// For op with no constraints on the maximum rank that library supports.
def NoMaxRankTrait : OpLibraryMaxRankTrait<-1>;

/// For op with no library function implemented.
def NoLibraryFunctionTrait : NativeOpTrait<"NoLibraryFunctionTrait">;

/// whether a op is commutative (note: only applies to the input operands)
def CommutativeOpTrait : NativeOpTrait<"CommutativeOpTrait">;

/// Op Trait for operation with operands that only support
/// shaped-typed (i.e., vector) inputs.
class VectorOnlyTrait<int idx>
    : ParamNativeOpTrait<"VectorOnlyTrait", !cast<string>(idx)>;

/// Op Trait for operation with operands that only support
/// scalar-typed inputs (for simplicity, consider non-shaped
/// types as scalar-typed).
class ScalarOnlyHWTrait<int idx>
    : ParamNativeOpTrait<"ScalarOnlyHWTrait", !cast<string>(idx)>;

/// Op Trait for operation with operands that only support certain element types.
class OperElemTypeConstraints<list<int> indices,
                              list<Type> allowedTypes>
  : PredOpTrait
    <"operand at idx " #
     !interleave(!foreach(i, indices, i), " and ") #
     " should have element type " #
     !interleave(!foreach(t, allowedTypes, t.summary), " or "),
     And<
       !foreach(idx, indices,
         And<[CPred<"$_op.getNumOperands() > " #idx>,
              SubstLeaves<"$_self",
                "getElementTypeOrSelf($_op.getOperand(" #idx #").getType())",
                AnyTypeOf<allowedTypes>.predicate>,
             ]>
         )
        >
    >;

/// Op Trait for operation that can broadcast input operands on-the-fly.
/// - Semantic:
///   ```cpp
///   for i                                        // <- broadcast dim
///     for j
///       dst[i, j] = some_op(src1[0, j], src2[0, j], ..., srcN[0, j])
///   ```
/// - Constraints:
///   1. Op implements DestinationStyleOpInterface.
///   2. Op must have a `DenseI64ArrayAttr` named `"broadcast"`, indicating the
///      dimensions to be broadcasted.
///   3. (B1) Op has buffer semantic.
///   4. (B2) Dimensions in `broadcast` are unique.
///   5. (B3) For all `d` in `broadcast`, `0 <= d < rank(dst)`.
///   6. For all `d` in `axes(dst)`:
///       - (B4) If `d` is in `broadcast`, for all `src_i` for `i` in [1, N]:
///              `dim(src_i, d) = 1 || dim(src_i, d) = dim(dst, d)`.
///       - (B5) If `d` is not in `broadcast`, for all `src_i` for `i` in [1, N]:
///              `dim(src_i, d) = dim(dst, d)`.
/// - Dependency:
///   1. HIVMStructuredOpInterface
///   2. HIVMOpSameOperandsAndResultRank
def BroadcastableOTF :
  NativeOpTrait<"BroadcastableOTF", [HIVMStructuredOpInterface,
                                     HIVMOpSameOperandsAndResultRank]>;

/// Op Trait for operation that can transpose input operands on-the-fly.
/// - Semantic:
///   ```cpp
///   for i                                        // transpose = [1, 0]
///     for j
///       dst[i, j] = some_op(src1[j, i], src2[j, i], ..., srcN[j, i])
///   ```
/// - Constraints:
///   1. Op implements DestinationStyleOpInterface.
///   2. Op must have a `DenseI64ArrayAttr` named `"transpose"`, indicating
///      the order of axes after transpose.
///   3. (T1) Op has buffer semantic.
///   4. (T2) `transpose` is a permutation of `range(rank(dst))`.
///   5. (T3) `transpose[rank(dst) - 1] = rank(dst) - 1`
///   6. For all `d` in `axes(dst)`, for all `src_i` for `i` in [1, N]:
////     - (T4) `dim(dst, d) = dim(src_i, transpose[d])`
/// - Dependency:
///   1. HIVMStructuredOpInterface
///   2. HIVMOpSameOperandsAndResultRank
def TransposableOTF :
  NativeOpTrait<"TransposableOTF", [HIVMStructuredOpInterface,
                                    HIVMOpSameOperandsAndResultRank]>;

def SinglePipeOpTrait : NativeOpTrait<"SinglePipeOpTrait">;

/// Op Trait for operation with single pipe information.
class OpPipeTrait<string Pipe>
    : ParamNativeOpTrait<"OpPipeTrait", Pipe, [SinglePipeOpTrait]>;

/// Op Trait for macro operations with multiple kinds of pipe
def MacroOpTrait : NativeOpTrait<"MacroOpTrait">;

/// Op Trait for macro operations to mark in pipe of its input operation
/// and out pipe of its out operations
class MacroOpPipeTrait<string InOutPipes>
      : ParamNativeOpTrait<"MacroOpPipeTrait", InOutPipes,
                           [MacroOpTrait]>;

class CoreTypeTrait<string CoreType>
    : ParamNativeOpTrait<"CoreTypeTrait", CoreType>;

def VectorCoreTypeTrait : CoreTypeTrait<"TCoreType::VECTOR">;
def CubeCoreTypeTrait : CoreTypeTrait<"TCoreType::CUBE">;
def CubeVectorCoreTypeTrait : CoreTypeTrait<"TCoreType::CUBE_OR_VECTOR">;

/// Operations with this trait can have their dimensions collapsed (flattened)
/// uniformly across all their operands and results. This implies that the same
/// grouping of dimensions (reassociation) can be applied to every tensor
/// involved in the operation.
def UniformReassociationFlattenTrait : NativeOpTrait<"UniformReassociationFlattenTrait", [HIVMStructuredOpInterface]>;

/// A marker trait for operations that must preserve the rank and individuality
/// of their target dimensions during flattening.
/// For example, a permutation operation might be target-preserving. If it
/// permutes dimensions `b` and `c`, these two dimensions cannot be collapsed
/// together, even if they are adjacent. This trait enforces that each target
/// dimension forms its own reassociation group
def CollapsibleConsecutiveTargetDimsTrait
    : NativeOpTrait<"CollapsibleConsecutiveTargetDimsTrait", [HIVMStructuredOpInterface, UniformReassociationFlattenTrait]>;

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMTRAITS_TD
