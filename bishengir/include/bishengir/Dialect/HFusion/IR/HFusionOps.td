//===- HFusionOps.td - HFusion op definitions --------------*- tablegen -*-===//
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
// This is the operation definition file for hybrid fusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "bishengir/Interfaces/AggregatedOpInterface.td"
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionAttrs.td"

// Base class for HFusion dialect ops that do not correspond to library calls.
class HFusion_Op<string mnemonic, list<Trait> traits = []> :
    Op<HFusion_Dialect, mnemonic, traits>;

// Shaped type whose element type is from the given `allowedTypes` list,
// and which additionally satisfies an optional list of predicates.
class ShapedTypeOf<list<Type> allowedTypes, list<Pred> preds = [],
                     string summary = "ranked tensor">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([IsShapedTypePred], preds)>,
      summary, "::mlir::ShapedType">;

//===----------------------------------------------------------------------===//
// HFusion Types
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

//===----------------------------------------------------------------------===//
// HFusion Debug op definitions.
//===----------------------------------------------------------------------===//

def PrintOp : HFusion_Op<"print">,
  Arguments<(ins StrAttr:$prefix, BoolAttr:$hex,
             AnyTypeOf<[AnyInteger, AnyFloat, AnyRankedTensor]>:$arg)> {
  let summary = "Device-side print for debugging";
  let description = [{
    `hfusion.print` takes a literal string `prefix` and an argument of
    scalar or tensor that should be printed. The optional arg `hex` configs
    if printing in hex format.
  }];
  let assemblyFormat = [{
    $prefix attr-dict $arg `:` type($arg)
  }];
}

def AssertOp : HFusion_Op<"assert">,
  Arguments<(ins StrAttr:$msg,
                 AnyTypeOf<[AnyInteger, AnyRankedTensor]>:$cond)> {
  let summary = "Device-side assert for debugging";
  let description = [{
    `hfusion.assert` takes a literal string `msg` and an argument of
    scalar or tensor that should be asserted.
  }];
  let assemblyFormat = [{
    $msg attr-dict $cond `:` type($cond)
  }];
}

def BarrierOp : HFusion_Op<"barrier"> {
  let summary = "Synchronizes all pipelines of a core.";
  let description = [{
    The "barrier" op synchronizes all pipelines of a core.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// SymbolicDimOp
//===----------------------------------------------------------------------===//

def SymbolicDimOp : HFusion_Op<"symbolic_dim", [Pure]> {
  let summary = "Symbolic dimension reference returning an index";
  let description = [{
    The "symbolic_dim" operation references a symbolic dimension by name
    (via a symbol attribute) and returns it as an `index`-typed value.
  }];

  let arguments = (ins SymbolRefAttr:$symbolName);
  let results = (outs Index:$result);

  //   %0 = hfusion.symbolic_dim @SymName : index
  let assemblyFormat = "$symbolName attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// MulExtOp
//===----------------------------------------------------------------------===//

def MulExtOp : HFusion_Op<"mulext", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "low", "high"]>]> {
  let summary = [{
    extended signed integer multiplication operation
  }];

  let description = [{
    Performs (2*N)-bit multiplication on sign-extended operands. Returns two
    N-bit results: the low and the high halves of the product. The low half has
    the same value as the result of regular multiplication with the same operands.
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike:$low, SignlessIntegerLike:$high);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";

}

//===----------------------------------------------------------------------===//
// InterleaveOp
//===----------------------------------------------------------------------===//

def InterleaveOp : HFusion_Op<"interleave", [Pure, Commutative,
                                             SameOperandsAndResultRank,
                                             DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                                             AllTypesMatch<["input"]>]> {
  let summary = [{
      Constructs one tensor by interleaving n input tensors.
      Only support n = 2 now.
  }];

  let description = [{
      Interleaves the values of n tensors along their last dimension.
      N tensors must have the same shape.
      Input tensors and output tensor must have same rank.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[AnyRankedTensor]>>:$input);
  let results = (outs AnyTypeOf<[AnyRankedTensor]>:$output);
  let extraClassDeclaration = [{
    static int64_t getInterLeaveChannelNums() {return 2;}
  }];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeinterleaveOp
//===----------------------------------------------------------------------===//

def DeinterleaveOp : HFusion_Op<"deinterleave", [Pure, Commutative,
                                                 SameOperandsAndResultRank,
                                                 DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                                                 AllTypesMatch<["output"]>]> {
  let summary = [{
    Constructs two tensors by deinterleaving an input tensor
  }];

  let description = [{
    The deinterleave operation constructs two tensors from a single input
    tensor. The first result tensor contains the elements from even indexes, and
    the second contains elements from odd indexes(both indexes mean the last
    dimension index). And it constrains the last dimension size of input must be
    multiples of 2.

    The channelIndex attribute controls the output behavior:
    * -1: Output all channels (returns two tensors, one with even indexes and one with odd indexes)
    * 0: Output only channel 0 (even indexes)
    * 1: Output only channel 1 (odd indexes)
  }];

  let extraClassDeclaration = [{
    static int64_t getDeInterLeaveChannelNum() { return 2; }
    int64_t getDeInterLeaveChannelIdx();
  }];

  let arguments = (ins AnyTypeOf<[AnyRankedTensor]>:$input,
                       I64Attr:$channelIndex);
  let results = (outs Variadic<AnyTypeOf<[AnyRankedTensor]>>:$output);

  let assemblyFormat = "$input custom<HFusionDeinterleave>($channelIndex) attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FlipOp
//===----------------------------------------------------------------------===//

def FlipOp : HFusion_Op<"flip", [Pure, Commutative, AllTypesMatch<["output"]>]> {
  let summary = [{
      Flips a tensor x along the dimension dim.
  }];

  let description = [{
      Flips a tensor x along the dimension dim.
      currently only final dimension supported.
  }];

  let arguments = (ins AnyTypeOf<[AnyRankedTensor]>:$input,
                   I64Attr:$flip_axis);
  let results = (outs AnyTypeOf<[AnyRankedTensor]>:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
    `flip_axis` `=` $flip_axis
    `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// IsInfOp
//===----------------------------------------------------------------------===//

def IsInfOp : HFusion_Op<"isinf", [Pure, SameOperandsAndResultRank,
                                   AllTypesMatch<["output"]>]> {
  let summary = [{
    Calculates whether elements of a float type tensor equal positive
    infinity or negative infinity.
  }];
  let description = [{
    Calculates whether elements of a float type tensor is infinity.
    Both positive infinity and negative infinity work.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// IsNanOp
//===----------------------------------------------------------------------===//

def IsNanOp : HFusion_Op<"isnan", [Pure, SameOperandsAndResultRank,
                                   AllTypesMatch<["output"]>]> {
  let summary = [{
    Calculates whether elements of a float type tensor is NAN.
  }];
  let description = [{
    Calculates whether elements of a float type tensor is NAN.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

def IsFiniteOp : HFusion_Op<"isfinite", [Pure, SameOperandsAndResultRank,
                                         AllTypesMatch<["output"]>,
                                         DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
                                           ["decomposeOperation"]>]> {
  let summary = [{
    Calculates whether elements of a float type tensor is finite.
  }];
  let description = [{
    Calculates whether elements of a float type tensor is finite
    (i.e., not NaN or infinity).
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// CumsumOp
//===----------------------------------------------------------------------===//

def CumsumOp : HFusion_Op<"cumsum",
    [Pure, SameOperandsAndResultRank, AllTypesMatch<["output"]>]> {
  let summary = [{
    Calculate cumulative sum on a certain dim of the input tensor
  }];
  let description = [{
    The cumsum operation calculates cumulative sum on a certain dim of the input
    tensor.  The reverse argument specifies the direction of the cumulative sum.
    Currently only support one cumulation dim.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>:$input,
                       ConfinedAttr<DenseI64ArrayAttr,
                      [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$reverse);
  let results = (outs RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CumprodOp
//===----------------------------------------------------------------------===//

def CumprodOp : HFusion_Op<"cumprod",
    [Pure, SameOperandsAndResultRank, AllTypesMatch<["output"]>]> {
  let summary = [{
    Calculate cumulative product on a certain dim of the input tensor
  }];
  let description = [{
    The cumsum operation calculates cumulative product on a certain dim of
    the input tensor. The reverse argument specifies the direction of the cumulative product.
    Currently only support one cumulation dim.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>:$input,
                       ConfinedAttr<DenseI64ArrayAttr,
                      [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$reverse);
  let results = (outs RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AtomicCasOp
//===----------------------------------------------------------------------===//

def AtomicCasOp : HFusion_Op<"atomic_cas",
    [SameOperandsAndResultRank, AllTypesMatch<["output"]>, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{
    Atomic Compare-And-Swap (CAS) Op
  }];
  let description = [{
    Compare-And-Swap (CAS) is an atomic operation that consists of three operands:
    Memory location (V), Expected old value (A), New value (B).
    The semantics of the operation are: the value of V is updated to B,
    only if the value of memory location V is equal to the expected old value A. 
    The operation returns the original value of V regardless of whether it is updated or not.

    Constraints:
      1. The input and output must have the same rank
         and the same element type.

    Arguments:
      * `src0`: expected old value
      * `src1`: new value
      * `dst`: memory location in GM

    Examples:
    ```mlir
    hfusion.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
    %result = hfusion.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
    ```
  }];
  let arguments = (ins Variadic<AnyTypeOf<[TensorOrMemref]>>:$input,
                       TensorOrMemref:$dst);
  let results = (outs Optional<TensorOrMemref>:$output);
  let assemblyFormat = [{ attr-dict `ins` `(` $input `:` type($input) `)`
                       `outs` `(` $dst `:` type($dst) `)`
                       (`->` type($output)^)?
  }];
}

//===----------------------------------------------------------------------===//
// AtomicXchgOp
//===----------------------------------------------------------------------===//

def AtomicXchgOp : HFusion_Op<"atomic_xchg",
    [SameOperandsAndResultRank, AllTypesMatch<["output"]>, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{
    Atomic Exchange Op
  }];
  let description = [{
      Atomic exchange is an atomic operation that consists of three steps:
      1. Read the current value of the specified memory address
      2. Write the new value to the memory address
      3. Return the old value read previously
      The whole process is atomic, that is, it will not be interrupted by other threads during the operation.

      Constraints:
        1. The input memref and output memref must have the same rank
          and the same element type.

      Arguments:
        * `src`: new value
        * `dst`: memory location in GM
        * `mask` : mask the element

      Examples:
      ```mlir
      hfusion.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) mask(%m : memref<?xi1>)
      %result = hfusion.atomic_xchg ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) mask(%m : memref<?xi1>) -> tensor<?xf32>
      ```
  }];
  let arguments = (ins TensorOrMemref:$input,
                       TensorOrMemref:$dst, Optional<TensorOrMemref>:$mask);
  let results = (outs Optional<TensorOrMemref>:$output);
  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>,
  ];
  let assemblyFormat = [{ attr-dict `ins` `(` $input `:` type($input) `)`
                       `outs` `(` $dst `:` type($dst) `)`
                       (`mask` `(` $mask^ `:` type($mask) `)`)?
                       (`->` type($output)^)?
  }];
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

def AtomicRMWOp : HFusion_Op<"atomic_rmw",
    [SameOperandsAndResultRank, AllTypesMatch<["output"]>, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{
    Atomic RMW Op
  }];
  let description = [{
      Atomic RMW is an atomic operation that consists of three steps:
      1. Read the current value of the specified memory address
      2. Perform action depending on atomic_kind attr
      3. Return the old value read previously
      The whole process is atomic, that is, it will not be interrupted by other threads during the operation.

      Constraints:
        1. The input memref and output memref must have the same rank
          and the same element type.

      Arguments:
        * `src`: new value
        * `dst`: memory location in GM

      Examples:
      ```mlir
      hfusion.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
      %result = hfusion.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
      ```
  }];
  let arguments = (ins TensorOrMemref:$input,
                       TensorOrMemref:$dst,
                       AtomicKindAttr:$atomic_kind);
  let results = (outs Optional<TensorOrMemref>:$output);
  let assemblyFormat = [{ attr-dict `ins` `(` $input `:` type($input) `)`
                       `outs` `(` $dst `:` type($dst) `)`
                       `atomic_kind` `=` $atomic_kind
                       (`->` type($output)^)?
  }];
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//
def SortOp : HFusion_Op<"sort",
    [SameOperandsAndResultRank, AllTypesMatch<["output"]>]> {
  let summary = [{
    Sort Op
  }];
  let description = [{
    Sort the sorting axis of `src` in ascending or descending order, and output
    the sorted value and the index corresponding to the value.

    Constraints:
      1. The input vector and output vector must have the same rank.
      2. Currently only tail axis sorting is supported.

    Arguments:
      * `src`: the tensor/memref from which to be sorted
      * `dst_value`: the tensor/memref to store the sorted value
      * `dst_index`: the tensor/memref to store the index corresponding to dst_value
      * `descending`: determines whether to sort in ascending or descending
                      order. The default is false, which means ascending order
      * `sort_axis`: Axis to be sorted

    Examples:
      ```mlir
      %result = hfusion.sort ins(%src : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
      ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$descending,
                       I64Attr:$sort_axis);
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `descending` `=` $descending
    `sort_axis` `=` $sort_axis
    (`->` type($result)^)?
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Get Signed Sort Axis
    int64_t getSignedSortAxis();
  }];
}

//===----------------------------------------------------------------------===//
// HistogramOp
//===----------------------------------------------------------------------===//
def HistogramOp : HFusion_Op<"histogram", [
    DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
                              ["decomposeOperation"]>
  ]> {
  let summary = "Compute histogram of an integer tensor with optional mask";
  let description = [{
    For each element of the input tensor, increment the corresponding bin in
    the output histogram. The number of bins is a compile-time constant
    (`I64Attr`) and the output must be a 1D tensor of length = num_bins.
    If a mask tensor is provided, only elements with mask[i] = true are counted.
  }];

  let arguments = (ins
    RankedTensorOf<[I8, I16, I32, I64]>:$input,
    I64Attr:$num_bins,
    Optional<RankedTensorOf<[I1]>>:$mask
  );

  let results = (outs
    RankedTensorOf<[I32, I64]>:$output
  );

  let hasVerifier = 1;
  let assemblyFormat =
    "$input `,` $num_bins (`,` $mask^)? attr-dict `:` "
    "type($input) (`,` type($mask)^)? `->` type($output)";
}

//===----------------------------------------------------------------------===//
// MatMulMxOp
//===----------------------------------------------------------------------===//
def MatMulMxOp : HFusion_Op<"matmul_mx", [Pure, DestinationStyleOpInterface]> {
  let summary = "Performs a block-scaled matrix multiplication";
  let description = [{
  Performs matrix multiplication in microscaling format where the inputs are implicitly scaled using provided scale factors, which are often used in quantized data types like FP8/FP4. The operation computes C = (A * scale_a) dot (B * scale_b) directly in hardware (Tensor Cores).
  lhs and rhs use microscaling formats described here: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  }];

  let arguments = (ins
    ShapedTypeOf<[F8E4M3FN, F8E5M2]>:$inputA,
    ShapedTypeOf<[F8E4M3FN, F8E5M2]>:$inputB,
    ShapedTypeOf<[UI8, I8]>:$scaleA,
    ShapedTypeOf<[UI8, I8]>:$scaleB,
    ShapedTypeOf<[AnyFloat]>:$acc
  );

  let results = (outs
    ShapedTypeOf<[AnyFloat]>:$result
  );

  code hfusionMmadBaseDecls = [{
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getAccMutable();
    }
  }];

  let extraClassDeclaration = hfusionMmadBaseDecls;

  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict `ins` `(`
    $inputA
    `,` $inputB
    `,` $scaleA
    `,` $scaleB
    `:`
    type($inputA)
    `,` type($inputB)
    `,` type($scaleA)
    `,` type($scaleB)
    `)`
    `outs` `(` $acc `:` type($acc) `)`
    `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins "Value":$inputA,
                   "Value":$inputB,
                   "Value":$scaleA,
                   "Value":$scaleB,
                   "Value":$acc)>
  ];
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD
