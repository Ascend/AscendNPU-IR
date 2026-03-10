//===- HFusionOps.td - HFusion op definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    tensor.
    Currently only support one cumulation dim.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64, F8E4M3FN, F8E5M2]>:$input,
                       ConfinedAttr<DenseI64ArrayAttr,[DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims,
                       BoolAttr:$reverse);
  let results = (outs RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64, F8E4M3FN, F8E5M2]>:$output);
  let assemblyFormat = [{
      $input attr-dict `:` type($input)
      `cum_dims` `=` $cum_dims
      `reverse` `=` $reverse
      `->` type($output)
    }];
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
    The cumprod operation calculates cumulative product on a certain dim of
    the input tensor.
    Currently only support one cumulation dim.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>:$input,
                       ConfinedAttr<DenseI64ArrayAttr,[DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims,
                       BoolAttr:$reverse);
  let results = (outs RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
    `cum_dims` `=` $cum_dims
    `reverse` `=` $reverse
    `->` type($output)
  }];
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
  let results = (outs Variadic<AnyTypeOf<[TensorOrMemref]>>:$output);
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

      Examples:
      ```mlir
      hfusion.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)
      %result = hfusion.atomic_xchg ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
      ```
  }];
  let arguments = (ins Variadic<AnyTypeOf<[TensorOrMemref]>>:$input,
                       TensorOrMemref:$dst);
  let results = (outs Variadic<AnyTypeOf<[TensorOrMemref]>>:$output);
  let assemblyFormat = [{ attr-dict `ins` `(` $input `:` type($input) `)`
                       `outs` `(` $dst `:` type($dst) `)`
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
// EmbeddingGatherOp
//===----------------------------------------------------------------------===//

def EmbeddingGatherOp : HFusion_Op<"embedding_gather", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize,
]> {
  let summary = [{
    Embedding lookup operation using gather semantics
  }];

  let description = [{
    Performs embedding lookup (also known as embedding table lookup or embedding gather).
    This operation takes a 2D embedding table in global memory and a 1D/2D index tensor,
    and produces a 2D/3D output tensor by gathering embedding vectors corresponding to the indices.

    Semantic equivalent:
      result[b][i][d] = src[index[b][i]][d]   (with bounds checking)

    Where:
      - b: batch dimension
      - i: sequence/position dimension
      - d: embedding feature dimension
  }];

  let arguments = (ins AnyMemRef:$src,
                       AnyRankedTensor:$index,
                       AnyRankedTensor:$dst,
                       // Vocabulary size for bounds checking
                       AnyTypeOf<[I32, I64]>:$bound,
                       Variadic<AnyTypeOf<[I32, I64]>>:$offsets,
                       Variadic<AnyTypeOf<[I32, I64]>>:$numels
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `,` $index `:` type($index) `,`
    $bound `:` type($bound) `,`
    `[` $offsets `:` type($offsets) `]` `,`
    `[` $numels `:` type($numels) `]` `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "embedding_gather"; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GatherLoadOp
//===----------------------------------------------------------------------===//

def GatherLoadOp : HFusion_Op<"gather_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize,
]> {
  let summary = [{
    Perform sparse memory loading with optional mask and other
  }];

  let description = [{
    This operation takes a source memory buffer and a tensor of offsets,
    and produces an output tensor by gathering elements from the source
    at the specified offset positions. The operation supports masking
    and provides fallback values for masked-out positions.
  }];
  let arguments = (ins AnyMemRef:$base,
                       RankedTensorOf<[I32, I64]>:$indices,
                       AnyTypeOf<[I32, I64]>:$burst_len,
                       Optional<RankedTensorOf<[I1]>>:$mask,
                       Optional<AnyTypeOf<[AnyInteger, AnyFloat]>>:$other,

                       OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                       OptionalAttr<HFusion_PaddingOptionAttr>:$padding,
                       OptionalAttr<HFusion_CacheModifierAttr>:$cache,
                       OptionalAttr<HFusion_EvictionPolicyAttr>:$evict,
                       OptionalAttr<BoolAttr>:$isVolatile
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `ins` `(` $base `:` type($base) `,` $indices `:` type($indices) `,`
    $burst_len `:` type($burst_len)
    (`,` $mask `:` type($mask)^)? (`,` $other `:` type($other)^)? `)`
    attr-dict
    `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScatterStoreOp
//===----------------------------------------------------------------------===//
 
def ScatterStoreOp : HFusion_Op<"scatter_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize,
]> {
  let summary = [{
    Performs sparse memory storing with optional mask
  }];
 
  let description = [{
    This operation takes a source tensor and a tensor of offsets from UB,
    and stores elements from the source into the destination GM buffer
    at the specified offset positions. The operation 
    supports masking to conditionally control which elements are stored.
  }];
  let arguments = (ins AnyMemRef:$base,
                       RankedTensorOf<[I32, I64]>:$indices,
                       AnyRankedTensor:$data,
                       AnyTypeOf<[I32, I64]>:$burst_len,
                       Optional<RankedTensorOf<[I1]>>:$mask,
 
                       OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                       OptionalAttr<HFusion_CacheModifierAttr>:$cache,
                       OptionalAttr<HFusion_EvictionPolicyAttr>:$evict
  );
 
  let assemblyFormat = [{
    `ins` `(` $base `:` type($base) `,` $indices `:` type($indices) `,`
    $data `:` type($data) `,` $burst_len `:` type($burst_len)
    (`,` $mask `:` type($mask)^)? `)`
    attr-dict
  }];
 
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IndirectLoadOp
//===----------------------------------------------------------------------===//

def IndirectLoadOp : HFusion_Op<"indirect_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = [{
    Performs indirect memory loading with masking and fallback values.
  }];

  let description = [{
    This operation takes a source memory buffer and a tensor of offsets,
    and produces an output tensor by gathering elements from the source
    at the specified offset positions. The operation supports masking
    and provides fallback values for masked-out positions.
    This operation supports 1D-5D.

    For each position in the output tensor:
      1D:
          dst[i] = mask[i] ? src[offsets[i]] : other[i]
      2D:
          dst[i][j] = mask[i][j] ? src[offsets[i][j]] : other[i][j]
      3D:
          dst[i][j][k] = mask[i][j][k] ? src[offsets[i][j][k]] : other[i][j][k]
      4D:
          dst[i][j][k][l] = mask[i][j][k][l] ? src[offsets[i][j][k][l]] : other[i][j][k][l]
      5D:
          dst[i][j][k][l][m] = mask[i][j][k][l][m] ? src[offsets[i][j][k][l][m]] : other[i][j][k][l][m]

    Where:
      - src: source memory buffer to load from
      - offsets: indices specifying positions in the source buffer
      - mask: boolean mask controlling which elements to load
      - other: fallback values used when mask is false
      - dst: destination tensor specifying output shape and type

    This operation is useful for sparse data access patterns and
    gather operations with conditional loading semantics.
  }];

  let arguments = (ins AnyMemRef:$src,
                       TensorOf<[I32, I64]>:$offsets,
                       AnyRankedTensor:$dst,
                       TensorOf<[I1, I8]>:$mask,
                       TensorOf<[AnyInteger, AnyFloat]>:$other
  );
  let results = (outs Optional<AnyRankedTensor>:$result);

  let assemblyFormat = [{
      `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
      `,` $mask `:` type($mask)
      `,` $other `:` type($other) `)`
      `outs` `(` $dst `:` type($dst) `)`
      attr-dict
      (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "indirect_load"; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IndirectStoreOp
//===----------------------------------------------------------------------===//

def IndirectStoreOp : HFusion_Op<"indirect_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {

  let summary = [{
    Performs indirect memory storing with SIMT template.
  }];

  let description = [{
    This operation takes a source tensor and a tensor of offsets from UB,
    and stores elements from the source into the destination GM buffer
    at the specified offset positions with SIMT template. The operation 
    supports masking to conditionally control which elements are stored.
    This operation supports 1D-5D.

    For each position in the source tensor:
      1D:
          if (mask[i]) dst[offsets[i]] = src[i]
      2D:
          if (mask[i][j]) dst[offsets[i][j]] = src[i][j]
      3D:
          if (mask[i][j][k]) dst[offsets[i][j][k]] = src[i][j][k]
      4D:
          if (mask[i][j][k][l]) dst[offsets[i][j][k][l]] = src[i][j][k][l]
      5D:
          if (mask[i][j][k][l][m]) dst[offsets[i][j][k][l][m]] = src[i][j][k][l][m]

    Where:
      - src: source tensor containing values to store from UB
      - offsets: indices specifying positions in the destination buffer
      - dst: destination memory buffer to store into GM
      - mask: optional boolean mask controlling which elements to store

    When no mask is provided, all elements from the source tensor are stored
    to the corresponding offset positions in the destination buffer.
  }];

  let arguments = (ins AnyMemRef:$dst,
                       TensorOf<[I32, I64]>:$offsets,
                       AnyRankedTensor:$src,
                       Optional<TensorOf<[I1, I8]>>:$mask
  );

  let assemblyFormat = [{
      `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
      (`,` $mask^ `:` type($mask))? `)`
      `outs` `(` $dst `:` type($dst) `)`
      attr-dict
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "indirect_store"; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GatherTOp
// The name conflicts with GatherOp so added T
// GatherTOp is used for SIMT
// TODO: Merge GatherOp and GatherTOp
//===----------------------------------------------------------------------===//

def GatherTOp : HFusion_Op<"gatherT", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize
]> {
  let summary = [{
    Gather operation that selects elements from the input tensor based on index tensor along a specified axis.
  }];

  let description = [{
    This operation takes a source memory GM buffer and a UB tensor of index,
    and produces an output UB tensor by gathering elements from the source
    at the index positions with offsets. This operation supports 1D-5D.

    For each position in the output tensor:
      1D:
        dst[i] = src[index[i]]
      2D:
        dim=0:
          dst[i][j] = src[index[i][j]][j]
        dim=1:
          dst[i][j] = src[i][index[i][j]]
      3D:
        dim=0:
          dst[i][j][k] = src[index[i][j][k]][j][k]
        dim=1:
          dst[i][j][k] = src[i][index[i][j][k]][k]
        dim=2:
          dst[i][j][k] = src[i][j][index[i][j][k]]

    Where:
      - src : source memory GM buffer
      - index: UB tensor specifying the position in the src
      - dst : destination UB tensor specifying output shape and type
      - bound : size of the gather dim
      - dim : dimensions for the gather operation.
      - src_stride : stride of the source tensor.
      - index_shape : shape of the index tensor.
      - offsets : offset in the dst buffer
  }];

  let arguments = (ins AnyMemRef:$src,
                       AnyRankedTensor:$index,
                       AnyRankedTensor:$dst,
                       AnyTypeOf<[I32, I64]>:$bound,
                       AnyTypeOf<[I32, I64]>:$dim,
                       Variadic<AnyTypeOf<[I32, I64]>>:$src_stride,
                       Variadic<AnyTypeOf<[I32, I64]>>:$index_shape,
                       Variadic<AnyTypeOf<[I32, I64]>>:$offsets
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `,`
    $index `:` type($index) `,`
    $bound `:` type($bound) `,`
    $dim `:` type($dim) `,`
    `[` $src_stride `:` type($src_stride) `]` `,`
    `[` $index_shape `:` type($index_shape) `]` `,`
    `[` $offsets `:` type($offsets) `]` `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "gatherT"; }
  }];
}

//===----------------------------------------------------------------------===//
// IndexPutOp
//===----------------------------------------------------------------------===//

def IndexPutOp : HFusion_Op<"index_put", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize
]> {
  let summary = [{
    IndexPut operation that assigns values to specified positions in the destination tensor with SIMT template.
  }];

  let description = [{
    This operation IndexPut assigns values from the value UB tensor
    to the dst GM buffer at positions with offsets specified by the index UB tensor
    along the specified scatter dimension with SIMT template. This operation supports 2D-5D.

    For each position in the destination tensor:
    2D:
      for dim = 0:
      dst[index[i]][offset[1] + j] = value[i][j]
    3D:
      for dim = 0:
      dst[index[i]][offset[1] + j][offset[2] + k] = value[i][j][k]
      for dim = 1:
      dst[offset[0] + i][index[j]][offset[2] + k] = value[i][j][k]
    4D:
      for dim = 0:
      dst[index[i]][offset[1] + j][offset[2] + k][offset[3] + l] = value[i][j][k][l]
      for dim = 1:
      dst[offset[0] + i][index[j]][offset[2] + k][offset[3] + l] = value[i][j][k][l]
      for dim = 2:
      dst[offset[0] + i][offset[1] + j][index[k]][offset[3] + l] = value[i][j][k][l]

    Where:
      - dst : the pointer of destination tensor on GM.
      - index:  the index tensor tile pointer on UB specifying positions in the destination tensor. 
                must not be repeated.
                must contain valid indices within the bounds of 
                the destination tensor along the specified scatter_dim.
      - value : value tile pointer on UB.
      - scatter_dim : the gather dimension on which index is applied.
      - bound : upper bound of index value.
      - end_offset : end offsets of the value tile, end offset of index tile is No.'dim' value in the tuple.
      - dst_shape : start offsets of the value tile, end offset of index tile is No.'dim' value in the tuple.
      - dst_offset : the stride of target global tensor.
  }];

  let arguments = (ins AnyMemRef:$dst,
                       AnyRankedTensor:$index,
                       AnyRankedTensor:$value,
                       AnyTypeOf<[I32, I64]>:$scatter_dim,
                       AnyTypeOf<[I32, I64]>:$bound,
                       Variadic<AnyTypeOf<[I32, I64]>>:$end_offset,
                       Variadic<AnyTypeOf<[I32, I64]>>:$start_offset,
                       Variadic<AnyTypeOf<[I32, I64]>>:$dst_stride
  );

  let assemblyFormat = [{
    `ins` `(` $dst `:` type($dst) `,` 
    $index `:` type($index) `,` 
    $value `:` type($value) `,`
    $scatter_dim `:` type($scatter_dim) `,`
    $bound `:` type($bound) `,`
    `[` $end_offset `:` type($end_offset) `]` `,`
    `[` $start_offset `:` type($start_offset) `]` `,`
    `[` $dst_stride `:` type($dst_stride) `]` `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "index_put"; }
  }];
}


//===----------------------------------------------------------------------===//
// ScatterTOp
//===----------------------------------------------------------------------===//

def ScatterTOp : HFusion_Op<"scatterT", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = [{
    Scatter operation writes all values from the tensor value into dst at the indices specified in the index_tile
    tensor with SIMT template.
  }];

  let description = [{
    This operation Scatter assigns values from the UB tensor to the dst GM buffer at positions with offsets specified
    by the index_tile UB tensor with SIMT template. This operation supports 2D-5D.

    Using 3D as an example:
    for dim = 0:
    Dst[index_tile[i][j][k]][j][k] = value[i][j][k]
    for dim = 1:
    Dst[i][index_tile[i][j][k]][k] = value[i][j][k]
    for dim = 2:
    Dst[i][j][index_tile[i][j][k]] = value[i][j][k]

    Where:
      - dst : the base pointer of destination tensor GM memory buffer.
      - value : value tensor from UB to store.
      - index_tile: index tensor from UB specifying positions in the destination tensor.
                    should not be repeated.
      - index_boundary: upper bound of index
      - dim: dimension along which the assignment operation is performed.
      - dst_stride: tuple of destination strides
      - index_shape: shape of the index tensor
      - offsets: tuple of offsets on all dims of the index tile
  }];

  let arguments = (ins AnyMemRef:$dst,
                       AnyRankedTensor:$value,
                       AnyRankedTensor:$index_tile,
                       AnyTypeOf<[I32, I64]>:$index_boundary,
                       AnyTypeOf<[I32, I64]>:$dim,
                       Variadic<AnyTypeOf<[I32, I64]>>:$dst_stride,
                       Variadic<AnyTypeOf<[I32, I64]>>:$index_shape,
                       Variadic<AnyTypeOf<[I32, I64]>>:$offsets
  );

  let assemblyFormat = [{
      `ins` `(` $dst `:` type($dst) `,`
      $value `:` type($value)`,`
      $index_tile `:` type($index_tile)`,`
      $index_boundary `:` type($index_boundary)`,`
      $dim `:` type($dim)`,`
      `[` $dst_stride `:` type($dst_stride) `]` `,`
      `[` $index_shape `:` type($index_shape) `]` `,`
      `[` $offsets `:` type($offsets) `]` `)`
      attr-dict
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "scatterT"; }
  }];

  let hasVerifier = 1;
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
    RankedTensorOf<[I8, UI8, I16, UI16, I32, UI32, I64, UI64]>:$input,
    I64Attr:$num_bins,
    Optional<RankedTensorOf<[I1]>>:$mask
  );

  let results = (outs
    RankedTensorOf<[I8, I16, I32, I64]>:$output
  );

  let hasVerifier = 1;
  let assemblyFormat =
    "$input `,` $num_bins (`,` $mask^)? attr-dict `:` "
    "type($input) (`,` type($mask)^)? `->` type($output)";
}


#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD
