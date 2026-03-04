//===- HIVMOps.td - HIVM dialect operation definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for HIVM dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

//===----------------------------------------------------------------------===//
// HIVM Dialect operations.
//===----------------------------------------------------------------------===//

def GetBlockIdxOp : HIVM_Op<"get_block_idx", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get block idx of the current device thread used for parallelization.";
  let description = [{
    This op gets the block idx of the current device thread.
    This op will be lowered to `GetBlockIdxInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetBlockNumOp : HIVM_Op<"get_block_num", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get block number of the current device thread used for parallelization.";
  let description = [{
    This op gets the block number of the current device thread.
    This op will be lowered to `GetBlockNumInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetSubBlockIdxOp : HIVM_Op<"get_sub_block_idx", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sub block idx of the current device thread used for parallelization.";
  let description = [{
    This op gets the sub block idx of the current device thread.
    This op will be lowered to GetSubBlockIdxInstrOp.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetSubBlockNumOp : HIVM_Op<"get_sub_block_num", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sub block number of the current device thread used for parallelization.";
  let description = [{
    This op gets the sub block number of the current device thread.
    This op will be lowered to GetSubBlockNumInstrOp.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def SetAtomicOp : HIVM_Op<"set_atomic"> {
  let summary = "HIVM Set Atomic Operation";
  let description = [{
    This operation enables atomic operation for any data written to global memory.
    Reset atomic operation by setting kind = #hivm.atomic_kind<NONE>.

    The supported atomic operation's data type includes:
        - F32
        - F16
        - S16
        - S32
        - S8
        - BF16

    The supported atomic operation's type includes:
        - ADD
        - MAX
        - MIN

    For example:

    ``` 
    hivm.hir.set_atomic kind= #hivm.atomic_kind<ADD> type = i32
    // all subsequent store-like operations will enable atomic store with addition
    hivm.hir.store ins(...) outs(...)
    ...
    hivm.hir.set_atomic kind= #hivm.atomic_kind<NONE>
    hivm.hir.fixpipe ins(...) outs(...)
    ``` 
  }];
  let arguments = (ins HIVM_AtomicKindAttr:$kind,
                       OptionalAttr<TypeAttr>:$type);
  let assemblyFormat = "`kind` `=` $kind `[` `type` `=` $type `]` attr-dict";
}

def SetMaskNormOp : HIVM_Op<"set_mask_norm"> {
  let summary = "hivm set mask norm";
  let assemblyFormat = "attr-dict";
}

def SetCtrlOp : HIVM_Op<"set_ctrl"> {
  let summary = "hivm set ctrl operation";
  let description = "This op configures control bits by setting a specific bit to 0 or 1.";
  let arguments = (ins BoolAttr:$enable,
                       I64Attr:$idx);
  let assemblyFormat = "$enable `at` `ctrl``[` $idx `]` attr-dict";
}

def LoadScalarOp : HIVM_Op<"load_scalar"> {
  let summary = "hivm load scalar";
  let arguments = (ins LLVMPointerType:$addr);
  let results = (outs AnyTypeOf<[AnyInteger, AnyFloat]>:$result);
  let assemblyFormat = "attr-dict $addr `:` type($addr) `->` type($result)";
  let hasVerifier = 1;
}

def DCCIOp : HIVM_Op<"dcci"> {
  let summary = "hivm dcci op ";
  let description = [{
    This op cleans(writes back) and invalidates one cacheline or the entire data cache
  }];
  let arguments = (ins HIVM_DCCIModeAttr:$mode,
      HIVM_DataCacheKindAttr:$dataCacheKind, Optional<AnyMemRef>:$ptr);
  let assemblyFormat = [{
    attr-dict `(` $mode `,` $dataCacheKind (`,` $ptr^ `:` type($ptr))? `)`
  }];
}

def ConvertLayoutOp : HIVM_Op<"convert_layout",
   [Pure,
    ViewLikeOpInterface,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>]> {
  let summary = "HIVM layout conversion operation.";
  let description = [{
    The `convert_layout` operation converts a memref with one layout
    to another. The data is not copied or modified.
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source,
                       HIVM_DataLayoutAttr:$srcLayout,
                       HIVM_DataLayoutAttr:$dstLayout);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let assemblyFormat = [{
    $source attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    Value getViewSource() { return getSource(); }
  }];
  let hasCanonicalizer = 1;
}

def PointerCastOp : HIVM_Op<"pointer_cast",
  [CubeVectorCoreTypeTrait, AttrSizedOperandSegments]> {
  let summary = "HIVM pointer cast op at specific i64 addr";
  let description = [{
    The specific i64 addrs are stored in `$addrs`, which is variadic.

    Constraints:
    1. The type of each address should be i64.
    2. addrs should have at least one addr.

    Examples:
    ```mlir
    %addr = arith.constant 1234 : i64
    %tmp = hivm.hir.pointer_cast(%addr) : memref<32xf32>

    %addr2 = arith.constant 1600 : i64
    %addr3 = arith.constant 3200 : i64
    %tmp2 = hivm.hir.pointer_cast(%addr, %addr2) : memref<32xf32>
    %tmp3 = hivm.hir.pointer_cast(%addr, %addr2, %addr3) : memref<32xf32>
    ```
  }];
  let arguments = (ins Variadic<I64>:$addrs,
                       Variadic<Index>:$dynamicSizes
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `(`$addrs `)` (`[` $dynamicSizes^`]`)? attr-dict `:` type($result)
  }];
  let builders= [
    OpBuilder<(ins "Type":$result, "Value":$addr)>,
    OpBuilder<(ins "Type":$result, "ValueRange":$addrs)>,
    OpBuilder<(ins "Type":$result, "Value":$addr, "ValueRange":$dynamicSizes)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$addr, "ValueRange":$dynamicSizes)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    TypedValue<IntegerType> getSingleAddr();
  }];
}

def BitcastOp : HIVM_Op<"bitcast",
  [AlwaysSpeculatable, Elementwise, SameOperandsAndResultShape,
   NoMemoryEffect, Pure]> {
  let summary = "Reinterprets the bits of a shaped value without changing data";
  let description = [{
    The `bitcast` operation converts a tensor/memref from one element type to another
    while preserving the underlying bit representation. The operation requires:

    1. Same shape for input and output (2x3 != 3x2)
    2. Same total bit-width (element_bitwidth * num_elements)
    3. Same memory layout/strides (for memrefs)
  }];
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
      $src `:` type($src) `->` type($result) attr-dict
  }];
}

def SetFFTSBaseAddrOp : HIVM_Op<"set_ffts_base_addr", [CubeVectorCoreTypeTrait]> {
  let summary = "set base addr for ffts sync machenism.";
  let arguments = (ins I64:$ffts_base_addr);
  let assemblyFormat = [{
    attr-dict $ffts_base_addr
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Debug op definitions.
//===----------------------------------------------------------------------===//

def DebugOp :
  HIVM_Op<"debug",
          [StaticMaxRankTrait<4>,
           MemoryEffects<[MemRead, MemWrite]>,
           DeclareOpInterfaceMethods<LibraryFunctionOpInterface>,
           DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>]>,
  Arguments<(ins StrAttr:$debugtype,
                 StrAttr:$prefix,
                 BoolAttr:$hex,
                 AnyTypeOf<[AnyInteger, AnyFloat, TensorOrMemref]>:$arg,
                 OptionalAttr<HIVM_TCoreTypeAttr>:$tcoretype)> {
  let summary = "Device-side debugging";
  let assemblyFormat = [{
    attr-dict $arg `:` type($arg)
  }];
  let extraClassDeclaration = [{
    static StringRef getOpName() { return "debug"; }
  }];
}

def InitDebugOp : HIVM_Op<"init_debug",
                          [CubeVectorCoreTypeTrait,
                           NoMaxRankTrait,
                           DeclareOpInterfaceMethods<LibraryFunctionOpInterface>
                          ]> {
  let summary = "Init func for device-side debugging";
  let assemblyFormat = [{
    attr-dict
  }];
}

def FinishDebugOp : HIVM_Op<"finish_debug",
                          [CubeVectorCoreTypeTrait,
                           NoMaxRankTrait,
                           DeclareOpInterfaceMethods<LibraryFunctionOpInterface>
                          ]> {
  let summary = "Finish func for device-side debugging";
  let assemblyFormat = [{
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// EmbeddingGatherOp
//===----------------------------------------------------------------------===//

def EmbeddingGatherOp : HIVM_Op<"embedding_gather", [
    StaticMaxRankTrait<3>,
    SameVariadicOperandSize,
    DestinationStyleOpInterface,
    OpPipeInterface,
    SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<LibraryFunctionOpInterface>
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
                       TensorOrMemref:$index,
                       TensorOrMemref:$dst,
                       // Vocabulary size for bounds checking
                       AnyTypeOf<[I32, I64]>:$bound,
                       Variadic<AnyTypeOf<[I32, I64]>>:$offsets,
                       Variadic<AnyTypeOf<[I32, I64]>>:$numels
  );

  let results = (outs Variadic<TensorOrMemref>:$result);

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
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GatherLoadOp
//===----------------------------------------------------------------------===//

def GatherLoadOp : HIVM_Op<"gather_load", [
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
                       OptionalAttr<HIVM_PaddingOptionAttr>:$padding,
                       OptionalAttr<HIVM_CacheModifierAttr>:$cache,
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$evict,
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
 
def ScatterStoreOp : HIVM_Op<"scatter_store", [
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
                       OptionalAttr<HIVM_CacheModifierAttr>:$cache,
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$evict
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

def IndirectLoadOp : HIVM_Op<"indirect_load",[
                            StaticMaxRankTrait<5>,
                            DestinationStyleOpInterface,
                            OpPipeInterface,
                            SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                            DeclareOpInterfaceMethods<LibraryFunctionOpInterface>,
                            AttrSizedOperandSegments
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
                       TensorOrMemref:$offsets,
                       TensorOrMemref:$dst,
                       Optional<TensorOrMemref>:$mask,
                       Optional<TensorOrMemref>:$other
  );
  let results = (outs Optional<TensorOrMemref>:$result);

  let assemblyFormat = [{
      `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
      (`,` $mask^ `:` type($mask))?
      (`,` $other^ `:` type($other))? `)`
      `outs` `(` $dst `:` type($dst) `)`
      attr-dict
      (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "indirect_load"; }
      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getDstMutable();
      }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IndirectStoreOp
//===----------------------------------------------------------------------===//

def IndirectStoreOp : HIVM_Op<"indirect_store",[
                            StaticMaxRankTrait<5>,
                            DestinationStyleOpInterface,
                            OpPipeInterface,
                            SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                            DeclareOpInterfaceMethods<LibraryFunctionOpInterface>,
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
                       TensorOrMemref:$offsets,
                       TensorOrMemref:$src,
                       Optional<TensorOrMemref>:$mask
  );

  let assemblyFormat = [{
      `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
      (`,` $mask^ `:` type($mask))? `)`
      `outs` `(` $dst `:` type($dst) `)`
      attr-dict
  }];

  let extraClassDeclaration = [{
      static StringRef getOpName() { return "indirect_store"; }
      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getDstMutable();
      }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GatherTOp
//===----------------------------------------------------------------------===//

def GatherTOp : HIVM_Op<"gatherT",[
                            StaticMaxRankTrait<5>,
                            SameVariadicOperandSize,
                            DestinationStyleOpInterface,
                            OpPipeInterface,
                            SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                            DeclareOpInterfaceMethods<LibraryFunctionOpInterface>
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
                       TensorOrMemref:$index,
                       TensorOrMemref:$dst,
                       AnyTypeOf<[I32, I64]>:$bound,
                       AnyTypeOf<[I32, I64]>:$dim,
                       Variadic<AnyTypeOf<[I32, I64]>>:$src_stride,
                       Variadic<AnyTypeOf<[I32, I64]>>:$index_shape,
                       Variadic<AnyTypeOf<[I32, I64]>>:$offsets
  );
  let results = (outs Optional<TensorOrMemref>:$result);

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
      static StringRef getOpName() { return "gather_out_to_ub"; }
      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getDstMutable();
      }
  }];
}

//===----------------------------------------------------------------------===//
// IndexPutOp
//===----------------------------------------------------------------------===//

def IndexPutOp : HIVM_Op<"index_put",[
                            StaticMaxRankTrait<5>,
                            SameVariadicOperandSize,
                            DestinationStyleOpInterface,
                            OpPipeInterface,
                            SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                            DeclareOpInterfaceMethods<LibraryFunctionOpInterface>
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
                       TensorOrMemref:$index,
                       TensorOrMemref:$value,
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
      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getDstMutable();
      }
  }];
}

//===----------------------------------------------------------------------===//
// ScatterTOp
//===----------------------------------------------------------------------===//

def ScatterTOp : HIVM_Op<"scatterT",[
                            StaticMaxRankTrait<5>,
                            DestinationStyleOpInterface,
                            OpPipeInterface,
                            SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_V">,
                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                            DeclareOpInterfaceMethods<LibraryFunctionOpInterface>,
                            AttrSizedOperandSegments
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
                       TensorOrMemref:$value,
                       TensorOrMemref:$index_tile,
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
      static StringRef getOpName() { return "scatter_ub_to_out"; }
      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getDstMutable();
      }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

class HIVM_CustomOp<string mnemonic, list<Trait> traits = []>
    : HIVM_StructuredOp<mnemonic,
                        !listconcat([AttrSizedOperandSegments,
                                     MemoryEffects<[MemRead, MemWrite]>,
                                     HIVMInferCoreTypeInterface],
                                    traits)> {
  string customOpSummary = [{
    Custom operation is a generic op interface for users to write their own custom implementation.
      Scenarios:
        1. Existing operations could not fulfill the desired functionality.
        2. Existing operations could fulfill the functionality, but overall performance is not optimal.
        3. Desire for private operation.
  }];

  string customOpDescription = [{
    General interface for custom op, where:
      - name : unique op name.

               Note : there are names reserved for builtins, usually starts with "__builtin".
                      Compiler will link these builtins to self-contained template library,
                      which comes together within bishengir-compile.

                      For normal names/cases, user needs to specify implementation
                      location/compilation commands (TODO), and all ther necessary informations.

               Available builtin names:
                 "__builtin_gather_load"

      - inputs : input parameters.
      - outputs : output results, designated "init" operands, which act as initial values for the results
                  of the operation or the init locations to which the results of the op will be written.
  }];

  string customOpAttrsDescription = [{
    In order to adapt to future enhancements quickly and dynamically, custom op relies on attributes
    to retreive necessary information. (NOTE: for builtins, user could specify these informations or not,
    compiler will help to check the correctness and canonicalize)

    Required informations are:
      - CoreType : which core type to execute on, refer to TCoreTypeAttr.
      - VFMode   : which mode to run on vector units, refer to VFModeAttr.
                   this attribute is ignored when core type is cube.

  }];

  string customOpOptionalAttrsDescription = [{
    Optional informations are:
      - GMAddrArgsIndices : an indices array to indicate arguments that contain the pure address from GM,
                            lowering passes will preserve the address value. Default: null.
  }];

  string customOpTODODescription = [{
    TODO:
      - Impl : user provided implementation.
  }];

  dag args = (ins StrAttr:$name, Variadic<AnyType>:$inputs,
      Variadic<AnyType>:$outputs);
  dag res = (outs Variadic<AnyType>:$results);

  code customOpBaseDecl = [{
      static constexpr StringLiteral kBuiltinGatherLoadName = "__builtin_gather_load";
      static constexpr StringLiteral kBuiltinIndexSelectName = "__builtin_index_select";

      // TODO: Enable user provided customization
      static int getOpLibraryMaxRankImpl() { return 5; }

      std::optional<TCoreType> inferCoreType() {
        if (auto coreTypeAttr = getOperation()->template getAttrOfType<TCoreTypeAttr>(
                TCoreTypeAttr::name)) {
          return coreTypeAttr.getTcoretype();
        }

        return {};
      }

      PIPE getPipe() {
        if (auto pipAttr =
            getOperation()->template getAttrOfType<PipeAttr>(PipeAttr::name))
          return pipAttr.getPipe();

        return PIPE::PIPE_UNASSIGNED;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getOutputsMutable();
      }

      // Helper functions
      std::optional<TCoreType> getCoreType() {
        if (const auto coreTypeAttr =
            getOperation()->template getAttrOfType<TCoreTypeAttr>(
              TCoreTypeAttr::name)) {
          return coreTypeAttr.getTcoretype();
        }

        return {};
      }

      void setCoreType(const TCoreType& coreType) {
        getOperation()->setAttr(TCoreTypeAttr::name,
            TCoreTypeAttr::get(getContext(), coreType));
      }

      std::optional<VFMode> getVFMode() {
        if (const auto vfModeAttr =
            getOperation()->template getAttrOfType<VFModeAttr>(
              VFModeAttr::name)) {
          return vfModeAttr.getValue();
        }

        return {};
      }

      void setVFMode(const VFMode& vfMode) {
        getOperation()->setAttr(VFModeAttr::name,
            VFModeAttr::get(getContext(), vfMode));
      }

      static constexpr StringLiteral kGMAddrArgIndicesAttrName = "gm_addr_args_indices";

      std::optional<SmallVector<size_t>> getGMAddrArgsIndices() {
        if (const auto indicesAttr =
            getOperation()->getAttrOfType<DenseI32ArrayAttr>(
              kGMAddrArgIndicesAttrName)) {
          SmallVector<size_t> res;
          for (int32_t i : indicesAttr.asArrayRef())
            res.emplace_back(static_cast<size_t>(i));
          return res;
        }

        return {};
      }

      void setGMAddrArgsIndices(const SmallVector<size_t>& indices) {
        SmallVector<int32_t> i32Indices;
        for (size_t i : indices)
          i32Indices.emplace_back(i);
        getOperation()->setAttr(kGMAddrArgIndicesAttrName,
            DenseI32ArrayAttr::get(getContext(), i32Indices));
      }

      bool isBuiltin() { return kBuiltins.contains(getName()); }

      // Builtins helpers
      template<typename CustomOpT>
      struct BuiltinInfoBase {
        const TCoreType coreType;
        const VFMode vfMode;

        const std::function<std::string(CustomOpT)> getOpLibraryCallName;

        const SmallVector<size_t> gmAddrArgsIndices;

        BuiltinInfoBase(TCoreType _coreType, VFMode _vfMode,
                        std::function<std::string(CustomOpT)> _getOpLibraryCallName,
                        const SmallVector<size_t>& _gmAddrArgsIndices) :
          coreType(_coreType), vfMode(_vfMode), getOpLibraryCallName(_getOpLibraryCallName),
          gmAddrArgsIndices(_gmAddrArgsIndices) {}

        virtual ~BuiltinInfoBase() {}
      };
  }];
}

def CustomOp : HIVM_CustomOp<"custom", [SinglePipeOpTrait]> {
  let summary = customOpSummary;

  let description = customOpDescription#customOpAttrsDescription#[{
      - Pipe     : which pipe to execute on, refer to PipeAttr.
  }]#customOpOptionalAttrsDescription#customOpTODODescription;

  let arguments = args;
  let results = res;

  let extraClassDeclaration = customOpBaseDecl#[{
    // Helper functions
    void setPipe(PIPE pipe) {
      getOperation()->setAttr(PipeAttr::name, PipeAttr::get(getContext(), pipe));
    }

    // Builtins helpers
    struct BuiltinInfo : public BuiltinInfoBase<CustomOp> {
      const PIPE pipe;

      BuiltinInfo(TCoreType _coreType, PIPE _pipe, VFMode _vfMode,
                  std::function<std::string(CustomOp)> _getOpLibraryCallName,
                  const SmallVector<size_t>& _gmAddrArgsIndices) :
        BuiltinInfoBase(_coreType, _vfMode, _getOpLibraryCallName, _gmAddrArgsIndices),
        pipe(_pipe) {}
    };

    // Map <builtin in name> <-> {CORE_TYPE, PIPE, VF_MODE, getOpLibraryCallName}
    static const DenseMap<StringRef, BuiltinInfo> kBuiltins;
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def CustomMacroOp : HIVM_CustomOp<"custom_macro", [MacroOpTrait]> {
  let summary = customOpSummary;

  let description =
      customOpDescription#customOpAttrsDescription#customOpTODODescription;

  let description = customOpDescription#customOpAttrsDescription#[{
      - InPipe   : which input pipe to execute on, refer to PipeAttr.
      - OutPipe  : which output pipe to execute on, refer to PipeAttr.
  }]#customOpOptionalAttrsDescription#customOpTODODescription;

  let arguments = args;
  let results = res;

  let extraClassDeclaration = customOpBaseDecl#[{
    static constexpr StringLiteral inPipeName = "hivm.pipe_in";
    static constexpr StringLiteral outPipeName = "hivm.pipe_out";

    PIPE getInPipe() {
      if (auto pipAttr =
          getOperation()->template getAttrOfType<PipeAttr>(inPipeName))
        return pipAttr.getPipe();

      return PIPE::PIPE_UNASSIGNED;
    }

    PIPE getOutPipe() {
      if (auto pipAttr =
          getOperation()->template getAttrOfType<PipeAttr>(outPipeName))
        return pipAttr.getPipe();

      return PIPE::PIPE_UNASSIGNED;
    }

    // Helper functions
    void setInPipe(PIPE pipe) {
      getOperation()->setAttr(inPipeName, PipeAttr::get(getContext(), pipe));
    }

    void setOutPipe(PIPE pipe) {
      getOperation()->setAttr(outPipeName, PipeAttr::get(getContext(), pipe));
    }

    // Builtins helpers
    struct BuiltinInfo : public BuiltinInfoBase<CustomMacroOp> {
      const PIPE inPipe;
      const PIPE outPipe;

      BuiltinInfo(TCoreType _coreType, PIPE _inPipe, PIPE _outPipe, VFMode _vfMode,
                  std::function<std::string(CustomMacroOp)> _getOpLibraryCallName,
                  const SmallVector<size_t>& _gmAddrArgsIndices) :
        BuiltinInfoBase(_coreType, _vfMode, _getOpLibraryCallName, _gmAddrArgsIndices),
        inPipe(_inPipe), outPipe(_outPipe) {}
    };

    // Map <builtin in name> <-> {CORE_TYPE, IN_PIPE, OUT_PIPE VF_MODE, getOpLibraryCallName}
    static const DenseMap<StringRef, BuiltinInfo> kBuiltins;
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
