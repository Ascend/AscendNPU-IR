//===- HIVMOps.td - HIVM dialect operation definitions -----*- tablegen -*-===//
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
// This is the definition file for HIVM dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
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

def GetSysCntOp : HIVM_Op<"get_sys_cnt", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sys cnt of the current device";
  let description = [{
    This op get the sys cnt of the current device.
    This op will be lowered to `GetSysCntInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def SetMaskNormOp : HIVM_Op<"set_mask_norm"> {
  let summary = "hivm set mask norm";
  let assemblyFormat = "attr-dict";
}

def LoadScalarOp : HIVM_Op<"load_scalar"> {
  let summary = "hivm load scalar";
  let arguments = (ins LLVMPointerType:$addr);
  let results = (outs AnyTypeOf<[AnyInteger, AnyFloat]>:$result);
  let assemblyFormat = "attr-dict $addr `:` type($addr) `->` type($result)";
  let hasVerifier = 1;
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

def SetCtrlOp : HIVM_Op<"set_ctrl"> {
  let summary = "hivm set ctrl operation";
  let description = "This op configures control bits by setting a specific bit to 0 or 1.";
  let arguments = (ins BoolAttr:$enable,
                       I64Attr:$idx);
  let assemblyFormat = "$enable `at` `ctrl``[` $idx `]` attr-dict";
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
    DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>]> {
  let summary = "HIVM layout conversion operation.";
  let description = [{
    The `convert_layout` operation converts a memref with one layout
    to another. The data is not copied or modified.

    The operation supports mixed static/dynamic output shapes where static
    dimensions are specified in the `static_output_shape` attribute and
    dynamic dimensions are provided as SSA values in `output_shape`.
    Dynamic dimensions are indicated by `ShapedType::kDynamic` in
    the static shape attribute.
  }];

  let arguments = (ins TensorOrMemref:$source,
                       HIVM_DataLayoutAttr:$srcLayout,
                       HIVM_DataLayoutAttr:$dstLayout,
                       DenseI64ArrayAttr:$static_output_shape,
                       Variadic<Index>:$output_shape);
  let results = (outs TensorOrMemref:$result);

  let assemblyFormat = [{
    $source
    `output_shape` custom<DynamicIndexList>($output_shape, $static_output_shape)
    attr-dict `:` `(`type($source)`)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    Value getViewSource() { return getSource(); }

    /// Check if this operation has pure tensor semantics
    bool hasPureTensorSemantics() {
      return isa<TensorType>(getSource().getType()) &&
             isa<TensorType>(getResult().getType());
    }

    /// Return the output shape as a list of OpFoldResult (mixing static and dynamic)
    SmallVector<OpFoldResult> getMixedOutputShape() {
      Builder b(getContext());
      return getMixedValues(getStaticOutputShape(), getOutputShape(), b);
    }

    /// Return the number of dynamic dimensions
    int64_t getNumDynamicDims() {
      return getOutputShape().size();
    }

    /// Return a vector of all static shape dimensions
    SmallVector<int64_t> getStaticOutputShapeVec() {
      return SmallVector<int64_t>(getStaticOutputShape());
    }
  }];

  let builders = [
    // Convenience builder with mixed OpFoldResult shape
    OpBuilder<(ins "Type":$resultType,
                   "Value":$source,
                   "DataLayoutAttr":$srcLayout,
                   "DataLayoutAttr":$dstLayout,
                   "ArrayRef<OpFoldResult>":$outputShape)>,
    OpBuilder<(ins "Type":$resultType,
                   "Value":$source,
                   "DataLayoutAttr":$srcLayout,
                   "DataLayoutAttr":$dstLayout)>
  ];

  let hasVerifier = 1;
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
          [MemoryEffects<[MemRead, MemWrite]>,
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
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    static StringRef getOpName() { return "debug"; }
  }];
}

def InitDebugOp : HIVM_Op<"init_debug", [CubeVectorCoreTypeTrait]> {
  let summary = "Init func for device-side debugging";
  let assemblyFormat = [{
    attr-dict
  }];
}

def FinishDebugOp : HIVM_Op<"finish_debug", [CubeVectorCoreTypeTrait]> {
  let summary = "Finish func for device-side debugging";
  let assemblyFormat = [{
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// GatherLoadOp
//===----------------------------------------------------------------------===//

def GatherLoadOp : HIVM_Op<"gather_load", [
    DestinationStyleOpInterface,
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize,
    AllElementTypesMatch<["base", "dst"]>,
    AllRanksMatch<["indices", "dst"]>,
    AllShapesMatch<["indices", "dst"]>,
    OptionalTypesMatchWith<"base and other have same element type",
                           "base", "other", "getElementTypeOrSelf($_self)">
]> {
  let summary = [{
    Perform sparse memory loading with optional mask and other
  }];

  let description = [{
    This operation takes a source memory buffer and a tensor of offsets,
    and produces an output tensor by gathering elements from the source
    at the specified offset positions. The operation supports masking
    and provides fallback values for masked-out positions.
    The meanings of the arguments are as follows:
      - base : base address of the data to be loaded.
      - indices : indices of the data to be loaded.
      - burst_len : indicates the continuity of index values on the innermost dim.
      - mask : (optional) load mask that controls which positions should be
               actually load and which positions should not be loaded. It
               is meaningful in call cases except block pointers.
      - other : (optional) a default/fallback value when the corresponding
                mask element is false. It is meaningful in call cases except
                block pointers.
      - dst : destination buffer/tensor specifying output shape and type.
      - cache : (optional) a enum hint that controls how the NPU L1 caches the
                data being loaded from global memory.
      - evict : (optional) a enum hint that controls the priority of data being
                removed from NPU cache when cache space is full.
      - isVolatile : (optional) the load op marked as volatile is not allowed
                     to be optimized in any way.
  }];
  let arguments = (ins AnyMemRef:$base,
                       RankedTensorOf<[I32, I64]>:$indices,
                       AnyTypeOf<[I32, I64]>:$burst_len,
                       Optional<RankedTensorOf<[I1]>>:$mask,
                       Optional<AnyTypeOf<[AnyInteger, AnyFloat]>>:$other,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_CacheModifierAttr>:$cache,
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$evict,
                       OptionalAttr<BoolAttr>:$isVolatile);
  let results = (outs Optional<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $base `:` type($base) `,` $indices `:` type($indices) `,`
    $burst_len `:` type($burst_len)
    (`,` $mask `:` type($mask)^)? (`,` $other `:` type($other)^)? `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScatterStoreOp
//===----------------------------------------------------------------------===//

def ScatterStoreOp : HIVM_Op<"scatter_store", [
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize,
    AllElementTypesMatch<["base", "data"]>,
    AllRanksMatch<["indices", "data"]>,
    AllShapesMatch<["indices", "data"]>
]> {
  let summary = [{
    Performs sparse memory storing with optional mask
  }];

  let description = [{
    This operation takes a source tensor and a tensor of offsets from UB,
    and stores elements from the source into the destination GM buffer
    at the specified offset positions. The operation supports masking to
    conditionally control which elements are stored.
    The meanings of the arguments are the same as these of the gather_load.
  }];
  let arguments = (ins RankedTensorOf<[I32, I64]>:$indices,
                       AnyRankedTensor:$data,
                       AnyTypeOf<[I32, I64]>:$burst_len,
                       Optional<RankedTensorOf<[I1]>>:$mask,
                       TensorOrMemref:$base,
                       OptionalAttr<HIVM_CacheModifierAttr>:$cache,
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$evict);
  let results = (outs Optional<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $indices `:` type($indices) `,`
    $data `:` type($data) `,` $burst_len `:` type($burst_len)
    (`,` $mask `:` type($mask)^)? `)`
    `outs` `(` $base `:` type($base) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getBaseMutable();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

def CustomOp
    : HIVM_StructuredOp<
          "custom",
          [AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>,
           SinglePipeOpTrait, DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
           DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
                                     ["getIteratorTypesArray",
                                      "getIndexingMaps"]>,
           DeclareOpInterfaceMethods<
               HIVMInferCoreTypeInterface, ["inferCoreType"]>]> {
  let summary = [{
    Custom operation is a generic op interface for users to write their own custom implementation.

    Scenarios:
      1. Existing operations could not fulfill the desired functionality.
      2. Existing operations could fulfill the functionality, but overall performance is not optimal.
      3. Desire for private operation.
  }];

  let description = [{
    General interface for custom op, where:
      - name : unique op name.

               Note : there are names reserved for builtins, usually starts with "__builtin".
                      Compiler will link these builtins to self-contained template library,
                      which comes together within bishengir-compile.

                      For normal names/cases, user needs to specify implementation location/compilation command,
                      and all ther necessary informations.

      - inputs : input parameters.
      - outputs : output results, designated "init" operands, which act as initial values for the results
                  of the operation or the init locations to which the results of the op will be written.

    In order to adapt to future enhancements quickly and dynamically, custom op relies on attributes
    to retreive necessary information, required informations are:
      - CoreType : which core type to execute on, refer to TCoreTypeAttr.
      - Pipe     : which pipe to execute on, refer to PipeAttr.
      - VFMode   : which mode to run on vector units, refer to VFModeAttr.
                   this attribute is ignored when core type is cube.

                   Note : for builtins, user could specify these informations or not,
                          compiler will help to check the correctness and canonicalize.

    TODO:
      - Multi Pipe : custom op wants to use multiple pipes, which is a MacroOp in HIVM's context.
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$inputs,
      Variadic<AnyType>:$outputs, Variadic<AnyType>:$temp_buffers);

  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    // Convenience builder with empty temp_buffers.
    OpBuilder<(ins "StringRef":$name,
                   "TypeRange":$resultTypes,
                   "ValueRange":$inputs,
                   "ValueRange":$outputs)>
  ];

  let extraClassDeclaration = [{
      PIPE getPipe();

      ::mlir::MutableOperandRange getDpsInitsMutable() {
          return getOutputsMutable();
      }

      // Helper functions
      void setPipe(PIPE);

      std::optional<TCoreType> getCoreType();
      void setCoreType(TCoreType);

      std::optional<VFMode> getVFMode();
      void setVFMode(VFMode);

      bool isBuiltin();

      static constexpr StringLiteral kExtraBuffersTypesName = "extra_buffers_types";
      static constexpr StringLiteral kExtraBuffersSizesName = "extra_buffers_sizes";

      SmallVector<std::pair<Type, int64_t>> getExtraBuffersInfo();

      static constexpr StringLiteral kIteratorTypesName = "iterator_types";

      static constexpr StringLiteral kIndexingMapName = "indexing_map";

      static constexpr StringLiteral kArgAttrsName = "arg_attrs";

      static constexpr StringLiteral kAlignDimName = "align_dim";

      static constexpr StringLiteral kMaxRankAttrName = "max_rank";
      int getMaxRank();
      static constexpr StringLiteral kSymbolAttrName = "symbol";
      std::string getSymbol();

      // Builtins helpers
      struct BuiltinInfo {
        TCoreType coreType;
        PIPE pipe;
        VFMode vfMode;
      };

      // Map <builtin in name> <-> {CORE_TYPE, PIPE, VF_MODE}
      static const DenseMap<StringRef, BuiltinInfo> kBuiltins;
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

// TODO: Add CustomMacroOp

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
