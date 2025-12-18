//===- HIVMDMAOps.td - HIVM dialect DMA operations ---------*- tablegen -*-===//
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
// This is the definition file for HIVM dialect data movement action
// (DMA) operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMDMAOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMDMAOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/CopyOpInterface.td"

//===----------------------------------------------------------------------===//
// HIVM DMA Operations.
//===----------------------------------------------------------------------===//

class HIVM_DmaOp<string mnemonic, list<Trait> traits = []> :
  HIVM_StructuredOp<mnemonic, !listconcat(
    [AlwaysSpeculatable,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     DeclareOpInterfaceMethods<CopyOpInterface>
    ],
    traits)> {
  code DmaOpBaseDecl = [{
    ShapedType getSrcOperandType() {
      return cast<ShapedType>(getSrc().getType());
    }
    ShapedType getDstOperandType() {
      return cast<ShapedType>(getDst().getType());
    }
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }] # opNameDeclaration;
}

def LoadOp : HIVM_DmaOp<"load", [
  AttrSizedOperandSegments,
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  // Will automatically use the default uniform reassociation impl
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64]>,
  DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
    ["decomposeOperation", "getDecomposePhase"]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
]> {
  let summary = "HIVM data load operation";
  let description = [{
    Loads the data from the global memory to the local buffer.
    Currently only support loading to the unified buffer.

    Examples:
    ```mlir
    hivm.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - Supports both left and right padding.
    - `pad_value` should have the same element type as `src` and `dst`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_PadModeAttr>:$pad_mode,
                       Optional<AnyType>:$pad_value,
                       Optional<Index>:$left_padding_num,
                       Optional<AnyType>:$right_padding_num,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
                       Optional<AnyType>:$init_condition,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$may_implicit_transpose_with_last_axis
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "Value":$right_padding_num)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
               "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
               "Value":$left_padding_num, "bool":$init_out_buffer,
               "bool":$may_implicit_transpose_with_last_axis)>,               
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
    (`init_out_buffer` `=` $init_out_buffer^ )?
    (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
    (`init_condition` `=` $init_condition^ `:` type($init_condition))?
    (`may_implicit_transpose_with_last_axis` `=` $may_implicit_transpose_with_last_axis^ )?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl;
}

def StoreOp : HIVM_DmaOp<"store", [
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE3">,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  // Will automatically use the default uniform reassociation impl
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
]> {
  let summary = "HIVM data store operation";
  let description = [{
    Stores the data on local buffer to global memory.
    Currently only support storing data on the unified buffer.

    Examples:
    ```mlir
    hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `atomic_kind` is set, the kind is one of `add`, `max`, `min`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_AtomicKindAttr>:$atomic_kind,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$may_implicit_transpose_with_last_axis
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`atomic` `=` $atomic_kind^)?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl # [{
    // Return whether atomic store is enabled.
    bool isAtomic();

    // Return whether block-sync atomic store enabled is implemented by hardware.
    bool isHWAtomic();

    // Return whether block-sync atomic store enabled is implemented by software.
    bool isSWAtomic();
  }];
}

def CopyOp : HIVM_DmaOp<"copy", [
  SinglePipeOpTrait,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  // Will automatically use the default uniform reassociation impl
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I1, I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
]> {
  let summary = "HIVM data copy operation";
  let description = [{
    Copy the data between local memory hierarchies.
    Currently support:
      - UB to UB
      - UB to L1 (for Ascend910_95 series)

    Examples:
    ```mlir
    hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - Only support left padding.
    - `pad_value` should have the same element type as `src` and `dst`.

    ### Non-contigous reassocicative reshape
    `hivm.hir.copy` also supports copying non-contiguous data to contiguous storage, and vice versa.
    This can be seen as "expanding" or "collapsing" the data. The `collapse_reassociation` attribute is used to
    specify which axes are collapsed together.
    For example:
    ```mlir
    hivm.hir.copy ins(%src : memref<32x4xbf16, strided<[16, 1]>>) outs(%dst : memref<32x4xbf16, strided<[4, 1]>>)
      collapse_reassociation = [[0, 1]]
    ```
    Means that the 0th and 1st axes are collapsed contiguously.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_PadModeAttr>:$pad_mode,
                       Optional<AnyType>:$pad_value,
                       OptionalAttr<IndexListArrayAttr>:$collapse_reassociation
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`collapse_reassociation` `=` $collapse_reassociation^)?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl # [{
    // Declare functions necessary for SinglePipeOpTrait.
    PIPE getPipe();

    // Methods related to non-contiguous copy.
    SmallVector<AffineMap, 4> getReassociationMaps(bool isCollapse);
    SmallVector<ReassociationExprs, 4> getReassociationExprs(bool isCollapse);
    SmallVector<ReassociationIndices, 4> getReassociationIndices(bool isCollapse);
  }];
}

def FixpipeOp : HIVM_DmaOp<"fixpipe", [
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_FIX">,
  HIVMCoreTypeInterface, CubeCoreTypeTrait,
  HIVMUnitFlagEnabledInterface,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>
]> {
  let summary = "HIVM data copy operation from L0C to other memory hierarchies.";
#ifndef BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
  let description = [{
    Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies,
    with on-the-fly fixed function of pre-stage quantization,
    pre-stage ReLU, element-wise add, post-stage ReLU, post-stage quantization.
    Currently support:
      - L0C to OUT
      - L0C to L1
      - L0C to UB (for Ascend910_95 series)

    Additionally, Fixpipe is also capable of layout transform.
  }];
#else
  let description = [{
    Fixpipe is pipeline that performing data movement from L0C to OUT or L1,
    with on-the-fly fixed function of pre-stage quantization,
    pre-stage ReLU, element-wise add, post-stage ReLU, post-stage quantization.

    Additionally, Fixpipe is also capable of layout transform.

    ### Attributes

    #### 'dma_mode'
    HIVM data movement model from L0C to destination, There are three values: NZ2DN, NZ2ND, and NZ2NZ(normal).

    #### `dual_dst_mode`
    HIVM dual destination mode control. dual destination mode can be enabled only when nz2nd or normal data
    movement mode is enabled, and data movement is being performed from L0C to UB. Only supported on Ascend910_95 series.
  }];
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$dst,
                       Arg<Optional<I1>, [{
                        An optional condition to enable unit-flge mode,
                        useful if there is a dependency on a for loop to run at least once.
                       }]>:$unit_flag_cond,
#ifdef BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
                       DefaultValuedAttr<HIVM_FixpipeDMAModeAttr,
                         "FixpipeDMAMode::NZ2NZ">:$dma_mode,
                       DefaultValuedOptionalAttr<HIVM_FixpipeDualDstModeAttr,
                         "FixpipeDualDstMode::NO_DUAL">:$dual_dst_mode,
#else
                       OptionalAttr<UnitAttr>:$enable_nz2nd,
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
                       DefaultValuedOptionalAttr<HIVM_FixpipePreQuantModeAttr,
                         "FixpipePreQuantMode::NO_QUANT">:$pre_quant,
                       DefaultValuedOptionalAttr<HIVM_FixpipePreReluModeAttr,
                         "FixpipePreReluMode::NO_RELU">:$pre_relu,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$channel_split,
                       Arg<OptionalAttr<HIVM_UnitFlagAttr>, [{
                        Enable unit-flag for synchronization, it is
                        managed by inject-sync passes.
                       }]>:$unit_flag_mode
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$src, "Value":$dst,
#ifdef BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
                    CArg<"FixpipeDMAModeAttr", "nullptr">:$dma_mode,
                    CArg<"FixpipeDualDstModeAttr", "nullptr">:$dual_dst_mode,
#else
                    CArg<"UnitAttr", "nullptr">:$enable_nz2nd,
#endif
                    CArg<"FixpipePreQuantModeAttr", "nullptr">:$pre_quant,
                    CArg<"FixpipePreReluModeAttr", "nullptr">:$pre_relu, CArg<"BoolAttr", "nullptr">:$channel_split)
    >,
    OpBuilder<(ins "Type":$result, "Value":$src, "Value":$dst,
#ifdef BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
                    CArg<"FixpipeDMAModeAttr", "nullptr">:$dma_mode,
                    CArg<"FixpipeDualDstModeAttr", "nullptr">:$dual_dst_mode,
#else
                    CArg<"UnitAttr", "nullptr">:$enable_nz2nd,
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
                    CArg<"FixpipePreQuantModeAttr", "nullptr">:$pre_quant,
                    CArg<"FixpipePreReluModeAttr", "nullptr">:$pre_relu, CArg<"BoolAttr", "nullptr">:$channel_split)
    >
  ];
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
    (`->` type($result_tensor)^)?
  }];
#ifdef BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
  let hasVerifier = 1; // verify parameter according to hardware version
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
  let extraClassDeclaration = DmaOpBaseDecl # [{
    int getFixpipeState();
    int needFixpipePreFuse();
    bool hasStore();
    // Needed for unit-flag-enabled interface.
    ::std::optional<::mlir::hivm::UnitFlagAttr> getUnitFlagModeValue();
    ::std::optional<::mlir::Value> getUnitFlagModeCondition();
  }];
}

def ND2NZOp : HIVM_DmaOp<"nd2nz", [
  AttrSizedOperandSegments, SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
  HIVMCoreTypeInterface, CubeCoreTypeTrait,
  DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
  ["decomposeOperation", "getDecomposePhase"]>
]> {
  let summary = "HIVM data copy operation with on-the-fly ND to NZ layout transformation";
  let description = [{
    - `dst_continuous`: if present, signify that the source data is stored continuously
      in the destination buffer. This must be set in order for this op to be converted to
      library function call.
    Constraints:
    - if `init_out_buffer` is true, `pad_value` should have value.
  }];
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$dst,
                       OptionalAttr<UnitAttr>:$dst_continuous,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
                       Optional<AnyType>:$pad_value,
                       Optional<AnyType>:$init_condition
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`init_out_buffer` `=` $init_out_buffer^ )?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`init_condition` `=` $init_condition^ `:` type($init_condition))?
    (`->` type($result_tensor)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange" : $res, "Value" : $src, "Value" : $dst,
                   "UnitAttr" : $dst_continuous)>,
    OpBuilder<(ins "TypeRange" : $res, "Value" : $src, "Value" : $dst,
                   "UnitAttr" : $dst_continuous, "bool" : $init_out_buffer,
                   "Value" : $pad_value)>,
  ];
  let extraClassDeclaration = DmaOpBaseDecl;
}

def NZ2NDOp : HIVM_DmaOp<"nz2nd",
    [SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE3">,
     HIVMCoreTypeInterface, CubeCoreTypeTrait
    ]> {
  let summary = "HIVM data copy operation from L1 to Global Memory with NZ2ND conversion";
  let description = [{ NZ2ND does data movement from L1 to OUT with NZ2ND conversion. }];
  let arguments = (ins TensorOrMemref:$src, TensorOrMemref:$dst);
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
  let extraClassDeclaration = DmaOpBaseDecl;
}

def AtomicCasOp : HIVM_Op<"atomic_cas", [
  OperElemTypeConstraints<[0], [I8, I16, I32, F16, F32, I64]>
]> {
  let summary = "Atomic Compare-And-Swap (CAS) Op";
  let description = [{
    Compare-And-Swap (CAS) is an atomic operation that consists of three operands:
    Memory location (V), Expected old value (A), New value (B).
    The semantics of the operation are: the value of V is updated to B,
    only if the value of memory location V is equal to the expected old value A.
    The operation returns the original value of V regardless of whether it is updated or not.

    Constraints:
      1. The input memref and output memref must have the same rank
         and the same element type.

    Arguments:
      * `src0`: expected old value
      * `src1`: new value
      * `dst`: memory location in GM

    Examples:
    ```mlir
    hivm.hir.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
    %result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       TensorOrMemref:$dst
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
}

def AtomicXchgOp : HIVM_Op<"atomic_xchg", [
  OperElemTypeConstraints<[0], [I8, I16, I32, F16, F32, I64, BF16]>
]> {
  let summary = "Atomic Exchange Op";
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
    hivm.hir.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)
    %result = hivm.hir.atomic_cas ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       TensorOrMemref:$dst
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMDMAOPS_TD
