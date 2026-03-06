//===- HIVMDMAOps.td - HIVM dialect DMA operations ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  AttrSizedOperandSegments, StaticMaxRankTrait<3>,
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  // Will automatically use the default uniform reassociation impl
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64, F8E4M3FN, F8E5M2]>,
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
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$eviction_policy
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "Value":$left_padding_num)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "Value":$right_padding_num)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "bool":$init_out_buffer)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "bool":$init_out_buffer,
                   "Value":$init_condition)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "hivm::EvictionPolicyAttr":$eviction_policy)>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "hivm::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num,
                   "hivm::EvictionPolicyAttr":$eviction_policy)>,
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
    (`eviction_policy` `=` $eviction_policy^)?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl;
}

def StoreOp : HIVM_DmaOp<"store", [
  StaticMaxRankTrait<3>,
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE3">,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  // Will automatically use the default uniform reassociation impl
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64, F8E4M3FN, F8E5M2]>,
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
                       OptionalAttr<HIVM_AtomicKindAttr>:$atomic_kind
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
  SinglePipeOpTrait, StaticMaxRankTrait<3>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  // Will automatically use the default uniform reassociation impl
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I1, I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64, F8E4M3FN, F8E5M2]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
]> {
  let summary = "HIVM data copy operation";
  let description = [{
    Copy the data between local memory hierarchies.
    Currently support:
      - UB to UB
      - UB to L1 (for Ascend950 series)

    Examples:
    ```mlir
    hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - Only support left padding.
    - `pad_value` should have the same element type as `src` and `dst`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_PadModeAttr>:$pad_mode,
                       Optional<AnyType>:$pad_value
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
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl # [{
    // Declare functions necessary for SinglePipeOpTrait.
    PIPE getPipe();
  }];
}

def FixpipeOp : HIVM_DmaOp<"fixpipe", [
  AttrSizedOperandSegments,
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_FIX">,
  HIVMCoreTypeInterface, CubeCoreTypeTrait, NoMaxRankTrait,
  HIVMUnitFlagEnabledInterface, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>
]> {
  let summary =
      "HIVM data copy operation from L0C to other memory hierarchies.";
  let description = [{
    Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies,
    with on-the-fly fixed function of pre-stage quantization,
    pre-stage ReLU, element-wise add, post-stage ReLU, post-stage quantization.
    Currently support:
      - L0C to OUT
      - L0C to L1
      - L0C to UB (for Ascend950 series)

    Additionally, Fixpipe is also capable of layout transform.

    ### Attributes

    #### 'dma_mode'
    HIVM data movement model from L0C to destination, There are three values: NZ2DN, NZ2ND, and NZ2NZ(normal).

    #### `dual_dst_mode`
    HIVM dual destination mode control. dual destination mode can be enabled only when nz2nd or normal data
    movement mode is enabled, and data movement is being performed from L0C to UB. Only supported on Ascend950 series.
  }];
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$dst,
                       Arg<Variadic<I1>, [{
                        An optional condition to enable unit-flag mode, 
                        useful if there is a dependency on a for loop to run at least once.
                       }]>:$unit_flag_cond,
                       DefaultValuedAttr<HIVM_FixpipeDMAModeAttr,
                         "FixpipeDMAMode::NZ2NZ">:$dma_mode,
                       DefaultValuedOptionalAttr<HIVM_FixpipeDualDstModeAttr,
                         "FixpipeDualDstMode::NO_DUAL">:$dual_dst_mode,
                       DefaultValuedOptionalAttr<HIVM_FixpipePreQuantModeAttr,
                         "FixpipePreQuantMode::NO_QUANT">:$pre_quant,
                       DefaultValuedOptionalAttr<HIVM_FixpipePreReluModeAttr,
                         "FixpipePreReluMode::NO_RELU">:$pre_relu,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$channel_split,
                       OptionalAttr<UnitFlagArrayAttr>:$unit_flag_mode,
                       Optional<AnyFloat>:$quant_scale
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$src, "Value":$dst,
                    CArg<"FixpipeDMAModeAttr", "nullptr">:$dma_mode,
                    CArg<"FixpipeDualDstModeAttr", "nullptr">:$dual_dst_mode, CArg<"FixpipePreQuantModeAttr", "nullptr">:$pre_quant,
                    CArg<"FixpipePreReluModeAttr", "nullptr">:$pre_relu, 
                    CArg<"BoolAttr", "nullptr">:$channel_split,
                    CArg<"Value", "nullptr">:$quant_scale)
    >,
    OpBuilder<(ins "Type":$result, "Value":$src, "Value":$dst,
                    CArg<"FixpipeDMAModeAttr", "nullptr">:$dma_mode,
                    CArg<"FixpipeDualDstModeAttr", "nullptr">:$dual_dst_mode, CArg<"FixpipePreQuantModeAttr", "nullptr">:$pre_quant,
                    CArg<"FixpipePreReluModeAttr", "nullptr">:$pre_relu,
                    CArg<"BoolAttr", "nullptr">:$channel_split,
                    CArg<"Value", "nullptr">:$quant_scale)
    >
  ];
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`quant_scale` `=` $quant_scale^ `:` type($quant_scale) )?
    (`dual_dst_mode` `=` $dual_dst_mode^)?
    (`unit_flag_mode` `(` $unit_flag_mode^ `)` )?
    (`unit_flag_cond` `(` $unit_flag_cond^ `)` )?
    (`->` type($result_tensor)^)?
  }];
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = DmaOpBaseDecl # [{
    int getFixpipeState();
    int needFixpipePreFuse();
    bool hasStore();
  }];
}

def ND2NZOp : HIVM_DmaOp<"nd2nz", [
  AttrSizedOperandSegments, SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE2">,
  HIVMCoreTypeInterface, CubeCoreTypeTrait, NoMaxRankTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
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
    [StaticMaxRankTrait<2>,
     SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE3">,
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

def L12UBOp : HIVM_DmaOp<"l12ub",
    [StaticMaxRankTrait<2>,
     SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE1">,
     HIVMCoreTypeInterface, CubeCoreTypeTrait
    ]> {
  let summary = "HIVM data copy operation from L1 to UB Memory with NZ2ND conversion";
  let description = [{ NZ2ND does data movement from L1 to UB with NZ2ND conversion. }];
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

def AtomicCasOp : HIVM_Op<"atomic_cas",
    [NoLibraryFunctionTrait,
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

def AtomicXchgOp : HIVM_Op<"atomic_xchg",
    [NoLibraryFunctionTrait,
     OperElemTypeConstraints<[0], [I8, I16, I32, F16, F32, I64]>
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
