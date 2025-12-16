//===- HIVMMacroOps.td - HIVM dialect macro operations -----*- tablegen -*-===//
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
// This is the definition file for HIVM dialect macro operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMMACROOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMMACROOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// HIVM Macro Operations.
//===----------------------------------------------------------------------===//

class HIVM_MacroOp<string mnemonic, list<Trait> traits = []> :
  HIVM_StructuredOp<mnemonic, !listconcat(
    [MacroOpTrait,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>
    ],
    traits)
  > {
  code MacroOpBaseDecl = [{
    ArrayAttr getIndexingMaps() {
      return mlir::hivm::detail::getIndexingMapsImpl(*this);
    }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Local MMAD op definitions.
//===----------------------------------------------------------------------===//

class HIVM_LocalMmadOp<string mnemonic, list<Trait> traits = []> :
  HIVM_MacroOp<mnemonic, !listconcat(
    [AttrSizedOperandSegments,
     CubeCoreTypeTrait,
     HIVMUnitFlagEnabledInterface,
     MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">,
    ], traits)> {
  let arguments = (ins Arg<TensorOrMemref, "Matrix A">:$a,
                       Arg<TensorOrMemref, "Matrix B">:$b,
                       Arg<I1, [{
                        The condition under which
                        data in L0C is cleared before use
                       }]>:$init_condition,
                       Index:$real_m,
                       Index:$real_k,
                       Index:$real_n,
                       Arg<TensorOrMemref, "Matrix C">:$c,
                       Arg<Variadic<I64>, [{
                        Arguments required by mma_tile to insert 
                        synchronization instructions, they are 
                        managed by inject-sync passes.
                       }]>:$sync_related_args,
                       Arg<Optional<I1>, [{
                        An optional condition to enable unit-flge mode, 
                        useful if there is a dependency on a for loop to run at least once.
                       }]>:$unit_flag_cond,
                       Optional<TensorOrMemref>:$per_channel_bias,
                       Arg<OptionalAttr<UnitAttr>, [{
                        Matrix A is transposed before load
                       }]>:$a_transpose,
                       Arg<OptionalAttr<UnitAttr>, [{
                        Matrix B is transposed before load
                       }]>:$b_transpose,
                       Arg<OptionalAttr<UnitAttr>, [{
                        MMAD HF32 mode control.If true, FP32 data will be rounded
                        to HF32 by CUBE before matrix multiplication, which achieves
                        double performance of MMAD.F32 but losing precision.
                       }]>:$enable_HF32,
                       Arg<OptionalAttr<HIVM_UnitFlagAttr>, [{
                        Enable unit-flag for synchronization, it is 
                        managed by inject-sync passes.
                       }]>:$unit_flag_mode
                       );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  code localMmadBaseDecls = [{
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getCMutable();
    }
    bool isInitConstant(std::optional<bool> cst = std::nullopt);
    bool isInitFirstLoopIter();
    void setInitCondition(Value init);
    MatmulBiasMode getMatmulBiasMode();
    bool shouldDecomposeBiasByElementAdd();
    bool shouldDecomposeBiasByCrossLoopElementAdd();
    int getNumSyncRelatedArgs();
    // Needed for lower-hivm-to-std, should be updated if input operands got changed.
    SmallVector<Value> getInputOperands(bool includeSyncRelatedArgs = true);
    // Needed for unit-flag-enabled interface.
    ::std::optional<::mlir::hivm::UnitFlagAttr> getUnitFlagModeValue();
    ::std::optional<::mlir::Value> getUnitFlagModeCondition();
  }];
  string localMmadBaseDes = [{
    The computation logic is:

    ```
    C = C + A x B + (optional) channel_bias
    ```
  }];
  let assemblyFormat = [{
    attr-dict `ins` `(`
    $a
    `,` $b
    `,` $init_condition
    `,` $real_m
    `,` $real_k
    `,` $real_n
    (`,` $per_channel_bias^)?
    `:`
    type($a)
    `,` type($b)
    `,` type($init_condition)
    `,` type($real_m)
    `,` type($real_k)
    `,` type($real_n)
    (`,` type($per_channel_bias)^)? `)`
    `outs` `(` $c `:` type($c) `)`
    (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
    (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
    (`->` type($result_tensors)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result_tensors,
                   "Value":$a,
                   "Value":$b,
                   "Value":$init_condition,
                   "Value":$real_m,
                   "Value":$real_k,
                   "Value":$real_n,
                   "Value":$c,
                   CArg<"Value", "{}">:$per_channel_bias,
                   CArg<"UnitAttr", "{}">:$a_transpose,
                   CArg<"UnitAttr", "{}">:$b_transpose,
                   CArg<"UnitAttr", "{}">:$enable_HF32)>
  ];

  let hasVerifier = 1;
}

def MmadL1Op : HIVM_LocalMmadOp<"mmadL1", [
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
  OpLayoutInterface,
]> {
  let summary = [{
    Matrix Multiply and Add Op with inputs from L1 memory hierarchy.
  }];
  let description = localMmadBaseDes # [{
    Note: the rank of A, B, and C Matrix must be two.
  }];
  let extraClassDeclaration = localMmadBaseDecls # [{
    static StringRef getOpName() { return "mma_tile"; }

    // Declare functions necessary for OpLayoutInterface.
    FailureOr<DataLayoutAttr> getOperandALayout();
    FailureOr<DataLayoutAttr> getOperandBLayout();
    FailureOr<DataLayoutAttr> getOperandCLayout();
    FailureOr<DataLayoutAttr> getOperandBiasLayout();
    llvm::SmallDenseMap<Value, DataLayoutAttr> getOperandsTargetLayout();
    llvm::SmallDenseMap<Value, DataLayoutAttr> getOperandsCurrentLayout();
  }];
}

def BatchMmadL1Op : HIVM_LocalMmadOp<"batchMmadL1"> {
  let summary = [{
    Batch Matrix Multiply and Add Op with inputs from L1 memory hierarchy.
  }];
  let description = localMmadBaseDes # [{
    Note: the rank of A, B, and C Matrix must be three, where the 0-th dimension
    being the batch dimension.
  }];
  let extraClassDeclaration = localMmadBaseDecls;
}

//===----------------------------------------------------------------------===//
// HIVM Global MMAD op definitions.
//===----------------------------------------------------------------------===//

class HIVM_GlobalMmadOp<string mnemonic, list<Trait> traits = []> :
  HIVM_MacroOp<mnemonic,
    !listconcat([MacroOpPipeTrait<"PIPE::PIPE_MTE2, PIPE::PIPE_MTE3">,
                 DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface,
                   ["inferCoreType"]>,
                 AttrSizedOperandSegments
                ], traits)> {
  let results = (outs Variadic<AnyRankedTensor>:$result);
  string mmadOpBaseDescription = [{
    This operation takes three tiled matrices from the global memory as arguments:
      - `A` (ranked type): an `m x k` matrix
      - `B` (ranked type): an `k x n` matrix
      - `C` (ranked type): an `m x n` matrix

    Other arguments include:
      - `block_sizes`: data size of m, n, and k dimension processed on the L1 memory hierarchy
      - `process_sizes`: data size of m, n, and k dimension processed on the L0 memory hierarchy
      - (optional) `swizzle_offset`: continuous block number which swizzle schedule
      - (optional) `swizzle_direction`: block direction which swizzle schedule
      - (optional) `epilogue_p_tiles`: block number which compute attached op once handle

    The operation performed is represented as `C = A * B`.
    If `a_transpose` or `b_transpose` is present, the respective operand is
    loaded in a transposed manner.

    Optionally, this operation takes the following arguments:
      - `bias` (ranked type): bias value, which is a vector of shape `n`
      - `descale`: antiquantionzation value. Support 3 types:
      - `DescaleNull` : no descale.
      - `DescalePerChannel`: the shape of `descale` is equal to `n`.
      - `DescalePerTensor`: the shape of `descale` is equal to `1`.

    The operation performed is represented as `C = descale * (A * B + bias)`.
  }];

  code mmadOpBaseDecls = [{
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getCMutable();
    }
  }];

  let hasVerifier = 1;
}

def MatmulOp : HIVM_GlobalMmadOp<"matmul"> {
  let summary = "HIVM Matrix Multiply Op with inputs from global memory";
  let description =  mmadOpBaseDescription;
  let arguments = (ins AnyShaped:$a,
                       AnyShaped:$b,
                       Optional<AnyShaped>:$tilingParams,
                       Optional<AnyShaped>:$bias,
                       Optional<AnyShaped>:$descale,
                       OptionalAttr<UnitAttr>:$aTranspose,
                       OptionalAttr<UnitAttr>:$bTranspose,
                       OptionalAttr<HIVM_DescaleModeAttr>:$descaleMode,
                       Variadic<I64>:$blockSizes,
                       Variadic<I64>:$processSizes,
                       Optional<I64>:$swizzleOffset,
                       Optional<I64>:$swizzleDirection,
                       Optional<I64>:$epiloguePTiles,
                       AnyShaped:$c);

  let extraClassDeclaration = mmadOpBaseDecls # [{
    static StringRef getOpName() { return "matmul"; }
  }];

  let assemblyFormat = [{
      attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
      `outs` `(` $c `:` type($c) `)`
      (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
      (`bias` `=` $bias^ `:` type($bias) )?
      (`descale` `=` $descale^ `:` type($descale))?
      (`a_transpose` $aTranspose^)?
      (`b_transpose` $bTranspose^)?
      (`descale_mode` `=` $descaleMode^)?
      (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
      (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
      (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
      (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
      (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
      (`->` type($result)^)?
    }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$c,
                   "::mlir::UnitAttr":$aTranspose,
                   "::mlir::UnitAttr":$bTranspose), [{
      build($_builder, $_state, result, a, b, /*tilingParams=*/nullptr,
            /*bias=*/nullptr, /*descale=*/nullptr,
            aTranspose, bTranspose, /*descaleMode=*/nullptr, /*blockSizes=*/ValueRange{},
            /*processSizes=*/ValueRange{}, /*swizzleOffset=*/nullptr,
            /*swizzleDirection=*/nullptr, /*epiloguePTiles=*/nullptr, c);
    }]>,
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$c,
                   CArg<"Value", "Value()">:$bias,
                   CArg<"Value", "Value()">:$descale,
                   CArg<"::mlir::UnitAttr", "{}">:$aTranspose,
                   CArg<"::mlir::UnitAttr", "{}">:$bTranspose,
                   CArg<"::mlir::hivm::DescaleModeAttr", "{}">:$descaleMode), [{
      build($_builder, $_state, result, a, b, /*tilingParams=*/nullptr, bias, descale,
            aTranspose, bTranspose, descaleMode, /*blockSizes=*/ValueRange{},
            /*processSizes=*/ValueRange{}, /*swizzleOffset=*/nullptr,
            /*swizzleDirection=*/nullptr, /*epiloguePTiles=*/nullptr, c);
    }]>,
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$c,
                   CArg<"Value", "Value()">:$tilingParams,
                   CArg<"Value", "Value()">:$bias,
                   CArg<"Value", "Value()">:$descale,
                   CArg<"::mlir::UnitAttr", "{}">:$aTranspose,
                   CArg<"::mlir::UnitAttr", "{}">:$bTranspose,
                   CArg<"::mlir::hivm::DescaleModeAttr", "{}">:$descaleMode), [{
      build($_builder, $_state, result, a, b, tilingParams, bias, descale,
            aTranspose, bTranspose, descaleMode, /*blockSizes=*/ValueRange{},
          /*processSizes=*/ValueRange{}, /*swizzleOffset=*/nullptr,
          /*swizzleDirection=*/nullptr, /*epiloguePTiles=*/nullptr, c);
    }]>
  ];
}

def MixMatmulOp : HIVM_GlobalMmadOp<"mix_matmul"> {
  let summary = "HIVM (Mix) Matrix Multiply Op with inputs from global memory";
  let description = mmadOpBaseDescription # [{
    This operation also supports tile-level fusion with a post-vector
    function (hence it's a Mix op).
    `post_vector_func_ins` is used to specify the arguments.
    `comm_params` is used to specify communication related arguments (eg.
      topology, communicator, group, etc al) when fusing communication
      operators.
  }];
  let arguments = (ins AnyShaped:$a,
                       AnyShaped:$b,
                       Variadic<AnyShaped>:$postVecFuncIns,
                       Variadic<AnyShaped>:$workspaceIns,
                       Optional<AnyShaped>:$tilingParams,
                       Optional<AnyShaped>:$commParams,
                       Optional<AnyShaped>:$bias,
                       Optional<AnyShaped>:$descale,
                       OptionalAttr<UnitAttr>:$aTranspose,
                       OptionalAttr<UnitAttr>:$bTranspose,
                       OptionalAttr<HIVM_DescaleModeAttr>:$descaleMode,
                       Variadic<I64>:$blockSizes,
                       Variadic<I64>:$processSizes,
                       Optional<I64>:$swizzleOffset,
                       Optional<I64>:$swizzleDirection,
                       Optional<I64>:$epiloguePTiles,
                       AnyShaped:$c);

  let extraClassDeclaration = mmadOpBaseDecls # [{
    static StringRef getOpName() { return "mix_matmul"; }
  }];

  let assemblyFormat = [{
      attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
      (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
      (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
      `outs` `(` $c `:` type($c) `)`
      (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
      (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
      (`bias` `=` $bias^ `:` type($bias) )?
      (`descale` `=` $descale^ `:` type($descale))?
      (`a_transpose` $aTranspose^)?
      (`b_transpose` $bTranspose^)?
      (`descale_mode` `=` $descaleMode^)?
      (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
      (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
      (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
      (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
      (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
      (`->` type($result)^)?
    }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$c,
                   "::mlir::UnitAttr":$aTranspose,
                   "::mlir::UnitAttr":$bTranspose), [{
      build($_builder, $_state, result, a, b, /*postVecFuncIns=*/ValueRange{},
            /*workspaceIns=*/ValueRange{}, /*tilingParams=*/nullptr, /*commParams*/nullptr,
            /*bias=*/nullptr, /*descale=*/nullptr,
            aTranspose, bTranspose, /*descaleMode=*/nullptr,
            /*blockSizes=*/ValueRange{}, /*processSizes=*/ValueRange{},
            /*swizzleOffset=*/nullptr, /*swizzleDirection=*/nullptr, /*epiloguePTiles=*/nullptr, c);
    }]>,
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$c,
                   CArg<"Value", "Value()">:$bias,
                   CArg<"Value", "Value()">:$descale,
                   CArg<"::mlir::UnitAttr", "{}">:$aTranspose,
                   CArg<"::mlir::UnitAttr", "{}">:$bTranspose,
                   CArg<"::mlir::hivm::DescaleModeAttr", "{}">:$descaleMode), [{
      build($_builder, $_state, result, a, b, /*postVecFuncIns=*/ValueRange{},
            /*workspaceIns=*/ValueRange{}, /*tilingParams=*/nullptr, /*commParams*/nullptr, bias, descale,
            aTranspose, bTranspose, descaleMode,
            /*blockSizes=*/ValueRange{}, /*processSizes=*/ValueRange{},
            /*swizzleOffset=*/nullptr, /*swizzleDirection=*/nullptr,
            /*epiloguePTiles=*/nullptr, c);
    }]>,
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$c,
                   CArg<"ValueRange", "{}">:$postVecFuncIns,
                   CArg<"ValueRange", "{}">:$workspaceIns,
                   CArg<"Value", "Value()">:$tilingParams,
                   CArg<"Value", "Value()">:$commParams,
                   CArg<"Value", "Value()">:$bias,
                   CArg<"Value", "Value()">:$descale,
                   CArg<"::mlir::UnitAttr", "{}">:$aTranspose,
                   CArg<"::mlir::UnitAttr", "{}">:$bTranspose,
                   CArg<"::mlir::hivm::DescaleModeAttr", "{}">:$descaleMode), [{
      build($_builder, $_state, result, a, b, postVecFuncIns, workspaceIns, tilingParams, commParams, bias, descale,
            aTranspose, bTranspose, descaleMode,
            /*blockSizes=*/ValueRange{}, /*processSizes=*/ValueRange{},
            /*swizzleOffset=*/nullptr, /*swizzleDirection=*/nullptr,
            /*epiloguePTiles=*/nullptr, c);
    }]>
  ];
}

def MixGroupMatmulOp : HIVM_GlobalMmadOp<"mix_group_matmul"> {
  let summary = "HIVM (Mix) Matrix Group Multiply Op with inputs from global memory";
  let description = mmadOpBaseDescription # [{
    This operation also supports tile-level fusion with a post-vector
    function (hence it's a Mix op)
    `tokens_per_expert` specify how matmuls are distrubuted to different experts
    `post_vector_func_ins` is used to specify the arguments.
    `post_vector_func_outs` is used to specify the outputs.
    `comm_params` is used to specify communication related arguments (eg.
      topology, communicator, group, etc al) when fusing communication
      operators.
  }];
  let arguments = (ins AnyShaped:$a, // weight, 3D
                       AnyShaped:$b, // tokens, 2D
                       AnyShaped:$tokens_per_expert, // tokens_per_expert, 1D
                       Variadic<AnyShaped>:$postVecFuncIns,
                       Variadic<AnyShaped>:$postVecFuncOuts,
                       Variadic<AnyShaped>:$workspaceIns,
                       Optional<AnyShaped>:$tilingParams,
                       Optional<AnyShaped>:$commParams,
                       Optional<AnyShaped>:$bias,
                       Optional<AnyShaped>:$descale,
                       OptionalAttr<UnitAttr>:$aTranspose,
                       OptionalAttr<UnitAttr>:$bTranspose,
                       OptionalAttr<HIVM_DescaleModeAttr>:$descaleMode,
                       Variadic<I64>:$blockSizes,
                       Variadic<I64>:$processSizes,
                       Optional<I64>:$swizzleOffset,
                       Optional<I64>:$swizzleDirection,
                       Optional<I64>:$epiloguePTiles,
                       AnyShaped:$c);

  let extraClassDeclaration = mmadOpBaseDecls # [{
    static StringRef getOpName() { return "mix_group_matmul"; }
  }];

  let assemblyFormat = [{
      attr-dict `ins` `(` $a `,` $b `,` $tokens_per_expert `:` type($a) `,` type($b) `,` type($tokens_per_expert) `)`
      (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
      (`post_vector_func_outs` `(` $postVecFuncOuts^ `:` type($postVecFuncOuts) `)`) ?
      (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
      `outs` `(` $c `:` type($c) `)`
      (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
      (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
      (`bias` `=` $bias^ `:` type($bias) )?
      (`descale` `=` $descale^ `:` type($descale))?
      (`a_transpose` $aTranspose^)?
      (`b_transpose` $bTranspose^)?
      (`descale_mode` `=` $descaleMode^)?
      (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
      (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
      (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
      (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
      (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
      (`->` type($result)^)?
    }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$tokens_per_expert,
                   "Value":$c,
                   "::mlir::UnitAttr":$aTranspose,
                   "::mlir::UnitAttr":$bTranspose), [{
      build($_builder, $_state, result, a, b, tokens_per_expert, /*postVecFuncIns=*/ValueRange{}, /*postVecFuncOuts=*/ValueRange{},
            /*workspaceIns=*/ValueRange{}, /*tilingParams=*/nullptr, /*commParams*/nullptr,
            /*bias=*/nullptr, /*descale=*/nullptr,
            aTranspose, bTranspose, /*descaleMode=*/nullptr,
            /*blockSizes=*/ValueRange{}, /*processSizes=*/ValueRange{},
            /*swizzleOffset=*/nullptr, /*swizzleDirection=*/nullptr, /*epiloguePTiles=*/nullptr, c);
    }]>,
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$tokens_per_expert,
                   "Value":$c,
                   CArg<"Value", "Value()">:$bias,
                   CArg<"Value", "Value()">:$descale,
                   CArg<"::mlir::UnitAttr", "{}">:$aTranspose,
                   CArg<"::mlir::UnitAttr", "{}">:$bTranspose,
                   CArg<"::mlir::hivm::DescaleModeAttr", "{}">:$descaleMode), [{
      build($_builder, $_state, result, a, b, tokens_per_expert, /*postVecFuncIns=*/ValueRange{}, /*postVecFuncOuts=*/ValueRange{},
            /*workspaceIns=*/ValueRange{}, /*tilingParams=*/nullptr, /*commParams*/nullptr, bias, descale,
            aTranspose, bTranspose, descaleMode,
            /*blockSizes=*/ValueRange{}, /*processSizes=*/ValueRange{},
            /*swizzleOffset=*/nullptr, /*swizzleDirection=*/nullptr,
            /*epiloguePTiles=*/nullptr, c);
    }]>,
    OpBuilder<(ins "TypeRange":$result,
                   "Value":$a,
                   "Value":$b,
                   "Value":$tokens_per_expert,
                   "Value":$c,
                   CArg<"ValueRange", "{}">:$postVecFuncIns,
                   CArg<"ValueRange", "{}">:$postVecFuncOuts,
                   CArg<"ValueRange", "{}">:$workspaceIns,
                   CArg<"Value", "Value()">:$tilingParams,
                   CArg<"Value", "Value()">:$commParams,
                   CArg<"Value", "Value()">:$bias,
                   CArg<"Value", "Value()">:$descale,
                   CArg<"::mlir::UnitAttr", "{}">:$aTranspose,
                   CArg<"::mlir::UnitAttr", "{}">:$bTranspose,
                   CArg<"::mlir::hivm::DescaleModeAttr", "{}">:$descaleMode), [{
      build($_builder, $_state, result, a, b, tokens_per_expert, postVecFuncIns, postVecFuncOuts, workspaceIns, tilingParams, commParams, bias, descale,
            aTranspose, bTranspose, descaleMode,
            /*blockSizes=*/ValueRange{}, /*processSizes=*/ValueRange{},
            /*swizzleOffset=*/nullptr, /*swizzleDirection=*/nullptr,
            /*epiloguePTiles=*/nullptr, c);
    }]>
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMMACROOPS_TD
