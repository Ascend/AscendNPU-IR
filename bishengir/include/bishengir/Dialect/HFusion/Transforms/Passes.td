//===-- Passes.td - HFusion dialect pass definition file ------*-tablegen-*-==//
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

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_PASSES
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HFusionOpFusion : Pass<"hfusion-fuse-ops", "ModuleOp"> {
  let summary = "HFusion Fuse operations on tensors";
  let constructor = "hfusion::createHFusionOpFusionPass()";
  let dependentDialects = ["hfusion::HFusionDialect", ];
  let options =
      [Option<"outputMode", "output-mode", "::mlir::hfusion::OutputMode",
              "hfusion::OutputMode::Multiple",
              "Outlined function output mode (default is multi, can also use "
              "single or single-aggr)",
              [{::llvm::cl::values(
               clEnumValN(hfusion::OutputMode::Multiple, "multi",
                          "Outlined function with multiple outputs"),
               clEnumValN(hfusion::OutputMode::Single, "single",
                          "Outlined function with single output"),
               clEnumValN(hfusion::OutputMode::SingleAggressive, "single-aggr",
                          "Outlined function with single output, fusing "
                          "more aggressively by duplicating operations"))}]>,
       Option<"fusionMode", "fusion-mode", "::mlir::hfusion::FusionKind",
              "hfusion::FusionKind::Unknown",
              "Fusion kind is determined by label">,
       Option<"alwaysInline", "always-inline", "bool", /*default=*/"false",
              "Enable always inline for the outline function.">,
       Option<"moveOutToParam", "move-out-to-param", "bool",
              /*default=*/"true",
              "Whether move the tensor out to params or not">,
       Option<"maxHorizontalFusionSize", "max-horizontal-fusion-size", "int",
              /*default=*/"-1",
              "Maximum horizontal (non-dependent) fusioning allowed, -1 for "
              "unlimited attempt"
              "of horizontal fusion">,
       Option<"enableMultiKernel", "multi-kernel", "bool",
              /*default=*/"false",
              "When disabled, graph must fuse as single kernel; when enabled, "
              "outline multiple kernels.">,
       Option<"enableSymbolAnalysis", "enable-symbol-analysis", "bool",
              /*default=*/"false",
              "Enable symbol dialect analysis.">];
}

def AutoSchedule : Pass<"hfusion-auto-schedule", "ModuleOp"> {
  let summary = "Auto schedule fused kernels.";
  let constructor = "mlir::hfusion::createHFusionAutoSchedulePass()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect",
                           "transform::TransformDialect", "arith::ArithDialect",
                           "hacc::HACCDialect"];
  let options =
      [Option<"blockDim", "block-dim", "unsigned", "1",
              "Number of blocks to use">,
       Option<"enableAutoMultiBuffer", "enable-auto-multi-buffer", "bool",
              "false", "Enable auto multi buffer">,
       Option<"enableDeterministicComputing", "enable-deterministic-computing", "bool",
              "true", "Enable deterministic computing">,
       Option<"maxBufferCntTuning", "max-buffer-count-tuning", "int64_t", "0",
              "allow maxBufferCnt tuning">,
       Option<"enableCountBufferDmaOpt", "enable-count-buffer-dma-opt", "bool",
              "false",
              "If enabled, the buffer used by DMA operations will not be"
              "reused by Vector operations">,
       Option<"enableManageHostResources", "enable-manage-host-resources",
              "bool", "false", "Enable managing resource for Host functions">,
       ListOption<"cubeTilingTuning", "cube-tiling-tuning", "int64_t",
              "allow cube tiling params tuning">,
       Option<"externalTilingFuncPath", "external-tiling-func-path", "std::string",
              "\"-\"", "auto add external tiling func">,
       Option<"enableSymbolAnalysis", "enable-symbol-analysis", "bool",
              "false", "Enable symbol analysis for tiling and fusion">,
  ];
}

def AddFFTSAddr : Pass<"hfusion-add-ffts-addr", "ModuleOp"> {
  let summary = "Add ffts base address to func param and annotation";
  let constructor = "mlir::hfusion::createAddFFTSAddrPass()";
  let dependentDialects = ["hacc::HACCDialect"];
  let options = [Option<
      "forceAddFFTSAddr", "force-add-ffts-addr", "int",
      /*default=*/"-1",
      "Force adding FFTS base addr to the user specified param "
      "location. Default value -1 means no insertion. 0 means"
      "insert to the first param location.">];
}

def ConvertGenericToNamedOp
    : Pass<"hfusion-convert-generic-to-named", "func::FuncOp"> {
  let summary =
      "Convert linalg generic ops to linalg named ops and hfusion named ops.";
  let constructor = "mlir::hfusion::createConvertGenericToNamedOpPass()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect"];
}

def FlattenOps : Pass<"hfusion-flatten-ops", "func::FuncOp"> {
  let constructor = "mlir::hfusion::createFlattenOpsPass()";
  let summary = "Flatten linalg and hfusion ops.";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect",
                           "memref::MemRefDialect", "tensor::TensorDialect"];

  let options =
      [Option<
           "flattenMode", "flatten-mode", "hfusion::FlattenMode",
           "hfusion::FlattenMode::Greedy",
           "Flatten mode, tidy mode will do an analysis on the entire function",
           [{::llvm::cl::values(
               clEnumValN(hfusion::FlattenMode::Greedy, "greedy",
                          "Outlined function with greedy elemwise flatten"),
               clEnumValN(hfusion::FlattenMode::Tidy, "tidy",
                          "Outlined function with analysis"))}]>,
       Option<"skipHost", "skip-host", "bool", /*default=*/"false",
              "Whether to skip the host function or not">,
       Option<"multiDynamicShape", "multi-dynamic-shape", "bool", /*default=*/"true",
              "Whether to collapse multiple dynamic shape or not">,
  ];
}

def TensorResToOutParams
    : Pass<"hfusion-tensor-results-to-out-params", "ModuleOp"> {
  let summary = "Move tensor results to function output parameters";
  let constructor = "hfusion::createTensorResToOutParamsPass()";
  let dependentDialects = ["hacc::HACCDialect"];
  let options =
      [ListOption<
           "includeSymbols", "include-symbols", "std::string",
           "Comma separated list of symbols that should apply this "
           "transformation. "
           "If empty, the default behavior is to apply transformation to "
           "all functions.">,
       Option<"enableManageHostResources", "enable-manage-host-resources",
              "bool", "false", "Enable managing resource for Host functions">,
  ];
}

def HFusionInlineBrc : Pass<"hfusion-inline-brc", "func::FuncOp"> {
  let summary = "Inline broadcast-like ops.";
  let constructor = "hfusion::createHFusionInlineBrcPass()";
}

def OutlineSingleOp : Pass<"hfusion-outline-single-op", "func::FuncOp"> {
  let constructor = "mlir::hfusion::createOutlineSingleOpPass()";
  let summary = "Outline single linalg ops into kernels.";
  let dependentDialects = ["hfusion::HFusionDialect"];
  let options = [Option<"moveOutToParam", "move-out-to-param", "bool",
                        /*default=*/"true",
                        "Whether move the tensor out to params or not">,
  ];
}

def SimplifyOps : Pass<"hfusion-simplify-ops", "func::FuncOp"> {
  let summary = "Simplify operations";
  let constructor = "mlir::hfusion::createSimplifyOpsPass()";
  let dependentDialects = ["hfusion::HFusionDialect"];
}

def Normalize : Pass<"hfusion-normalize-ops", "func::FuncOp"> {
  let summary = "Normalize Hfusion";
  let constructor = "mlir::hfusion::createHFusionNormalizeOpsPass()";
  let dependentDialects = ["hfusion::HFusionDialect"];
}

def NormalizeSliceOps : Pass<"hfusion-normalize-slice-ops", "func::FuncOp"> {
  let summary = "Normalize Slice Ops.";
  let options = [
    Option<"skipAlignedSlice", "skip-aligned-slice", "bool", "false", 
           "Skip FoldInsertSliceToConcat pattern for aligned slice.">,
  ];
  let constructor = "mlir::hfusion::createHFusionNormalizeSliceOpsPass()";
  let dependentDialects = ["hfusion::HFusionDialect", "annotation::AnnotationDialect"];
}

def PackTilingData : Pass<"hfusion-pack-tiling-data", "ModuleOp"> {
  let summary = "Pack dynamic tiling information into a struct.";
  let description = [{Pack the tiling information into a struct.}];
  let constructor = "mlir::hfusion::createPackTilingDataPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "hacc::HACCDialect",
                           "LLVM::LLVMDialect"];
  let options =
      [ListOption<
           "includeSymbols", "include-symbols", "std::string",
           "Comma separated list of symbols that should apply this "
           "transformation. "
           "If empty, the default behavior is to apply transformation to "
           "all functions.">,
       Option<"emitGetTilingStructSizeFunction",
              "emit-get-tiling-struct-size-function", "bool", "false",
              "When enabled, a host function that returns the number of i64 "
              "tiling data is emitted.">,
       Option<
           "packTilingKey", "pack-tiling-key", "bool", "true",
           "When enabled, the tiling key would also be packed into the tiling "
           "struct."
           "Otherwise, the tiling key is directly written to a pointer.">,
  ];
}

def ConstantizeTilingData
    : Pass<"hfusion-constantize-tiling-data", "ModuleOp"> {
  let summary = "Propagate constants between tiling and device function";
  let description = [{
    Propagate constants from calculate tiling to the device function.

    Modifications made:
      * Constant tiling data are inlined into the device function.
      * Constant tiling data are removed from the tiling function.
      * Constant tiling data are removed from the arguments of the device
        function. And the call sites are modifed accordingly.
      * Constant tiling data are removed from the callers of the device function,
        and the callers of the callers, and so on.

    Constraints/Assumptions:
      * For all the device functions sharing the same tiling function, the order
        of tiling data argument is exactly the same.
      * The tiling arguments in device function's input arguments has the exact
        same order as the return values of the tiling function.

    Input

    ```mlir
    func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64, i64)
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
      %ret0 = "some_calculation"() : () -> i64
      %ret1 = arith.constant 42: i64
      return %ret0, %ret1: i64, i64
    }

    func.func @device_kernel_tiling_0(%arg0: tensor<?x?xf16>,
                                     %arg1: i64 {hacc.tiling_data},
                                     %arg2: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
      "some_use"(%arg1) : (i64) -> ()
      "some_use"(%arg2) : (i64) -> ()
      %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
      return %ret0 : tensor<?x?xf16>
    }

    func.func @device_kernel_tiling_1(%arg0: tensor<?x?xf16>,
                                      %arg1: i64 {hacc.tiling_data},
                                      %arg2: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
      "some_use"(%arg1) : (i64) -> ()
      "some_use"(%arg2) : (i64) -> ()
      %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
      return %ret0 : tensor<?x?xf16>
    }

    func.func @main(%arg0: tensor<?x?xf16>,
                    %arg1: i64 {hacc.tiling_data},
                    %arg2: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
      %0 = arith.index_castui %arg1 : i64 to index
      %1 = scf.index_switch %0 -> tensor<?x?xf16>
      case 1 {
        %2 = func.call @device_kernel_tiling_1(%arg0, %arg1, %arg2) : (tensor<?x?xf16>, i64, i64) -> tensor<?x?xf16>
        scf.yield %2 : tensor<?x?xf16>
      }
      case 0 {
        %2 = func.call @device_kernel_tiling_0(%arg0, %arg1, %arg2): (tensor<?x?xf16>, i64, i64) -> tensor<?x?xf16>
        scf.yield %2 : tensor<?x?xf16>
      }
      default {
        %false = arith.constant false
        cf.assert %false, "Invalid tiling key"
        %2 = ub.poison : tensor<?x?xf16>
        scf.yield %2 : tensor<?x?xf16>
      }
      return % 1 : tensor < ? x ? xf16 >
    }
    ```

    Output

    ```mlir
    func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64)
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
      %ret0 = "some_calculation"() : () -> i64
      return %ret0: i64
    }

    func.func @device_kernel_tiling_0(%arg0: tensor<?x?xf16>,
                                      %arg1: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
      "some_use"(%arg1) : (i64) -> ()
      %arg2 = arith.constant 32 : i64
      "some_use"(%arg2) : (i64) -> ()
      %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
      return %ret0 : tensor<?x?xf16>
    }

    func.func @device_kernel_tiling_1(%arg0: tensor<?x?xf16>,
                                      %arg1: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
      "some_use"(%arg1) : (i64) -> ()
      %arg2 = arith.constant 32 : i64
      "some_use"(%arg2) : (i64) -> ()
      %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
      return %ret0 : tensor<?x?xf16>
    }

    func.func @main(%arg0: tensor<?x?xf16>,
                    %arg1: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
      %0 = arith.index_castui %arg1 : i64 to index
      %1 = scf.index_switch %0 -> tensor<?x?xf16>
      case 1 {
        %2 = func.call @device_kernel_tiling_1(%arg0, %arg1) : (tensor<?x?xf16>, i64) -> tensor<?x?xf16>
        scf.yield %2 : tensor<?x?xf16>
      }
      case 0 {
        %2 = func.call @device_kernel_tiling_0(%arg0, %arg1): (tensor<?x?xf16>, i64) -> tensor<?x?xf16>
        scf.yield %2 : tensor<?x?xf16>
      }
      default {
        %false = arith.constant false
        cf.assert %false, "Invalid tiling key"
        %2 = ub.poison : tensor<?x?xf16>
        scf.yield %2 : tensor <?x?xf16>
      }
      return %1 : tensor<?x?xf16>
    }
   ```
  }];
  let constructor = "mlir::hfusion::createConstantizeTilingDataPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "arith::ArithDialect"];
}

def InferFuncFusionKind
    : Pass<"hfusion-infer-func-fusion-kind", "func::FuncOp"> {
  let constructor = "mlir::hfusion::createInferFuncFusionKind()";
  let summary = "Infer function for fusion kind";
  let dependentDialects = ["hfusion::HFusionDialect"];
}

def AdaptTritonKernel : Pass<"adapt-triton-kernel", "ModuleOp"> {
  let summary = "Adapt the triton kernel";
  let constructor = "mlir::hfusion::createAdaptTritonKernelPass()";
  let dependentDialects = ["hacc::HACCDialect", "hfusion::HFusionDialect"];
  let options = [Option<"hivmcVersion", "hivmc-version", "std::string", /*default=*/"",
                        "Specify hivmc version to resolve backward compatibility">,
  ];
}

def LegalizeBF16Pass : Pass<"hfusion-legalize-bf16", "func::FuncOp"> {
  let summary = "normalize BF16 to FP32.";
  let constructor = "mlir::hfusion::createLegalizeBF16Pass()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect"];
}

def LegalizeBoolPass : Pass<"hfusion-legalize-bool", "ModuleOp"> {
  let summary = "cast int8 to int1 for input and int1 to int8 for output.";
  let constructor = "mlir::hfusion::createLegalizeBoolPass()";
  let dependentDialects = ["arith::ArithDialect", "hfusion::HFusionDialect", "annotation::AnnotationDialect"];
}

def ReorderOpsByBFS : Pass<"hfusion-reorder-ops", "func::FuncOp"> {
  let summary = "reorder the ops by bfs.";
  let constructor = "mlir::hfusion::createReorderOpsByBFS()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect",
                           "mlir::tensor::TensorDialect"];
}

def DowngradeFP64CstOpPass : Pass<"hfusion-downgrade-fp64", "func::FuncOp"> {
  let summary = "downgrade fp64 constant to fp32";
  let constructor = "mlir::hfusion::createDowngradeFP64CstOpPass()";
  let dependentDialects = ["arith::ArithDialect", "hfusion::HFusionDialect"];
}

def InferOutShapesPass : Pass<"hfusion-infer-out-shapes", "ModuleOp"> {
  let summary = "generate out tensor's shape function for kernel";
  let constructor = "mlir::hfusion::createInferOutShapesPass()";
  let dependentDialects = ["arith::ArithDialect", "shape::ShapeDialect"];
}

def ComposeMultiReduce : Pass<"hfusion-compose-multi-reduce", "func::FuncOp"> {
  let summary = "Compose multi reduce optimization";
  let constructor = "mlir::hfusion::createComposeMultiReduce()";
  let dependentDialects = [];
  let options =
      [Option<"maxCompose", "max-compose", "int", "-1",
              "Maximum reduce composed into single operation, -1 is limitless">,
       Option<"maxDistDiff", "max-dist-diff", "int", "-1",
              "Maximum distance difference from common ancestor">,
       Option<
           "aggressive", "aggressive", "bool", "false",
           "Aggressive mode will try to reshape if shape are loosely matched">,
  ];
}

def DecomposeMulti : Pass<"hfusion-decompose-multi", "func::FuncOp"> {
  let summary = "Decompose multi ops into single ones";
  let constructor = "mlir::hfusion::createDecomposeMulti()";
  let dependentDialects = [];
}

def CacheIO : Pass<"hfusion-cache-io", "func::FuncOp"> {
  let summary = "cache input and output argument";
  let constructor = "::mlir::hfusion::createCacheIO()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect",
                           "mlir::tensor::TensorDialect"];
}

def CacheIOForReturnArg : Pass<"hfusion-cache-io-for-return-arg", "func::FuncOp"> {
  let summary = "cache argument that returns directly";
  let constructor = "::mlir::hfusion::createCacheIOForReturnArg()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect",
                           "mlir::tensor::TensorDialect"];
}

def ReCacheIO
    : Pass<"hfusion-recache-io", "func::FuncOp"> {
  let summary = "recache io";
  let constructor = "::mlir::hfusion::createReCacheIO()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect",
                           "mlir::tensor::TensorDialect"];
}

def HoistTensorEmpty : Pass<"hfusion-hoist-tensor-empty", "ModuleOp"> {
  let summary =
      "Hoist tensor empty to func parameters and merge into one parameter";
  let description = [{This pass merge all tensor.empty to one func parameter}];
  let constructor = "mlir::hfusion::createHoistTensorEmptyPass()";
  let dependentDialects = ["tensor::TensorDialect",
                           "bufferization::BufferizationDialect",
                           "hfusion::HFusionDialect", "memref::MemRefDialect",
                           "hacc::HACCDialect",
  ];
}

def WrapHostFunc : Pass<"hfusion-wrap-host-func", "ModuleOp"> {
  let summary = "Create wrappers for certain host related functions ";
  let description = [{
      This pass creates wrapper functions for host tiling func, infer shape func, etc.
  }];
  let constructor = "mlir::hfusion::createWrapHostFuncPass()";
  let dependentDialects = ["func::FuncDialect", "hfusion::HFusionDialect",
                           "hacc::HACCDialect",
  ];
  let options = [Option<"removeUnusedArguments", "remove-unused-arguments",
                        "bool",
                        /*default=*/"false",
                        "Whether to remove unused arguments in host wrapper "
                        "function or not">,
  ];
}

def FoldSymbolicDim : Pass<"hfusion-fold-symbolic-dim", "func::FuncOp"> {
  let summary = "Replace tensor.dim source operands to hfusion::SymbolicDimOp";
  let constructor = "mlir::hfusion::createFoldSymbolicDimPass()";
}

def UnfoldSymbolicDim : Pass<"hfusion-unfold-symbolic-dim", "func::FuncOp"> {
  let summary = "Replace hfusion::SymbolicDimOp to same symbolic arguments";
  let constructor = "mlir::hfusion::createUnfoldSymbolicDimPass()";
}

def DropSymbols : Pass<"hfusion-drop-symbols", "func::FuncOp"> {
  let summary = "Drop ranked tensor symbols from operations";
  let constructor = "mlir::hfusion::createDropSymbolsPass()";
}

def EliminateDuplicateFuncs : Pass<"hfusion-eliminate-duplicate-funcs", "ModuleOp"> {
  let summary = "Eliminate duplicate functions after fusion";
  let constructor = "mlir::hfusion::createEliminateDuplicateFuncsPass()";
}

def Decompose : Pass<"hfusion-decompose", "func::FuncOp">{
  let summary = "Decompose ops that implemented AggregatedOpInterface.";
  let constructor = "mlir::hfusion::createDecomposePass()";
  let options = [Option<"hfusionDecomposePhase",
           "hfusion-decompose-phase", "::bishengir::DecomposePhase",
           /*default=*/"::bishengir::DecomposePhase::NO_CONSTRAINT",
           "Specify which decompose phase to apply.", [{
            ::llvm::cl::values(
        clEnumValN(::bishengir::DecomposePhase::NO_CONSTRAINT, "no-constraint",
                   "decompose with no constraint specified"),
        clEnumValN(::bishengir::DecomposePhase::AFTER_HFUSION_FLATTEN, "after-hfusion-flatten",
                   "for ops that get decompose after tidy flatten")
    )}]>,
  ];
}

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_PASSES
