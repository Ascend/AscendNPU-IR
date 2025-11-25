//===- Options.td - BiShengIR Compile-related Options ---------*-tablegen-*-==//
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

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_OPTIONS_TD
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_OPTIONS_TD

include "bishengir/Tools/ConfigOptions/Options.td"

class BiShengIRCompileOpt<
  string desc,
  string ty,
  string defaultVal
> : Option {
  let description = desc;
  let type = ty;
  let defaultValue = defaultVal;
  string compileConfigName = "BiShengIRCompileMainConfig";
}

class BiShengIRCompileListOpt<
  string desc,
  string ty
> : ListOption {
  let description = desc;
  let type = ty;
  let defaultValue = "";
  string compileConfigName = "BiShengIRCompileMainConfig";
}

//===----------------------------------------------------------------------===//
// Feature Control Options
//===----------------------------------------------------------------------===//

class FeatureControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal> {
  let compileOptionCategories = ["featCtrlCategory"];
}

def EnableTritonKernelCompile : FeatureControlOptions<
  "Enable Triton kernel compilation.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION", "HFUSION_TO_HIVM"];
  bit isSharedWithDownstreamToolchain = 1;
}

#ifdef BISHENGIR_ENABLE_TORCH_CONVERSIONS
def EnableTorchCompile : FeatureControlOptions<
  "Enable Torch-MLIR compilation.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}
#endif

def EnableHIVMCompile : FeatureControlOptions<
  "Enable BiShengHIR HIVM compilation.",
  "bool",
  "true"
>;

def EnableHfusionCompile : FeatureControlOptions<
  "Enable BiShengHIR HFusion compilation.",
  "bool",
  "false"
>;

def EnableSymbolAnalysis : FeatureControlOptions<
  "Enable symbol analysis.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def EnableMultiKernel : FeatureControlOptions<
  "When disabled, graph must fuse as single kernel;"
  "when enabled, outline multiple kernels.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def EnableManageHostResources : FeatureControlOptions<
  "Enable managing resource for Host functions.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def EnableStaticBarePtr : FeatureControlOptions<
  "Enable generating bare ptr calling convention for static shaped kernels.",
  "bool",
  "true"
> {
  let emitGetterSetter = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableBinRelocation : FeatureControlOptions<
  "Enable binary relocation.",
  "bool",
  "true"
> {
  let emitGetterSetter = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

#ifdef BISHENGIR_ENABLE_TORCH_CONVERSIONS
def EnsureNoImplicitBroadcast : FeatureControlOptions<
  "Whether to ensure that there is no implicit broadcast "
  "semantics. If there is a dynamic to dynamic dim "
  "broadcast, raise a runtime error.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}
#endif

#ifndef BISHENGIR_PUBLISH
def EnableLIRCompile : FeatureControlOptions<
  "Enable BiShengLIR compilation.",
  "bool",
  "true"
> {
  let emitGetterSetter = 0;
  bit isSharedWithDownstreamToolchain = 1;
}
#endif

def DisableAutoInjectBlockSync : FeatureControlOptions<
  "Disable generating blocksync wait/set by injectBlockSync pass.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def DisableAutoCVWorkSpaceManage : FeatureControlOptions<
  "Disable automatically managing CV workspace. "
  "Used in combination with the DisableAutoInjectBlockSync option.",
  "bool",
  "false"
>;

def EnableHIVMGraphSyncSolver : FeatureControlOptions<
  "Use hivm graph-sync-solver instead of inject-sync.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def DisableHIVMAutoInjectSync : FeatureControlOptions<
  "Disable auto inject sync intra core.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def DisableHIVMTensorCompile : FeatureControlOptions<
  "Disable BiShengHIR HIVM Tensor compilation.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

//===----------------------------------------------------------------------===//
// DFX Control Options
//===----------------------------------------------------------------------===//

class DFXControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal> {
  let compileOptionCategories = ["dfxCtrlCategory"];
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableSanitizer : DFXControlOptions<
  "Enable ascend sanitizer.",
  "bool",
  "false"
>;

def EnableDebugInfo : DFXControlOptions<
  "Enable debug info.",
  "bool",
  "false"
>;

#ifndef BISHENGIR_PUBLISH
def EnableCpuTraceIntrinsic : DFXControlOptions<
  "Enable to generate host-accepted IR by eliminating HIVM special traits.",
  "bool",
  "false"
>;
#endif

//===----------------------------------------------------------------------===//
// General Optimization Control Options
//===----------------------------------------------------------------------===//

class GeneralOptimizationControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal> {
  let compileOptionCategories = ["generalOptCategory"];
}

def EnableAutoMultiBuffer : GeneralOptimizationControlOptions<
  "Enable auto multi buffer.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
  bit isSharedWithDownstreamToolchain = 1;
}

def LimitAutoMultiBufferOnlyForLocalBuffer : GeneralOptimizationControlOptions<
  "When enable-auto-multi-buffer = true, limit it only to work for local buffer",
  "bool",
  "true"
> {
  let emitGetterSetter = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableTuningMode : GeneralOptimizationControlOptions<
  "Enable tuning mode and will not try compile multi times in case of plan memory failure",
  "bool",
  "false"
>;

def BlockDim : GeneralOptimizationControlOptions<
  "Number of blocks to use",
  "unsigned",
  "1"
> {
  let passGroup = ["HFUSION"];
}

//===----------------------------------------------------------------------===//
// HFusion Optimization Control Options
//===----------------------------------------------------------------------===//

class HFusionOptimizationControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal> {
  let passGroup = ["HFUSION"];
  let compileOptionCategories = ["hfusionOptCategory"];
}

def EnableDeterministicComputing : HFusionOptimizationControlOptions<
  "If enabled, the computation result is deterministic. If "
  "disabled, we will enable extra optimizations that might "
  "boost performance, e.g. bind reduce to multiple cores. "
  "However, the result will be non-deterministic.",
  "bool",
  "true"
>;

def EnableOpsReorder : HFusionOptimizationControlOptions<
  "Enable ops reorder to opt pipeline.",
  "bool",
  "true"
>;

def HfusionMaxHorizontalFusionSize : HFusionOptimizationControlOptions<
  "Number of horizontal fusion attempt (Default: unlimited).",
  "int32_t",
  "-1"
>;

def HfusionMaxBufferCountTuning : HFusionOptimizationControlOptions<
  "Max buffer count tuning in HFusion auto schedule.",
  "int64_t",
  "0"
>;

def CubeTilingTuning : BiShengIRCompileListOpt<
  "Cube block size tuning in HFusion auto schedule",
  "int64_t"
> {
  let passGroup = ["HFUSION"];
  let compileOptionCategories = ["hfusionOptCategory"];
}

def enableHfusionCountBufferDmaOpt : HFusionOptimizationControlOptions<
  "If enabled, the buffer used by DMA operations will not "
  "be reused by Vector operations.",
  "bool",
  "false"
>;

//===----------------------------------------------------------------------===//
// HIVM Optimization Control Options
//===----------------------------------------------------------------------===//

class HIVMOptimizationControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal> {
  let compileOptionCategories = ["hivmOptCategory"];
  let emitGetterSetter = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

def LimitAutoMultiBufferOfLocalBuffer : HIVMOptimizationControlOptions<
  "When enable-auto-multi-buffer = true, limit local buffer mode.",
  "MultiBufferStrategy",
  "MultiBufferStrategy::CUBE_NO_L0C"
> {
  let additionalOptFlags = [{
llvm::cl::values(
  clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit", "No limit"),
  clEnumValN(MultiBufferStrategy::CUBE_NO_L0C, "no-l0c", "Disable l0c multi buffer")
)
  }];
}

def LimitAutoMultiBufferBuffer : HIVMOptimizationControlOptions<
  "When enable-auto-multi-buffer = true, limit it to only cube, only vector or no limit.",
  "MultiBufferStrategy",
  "MultiBufferStrategy::ONLY_CUBE"
> {
  let additionalOptFlags = [{
llvm::cl::values(
    clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit", "No limit"),
    clEnumValN(MultiBufferStrategy::ONLY_CUBE, "only-cube", "Limit to only cube"),
    clEnumValN(MultiBufferStrategy::ONLY_VECTOR, "only-vector", "Limit to only vector")
)
  }];
}

def EnableAutoBindSubBlock : HIVMOptimizationControlOptions<
  "Enable auto bind sub block.",
  "bool",
  "true"
>;

def EnableCodeMotion : HIVMOptimizationControlOptions<
  "Enable code-motion/subset-hoist.",
  "bool",
  "true"
>;

def EnableHIVMInjectBarrierAllSync : HIVMOptimizationControlOptions<
  "Enable barrier all mode for HIVM inject sync.",
  "bool",
  "false"
>;

def EnableHIVMUnitFlagSync : HIVMOptimizationControlOptions<
  "Enable inject sync pass to use unit-flag modes for synchronization.",
  "bool",
  "false"
>;

def EnableHIVMAssumeAliveLoops : HIVMOptimizationControlOptions<
  "Assume that all loops (forOp whileOp) will execute at least once.",
  "bool",
  "false"
>;

def EnableHIVMInjectBlockAllSync : HIVMOptimizationControlOptions<
  "Enable inject all block sync for HIVM inject block sync.",
  "bool",
  "false"
>;

def SetWorkspaceMultibuffer : HIVMOptimizationControlOptions<
  "Override number of multibuffers for workspace, defaults to 2.",
  "unsigned",
  "2"
>;

def EnableHIVMGlobalWorkspaceReuse : HIVMOptimizationControlOptions<
  "Enable global workspace reuse.",
  "bool",
  "false"
>;

def EnableHIVMAutoCVBalance : HIVMOptimizationControlOptions<
  "Enable balancing during cv-pipelining.",
  "bool",
  "false"
>;

def EnableHIVMAutoStorageAlign : HIVMOptimizationControlOptions<
  "Enable mark/enable storage align.",
  "bool",
  "true"
>;

def EnableHivmNd2nzOnVector : HIVMOptimizationControlOptions<
  "Enable nd2nz on vector.",
  "bool",
  "false"
>;

def EnableAutoBlockifyLoop : HIVMOptimizationControlOptions<
  "Enable auto loop on blocks for all parallel.",
  "bool",
  "false"
>;

def TileMixVectorLoop : HIVMOptimizationControlOptions<
  "The trip count of the tiled vector loop for mix kernels.",
  "unsigned",
  "1"
>;

def TileMixCubeLoop : HIVMOptimizationControlOptions<
  "The trip count of the tiled cube loop for mix kernels.",
  "unsigned",
  "1"
>;

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def Target : BiShengIRCompileOpt<
  "Target device name.",
  "mlir::hacc::TargetDevice",
  "mlir::hacc::TargetDevice::Ascend910B1"
> {
  let additionalOptFlags = [{
llvm::cl::values(
    #define TO_STRING(x) #x
    #define REGISTER_TARGET(TARGET)                                              \
    clEnumValN(mlir::hacc::TargetDevice::TARGET, TO_STRING(TARGET),              \
                TO_STRING(TARGET))
                // Ascend910B series
                REGISTER_TARGET(Ascend910B1), REGISTER_TARGET(Ascend910B2),
                REGISTER_TARGET(Ascend910B3), REGISTER_TARGET(Ascend910B4),
                REGISTER_TARGET(Ascend910B2C),
                // Ascend910_93 series
                REGISTER_TARGET(Ascend910_9362), REGISTER_TARGET(Ascend910_9372),
                REGISTER_TARGET(Ascend910_9381), REGISTER_TARGET(Ascend910_9382),
                REGISTER_TARGET(Ascend910_9391), REGISTER_TARGET(Ascend910_9392),
                REGISTER_TARGET(Unknown)
    #undef REGISTER_TARGET
    #undef TO_STRING
)
  }];
  let compileOptionCategories = ["targetCategory"];
}

//===----------------------------------------------------------------------===//
// Other Options
//===----------------------------------------------------------------------===//

/// Allow operation with no registered dialects.
/// This option is for convenience during testing only and discouraged in
/// general.
def AllowUnregisteredDialects : BiShengIRCompileOpt<
  "Allow operation with unregistered dialects.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def HivmCompileArgs : BiShengIRCompileListOpt<
  "List of arguments passed to bishengir-hivm-compile.",
  "std::string"
>;

/// Note that we don't own `mlir-print-ir-after-all` and
/// `mlir-print-ir-before-all` as they are upstream options.
/// We add it here because we can capture their values and forward
/// them to `bishengir-hivm-compile` tool.
def MlirPrintIrAfterAll : BiShengIRCompileOpt<
  "Print IR after each pass.",
  "bool",
  "false"
> {
  let emitOptionRegistration = 0;
  let externalStorage = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

def MlirPrintIrBeforeAll : BiShengIRCompileOpt<
  "Print IR before each pass.",
  "bool",
  "false"
> {
  let emitOptionRegistration = 0;
  let externalStorage = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_OPTIONS_TD
