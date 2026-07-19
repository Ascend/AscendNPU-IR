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

class BiShengIRCompileInternalStateOpt<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal>, InternalStateOption;

// A5 options whose implementation/types have not been migrated to A3 yet.
// This intentionally does not inherit Option, so bishengir-options-tblgen keeps
// the record name for future migration but emits no C++ option surface.
class A5UnmigratedOptionStub {
  list<string> optionScope = ["A5_ONLY"];
}

//===----------------------------------------------------------------------===//
// Shared / Public Options
// Ordered to match A3's bishengir-compile/Options.td.
//===----------------------------------------------------------------------===//

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
  let passGroup = ["HFUSION", "HIVM", "HFUSION_TO_HIVM"];
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableUbufSaving : FeatureControlOptions<
  "Enable ub buffer saving mode.",
  "bool",
  "false"
>{
  let optionScope = ["A3_ONLY"];
  let passGroup = ["HFUSION", "HIVM", "HFUSION_TO_HIVM"];
}

def DisableSizeAlignForCast : FeatureControlOptions<
  "Disable VCast alloc size alignment and use new temp buffer scheme (s32/s16->s8).",
  "bool",
  "false"
>{
  let optionScope = ["A3_ONLY"];
  let passGroup = ["HFUSION", "HIVM", "HFUSION_TO_HIVM"];
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

def EnableLayoutOptimization : FeatureControlOptions<
  "Enable layout optimization (insert/propagate convert_layout around cube ops).",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION", "HIVM"];
}

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
  let passGroup = ["HIVM"];
}

def EnableHIVMGraphSyncSolver : FeatureControlOptions<
  "Use hivm graph-sync-solver instead of inject-sync.",
  "bool",
  "true"
> {
  let passGroup = ["HIVM"];
}

def DisableAutoCVWorkSpaceManage : FeatureControlOptions<
  "In combination with the disableAutoInjectBlockSync option.",
  "bool",
  "false"
> {
  let optionScope = ["A3_ONLY"];
  let passGroup = ["HFUSION", "HIVM"];
}

def EnableHIVMCrossCoreGSS : FeatureControlOptions<
  "Use hivm cross-core graph-sync-solver instead of inject-block-sync.",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION", "HIVM"];
}

def EnableHIVMDelayedCrossCoreGSS : FeatureControlOptions<
  "Use hivm delayed cross-core graph-sync-solver instead of inject-block-sync.",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION", "HIVM"];
}

def DisableHIVMAutoInjectSync : FeatureControlOptions<
  "Disable auto inject sync intra core.",
  "bool",
  "false"
> {
  let optionScope = ["A3_ONLY"];
  let passGroup = ["HFUSION", "HIVM"];
}

def DisableHIVMTensorCompile : FeatureControlOptions<
  "Disable BiShengHIR HIVM Tensor compilation.",
  "bool",
  "false"
> {
  let optionScope = ["A3_ONLY"];
  let passGroup = ["HFUSION", "HIVM"];
}

def EnableLegacyInsertLoadStoreForMixCV : FeatureControlOptions<
  "Use legacy InsertLoadStoreForMixCV.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
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
  let passGroup = ["HFUSION", "HIVM"];
}

def EnableSanitizer : DFXControlOptions<
  "Enable ascend sanitizer.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableMemoryDisplay : DFXControlOptions<
  "Enable ascend memory display tools.",
  "bool",
  "false"
>;

def EnableDebugInfo : DFXControlOptions<
  "Enable debug info.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

#ifndef BISHENGIR_PUBLISH
def EnableCpuTraceIntrinsic : DFXControlOptions<
  "Enable to generate host-accepted IR by eliminating HIVM special traits.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}
#endif

def InjectIRFromFile : DFXControlOptions<
  "Path to IR file for debug: load functions from this file and replace "
  "matching functions in the current module (inject-ir pass).",
  "std::string",
  "\"\""
> {
  let passGroup = ["HFUSION", "HIVM"];
}

def SaveTemps : DFXControlOptions<
  "Save intermediate compilation results to specified directory.",
  "std::string",
  "\"\""
>;

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
  let passGroup = ["HFUSION", "HIVM"];
}

def LimitAutoMultiBufferOnlyForLocalBuffer : GeneralOptimizationControlOptions<
  "When enable-auto-multi-buffer = true, limit it only to work for local buffer",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
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
  let passGroup = ["HIVM"];
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

// Fine-grained per-buffer multi-buffer disables. These act AND-wise on top of
// the existing LimitAutoMultiBufferOfLocalBuffer / LimitAutoMultiBufferBuffer
// switches: any of them being true skips marking on that address space
// regardless of the broader strategy. They are also the knobs the compile-
// time fallback in BiShengIRCompileMain flips when a specific memory space
// reports overflow:
//   ub overflow   -> DisableMultiBufferOnUB
//   cc overflow   -> DisableMultiBufferOnL0C
//   cbuf overflow -> DisableMultiBufferOnL1
def DisableMultiBufferOnUB : HIVMOptimizationControlOptions<
  "Disable auto multi-buffer for UB (Vector) buffers (gates Load/Store mark).",
  "bool",
  "false"
>;

def DisableMultiBufferOnL0C : HIVMOptimizationControlOptions<
  "Disable auto multi-buffer for L0C (Cube accumulator) buffers (gates Fixpipe mark).",
  "bool",
  "false"
>;

def DisableMultiBufferOnL1 : HIVMOptimizationControlOptions<
  "Disable auto multi-buffer for L1 (cbuf) buffers (gates ND2NZ mark).",
  "bool",
  "false"
>;

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
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableHIVMUnitFlagSync : HIVMOptimizationControlOptions<
  "Enable inject sync pass to use unit-flag modes for synchronization.",
  "bool",
  "false"
>;

def EnableHIVMAssumeAliveLoops : HIVMOptimizationControlOptions<
  "Assume that all loops (forOp whileOp) will execute at least once.",
  "bool",
  "false"
> {
  let optionScope = ["A3_ONLY"];
}

def EnableHIVMInjectBlockAllSync : HIVMOptimizationControlOptions<
  "Enable inject all block sync for HIVM inject block sync.",
  "bool",
  "false"
>;

def SetWorkspaceMultibuffer : HIVMOptimizationControlOptions<
  "Override number of multibuffers for workspace, defaults to 4.",
  "unsigned",
  "4"
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

def EnablePreload : HIVMOptimizationControlOptions<
  "Enable preload.",
  "bool",
  "false"
>;

def EnableLazyLoading : HIVMOptimizationControlOptions<
  "Enable lazy loading in CV pipelining to reduce buffer expansion.",
  "bool",
  "false"
>;

// TODO: Restore this A5 option when A5 CVPipelineMode and the matching HIVM
// CV-pipelining pipeline wiring are migrated to A3. A3 currently exposes
// preload/skew through EnablePreload and the CVPipelining pass uses
// enableSkewMode/setDepthInUnrollMode, so generating this enum option would
// reference an unmigrated type and imply unsupported behavior.
def SetCVPipelineMode : A5UnmigratedOptionStub;

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

def SuperBlockFactor : HIVMOptimizationControlOptions<
  "The factor of super-blocking (must be a power of 2, >= 1). "
  "1 means no super-blocking.",
  "unsigned",
  "1"
>;

def SuperBlockBarrier : HIVMOptimizationControlOptions<
  "Enable/disable super-block barrier lowering.",
  "bool",
  "false"
>;

def TileMixVectorLoop : HIVMOptimizationControlOptions<
  "The trip count of the tiled vector loop for mix kernels.",
  "unsigned",
  "2"
>;

def TileMixCubeLoop : HIVMOptimizationControlOptions<
  "The trip count of the tiled cube loop for mix kernels.",
  "unsigned",
  "2"
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
                clEnumValN(mlir::hacc::TargetDevice::Ascend910B4_1,
                              "Ascend910B4-1", "Ascend910B4-1"),
                REGISTER_TARGET(Ascend910B2C),
                // Ascend910_93 series
  REGISTER_TARGET(Ascend910_9362), REGISTER_TARGET(Ascend910_9372),
  REGISTER_TARGET(Ascend910_9381), REGISTER_TARGET(Ascend910_9382),
  REGISTER_TARGET(Ascend910_9391), REGISTER_TARGET(Ascend910_9392),
                // Ascend910_95 series
  REGISTER_TARGET(Ascend910_950z), REGISTER_TARGET(Ascend910_9579),
  REGISTER_TARGET(Ascend910_957b), REGISTER_TARGET(Ascend910_957d),
  REGISTER_TARGET(Ascend910_9581), REGISTER_TARGET(Ascend910_9589),
  REGISTER_TARGET(Ascend910_958a), REGISTER_TARGET(Ascend910_958b),
                REGISTER_TARGET(Ascend910_9599),
                // Ascend950PR series
                REGISTER_TARGET(Ascend950PR_950z), REGISTER_TARGET(Ascend950PR_9579),
                REGISTER_TARGET(Ascend950PR_957a), REGISTER_TARGET(Ascend950PR_957b),
                REGISTER_TARGET(Ascend950PR_957c), REGISTER_TARGET(Ascend950PR_957d),
                REGISTER_TARGET(Ascend950PR_9589), REGISTER_TARGET(Ascend950PR_958a),
                REGISTER_TARGET(Ascend950PR_958b), REGISTER_TARGET(Ascend950PR_958c),
                REGISTER_TARGET(Ascend950PR_958d), REGISTER_TARGET(Ascend950PR_9599),
                REGISTER_TARGET(Ascend950PR_959a), REGISTER_TARGET(Ascend950PR_959b),
                // Ascend950DT series
                REGISTER_TARGET(Ascend950DT_950x), REGISTER_TARGET(Ascend950DT_950y),
                REGISTER_TARGET(Ascend950DT_9571), REGISTER_TARGET(Ascend950DT_9572),
                REGISTER_TARGET(Ascend950DT_9573), REGISTER_TARGET(Ascend950DT_9574),
                REGISTER_TARGET(Ascend950DT_9575), REGISTER_TARGET(Ascend950DT_9576),
                REGISTER_TARGET(Ascend950DT_9577), REGISTER_TARGET(Ascend950DT_9578),
                REGISTER_TARGET(Ascend950DT_9581), REGISTER_TARGET(Ascend950DT_9582),
                REGISTER_TARGET(Ascend950DT_9583), REGISTER_TARGET(Ascend950DT_9584),
                REGISTER_TARGET(Ascend950DT_9585), REGISTER_TARGET(Ascend950DT_9586),
                REGISTER_TARGET(Ascend950DT_9587), REGISTER_TARGET(Ascend950DT_9588),
                REGISTER_TARGET(Ascend950DT_9591), REGISTER_TARGET(Ascend950DT_9592),
                REGISTER_TARGET(Ascend950DT_9595), REGISTER_TARGET(Ascend950DT_9596),
                REGISTER_TARGET(Ascend950DT_95A1), REGISTER_TARGET(Ascend950DT_95A2),
                REGISTER_TARGET(Unknown)
#undef REGISTER_TARGET
#undef TO_STRING
)
  }];
  let compileOptionCategories = ["targetCategory"];
  let passGroup = ["HIVM"];
  bit isSharedWithDownstreamToolchain = 1;
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
>;

def HIVMCVersion : BiShengIRCompileOpt<
  "Version string for hivmc to resolve backward compatibility.",
  "std::string",
  ""
> {
  let passGroup = ["HFUSION", "HIVM"];
}

def HIVMCArgs : BiShengIRCompileListOpt<
  "List of arguments passed to hivmc.",
  "std::string"
>;

def LinkAicoreBitcode : BiShengIRCompileListOpt<
  "Aicore bitcode file(s) to link. Accepts one or more absolute paths, "
  "comma-separated. Passed to hivmc as --link-aicore-bitcode.",
  "std::string"
> {
  let compileOptionCategories = ["targetCategory"];
  bit isSharedWithDownstreamToolchain = 1;
}

/// Note that we don't own `mlir-print-ir-after-all` and
/// `mlir-print-ir-before-all` as they are upstream options.
/// We add it here because we can capture their values and forward
/// them to `hivmc` tool.
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

def AppendBishengOptions : BiShengIRCompileOpt<
  "Append options when calling bisheng.",
  "std::string",
  "\"\""
> {
  bit isSharedWithDownstreamToolchain = 1;
}

class SIMTOptimizationControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal> {
  let compileOptionCategories = ["simtOptCategory"];
}

class ProtonOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRCompileOpt<desc, ty, defaultVal>, ExternalStateOption {
  let compileOptionCategories = ["protonCategory"];
  bit isSharedWithDownstreamToolchain = 0;
}

class ProtonFieldOption<
  string desc,
  string ty,
  string defaultVal,
  string fieldName
> : ProtonOptions<desc, ty, defaultVal> {
  let externalStateObject = "protonGPUCompileConfig";
  let externalStateField = fieldName;
}

//===----------------------------------------------------------------------===//
// A5-only Options
//===----------------------------------------------------------------------===//
let optionScope = ["A5_ONLY"] in {

//===----------------------------------------------------------------------===//
// Feature Control Options
//===----------------------------------------------------------------------===//

#ifdef BISHENGIR_ENABLE_TRITON_COMPILE
def EnableTritonIRCompile : FeatureControlOptions<
  "Enable compile from Triton dialect",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}
#endif

def DisableTightCoupledBuffer : FeatureControlOptions<
  "Disable Tight Coupled Buffer Allocation",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION", "HIVM"];
}

def EnableMixedCV : FeatureControlOptions<
  "Enable mixed CV compilation",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION", "HIVM"];
  let additionalOptFlags = [{
llvm::cl::Hidden,
    llvm::cl::callback([](const bool &) {
      llvm::errs() << "[WARNING] --enable-mixed-cv is deprecated.\n";
    })
  }];
}

def DisableTightlyCoupledBufferReuse : FeatureControlOptions<
  "disable tightly coupled buffer reuse",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def DisableVFReachableCheck : FeatureControlOptions<
  "disable VF reachable check",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def DisableFFTS : FeatureControlOptions<
  "Force disabling FFTS.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def DisableFMA : FeatureControlOptions<
  "Force disabling FMA.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def DisableHfusionVectorize : FeatureControlOptions<
  "Disable hfusion auto vectorize.",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def PureSimt : FeatureControlOptions<
  "Full SIMT compile.",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def SimtVFDynamicSize : FeatureControlOptions<
  "Dynamic ub size(KB) for simt VF. Default is 216.",
  "int",
  "216"
> {
  let passGroup = ["HIVM"];
}

def SimtTritonGrid : BiShengIRCompileListOpt<
  "Grid dimensions (e.g., -simt-triton-grid=2,2,2). Default is 1.",
  "int64_t"
>;

def OnlyRunHIVMPipeline : FeatureControlOptions<
  "Only run BiShengHIR HIVM pipeline",
  "bool",
  "false"
>;

def EnableTreeReduce : FeatureControlOptions<
  "Enable tree reduce",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def SkipScope : FeatureControlOptions<
  "Skip passes like flattenOps when scope exists",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION"];
}

def EnableAutoVectorizeV2 : FeatureControlOptions<
  "Enable auto vectorize v2",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION"];
}

def EnableVFFusion : FeatureControlOptions<
  "Enable vf fusion",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION"];
}

def EnableVFStackLimit : FeatureControlOptions<
  "Enable conservative vector function stack slot limit",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

// TODO: Restore this A5 option when the A5 VFFusion implementation and
// mlir::analysis::FusionMode are migrated to A3.
def VfFusionMode : A5UnmigratedOptionStub;

// TODO: Restore this A5 option when the A5 TreeReduce pipeline mode and
// mlir::hfusion::TreeReduceMode are migrated to A3.
def EnableTreeReduceMode : A5UnmigratedOptionStub;

def HfusionEnableMultipleConsumerFusion : FeatureControlOptions<
  "Enable multiple consumer fusion in AutoVectorizeV2",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def EnableHighPrecision : FeatureControlOptions<
  "Enable high precision calculation for sin/cos in HFusion",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION"];
}

def SaveLinkedIR : FeatureControlOptions<
  "Enable saving linked IR before compile to binary",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

#ifdef BISHENGIR_ENABLE_TRITON_COMPILE
def NumWarps : FeatureControlOptions<
  "Number of warps",
  "int32_t",
  "4"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def ThreadsPerWarp : FeatureControlOptions<
  "Number of threads per warp",
  "int32_t",
  "32"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def SharedMemDynamicSize : FeatureControlOptions<
  "Dynamic size of shared memory (in bytes)",
  "int32_t",
  "122880"
> {
  bit isSharedWithDownstreamToolchain = 1;
}
#endif

def EnableSimdSimtMixCompile : HIVMOptimizationControlOptions<
  "Enable simd-simt mix kernel compile",
  "bool",
  "false"
> {
  let passGroup = [];
  bit isSharedWithDownstreamToolchain = 1;
}

//===----------------------------------------------------------------------===//
// DFX Control Options
//===----------------------------------------------------------------------===//

def EnablePrintMemoryAllocatedSize : DFXControlOptions<
  "Enable print memory allocated size",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def PrintPassId : DFXControlOptions<
  "Print pass ID before each pass",
  "bool",
  "false"
>;

def VerifyEach : DFXControlOptions<
  "Verify the IR after every pass. When disabled (default), the IR is only "
  "verified once after the last pass of each pipeline.",
  "bool",
  "false"
>;

def EnableLibCallNoInline : DFXControlOptions<
  "Mark the library-call function declarations generated during HIVM-to-"
  "Standard conversion as hacc.noinline instead of hacc.always_inline.",
  "bool",
  "true"
>;

def InjectIrBefore : DFXControlOptions<
  "Inject IR from file before a specific pass. Format: pass_id@file_path",
  "std::string",
  "\"\""
>;

def InjectIrAfter : DFXControlOptions<
  "Inject IR from file after a specific pass. Format: pass_id@file_path",
  "std::string",
  "\"\""
>;

//===----------------------------------------------------------------------===//
// General Optimization Control Options
//===----------------------------------------------------------------------===//

def EnableDropUnitDims : GeneralOptimizationControlOptions<
  "Enable drop-unit-dims pass",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION"];
}

def EnableFlatten : GeneralOptimizationControlOptions<
  "Enable flatten pass",
  "bool",
  "true"
> {
  let passGroup = ["HFUSION"];
}

def EnableFuseReductionIntoLoop : GeneralOptimizationControlOptions<
  "Enable fuse post-loop reductions into the loop body",
  "bool",
  "false"
> {
  let passGroup = ["HFUSION"];
}

def EnableBishengirSimtOptimization : GeneralOptimizationControlOptions<
  "enable bishengir simt optimization",
  "int",
  "900101"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableVfMergeLevel : GeneralOptimizationControlOptions<
  "Enable vector function merge with level",
  "int32_t",
  "1"
> {
  let passGroup = ["HIVM"];
}

//===----------------------------------------------------------------------===//
// HFusion Optimization Control Options
//===----------------------------------------------------------------------===//

def HfusionMaxFusedOpsInAutoVectorizeV2 : HFusionOptimizationControlOptions<
  "Maximum number of ops to fuse in AutoVectorizeV2 (Default: pass default)",
  "int32_t",
  "-1"
> {
  let passGroup = ["HFUSION"];
}

def HfusionMaxFusedElementwiseOps : HFusionOptimizationControlOptions<
  "Maximum number of elementwise ops to fuse in PreVectorizationFusion (Default: unlimited)",
  "int32_t",
  "-1"
> {
  let passGroup = ["HFUSION"];
}

//===----------------------------------------------------------------------===//
// HIVM Optimization Control Options
//===----------------------------------------------------------------------===//

def EnableFusedMultiplyAdd : HIVMOptimizationControlOptions<
  "Enable fused multiply add",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def MaxReductionSplit : HIVMOptimizationControlOptions<
  "Max split times for reductionLoop.",
  "int",
  "1"
> {
  let passGroup = ["HIVM"];
}

def InlineQuantScaleInFixpipe : HIVMOptimizationControlOptions<
  "Inline scalar quant scale from vmul into fixpipe op during inline-fixpipe pass.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

//===----------------------------------------------------------------------===//
// SIMT / Proton / Other Options
//===----------------------------------------------------------------------===//

def TritonMetadataOutput : BiShengIRCompileOpt<
  "File to dump triton metadata. -- means stdout",
  "std::string",
  "\"\""
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def DisableDecomposeReduction : SIMTOptimizationControlOptions<
  "Disable SIMT decompose reduction pass",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def DisableReorderInstruction : SIMTOptimizationControlOptions<
  "Disable reorder instruction pass",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableSinkDPXLoad : SIMTOptimizationControlOptions<
  "Enable post-lowering instruction scheduling that reduces register pressure by interleaving load-compute-store chains",
  "bool",
  "true"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableOptimizeMath : SIMTOptimizationControlOptions<
  "Enable optimize math pass",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 0;
}

def EnableSimtReorderInstruction : SIMTOptimizationControlOptions<
  "Enable SIMT reorder instruction pass",
  "bool",
  "false"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableSIMTFastDiv : SIMTOptimizationControlOptions<
  "Enable SIMT fast division optimization",
  "bool",
  "true"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

#ifdef BISHENGIR_ENABLE_TRITON_COMPILE
def ProtonMetricType : ProtonFieldOption<
  "The performance counter metric type we are profiling",
  "proton::MetricType",
  "proton::MetricType::CYCLE",
  "metricType"
> {
  let additionalOptFlags = [{
llvm::cl::values(
  clEnumValN(proton::MetricType::CYCLE, "cycle", "Cycle")
)
  }];
}

def ProtonSamplingStrategy : ProtonFieldOption<
  "Profiling sampling strategy",
  "proton::SamplingStrategy",
  "proton::SamplingStrategy::NONE",
  "samplingStrategy"
> {
  let additionalOptFlags = [{
llvm::cl::values(
  clEnumValN(proton::SamplingStrategy::NONE, "none", "No Sampling"),
  clEnumValN(proton::SamplingStrategy::SELECTIVE, "selective",
             "Selective Sampling")
)
  }];
}

def ProtonSamplingOptions : ProtonFieldOption<
  "Profiling sampling options",
  "std::string",
  "\"\"",
  "samplingOptions"
>;

def ProtonGranularity : ProtonFieldOption<
  "Profiling granularity: warp, warp_group, or cta",
  "proton::gpu::Granularity",
  "proton::gpu::Granularity::WARP",
  "granularity"
> {
  let additionalOptFlags = [{
llvm::cl::values(
  clEnumValN(proton::gpu::Granularity::THREAD, "thread", "Thread"),
  clEnumValN(proton::gpu::Granularity::WARP, "warp", "Warp"),
  clEnumValN(proton::gpu::Granularity::WARP_2, "warp-2", "2 Warps"),
  clEnumValN(proton::gpu::Granularity::WARP_4, "warp-4", "4 Warps"),
  clEnumValN(proton::gpu::Granularity::WARP_8, "warp-8", "8 Warps"),
  clEnumValN(proton::gpu::Granularity::CTA, "cta", "CTA"),
  clEnumValN(proton::gpu::Granularity::WARP_GROUP, "warp-group",
             "Warp Group"),
  clEnumValN(proton::gpu::Granularity::WARP_GROUP_2, "warp-group-2",
             "2 Warp Groups"),
  clEnumValN(proton::gpu::Granularity::WARP_GROUP_4, "warp-group-4",
             "4 Warp Groups"),
  clEnumValN(proton::gpu::Granularity::WARP_GROUP_8, "warp-group-8",
             "8 Warp Groups")
)
  }];
}

def ProtonBufferStrategy : ProtonFieldOption<
  "Profiler buffer recording strategy (circular or flush)",
  "proton::gpu::BufferStrategy",
  "proton::gpu::BufferStrategy::CIRCULAR",
  "bufferStrategy"
> {
  let additionalOptFlags = [{
llvm::cl::values(
  clEnumValN(proton::gpu::BufferStrategy::CIRCULAR, "circular",
             "Circular Buffer"),
  clEnumValN(proton::gpu::BufferStrategy::FLUSH, "flush", "Flush Buffer")
)
  }];
}

def ProtonBufferType : ProtonFieldOption<
  "Internal buffer type (SHARED, GLOBAL) that stores the profiling data",
  "proton::gpu::BufferType",
  "proton::gpu::BufferType::SHARED",
  "bufferType"
> {
  let additionalOptFlags = [{
llvm::cl::values(
  clEnumValN(proton::gpu::BufferType::SHARED, "shared", "Shared Memory"),
  clEnumValN(proton::gpu::BufferType::GLOBAL, "global", "Global Memory")
)
  }];
}

def ProtonBufferSize : ProtonFieldOption<
  "Internal buffer byte size that stores the profiling data. 0 means auto-size based on the device's `maxSharedMemSize`",
  "int32_t",
  "0",
  "bufferSize"
>;

def ProtonMaxSharedMem : ProtonFieldOption<
  "Maximum available shared memory size per CTA",
  "int32_t",
  "32768",
  "maxSharedMemSize"
>;

def ProtonProfileScratchSize : ProtonFieldOption<
  "Profiler global scratch memory size per CTA",
  "int64_t",
  "32768",
  "profileScratchSize"
>;

def ProtonProfileScratchAlignment : ProtonFieldOption<
  "Profiler global scratch memory alignment",
  "int32_t",
  "128",
  "profileScratchAlignment"
>;

def ProtonClkExt : ProtonFieldOption<
  "Use long clock if true, otherwise use 32-bit clock",
  "bool",
  "false",
  "clockExtension"
>;
#endif

def EnableDirectHIVMLowering : BiShengIRCompileOpt<
  "enable-direct-hivm-lowering",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

// These A5-specific compile states are still real options/config entries and
// therefore stay in Options.td, even though they are not currently surfaced as
// normal generated CLI/pass options.
def CompileHost : BiShengIRCompileInternalStateOpt<
  "Compile host-side code path.",
  "bool",
  "false"
>;

def LowerToLLVM : BiShengIRCompileInternalStateOpt<
  "Enable lowering to LLVM.",
  "bool",
  "true"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def HostOutputFile : BiShengIRCompileInternalStateOpt<
  "Host output file path.",
  "std::string",
  "\"-\""
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def DeviceMaxInputParamSizeInBytes : BiShengIRCompileInternalStateOpt<
  "Device max input parameter size in bytes.",
  "size_t",
  "1536"
>;

def SimtStackLimit : BiShengIRCompileInternalStateOpt<
  "Per-thread stack size limit (bytes) for SIMT kernels. The compiler fails "
  "compilation if a kernel's per-thread stack usage exceeds this limit. If "
  "unset, the check is skipped — Triton-Ascend owns the policy (env var, "
  "default) and always forwards a resolved value. Set to a negative value "
  "to disable the check; 0 is a valid (strict) limit.",
  "int32_t",
  "-1"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def UseDPX : BiShengIRCompileOpt<
  "Enable SIMT lowering through DPX Dialect.",
  "bool",
  "true"
> {
  bit isSharedWithDownstreamToolchain = 1;
}
} // let optionScope = ["A5_ONLY"]

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_OPTIONS_TD
