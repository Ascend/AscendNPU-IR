//===- Options.td - BiShengIR HIVM Compile-related Options ----*-tablegen-*-==//
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

#ifndef BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_OPTIONS_TD
#define BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_OPTIONS_TD

include "bishengir/Tools/ConfigOptions/Options.td"

class BiShengIRHIVMCompileOpt<
  string desc,
  string ty,
  string defaultVal
> : Option {
  let description = desc;
  let type = ty;
  let defaultValue = defaultVal;
  string compileConfigName = "BiShengIRHIVMCompileMainConfig";
}

class BiShengIRHIVMCompileListOpt<
  string desc,
  string ty
> : ListOption {
  let description = desc;
  let type = ty;
  let defaultValue = "";
  string compileConfigName = "BiShengIRHIVMCompileMainConfig";
}

//===----------------------------------------------------------------------===//
// Feature Control Options
//===----------------------------------------------------------------------===//

class FeatureControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRHIVMCompileOpt<desc, ty, defaultVal> {
  let compileOptionCategories = ["featCtrlCategory"];
}

def EnableTritonKernelCompile : FeatureControlOptions<
  "Enable Triton kernel compilation.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM", "HFUSION_TO_HIVM"];
}

def EnableHIVMCompile : FeatureControlOptions<
  "Enable BiShengHIR HIVM compilation.",
  "bool",
  "true"
>;

def EnableStaticBarePtr : FeatureControlOptions<
  "Enable generating bare ptr calling convention for static shaped kernels.",
  "bool",
  "true"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def EnableBinRelocation : FeatureControlOptions<
  "Enable binary relocation.",
  "bool",
  "true"
> {
  bit isSharedWithDownstreamToolchain = 1;
}

def ConvertHIRToLIR : FeatureControlOptions<
  "Enable converting from BiShengHIR to BiShengLIR.",
  "bool",
  "false"
> {
  // This option is still registered in the command line becuase
  // `bishengir-hivm-compile` relies on it.
  // However, it's hidden.
  let additionalOptFlags = [{
    llvm::cl::Hidden
  }];
}

def EnableLIRCompile : FeatureControlOptions<
  "Enable BiShengLIR compilation.",
  "bool",
  "true"
> {
#ifdef BISHENGIR_PUBLISH
  // This option is not exposed to users when published.
  let emitOptionRegistration = 0;
#endif
  bit isSharedWithDownstreamToolchain = 1;
}

def DisableAutoInjectBlockSync : FeatureControlOptions<
  "Enable generating blocksync wait/set by injectBlockSync pass.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def EnableHIVMGraphSyncSolver : FeatureControlOptions<
  "Use hivm graph-sync-solver instead of inject-sync.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def DisableAutoCVWorkSpaceManage : FeatureControlOptions<
  "In combination with the disableAutoInjectBlockSync option.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def DisableHIVMAutoInjectSync : FeatureControlOptions<
  "Disable auto inject sync intra core.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
}

def DisableHIVMTensorCompile : FeatureControlOptions<
  "Disable BiShengHIR HIVM Tensor compilation.",
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
> : BiShengIRHIVMCompileOpt<desc, ty, defaultVal> {
  let compileOptionCategories = ["dfxCtrlCategory"];
  let passGroup = ["HIVM"];
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

def EnableAutoMultiBuffer : BiShengIRHIVMCompileOpt<
  "Enable auto multi buffer.",
  "bool",
  "false"
> {
  let passGroup = ["HIVM"];
  let compileOptionCategories = ["generalOptCategory"];
}

def LimitAutoMultiBufferOnlyForLocalBuffer : BiShengIRHIVMCompileOpt<
  "When enable-auto-multi-buffer = true, limit it only to work for local buffer",
  "bool",
  "true"
> {
  let passGroup = ["HIVM"];
  let compileOptionCategories = ["generalOptCategory"];
}

//===----------------------------------------------------------------------===//
// HIVM Optimization Control Options
//===----------------------------------------------------------------------===//

class HIVMOptimizationControlOptions<
  string desc,
  string ty,
  string defaultVal
> : BiShengIRHIVMCompileOpt<desc, ty, defaultVal> {
  let passGroup = ["HIVM"];
  let compileOptionCategories = ["hivmOptCategory"];
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

def Target : BiShengIRHIVMCompileOpt<
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
                // Ascend910_95 series
                REGISTER_TARGET(Ascend910_950z), REGISTER_TARGET(Ascend910_9579),
                REGISTER_TARGET(Ascend910_957b), REGISTER_TARGET(Ascend910_957d),
                REGISTER_TARGET(Ascend910_9581), REGISTER_TARGET(Ascend910_9589),
                REGISTER_TARGET(Ascend910_958a), REGISTER_TARGET(Ascend910_958b),
                REGISTER_TARGET(Ascend910_9599),
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

/// Note that we don't own `mlir-print-ir-after-all` and
/// `mlir-print-ir-before-all` as they are upstream options.
/// We add it here because we can capture their values and forward
/// them to `bishengir-hivm-compile` tool.
def MlirPrintIrAfterAll : BiShengIRHIVMCompileOpt<
  "Print IR after each pass.",
  "bool",
  "false"
> {
  let emitOptionRegistration = 0;
  let externalStorage = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

def MlirPrintIrBeforeAll : BiShengIRHIVMCompileOpt<
  "Print IR before each pass.",
  "bool",
  "false"
> {
  let emitOptionRegistration = 0;
  let externalStorage = 0;
  bit isSharedWithDownstreamToolchain = 1;
}

/// Allow operation with no registered dialects.
/// This option is for convenience during testing only and discouraged in
/// general.
def AllowUnregisteredDialects : BiShengIRHIVMCompileOpt<
  "Allow operation with unregistered dialects.",
  "bool",
  "false"
>;

def HIVMCArgs : BiShengIRHIVMCompileListOpt<
  "List of arguments passed to hivmc.",
  "std::string"
>;

#endif // BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_OPTIONS_TD
