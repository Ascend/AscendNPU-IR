//===- NPUArchitectureFeatures.td - NPU features ---------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef BISHENGIR_TARGET_NPU_NPUARCHITECTUREFEATURES_TD
#define BISHENGIR_TARGET_NPU_NPUARCHITECTUREFEATURES_TD

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

//===----------------------------------------------------------------------===//
// Execution model
//===----------------------------------------------------------------------===//

// Describes how Vector compute instructions access their data operands. This
// is independent of Cube/Vector core organization and data-exchange paths.
def ExecutionModelFeature : EnumArchitectureFeature<"ExecutionModel">;

// Vector compute instructions read and write data directly in on-chip Unified
// Buffer (UB), without an explicit vector-register load/compute/store sequence.
def MemBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "MemBased">;

// Vector compute instructions operate on vector registers. Explicit vector
// loads and stores transfer data between those registers and UB.
def RegBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "RegBased">;

//===----------------------------------------------------------------------===//
// Cube/Vector core organization
//===----------------------------------------------------------------------===//

// Describes whether Cube and Vector execution units share Scalar control.
// This does not describe the data paths available between the units.
def CubeVectorModeFeature : EnumArchitectureFeature<"CubeVectorMode">;

// Cube and Vector units belong to a coupled core and share a Scalar control
// unit for instruction dispatch.
def Coupled : EnumArchitectureFeatureValue<CubeVectorModeFeature, "Coupled">;

// Cube and Vector reside in separate cores, each with its own Scalar control
// unit for instruction dispatch.
def Separated
    : EnumArchitectureFeatureValue<CubeVectorModeFeature, "Separated">;

//===----------------------------------------------------------------------===//
// Cube/Vector data exchange
//===----------------------------------------------------------------------===//

// Describes hardware paths for exchanging data between Cube-side and
// Vector-side storage, independently of whether the units share a core.
// This does not select a compiler pipeline or synchronization mechanism.
// Individual transfer legality still depends on Op/type/layout/attribute
// constraints.
def CubeVectorDataExchangeFeature
    : EnumArchitectureFeature<"CubeVectorDataExchange">;

// Data exchange requires staging through global memory (GM); no direct
// on-chip Cube/Vector data-exchange path is available.
def GlobalMemoryOnly
    : EnumArchitectureFeatureValue<CubeVectorDataExchangeFeature,
                                   "GlobalMemoryOnly">;

// Direct on-chip exchange is available through UB <-> L1 and Fixpipe -> UB
// paths, without staging through GM. Exchange through GM remains possible.
def OnChipSupported
    : EnumArchitectureFeatureValue<CubeVectorDataExchangeFeature,
                                   "OnChipSupported">;

//===----------------------------------------------------------------------===//
// Architecture feature profiles
//===----------------------------------------------------------------------===//

// Keep the three dimensions explicit: none is inferred from another.
def Dav_C220FeatureProfile
    : ArchitectureFeatureProfile<Dav_C220, [MemBased, Separated,
                                            GlobalMemoryOnly]>;
def Dav_M300FeatureProfile
    : ArchitectureFeatureProfile<Dav_M300, [RegBased, Coupled,
                                            OnChipSupported]>;
def Dav_C310FeatureProfile
    : ArchitectureFeatureProfile<Dav_C310, [RegBased, Separated,
                                            OnChipSupported]>;
// dav-920r1 (Ascend960 / A6) follows the dav-c310 reg-based lineage: vector
// register execution model, separated Cube/Vector cores, and on-chip
// Cube/Vector data exchange paths (UB <-> L1, Fixpipe -> UB).
def Dav_920R1FeatureProfile
    : ArchitectureFeatureProfile<Dav_920R1, [RegBased, Separated,
                                             OnChipSupported]>;

#endif // BISHENGIR_TARGET_NPU_NPUARCHITECTUREFEATURES_TD
