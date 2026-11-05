//===- L12UBCapabilityProfile.td - L12UB support rows -------*- tablegen
//-*-===//
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
//
// L1 to UB. C310 registers the nine common types plus i64 and u64.

#ifndef BISHENGIR_TARGET_NPU_DMA_CUBELOAD_L12UBCAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_DMA_CUBELOAD_L12UBCAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/L12UBCapability.td"
include "bishengir/Target/NPU/DMA/CubeLoad/CubeLoadProfileBase.td"

def C310L12UB : CubeLoadSameTypeRow<L12UB, cubeLoadWideTypes,
                                    HIVM_AddressSpace_L1, HIVM_AddressSpace_UB>;

def C310L12UBCoverage
    : ArchitectureCapabilityCoverage<L12UB, [C310L12UB], PartialProfile>;

#endif // BISHENGIR_TARGET_NPU_DMA_CUBELOAD_L12UBCAPABILITYPROFILE_TD
