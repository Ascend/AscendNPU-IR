//===- Nz2NdCapabilityProfile.td - NZ2ND support rows -------*- tablegen
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
// L1 to GM. C220 registers the nine common types plus i64 and u64. C310 does
// not instantiate that library.

#ifndef BISHENGIR_TARGET_NPU_DMA_CUBELOAD_NZ2NDCAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_DMA_CUBELOAD_NZ2NDCAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/Nz2NdCapability.td"
include "bishengir/Target/NPU/DMA/CubeLoad/CubeLoadProfileBase.td"

def C220Nz2Nd : CubeLoadSameTypeRow<Nz2Nd, cubeLoadWideTypes,
                                    HIVM_AddressSpace_L1, HIVM_AddressSpace_GM>;

def C220Nz2NdCoverage
    : ArchitectureCapabilityCoverage<Nz2Nd, [C220Nz2Nd], PartialProfile>;

#endif // BISHENGIR_TARGET_NPU_DMA_CUBELOAD_NZ2NDCAPABILITYPROFILE_TD
