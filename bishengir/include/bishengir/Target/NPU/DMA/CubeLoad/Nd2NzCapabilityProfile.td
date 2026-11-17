//===- Nd2NzCapabilityProfile.td - ND2NZ support rows -------*- tablegen
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
// GM to L1, same element type. C220 and C310 register half, float, bf16,
// i32, u32, i16, u16, i8, and u8 for the plain form. C310 also registers the
// two fp8 types for that form. The batch form has those nine types and no
// fp8 symbol.

#ifndef BISHENGIR_TARGET_NPU_DMA_CUBELOAD_ND2NZCAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_DMA_CUBELOAD_ND2NZCAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/Nd2NzCapability.td"
include "bishengir/Target/NPU/DMA/CubeLoad/CubeLoadProfileBase.td"

defvar cubeLoadC310Nd2NzTypes =
    !listconcat(cubeLoadCommonTypes, [CubeLoadF8E4M3FN, CubeLoadF8E5M2]);

def C220Nd2Nz
    : CubeLoadSameTypeRow<Nd2Nz, cubeLoadCommonTypes, HIVM_AddressSpace_GM,
                          HIVM_AddressSpace_L1, [AnyValue<Nd2NzInitOutBuffer>]>;
def C310Nd2Nz
    : CubeLoadSameTypeRow<Nd2Nz, cubeLoadC310Nd2NzTypes, HIVM_AddressSpace_GM,
                          HIVM_AddressSpace_L1, [AnyValue<Nd2NzInitOutBuffer>]>;
// The batch library has the nine common types and not the two fp8 types.
def C310Nd2NzBatch
    : CubeLoadSameTypeRow<Nd2NzBatch, cubeLoadCommonTypes, HIVM_AddressSpace_GM,
                          HIVM_AddressSpace_L1>;

def C220Nd2NzCoverage
    : ArchitectureCapabilityCoverage<Nd2Nz, [C220Nd2Nz], PartialProfile>;
def C310Nd2NzCoverage
    : ArchitectureCapabilityCoverage<Nd2Nz, [C310Nd2Nz], PartialProfile>;
def C310Nd2NzBatchCoverage
    : ArchitectureCapabilityCoverage<Nd2NzBatch, [C310Nd2NzBatch],
                                     PartialProfile>;

#endif // BISHENGIR_TARGET_NPU_DMA_CUBELOAD_ND2NZCAPABILITYPROFILE_TD
