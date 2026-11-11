//===- LoadScaleCapabilityProfile.td - load_scale rows -----*- tablegen -*-===//
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
// GM to L1, i8, is_transposed=false. C310 only. The core does not read a
// transpose flag.

#ifndef BISHENGIR_TARGET_NPU_DMA_CUBELOAD_LOADSCALECAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_DMA_CUBELOAD_LOADSCALECAPABILITYPROFILE_TD

include
    "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/LoadScaleCapability.td"
include "bishengir/Target/NPU/DMA/CubeLoad/CubeLoadProfileBase.td"

def C310LoadScale
    : CubeLoadSameTypeRow<LoadScale, [CubeLoadI8], HIVM_AddressSpace_GM,
                          HIVM_AddressSpace_L1,
                          [BoolValueConstraint<LoadScaleIsTransposed, 0>]>;

def C310LoadScaleCoverage
    : ArchitectureCapabilityCoverage<LoadScale, [C310LoadScale],
                                     PartialProfile>;

#endif // BISHENGIR_TARGET_NPU_DMA_CUBELOAD_LOADSCALECAPABILITYPROFILE_TD
