//===- CubeLoadProfileBase.td - Shared CubeLoad rows --------*- tablegen
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
// Same-type GM/L1/UB row used by the per-op CubeLoad profiles.

#ifndef BISHENGIR_TARGET_NPU_DMA_CUBELOAD_CUBELOADPROFILEBASE_TD
#define BISHENGIR_TARGET_NPU_DMA_CUBELOAD_CUBELOADPROFILEBASE_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadTypes.td"
include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

class CubeLoadSameTypeRow<CapabilityKey capability, list<TypeSignature> types,
                          I32EnumAttrCase srcSpace, I32EnumAttrCase dstSpace,
                          list<CapabilityConstraint> extras = []>
    : SupportCase<capability, !listconcat(types,
                                          [EnumEquals<SrcSpace, srcSpace>,
                                           EnumEquals<DstSpace, dstSpace>],
                                          extras)>;

#endif // BISHENGIR_TARGET_NPU_DMA_CUBELOAD_CUBELOADPROFILEBASE_TD
