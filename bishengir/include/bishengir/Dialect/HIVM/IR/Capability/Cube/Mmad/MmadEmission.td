//===- MmadEmission.td - MMAD catalogue scope ---------------*- tablegen
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
// Puts the MMAD catalogue in bishengir::npu::mmad. Fixpipe stays in
// bishengir::npu.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADEMISSION_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADEMISSION_TD

include "bishengir/Target/Capability/CapabilityBase.td"

def MmadEmission : CapabilityEmissionScope<
                       "mmad", "BISHENGIR_TARGET_NPU_CUBE_MMAD_CAPABILITY_H",
                       "BISHENGIR_TARGET_NPU_CUBE_MMAD_CAPABILITY_BINDING_H">;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_CUBE_MMAD_MMADEMISSION_TD
