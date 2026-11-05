//===- NCHW2NC1HWC0Capability.td - NCHW2NC1HWC0 schema ------*- tablegen
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
// Grouped NCHW to NC1HWC0. The groups count is outside this schema.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_NCHW2NC1HWC0CAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_NCHW2NC1HWC0CAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadTypes.td"

def Nchw2Nc1hwc0Schema : RequestSchema<[Src, Dst]>;
def Nchw2Nc1hwc0 : CapabilityKey<Nchw2Nc1hwc0Schema> {
  let DocNote = "GM NCHW to L1 NC1HWC0. groups is outside this schema. "
                "A missing memory scope is Unknown.";
}

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_NCHW2NC1HWC0CAPABILITY_TD
