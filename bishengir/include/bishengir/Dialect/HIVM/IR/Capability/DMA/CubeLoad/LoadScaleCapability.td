//===- LoadScaleCapability.td - load_scale execution schema -*- tablegen
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
// MX scale load, GM to L1. The registered library symbol does not take a
// transpose flag, so is_transposed=true stays off the support rows.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_LOADSCALECAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_LOADSCALECAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadTypes.td"

def LoadScaleIsTransposed
    : BoolAttrCapabilityParameter<"IsTransposed", "is_transposed">;

def LoadScaleSchema : RequestSchema<[Src, Dst], [LoadScaleIsTransposed]>;
def LoadScale : CapabilityKey<LoadScaleSchema> {
  let DocNote = "GM to L1 MX scale. is_transposed=true is not a registered "
                "symbol. A missing memory scope is Unknown.";
}

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_LOADSCALECAPABILITY_TD
