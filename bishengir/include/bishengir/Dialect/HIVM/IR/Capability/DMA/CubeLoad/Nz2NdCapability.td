//===- Nz2NdCapability.td - NZ2ND execution schema ----------*- tablegen
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
// L1-to-GM NZ-to-ND. This is hivm.nz2nd, not a Fixpipe dma_mode. Rank is
// outside the schema.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_NZ2NDCAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_NZ2NDCAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadTypes.td"

def Nz2NdSchema : RequestSchema<[Src, Dst]>;
def Nz2Nd : CapabilityKey<Nz2NdSchema> {
  let DocNote = "L1 to GM NZ-to-ND. Rank is outside this schema. "
                "A missing memory scope is Unknown.";
}

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_NZ2NDCAPABILITY_TD
