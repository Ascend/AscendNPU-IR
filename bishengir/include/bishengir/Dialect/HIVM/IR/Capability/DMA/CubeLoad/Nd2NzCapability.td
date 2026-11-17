//===- Nd2NzCapability.td - ND2NZ execution schema ----------*- tablegen
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
// ND2NZ execution vocabulary. Rank, the forbias user, dst_continuous, and
// l2_cache_mode are outside this schema. Support rows live in
// Target/NPU/DMA/CubeLoad/Nd2NzCapabilityProfile.td.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_ND2NZCAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_ND2NZCAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadTypes.td"

def Nd2NzInitOutBuffer
    : BoolAttrCapabilityParameter<"InitOutBuffer", "init_out_buffer">;

def Nd2NzSchema : RequestSchema<[Src, Dst], [Nd2NzInitOutBuffer]>;
def Nd2Nz : CapabilityKey<Nd2NzSchema> {
  let DocNote = "GM to L1 ND-to-NZ. Rank and the forbias form are outside "
                "this schema. A missing memory scope is Unknown.";
}

// Rank-3 source folded into one MTE2 descriptor. This is a separate key
// because the HIVM op has no batch attribute: the caller asks only when the
// source rank is 3.
def Nd2NzBatchSchema : RequestSchema<[Src, Dst]>;
def Nd2NzBatch : CapabilityKey<Nd2NzBatchSchema> {
  let DocNote = "Batched GM to L1 ND-to-NZ. fp8 has no batch symbol. "
                "A missing memory scope is Unknown.";
}

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_ND2NZCAPABILITY_TD
