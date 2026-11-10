//===- CubeLoadCapabilityBinding.td - CubeLoad Op bindings ---*- tablegen
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
// Reads each CubeLoad execution schema from its Op. dst is an operand. A
// missing memref memory scope does not fill src or dst address space.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYBINDING_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYBINDING_TD

include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadEmission.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/Nd2NzCapability.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/Nz2NdCapability.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/L12UBCapability.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/NCHW2NC1HWC0Capability.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/NCHW2C1HWNC0Capability.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/LoadScaleCapability.td"

def Nd2NzBatchCapabilityBinding : DmaCapabilityBinding<Nd2NzBatch, ND2NZOp, []> {
  // The same op is already checked by Nd2Nz. This key is only for an explicit
  // rank-3 query.
  let Verify = 0;
}

def Nd2NzCapabilityBinding
    : DmaCapabilityBinding<
          Nd2Nz, ND2NZOp,
          [CapabilityExtractor<AttributeBindingSource<"init_out_buffer">,
                               IdentityProjection, Nd2NzInitOutBuffer>]>;

def Nz2NdCapabilityBinding : DmaCapabilityBinding<Nz2Nd, NZ2NDOp, []>;
def L12UBCapabilityBinding : DmaCapabilityBinding<L12UB, L12UBOp, []>;
def Nchw2Nc1hwc0CapabilityBinding
    : DmaCapabilityBinding<Nchw2Nc1hwc0, NCHW2NC1HWC0Op, []>;
def Nchw2C1hwnc0CapabilityBinding
    : DmaCapabilityBinding<Nchw2C1hwnc0, NCHW2C1HWNC0Op, []>;

def LoadScaleCapabilityBinding
    : DmaCapabilityBinding<
          LoadScale, LoadMXScaleOp,
          [CapabilityExtractor<AttributeBindingSource<"is_transposed">,
                               IdentityProjection, LoadScaleIsTransposed>]>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYBINDING_TD
