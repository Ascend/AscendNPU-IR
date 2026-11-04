//===- FixpipeCapabilityBinding.td - Fixpipe Op binding -*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
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
// Reads the Fixpipe execution schema from FixpipeOp. dst is an operand.
// quant_scale presence is whether that optional operand exists, not its value.
// A missing memref memory scope does not fill src or dst address space.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_FIXPIPE_FIXPIPECAPABILITYBINDING_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_FIXPIPE_FIXPIPECAPABILITYBINDING_TD

include "bishengir/Dialect/HIVM/IR/HIVMDoc.td"
include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapability.td"

def FixpipeCapabilityBinding : DmaCapabilityBinding<Fixpipe, FixpipeOp, [
  CapabilityExtractor<AttributeBindingSource<"dma_mode">, IdentityProjection, DmaMode>,
  CapabilityExtractor<AttributeBindingSource<"pre_quant">, IdentityProjection,
                      PreQuantMode>,
  CapabilityExtractor<AttributeBindingSource<"pre_relu">, IdentityProjection, PreRelu>,
  CapabilityExtractor<AttributeBindingSource<"channel_split">, IdentityProjection,
                      ChannelSplit>,
  CapabilityExtractor<OperandBindingSource<"quant_scale">, PresenceProjection,
                      QuantScalePresent>
]>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_FIXPIPE_FIXPIPECAPABILITYBINDING_TD
