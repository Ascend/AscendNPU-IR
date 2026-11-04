//===- Common.td - Shared DMA capability slots -*- tablegen -*-===//
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
// Source and destination slots shared by DMA capability schemas. Each slot
// queries the common ElementType property.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_COMMON_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_COMMON_TD

include "bishengir/Dialect/HIVM/IR/Capability/Common.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

// Address space of the SSA value. Read from a memref memory space. An unscoped
// tensor leaves the field unset.
def AddressSpace
    : CapabilitySlotProperty<EnumAttrCapabilityDomain<HIVM_AddressSpaceAttr>,
                             "AddressSpace", "address_space">;

def Src : CapabilitySlot<[ElementType, AddressSpace], "Src", "src">;
def Dst : CapabilitySlot<[ElementType, AddressSpace], "Dst", "dst">;
def SrcType : SlotField<Src, ElementType>;
def DstType : SlotField<Dst, ElementType>;
def SrcSpace : SlotField<Src, AddressSpace>;
def DstSpace : SlotField<Dst, AddressSpace>;

// ODS operands named src and dst. A DMA binding inherits these and adds the
// op-specific parameters.
defvar DmaSrcDstExtractors = [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"src">, AddressSpaceProjection, SrcSpace>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<OperandBindingSource<"dst">, AddressSpaceProjection, DstSpace>
];

class DmaCapabilityBinding<CapabilityKey capability, Op operation,
                           list<CapabilityExtractor> extraExtractors>
    : CapabilityBinding<capability, operation,
                        !listconcat(DmaSrcDstExtractors, extraExtractors)>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_COMMON_TD
