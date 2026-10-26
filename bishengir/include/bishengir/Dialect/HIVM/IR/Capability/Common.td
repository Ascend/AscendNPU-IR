//===- Common.td - Shared capability properties -*- tablegen -*-===//
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
// Properties shared by HIVM capability schemas. An operation puts this record
// on its own slots. The generated names are ElementType and element_type.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_COMMON_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_COMMON_TD

include "bishengir/Target/Capability/CapabilityBase.td"

def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_COMMON_TD
