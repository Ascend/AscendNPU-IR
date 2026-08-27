//===- HFusionTransformEnums.td - HFusion transform enums ---*- tablegen-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMENUMS
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMENUMS

include "mlir/IR/EnumAttr.td"

def SetBufferSizeModeEnum : I32EnumAttr<"SetBufferSizeMode",
    "Set buffer size mode for SetBufferSizeOp",
    [
      I32EnumAttrCase<"kPerByte", 0, "per_byte">,
      I32EnumAttrCase<"kPerElement", 1, "per_element">,
    ]>{
  let cppNamespace = "mlir::transform";
}

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMENUMS
