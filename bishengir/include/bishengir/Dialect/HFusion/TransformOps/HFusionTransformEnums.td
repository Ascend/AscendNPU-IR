//===- HFusionTransformEnums.td - HFusion transform enums ---*- tablegen-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMENUMS
#define HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMENUMS

include "mlir/IR/EnumAttr.td"

def SetBufferSizeModeEnum : I32EnumAttr<"SetBufferSizeMode",
    "Set buffer size mode for SetBufferSizeOp",
    [
      I32EnumAttrCase<"kPerByte", 0, "per_byte">,
      I32EnumAttrCase<"kPerElement", 1, "per_element">,
    ]>{
  let cppNamespace = "mlir::transform";
}

#endif // HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMENUMS