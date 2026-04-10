//===- HFusionAttrs.td - HFusion dialect base support ------*- tablegen -*-===//
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
//
// This is the definition file for enumeration and attributes used in hybrid
// fusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD

include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// HFusion Attributes.
//===----------------------------------------------------------------------===//

class HFusion_Attr<string attrName, string attrMnemonic, list<Trait> traits = [
]> : AttrDef<HFusion_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def HFusion_FusionKindAttr : HFusion_Attr<"FusionKind", "fusion_kind"> {
  let parameters = (ins
    EnumParameter<HFusion_FusionKindEnum>:$fusion_kind
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HFusion fused kernel kind.
  }];
}

def HFusion_StrideAlignDimsAttr :  HFusion_Attr<"StrideAlignDims", "stride_align_dims"> {
  let description = [{
    HFusion stride align dims.
  }];
}

def HFusion_StrideAlignValueInByteAttr :  HFusion_Attr<"StrideAlignValueInByte", "stride_align_value_in_byte"> {
  let description = [{
    HFusion stride align value in byte.
  }];
}

def HFusion_MultiBufferAttr : HFusion_Attr<"MultiBuffer", "multi_buffer"> {
  let description = [{
    HFusion multi buffer attribute for target op.
  }];
}

def HFusion_BindSubBlockAttr : HFusion_Attr<"BindSubBlock", "bind_sub_block"> {
  let description = [{
    Specific operations for bind sub block.
  }];
}

def HFusion_ReduceComposeAttr : HFusion_Attr<"ReduceCompose", "reduce_composed"> {
  let description = [{
    HFusion reduced composed.
  }];
}

def HFusion_ReduceWithIndexOpAttr : HFusion_Attr<"ReduceWithIndexKind", "reduce_with_index_kind"> {
  let parameters = (ins
    EnumParameter<ReduceWithIndexKind>:$reduce_with_index_kind
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    The kind of reduce with index.
  }];
}

def HFusion_ReturnOperandNumAttr : HFusion_Attr<"ReturnOperandNum", "return_operand_num"> {
  let description = [{
    Specifies which operand this corresponds to in the function return
  }];
}

def HFusion_InsertSliceSourceIndexAttr : HFusion_Attr<"InsertSliceSourceIndex", "insert_slice_source_index"> {
  let description = [{
    Specifies which operand is insert_slice source in concat op
  }];
}

def HFusion_PaddingOptionAttr : HFusion_Attr<"PaddingOption", "padding_option"> {
  let parameters = (ins
    EnumParameter<HFusion_PaddingOptionEnum>:$option
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Padding option.
  }];
}

def HFusion_CacheModifierAttr : HFusion_Attr<"CacheModifier", "cache_modifier"> {
  let parameters = (ins
    EnumParameter<HFusion_CacheModifierEnum>:$policy
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Cache modification policy.
  }];
}

def HFusion_EvictionPolicyAttr : HFusion_Attr<"EvictionPolicy", "eviction_policy"> {
  let parameters = (ins
    EnumParameter<HFusion_EvictionPolicyEnum>:$policy
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Eviction policy.
  }];
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD
