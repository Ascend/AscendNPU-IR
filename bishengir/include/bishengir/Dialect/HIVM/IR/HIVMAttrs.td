//===- HIVMAttrs.td - HIVM dialect attributes definitions --*- tablegen -*-===//
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
// This is the definition file for HIVM dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMATTRS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMATTRS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"

include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// HIVM Enums and Attributes.
//===----------------------------------------------------------------------===//

class HIVM_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hivm";
}

class HIVM_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<HIVM_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class HIVM_I32EnumAttr<string mnemonic, HIVM_I32Enum enumInfo>
  : EnumAttr<HIVM_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Iterator Type
//===----------------------------------------------------------------------===//

def HIVM_IteratorType_Parallel : I32EnumAttrCase<"kParallel", 0, "parallel">;
def HIVM_IteratorType_Broadcast : I32EnumAttrCase<"kBroadcast", 1, "broadcast">;
def HIVM_IteratorType_Transpose : I32EnumAttrCase<"kTranspose", 2, "transpose">;
def HIVM_IteratorType_Reduction : I32EnumAttrCase<"kReduction", 3, "reduction">;
def HIVM_IteratorType_Interleave : I32EnumAttrCase<"kInterleave", 4, "interleave">;
def HIVM_IteratorType_Deinterleave : I32EnumAttrCase<"kDeinterleave", 5, "deinterleave">;
def HIVM_IteratorType_Inverse : I32EnumAttrCase<"kInverse", 6, "inverse">;
def HIVM_IteratorType_Pad : I32EnumAttrCase<"kPad", 7, "pad">;
def HIVM_IteratorType_Concat : I32EnumAttrCase<"kConcat", 8, "concat">;
def HIVM_IteratorType_Gather : I32EnumAttrCase<"kGather", 9, "gather">;
def HIVM_IteratorType_Cumulative : I32EnumAttrCase<"kCumulative", 10, "cumulative">;
def HIVM_IteratorType_Opaque : I32EnumAttrCase<"kOpaque", 99, "opaque">;

def HIVM_IteratorTypeEnum : HIVM_I32Enum<
  "IteratorType", "HIVM Structured Op Iterator Type", [
    HIVM_IteratorType_Parallel,
    HIVM_IteratorType_Broadcast,
    HIVM_IteratorType_Transpose,
    HIVM_IteratorType_Reduction,
    HIVM_IteratorType_Interleave,
    HIVM_IteratorType_Deinterleave,
    HIVM_IteratorType_Inverse,
    HIVM_IteratorType_Pad,
    HIVM_IteratorType_Concat,
    HIVM_IteratorType_Gather,
    HIVM_IteratorType_Cumulative,
    HIVM_IteratorType_Opaque
  ]>;

//===----------------------------------------------------------------------===//
// Data Layout
//===----------------------------------------------------------------------===//

def HIVM_DataLayoutDotAND : I32EnumAttrCase<"DOTA_ND", 1, "dotA_ND">;
def HIVM_DataLayoutDotBND : I32EnumAttrCase<"DOTB_ND", 2, "dotB_ND">;
def HIVM_DataLayoutDotCND : I32EnumAttrCase<"DOTC_ND", 3, "dotC_ND">;
def HIVM_DataLayoutnZ : I32EnumAttrCase<"nZ", 4, "nZ">;
def HIVM_DataLayoutzN : I32EnumAttrCase<"zN", 5, "zN">;
def HIVM_DataLayoutND : I32EnumAttrCase<"ND", 6, "ND">;

def HIVM_DataLayoutEnum : HIVM_I32Enum<
  "DataLayout", "HIVM data layout", [
    HIVM_DataLayoutDotAND,
    HIVM_DataLayoutDotBND,
    HIVM_DataLayoutDotCND,
    HIVM_DataLayoutnZ,
    HIVM_DataLayoutzN,
    HIVM_DataLayoutND
  ]>;

def HIVM_DataLayoutAttr : HIVM_Attr<"DataLayout", "data_layout"> {
  let parameters = (ins
    EnumParameter<HIVM_DataLayoutEnum>:$data_layout,
    OptionalParameter<"std::optional<bool>">:$transpose,
    OptionalParameter<"std::optional<DenseI64ArrayAttr>">:$fractalSizes
  );
  let description = [{
    HIVM data layout mapping attribute. Maps to DOTA_ND, DOTB_ND, DOTC_ND, zN, nZ and ND.
      - `transpose`: Indicates that the layout is transposed.
                     Only valid and must be present for DOTA_ND and DOTB_ND layout.
  }];
  let builders = [
    AttrBuilder<(ins "hivm::DataLayout":$data_layout)>,
    AttrBuilder<(ins "hivm::DataLayout":$data_layout, "std::optional<bool>":$transpose)>
  ];
  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` $data_layout (`,` `transpose` `=` $transpose^)? (`,` `fractalSizes` `=` $fractalSizes^)?`>`
  }];
}

//===----------------------------------------------------------------------===//
// Address Space
//===----------------------------------------------------------------------===//

def HIVM_AddressSpace_Default : I32EnumAttrCase<"Zero", 0, "zero">;
def HIVM_AddressSpace_GM : I32EnumAttrCase<"GM", 1, "gm">;
def HIVM_AddressSpace_L1 : I32EnumAttrCase<"L1", 2, "cbuf">;
def HIVM_AddressSpace_L0A : I32EnumAttrCase<"L0A", 3, "ca">;
def HIVM_AddressSpace_L0B : I32EnumAttrCase<"L0B", 4, "cb">;
def HIVM_AddressSpace_L0C : I32EnumAttrCase<"L0C", 5, "cc">;
def HIVM_AddressSpace_UB : I32EnumAttrCase<"UB", 6, "ub">;

def HIVM_AddressSpaceEnum : HIVM_I32Enum<
  "AddressSpace", "HIVM Address Space", [
    HIVM_AddressSpace_Default,
    HIVM_AddressSpace_GM,
    HIVM_AddressSpace_L1,
    HIVM_AddressSpace_L0A,
    HIVM_AddressSpace_L0B,
    HIVM_AddressSpace_L0C,
    HIVM_AddressSpace_UB
  ]>;

def HIVM_AddressSpaceAttr : HIVM_Attr<"AddressSpace", "address_space",
   [DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>]> {
  let parameters = (ins EnumParameter<HIVM_AddressSpaceEnum>:$address_space);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM address space mapping attribute. Maps to GM, L1, L0A, L0B, L0C and UB.
  }];
}

//===----------------------------------------------------------------------===//
// Op Pipe
//===----------------------------------------------------------------------===//

def HIVM_PIPE_S : I32EnumAttrCase<"PIPE_S", 0>;
def HIVM_PIPE_V : I32EnumAttrCase<"PIPE_V", 1>;
def HIVM_PIPE_M : I32EnumAttrCase<"PIPE_M", 2>;
def HIVM_PIPE_MTE1 : I32EnumAttrCase<"PIPE_MTE1", 3>;
def HIVM_PIPE_MTE2 : I32EnumAttrCase<"PIPE_MTE2", 4>;
def HIVM_PIPE_MTE3 : I32EnumAttrCase<"PIPE_MTE3", 5>;
def HIVM_PIPE_ALL : I32EnumAttrCase<"PIPE_ALL", 6>;
def HIVM_PIPE_MTE4 : I32EnumAttrCase<"PIPE_MTE4", 7>;
def HIVM_PIPE_MTE5 : I32EnumAttrCase<"PIPE_MTE5", 8>;
def HIVM_PIPE_V2 : I32EnumAttrCase<"PIPE_V2", 9>;
def HIVM_PIPE_FIX : I32EnumAttrCase<"PIPE_FIX", 10>;
def HIVM_VIRTUAL_PIPE_MTE2_L1A : I32EnumAttrCase<"VIRTUAL_PIPE_MTE2_L1A", 11>;
def HIVM_VIRTUAL_PIPE_MTE2_L1B : I32EnumAttrCase<"VIRTUAL_PIPE_MTE2_L1B", 12>;
def HIVM_PIPE_NUM : I32EnumAttrCase<"PIPE_NUM", 13>;
def HIVM_PIPE_UNASSIGNED : I32EnumAttrCase<"PIPE_UNASSIGNED", 99>;

def HIVM_PipeEnum : HIVM_I32Enum<
  "PIPE", "HIVM Op Pipe", [
    HIVM_PIPE_S,
    HIVM_PIPE_V,
    HIVM_PIPE_M,
    HIVM_PIPE_MTE1,
    HIVM_PIPE_MTE2,
    HIVM_PIPE_MTE3,
    HIVM_PIPE_ALL,
    HIVM_PIPE_MTE4,
    HIVM_PIPE_MTE5,
    HIVM_PIPE_V2,
    HIVM_PIPE_FIX,
    HIVM_VIRTUAL_PIPE_MTE2_L1A,
    HIVM_VIRTUAL_PIPE_MTE2_L1B,
    HIVM_PIPE_NUM,
    HIVM_PIPE_UNASSIGNED
  ]>;

def HIVM_PipeAttr : HIVM_Attr<"Pipe", "pipe"> {
  let parameters = (ins EnumParameter<HIVM_PipeEnum>:$pipe);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM Op pipe attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Function and Module Core Type
//===----------------------------------------------------------------------===//

def HIVM_AIC : I32EnumAttrCase<"AIC", 1>;
def HIVM_AIV : I32EnumAttrCase<"AIV", 2>;
def HIVM_MIX : I32EnumAttrCase<"MIX", 3>;
def HIVM_AIC_OR_AIV : I32EnumAttrCase<"AIC_OR_AIV", 4>;

def HIVM_TFuncCoreTypeEnum : HIVM_I32Enum<
  "TFuncCoreType", "HIVM Function Core Type", [
    HIVM_AIC,
    HIVM_AIV,
    HIVM_MIX,
    HIVM_AIC_OR_AIV
  ]>;

def HIVM_TFuncCoreTypeAttr : HIVM_Attr<"TFuncCoreType", "func_core_type"> {
  let parameters = (ins EnumParameter<HIVM_TFuncCoreTypeEnum>:$funcCoreType);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM function core type attribute.
  }];
}

def HIVM_TModuleCoreTypeEnum : HIVM_I32Enum<
  "TModuleCoreType", "HIVM Module Core Type", [
    HIVM_AIC,
    HIVM_AIV,
    HIVM_MIX
  ]>;

def HIVM_TModuleCoreTypeAttr : HIVM_Attr<"TModuleCoreType", "module_core_type"> {
  let parameters = (ins EnumParameter<HIVM_TModuleCoreTypeEnum>:$moduleCoreType);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM module core type attribute.

    If all of the functions within the module has `AIV` func core type , the
    module core type is `AIV`.

    If all of the functions within the module has `AIC` func core type , the
    module core type is `AIC`.

    Otherwise, the module core type is `MIX`.
  }];
}

//===----------------------------------------------------------------------===//
// Op Core Type
//===----------------------------------------------------------------------===//

def HIVM_CUBE : I32EnumAttrCase<"CUBE", 1>;
def HIVM_VECTOR : I32EnumAttrCase<"VECTOR", 2>;
def HIVM_CUBE_OR_VECTOR : I32EnumAttrCase<"CUBE_OR_VECTOR", 3>;
def HIVM_CUBE_AND_VECTOR : I32EnumAttrCase<"CUBE_AND_VECTOR", 4>;

def HIVM_TCoreTypeEnum : HIVM_I32Enum<
  "TCoreType", "HIVM Op Core Type", [
    HIVM_CUBE,
    HIVM_VECTOR,
    HIVM_CUBE_OR_VECTOR,
    HIVM_CUBE_AND_VECTOR
  ]>;

def HIVM_TCoreTypeAttr : HIVM_Attr<"TCoreType", "tcore_type"> {
  let parameters = (ins EnumParameter<HIVM_TCoreTypeEnum>:$tcoretype);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM op core type attribute.
  }];
}

def HIVM_TCoreTypeMarkerAttr : HIVM_Attr<"TCoreTypeMarker", "tcore_type_marker"> {
  let parameters = (ins EnumParameter<HIVM_TCoreTypeEnum>:$tcoretype);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM op core type marker attribute.
  }];
}

def HIVM_TPartOfMixAttr : HIVM_Attr<"TPartOfMix", "part_of_mix"> {
  let description = [{
      HIVM function is a part of mix kernel.
  }];
}

//===----------------------------------------------------------------------===//
// Pad Mode for LoadOp
//===----------------------------------------------------------------------===//

// No padding
def HIVM_PadNull : I32EnumAttrCase<"PadNull", 0>;
// Pad using the first element
def HIVM_PadFirstElem : I32EnumAttrCase<"PadFirstElem", 1>;
// Pad using specified pad value
def HIVM_PadValue : I32EnumAttrCase<"PadValue", 2>;

def HIVM_PadModeEnum : HIVM_I32Enum<
  "PadMode", "Pad Mode for LoadOp", [
    HIVM_PadNull,
    HIVM_PadFirstElem,
    HIVM_PadValue
  ]>;

def HIVM_PadModeAttr : HIVM_Attr<"PadMode", "padmode"> {
  let parameters = (ins EnumParameter<HIVM_PadModeEnum>:$padmode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM pad mode attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Round Mode for VCastOp
//===----------------------------------------------------------------------===//

def HIVM_RINT : I32EnumAttrCase<"RINT", 0, "rint">;
def HIVM_ROUND : I32EnumAttrCase<"ROUND", 1, "round">;
def HIVM_FLOOR : I32EnumAttrCase<"FLOOR", 2, "floor">;
def HIVM_CEIL : I32EnumAttrCase<"CEIL", 3, "ceil">;
def HIVM_TRUNC : I32EnumAttrCase<"TRUNC", 4, "trunc">;
def HIVM_ODD : I32EnumAttrCase<"ODD", 5, "odd">;
def HIVM_TRUNCWITHOVERFLOW : I32EnumAttrCase<"TRUNCWITHOVERFLOW", 6, "truncwithoverflow">;

def HIVM_RoundModeEnum : HIVM_I32Enum<
  "RoundMode", "Round Mode for VCastOp", [
    HIVM_RINT,
    HIVM_ROUND,
    HIVM_FLOOR,
    HIVM_CEIL,
    HIVM_TRUNC,
    HIVM_ODD,
    HIVM_TRUNCWITHOVERFLOW
  ]>;

def HIVM_RoundModeAttr : HIVM_I32EnumAttr<"round_mode", HIVM_RoundModeEnum> {
  let description = [{
    - RINT: round to nearest, tie to even (c language rint)
    - ROUND: round to nearest, tie away from zero (c language round)
    - FLOOR: round to minus infinity (c language floor)
    - CEIL: round to positive infinity (c language ceil)
    - TRUNC: round to zero (c language trunc)
    - ODD: round to odd (Von Neumann rounding)
  }];
}

//===----------------------------------------------------------------------===//
// Cast for VCastOp
//===----------------------------------------------------------------------===//

def HIVM_cast_signed : I32EnumAttrCase<"cast_signed", 0>;
def HIVM_cast_unsigned : I32EnumAttrCase<"cast_unsigned", 1>;
def HIVM_bitcast : I32EnumAttrCase<"bitcast", 2>;

def HIVM_TypeFnEnum : HIVM_I32Enum<
  "TypeFn", "Cast for VCastOp", [
    HIVM_cast_signed,
    HIVM_cast_unsigned,
    HIVM_bitcast
  ]>;

def HIVM_TypeFnAttr : HIVM_I32EnumAttr<"cast", HIVM_TypeFnEnum> {
  let description = [{
    HIVM cast attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Compare Mode for VCmpOp
//===----------------------------------------------------------------------===//

def HIVM_EQ : I32EnumAttrCase<"EQ", 0, "eq">;
def HIVM_NE : I32EnumAttrCase<"NE", 1, "ne">;
def HIVM_LT : I32EnumAttrCase<"LT", 2, "lt">;
def HIVM_GT : I32EnumAttrCase<"GT", 3, "gt">;
def HIVM_GE : I32EnumAttrCase<"GE", 4, "ge">;
def HIVM_LE : I32EnumAttrCase<"LE", 5, "le">;

def HIVM_CmpModeEnum : HIVM_I32Enum<
  "CompareMode", "Compare Mode for VCmpOp", [
    HIVM_EQ,
    HIVM_NE,
    HIVM_LT,
    HIVM_GT,
    HIVM_GE,
    HIVM_LE
  ]>;

def HIVM_CmpModeAttr : HIVM_I32EnumAttr<"compare_mode", HIVM_CmpModeEnum> {
  let description = [{
    HIVM compare mode attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Event ID
//===----------------------------------------------------------------------===//

def HIVM_EVENT_ID0 : I32EnumAttrCase<"EVENT_ID0", 0>;
def HIVM_EVENT_ID1 : I32EnumAttrCase<"EVENT_ID1", 1>;
def HIVM_EVENT_ID2 : I32EnumAttrCase<"EVENT_ID2", 2>;
def HIVM_EVENT_ID3 : I32EnumAttrCase<"EVENT_ID3", 3>;
def HIVM_EVENT_ID4 : I32EnumAttrCase<"EVENT_ID4", 4>;
def HIVM_EVENT_ID5 : I32EnumAttrCase<"EVENT_ID5", 5>;
def HIVM_EVENT_ID6 : I32EnumAttrCase<"EVENT_ID6", 6>;
def HIVM_EVENT_ID7 : I32EnumAttrCase<"EVENT_ID7", 7>;

def HIVM_EventEnum : HIVM_I32Enum<
  "EVENT", "Event ID for Synchronization", [
    HIVM_EVENT_ID0,
    HIVM_EVENT_ID1,
    HIVM_EVENT_ID2,
    HIVM_EVENT_ID3,
    HIVM_EVENT_ID4,
    HIVM_EVENT_ID5,
    HIVM_EVENT_ID6,
    HIVM_EVENT_ID7
  ]>;

def HIVM_EventAttr : HIVM_Attr<"Event", "event"> {
  let parameters = (ins EnumParameter<HIVM_EventEnum>:$event);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM event attribute for synchronization.
  }];
}

//===----------------------------------------------------------------------===//
// Unit Flag Mode
//===----------------------------------------------------------------------===//

def HIVM_UNIT_FLAG_DISABLED : I32EnumAttrCase<"DISABLED", 0>;
def HIVM_UNIT_FLAG_RESERVED : I32EnumAttrCase<"RESERVED", 1>;
def HIVM_UNIT_FLAG_ENABLED_WITHOUT_UPDATE : I32EnumAttrCase<"ENABLED_WITHOUT_UPDATE", 2>;
def HIVM_UNIT_FLAG_ENABLED_WITH_UPDATE : I32EnumAttrCase<"ENABLED_WITH_UPDATE", 3>;
def HIVM_UNIT_FLAG_ENABLED_ONLY_LAST_ITER : I32EnumAttrCase<"ENABLED_ONLY_LAST_ITER", 4>;
def HIVM_UNIT_FLAG_ENABLED_ONLY_FIRST_ITER : I32EnumAttrCase<"ENABLED_ONLY_FIRST_ITER", 5>;
def HIVM_UNIT_FLAG_ENABLED_ONLY_FIRST_AND_LAST_ITERS : I32EnumAttrCase<"ENABLED_ONLY_FIRST_AND_LAST_ITERS", 6>;

def HIVM_UnitFlagEnum : HIVM_I32Enum<
  "UNIT_FLAG", "Unit Flag Mode for Synchronization", [
    HIVM_UNIT_FLAG_DISABLED,
    HIVM_UNIT_FLAG_RESERVED,
    HIVM_UNIT_FLAG_ENABLED_WITHOUT_UPDATE,
    HIVM_UNIT_FLAG_ENABLED_WITH_UPDATE,
    HIVM_UNIT_FLAG_ENABLED_ONLY_LAST_ITER,
    HIVM_UNIT_FLAG_ENABLED_ONLY_FIRST_ITER,
    HIVM_UNIT_FLAG_ENABLED_ONLY_FIRST_AND_LAST_ITERS
  ]>;

def HIVM_UnitFlagAttr : HIVM_Attr<"UnitFlag", "unit_flag"> {
  let parameters = (ins EnumParameter<HIVM_UnitFlagEnum>:$unit_flag);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM unit flag attribute for synchronization.
  }];
}

//===----------------------------------------------------------------------===//
// Sync Block Mode
//===----------------------------------------------------------------------===//

def HIVM_ALL_CUBE : I32EnumAttrCase<"ALL_CUBE", 0>;
def HIVM_ALL_VECTOR : I32EnumAttrCase<"ALL_VECTOR", 1>;
def HIVM_ALL_SUB_VECTOR : I32EnumAttrCase<"ALL_SUB_VECTOR", 2>;
def HIVM_BARRIER_CUBE : I32EnumAttrCase<"BARRIER_CUBE", 3>;
def HIVM_BARRIER_VECTOR : I32EnumAttrCase<"BARRIER_VECTOR", 4>;
def HIVM_ALL : I32EnumAttrCase<"ALL", 5>;


def HIVM_SyncBlockModeEnum : HIVM_I32Enum<
  "SyncBlockMode", "HIVM SyncBlockMode", [
    HIVM_ALL_CUBE,
    HIVM_ALL_VECTOR,
    HIVM_ALL_SUB_VECTOR,
    HIVM_BARRIER_CUBE,
    HIVM_BARRIER_VECTOR,
    HIVM_ALL,
  ]>;

def HIVM_SyncBlockModeAttr : HIVM_Attr<"SyncBlockMode", "sync_block_mode"> {
  let parameters = (ins EnumParameter<HIVM_SyncBlockModeEnum>:$sync_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM synchronization block mode attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Sync Block Instruction Mode
//===----------------------------------------------------------------------===//

def HIVM_INTER_BLOCK_SYNCHRONIZATION : I32EnumAttrCase<"INTER_BLOCK_SYNCHRONIZATION", 0>;
def HIVM_INTER_SUBBLOCK_SYNCHRONIZATION : I32EnumAttrCase<"INTER_SUBBLOCK_SYNCHRONIZATION", 1>;
def HIVM_INTRA_BLOCK_SYNCHRONIZATION : I32EnumAttrCase<"INTRA_BLOCK_SYNCHRONIZATION", 2>;

def HIVM_SyncBlockInstrModeEnum : HIVM_I32Enum<
  "SyncBlockInstrMode", "HIVM SyncBlockInstrMode", [
    HIVM_INTER_BLOCK_SYNCHRONIZATION,
    HIVM_INTER_SUBBLOCK_SYNCHRONIZATION,
    HIVM_INTRA_BLOCK_SYNCHRONIZATION
  ]>;

def HIVM_SyncBlockInstrModeAttr : HIVM_Attr<
  "SyncBlockInstrMode", "sync_block_instr_mode"> {
  let parameters = (ins EnumParameter<HIVM_SyncBlockInstrModeEnum>:$sync_instr_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM synchronization block instruction mode attribute.
  }];
  let constBuilderCall =
      "::mlir::hivm::SyncBlockInstrModeAttr::get($_builder.getContext(), "
      "::mlir::hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION)";
}

//===----------------------------------------------------------------------===//
// Reduction Op Kind
//===----------------------------------------------------------------------===//

def HIVM_ReduceOp_SUM : I32EnumAttrCase<"sum", 1>;
def HIVM_ReduceOp_PROD : I32EnumAttrCase<"prod", 2>;
def HIVM_ReduceOp_MAX : I32EnumAttrCase<"max", 3>;
def HIVM_ReduceOp_MIN : I32EnumAttrCase<"min", 4>;
def HIVM_ReduceOp_MAX_WITH_INDEX_LEFT : I32EnumAttrCase<"max_with_index_left", 5>;
def HIVM_ReduceOp_MAX_WITH_INDEX_RIGHT : I32EnumAttrCase<"max_with_index_right", 6>;
def HIVM_ReduceOp_MIN_WITH_INDEX_LEFT : I32EnumAttrCase<"min_with_index_left", 7>;
def HIVM_ReduceOp_MIN_WITH_INDEX_RIGHT : I32EnumAttrCase<"min_with_index_right", 8>;
def HIVM_ReduceOp_ANY : I32EnumAttrCase<"any", 9>;
def HIVM_ReduceOp_ALL : I32EnumAttrCase<"all", 10>;
def HIVM_ReduceOp_XORI : I32EnumAttrCase<"xori", 11>;
def HIVM_ReduceOp_ORI : I32EnumAttrCase<"ori", 12>;
def HIVM_ReduceOp_ANDI : I32EnumAttrCase<"andi", 13>;
def HIVM_ReduceOp_NONE : I32EnumAttrCase<"none", 0>;

def HIVM_ReduceOpEnum : HIVM_I32Enum<
  "ReduceOperation", "Reduction kind for VReduceOp", [
    HIVM_ReduceOp_SUM,
    HIVM_ReduceOp_PROD,
    HIVM_ReduceOp_MAX,
    HIVM_ReduceOp_MIN,
    HIVM_ReduceOp_MAX_WITH_INDEX_LEFT,
    HIVM_ReduceOp_MAX_WITH_INDEX_RIGHT,
    HIVM_ReduceOp_MIN_WITH_INDEX_LEFT,
    HIVM_ReduceOp_MIN_WITH_INDEX_RIGHT,
    HIVM_ReduceOp_ANY,
    HIVM_ReduceOp_ALL,
    HIVM_ReduceOp_XORI,
    HIVM_ReduceOp_ORI,
    HIVM_ReduceOp_ANDI,
    HIVM_ReduceOp_NONE
  ]>;

def HIVM_ReduceOpAttr : HIVM_Attr<"ReduceOp", "reduce_op"> {
  let parameters = (ins EnumParameter<HIVM_ReduceOpEnum>:$reduce_op);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM reduction arithmetic operation attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Atomic Operation Kind for StoreOp
//===----------------------------------------------------------------------===//

def HIVM_ATOMIC_NONE : I32EnumAttrCase<"NONE", 0, "none">;
def HIVM_ATOMIC_ADD : I32EnumAttrCase<"ADD", 1, "add">;
def HIVM_ATOMIC_MAX : I32EnumAttrCase<"MAX", 2, "max">;
def HIVM_ATOMIC_MIN : I32EnumAttrCase<"MIN", 3, "min">;
def HIVM_ATOMIC_AND : I32EnumAttrCase<"AND", 4, "and">;
def HIVM_ATOMIC_OR : I32EnumAttrCase<"OR", 5, "or">;
def HIVM_ATOMIC_XOR : I32EnumAttrCase<"XOR", 6, "xor">;
def HIVM_ATOMIC_CAS : I32EnumAttrCase<"CAS", 7, "or">;
def HIVM_ATOMIC_XCHG : I32EnumAttrCase<"XCHG", 8, "xor">;
def HIVM_ATOMIC_UMAX : I32EnumAttrCase<"UMAX", 9, "umax">;
def HIVM_ATOMIC_UMIN : I32EnumAttrCase<"UMIN", 10, "umin">;

def HIVM_AtomicKindEnum : HIVM_I32Enum<
  "AtomicKind", "Atomic Operation Kind for StoreOp", [
    HIVM_ATOMIC_NONE,
    HIVM_ATOMIC_ADD,
    HIVM_ATOMIC_MAX,
    HIVM_ATOMIC_MIN,
    HIVM_ATOMIC_AND,
    HIVM_ATOMIC_OR,
    HIVM_ATOMIC_XOR,
    HIVM_ATOMIC_CAS,
    HIVM_ATOMIC_XCHG,
    HIVM_ATOMIC_UMAX,
    HIVM_ATOMIC_UMIN,
  ]>;

def HIVM_AtomicKindAttr : HIVM_I32EnumAttr<"atomic_kind", HIVM_AtomicKindEnum> {
  let description = [{
    HIVM atomic store kind attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Alignment Kind
//===----------------------------------------------------------------------===//

def HIVM_ALIGNMENT_ALIGN : I32EnumAttrCase<"ALIGN", 0, "align">;
def HIVM_ALIGNMENT_UNALIGNED : I32EnumAttrCase<"UNALIGNED", 1, "unaligned">;
def HIVM_ALIGNMENT_UNKNOWN : I32EnumAttrCase<"UNKNOWN", 2, "unknown">;

def HIVM_AlignKindEnum : HIVM_I32Enum<"AlignKind", "alignment kind information", [
  HIVM_ALIGNMENT_ALIGN,
  HIVM_ALIGNMENT_UNALIGNED,
  HIVM_ALIGNMENT_UNKNOWN
]>;

def HIVM_AlignKindAttr : HIVM_I32EnumAttr<"align_kind", HIVM_AlignKindEnum> {
  let description = "HIVM alignment kind attribute.";
}

//===----------------------------------------------------------------------===//
// Axis Kind
//===----------------------------------------------------------------------===//

def HIVM_AXIS_FIRST : I32EnumAttrCase<"FIRST", 0, "first">;
def HIVM_AXIS_MIDDLE : I32EnumAttrCase<"MIDDLE", 1, "middle">;
def HIVM_AXIS_LAST : I32EnumAttrCase<"LAST", 2, "last">;

def HIVM_AxisKindEnum : HIVM_I32Enum<"AxisKind", "hivm op axis kind information", [
  HIVM_AXIS_FIRST,
  HIVM_AXIS_MIDDLE,
  HIVM_AXIS_LAST
]>;

def HIVM_AxisKindAttr : HIVM_I32EnumAttr<"axis_kind", HIVM_AxisKindEnum> {
  let description = "HIVM op axis kind attribute.";
}

//===----------------------------------------------------------------------===//
// Block Mapping
//===----------------------------------------------------------------------===//

def DimX : I64EnumAttrCase<"DimX", 0, "x">;

def MappingIdEnum : I64EnumAttr<"MappingId", "Mapping ids for loop mapping", [DimX]> {
  let cppNamespace = "::mlir::hivm";
}

def HIVM_BlockMappingAttr
    : HIVM_Attr<"HIVMBlockMapping", "block",
                [DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>]> {
  let parameters = (ins OptionalParameter<"std::optional<int32_t>">:$order);
  let assemblyFormat = "(`<` `linear_dim` `=` $order^ `>`)?";
  let builders = [
    AttrBuilder<(ins), "return $_get($_ctxt, std::nullopt);">
  ];
}

def HIVM_SubBlockMappingAttr : HIVM_Attr<"HIVMSubBlockMapping", "sub_block", [
  DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>
]> {
  let parameters = (ins EnumParameter<MappingIdEnum>:$sub_block);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM sub block mapping attribute for the cv block dim ratio of mix func.
  }];
}

//===----------------------------------------------------------------------===//
// Data Cache
//===----------------------------------------------------------------------===//

def HIVM_DATACACHE_ALL : I32EnumAttrCase<"ALL", 0, "all">;
def HIVM_DATACACHE_UB : I32EnumAttrCase<"UB", 1, "ub">;
def HIVM_DATACACHE_OUT : I32EnumAttrCase<"OUT", 2, "out">;
def HIVM_DATACACHE_ATOMIC : I32EnumAttrCase<"ATOMIC", 3, "atomic">;

def HIVM_DataCacheKindEnum : HIVM_I32Enum<
  "DataCacheKind", "hivm data cache kind", [
    HIVM_DATACACHE_ALL,
    HIVM_DATACACHE_UB,
    HIVM_DATACACHE_OUT,
    HIVM_DATACACHE_ATOMIC
  ]>;

def HIVM_DataCacheKindAttr : HIVM_I32EnumAttr<"DataCacheKind", HIVM_DataCacheKindEnum> {
  let description = "HIVM data cache kind attribute.";
}

def HIVM_DCCI_SINGLE_CACHE_LINE : I32EnumAttrCase<"SINGLE_CACHE_LINE", 0, "single_cache_line">;
def HIVM_DCCI_ALL_CACHE_LINES : I32EnumAttrCase<"ALL_CACHE_LINES", 1, "all_cache_lines">;

def HIVM_DCCIModeEnum : HIVM_I32Enum<
  "DCCIMode", "hivm dcci mode", [
    HIVM_DCCI_SINGLE_CACHE_LINE,
    HIVM_DCCI_ALL_CACHE_LINES
  ]>;

def HIVM_DCCIModeAttr : HIVM_I32EnumAttr<"DCCIMode", HIVM_DCCIModeEnum> {
  let description = "HIVM DCCI mode attribute.";
}

//===----------------------------------------------------------------------===//
// Plan Memory Mode
//===----------------------------------------------------------------------===//

def HIVM_LOCALMEMPLAN : I32EnumAttrCase<"LOCAL_MEM_PLAN", 0>;
def HIVM_GLOBALWORKSPACEPLAN : I32EnumAttrCase<"GLOBAL_WORKSPACE_PLAN", 1>;

def HIVM_MemPlanModeEnum : HIVM_I32Enum<
  "MemPlanMode", "Mem Plan Mode", [
    HIVM_LOCALMEMPLAN,
    HIVM_GLOBALWORKSPACEPLAN
]>;

//===----------------------------------------------------------------------===//
// Fixpipe
//===----------------------------------------------------------------------===//

def HIVM_FIXPIPEPREQUANT_NO_QUANT : I32EnumAttrCase<"NO_QUANT", 0>;
def HIVM_FIXPIPEPREQUANT_F322F16 : I32EnumAttrCase<"F322F16", 1>;
def HIVM_FIXPIPEPREQUANT_S322I8 : I32EnumAttrCase<"S322I8", 9>;
def HIVM_FIXPIPEPREQUANT_F322BF16 : I32EnumAttrCase<"F322BF16", 16>;

def HIVM_FixpipePreQuantModeEnum : HIVM_I32Enum<
  "FixpipePreQuantMode", "HIVM fixpipe pre_quant mode", [
    HIVM_FIXPIPEPREQUANT_NO_QUANT,
    HIVM_FIXPIPEPREQUANT_S322I8,
    HIVM_FIXPIPEPREQUANT_F322F16,
    HIVM_FIXPIPEPREQUANT_F322BF16
  ]>;

def HIVM_FixpipePreQuantModeAttr : HIVM_I32EnumAttr<
  "fixpipe_pre_quant_mode", HIVM_FixpipePreQuantModeEnum> {
  let description = "HIVM fixpipe pre_quant mode";
}

def HIVM_FIXPIPEPRERELU_NO_RELU : I32EnumAttrCase<"NO_RELU", 0>;
def HIVM_FIXPIPEPRERELU_NORMAL_RELU : I32EnumAttrCase<"NORMAL_RELU", 1>;
def HIVM_FIXPIPEPRERELU_LEAKY_RELU : I32EnumAttrCase<"LEAKY_RELU", 2>;
def HIVM_FIXPIPEPRERELU_P_RELU : I32EnumAttrCase<"P_RELU", 3>;

def HIVM_FixpipePreReluModeEnum : HIVM_I32Enum<
  "FixpipePreReluMode", "HIVM fixpipe pre_relu mode", [
    HIVM_FIXPIPEPRERELU_NO_RELU,
    HIVM_FIXPIPEPRERELU_NORMAL_RELU,
    HIVM_FIXPIPEPRERELU_LEAKY_RELU,
    HIVM_FIXPIPEPRERELU_P_RELU
  ]>;

def HIVM_FixpipePreReluModeAttr : HIVM_I32EnumAttr<
  "fixpipe_pre_relu_mode", HIVM_FixpipePreReluModeEnum> {
  let description = "HIVM fixpipe pre_relu mode";
}

#ifdef BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
def HIVM_FIXPIPEDUALDST_NO_DUAL : I32EnumAttrCase<"NO_DUAL", 0>;
def HIVM_FIXPIPEDUALDST_ROW_SPLIT : I32EnumAttrCase<"ROW_SPLIT", 1>;
def HIVM_FIXPIPEDUALDST_COLUMN_SPLIT : I32EnumAttrCase<"COLUMN_SPLIT", 2>;

def HIVM_FixpipeDualDstModeEnum : HIVM_I32Enum<
  "FixpipeDualDstMode", "HIVM fixpipe dual_dst mode", [
    HIVM_FIXPIPEDUALDST_NO_DUAL,
    HIVM_FIXPIPEDUALDST_ROW_SPLIT,
    HIVM_FIXPIPEDUALDST_COLUMN_SPLIT
  ]>;

def HIVM_FixpipeDualDstModeAttr : HIVM_Attr<
  "FixpipeDualDstMode", "fixpipe_dual_dst_mode"> {
  let parameters = (ins EnumParameter<HIVM_FixpipeDualDstModeEnum>:$dual_dst_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HIVM fixpipe dual destination mode control. For a M x N matrix in L0C, supports the following modes:
    - `NO_DUAL`: single destination mode, write the whole matrix to the target UB indicating by SUB BLOCKID.
    - `ROW_SPLIT`: split in M dimension, M/2 x N is written to each UB, M must be multiples of 2.
    - `COLUMN_SPLIT`: split in N dimension, M x N/2 is written to each UB, N must be multiples of 32.
  }];
  let constBuilderCall =
      "::mlir::hivm::FixpipeDualDstModeAttr::get($_builder.getContext(), "
      "::mlir::hivm::FixpipeDualDstMode::NO_DUAL)";
}

def HIVM_FIXPIPE_DMA_MODE_NZ2ND : I32EnumAttrCase<"NZ2ND", 0, "nz2nd">;
def HIVM_FIXPIPE_DMA_MODE_NZ2DN : I32EnumAttrCase<"NZ2DN", 1, "nz2dn">;
def HIVM_FIXPIPE_DMA_MODE_NZ2NZ : I32EnumAttrCase<"NZ2NZ", 2, "normal">;

def HIVM_FixpipeDMAModeEnum : HIVM_I32Enum<
  "FixpipeDMAMode", "HIVM FixpipeDMA mode with int32_t_storage", [
    HIVM_FIXPIPE_DMA_MODE_NZ2ND, HIVM_FIXPIPE_DMA_MODE_NZ2DN, HIVM_FIXPIPE_DMA_MODE_NZ2NZ
  ]>;

def HIVM_FixpipeDMAModeAttr : HIVM_I32EnumAttr<
  "dma_mode", HIVM_FixpipeDMAModeEnum> {
  let description = "HIVM Fixpipe DMA mode";
}
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES

//===----------------------------------------------------------------------===//
// Deinterleave Mode for VDeinterleaveOp
//===----------------------------------------------------------------------===//

def HIVM_DEINTERLEAVE_CHANNEL_0 : I32EnumAttrCase<"CHANNEL_0", 0>;
def HIVM_DEINTERLEAVE_CHANNEL_1 : I32EnumAttrCase<"CHANNEL_1", 1>;
def HIVM_DEINTERLEAVE_ALL_CHANNELS : I32EnumAttrCase<"ALL_CHANNELS", 999>;

def HIVM_DeinterleaveModeEnum : HIVM_I32Enum<
  "DeinterleaveMode", "HIVM deinterleave mode", [
    HIVM_DEINTERLEAVE_CHANNEL_0,
    HIVM_DEINTERLEAVE_CHANNEL_1,
    HIVM_DEINTERLEAVE_ALL_CHANNELS
  ]>;

def HIVM_DeinterleaveModeAttr : HIVM_I32EnumAttr<
  "deinterleave_mode", HIVM_DeinterleaveModeEnum> {
  let description = "HIVM deinterleave index mode";
}

//===----------------------------------------------------------------------===//
// Macro Op Related
//===----------------------------------------------------------------------===//

// Descale Mode for matmul
def HIVM_DescaleNull : I32EnumAttrCase<"DescaleNull", 0>;
// Padding using n scale num
def HIVM_DescalePerChannel : I32EnumAttrCase<"DescalePerChannel", 1>;
// Padding using one scale num
def HIVM_DescalePerTensor : I32EnumAttrCase<"DescalePerTensor", 2>;
def HIVM_DescaleModeEnum : HIVM_I32Enum<"DescaleMode", "descale mode for matmul", [
  HIVM_DescaleNull,
  HIVM_DescalePerChannel,
  HIVM_DescalePerTensor
]>;

def HIVM_DescaleModeAttr : HIVM_I32EnumAttr<"descale_mode", HIVM_DescaleModeEnum> {
  let description = [{
    HIVM descale mode attribute for matmul op.
  }];
}

def HIVM_LocalMatmul_NoBias : I32EnumAttrCase<"NoBias", 0>;
def HIVM_LocalMatmul_PerChannelAdd : I32EnumAttrCase<"PerChannelAdd", 1>;
def HIVM_LocalMatmul_PerChannelAddWithSplitK : I32EnumAttrCase<"PerChannelAddWithSplitK", 2>;
def HIVM_LocalMatmul_ElementwiseAdd : I32EnumAttrCase<"ElementwiseAdd", 3>;
def HIVM_LocalMatmul_ElementwiseCrossLoopAdd : I32EnumAttrCase<"ElementwiseCrossLoopAdd", 4>;
def HIVM_LocalMatmul_BiasModeEnum : HIVM_I32Enum<"MatmulBiasMode", "bias mode for local matmul op", [
  HIVM_LocalMatmul_NoBias,
  HIVM_LocalMatmul_PerChannelAdd,
  HIVM_LocalMatmul_PerChannelAddWithSplitK,
  HIVM_LocalMatmul_ElementwiseCrossLoopAdd,
  HIVM_LocalMatmul_ElementwiseAdd
]>;

//===----------------------------------------------------------------------===//
// Metadata
//===----------------------------------------------------------------------===//

def HIVM_StorageAligned : HIVM_Attr<"StorageAligned", "storage_aligned"> {
  let description = [{
    If a module is tagged with this attribute, it means that all of the operations
    within all device functions in this module are aligned.
    If a function is tagged with this attribute, it means that all of the operations
    in this function are aligned.
  }];
}

//===----------------------------------------------------------------------===//
// Misc.
//===----------------------------------------------------------------------===//

def HIVM_VF_SIMD : I32EnumAttrCase<"SIMD", 0>;
def HIVM_VF_SIMT : I32EnumAttrCase<"SIMT", 1>;
def HIVM_VF_MIX : I32EnumAttrCase<"MIX", 2>;

def HIVM_VFModeEnum
    : HIVM_I32Enum<"VFMode",
                   "HIVM VF Mode", [HIVM_VF_SIMD, HIVM_VF_SIMT, HIVM_VF_MIX]>;

def HIVM_VFModeAttr : HIVM_I32EnumAttr<"vf_mode", HIVM_VFModeEnum> {
  let description = [{
    HIVM VF mode attribute.
  }];
}

def HIVM_MultiBufferAttr : HIVM_Attr<"MultiBuffer", "multi_buffer"> {
  let description = [{
    HIVM multi-buffer attribute.
  }];
}

def HIVM_StrideAlignValueAttr : HIVM_Attr<"StrideAlignDims", "stride_align_dims"> {
  let description = [{
    HIVM stride align dims.
  }];
}

def HIVM_StrideAlignValueInByteAttr :
  HIVM_Attr<"StrideAlignValueInByte", "stride_align_value_in_byte"> {
  let description = [{
    HIVM stride align value in bytes.
  }];
}

def HIVM_AllocAlignValueAttr : HIVM_Attr<"AllocAlignDims", "alloc_align_dims"> {
  let description = [{
    HIVM alloc align dims.
  }];
}

def HIVM_AllocAlignValueInByteAttr :
  HIVM_Attr<"AllocAlignValueInByte", "alloc_align_value_in_byte"> {
  let description = [{
    HIVM alloc align value in bytes.
  }];
}

def HIVM_FuncDynMemrefArgsAttr
    : HIVM_Attr<"HIVMFuncDynMemrefArgs", "func_dyn_memref_args"> {
  let description = [{
    HIVM FuncDynMemrefArgs to mark the index array of dynamic
    memref arguments of function.
  }];
}

def HIVM_InsertSliceSourceIndexAttr
    : HIVM_Attr<"InsertSliceSourceIndex", "insert_slice_source_index"> {
  let description = [{
    Specifies which operand is insert_slice source in vconcat op
  }];
}

def HIVM_ParallelLoopAttr : HIVM_Attr<"ParallelLoop", "parallel_loop"> {
  let description = [{
    Specifies that marked loop can run in parallel.
  }];
}

def HIVM_UnlikelyConditionAttr
    : HIVM_Attr<"UnlikelyCondition", "unlikely_condition"> {
  let description = [{
    Specifies that marked condition is unlikely to evaluate to true.
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMATTRS_TD
