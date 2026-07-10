//===- AscendDPXAttrs.td - SIMT DPX attrs definitions ------*- tablegen -*-===//
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
// Defines MLIR Ascend DPX dialect regbased attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDDPX_ATTRS
#define ASCENDDPX_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "bishengir/Dialect/AscendDPX/IR/AscendDPXBase.td"

//===----------------------------------------------------------------------===//
// AscendDPX Attributes definitions
//===----------------------------------------------------------------------===//

def AscendDPX_LoadCachePolicy : I32EnumAttr<"AscendDPXLoadCachePolicy",
    "L2 Cache policy for David A5 load operations",
    [
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_FV",        0>,
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_LV",        1>,
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_PERS",      2>,
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_PREF",      3>,
      I32EnumAttrCase<"L2_CACHE_HINT_NOTALLOC_KEEP",    4>,
      I32EnumAttrCase<"L2_CACHE_HINT_NOTALLOC_CLEAN",   5>,
      I32EnumAttrCase<"L2_CACHE_HINT_NOTALLOC_DROP",    6>,
      I32EnumAttrCase<"L2_CACHE_HINT_IDS_FV",           8>,
      I32EnumAttrCase<"L2_CACHE_HINT_IDS_LV",           9>,
      I32EnumAttrCase<"L2_CACHE_HINT_IDS_PERS",         10>,
      I32EnumAttrCase<"L2_CACHE_HINT_IDS_PREF",         11>,
      I32EnumAttrCase<"L2_CACHE_HINT_EXCLUSIV_FV",      12>,
      I32EnumAttrCase<"L2_CACHE_HINT_EXCLUSIV_LV",      13>,
      I32EnumAttrCase<"L2_CACHE_HINT_EXCLUSIV_PERS",    14>,
      I32EnumAttrCase<"L2_CACHE_HINT_EXCLUSIV_PREF",    15>,
      I32EnumAttrCase<"L2_CACHE_HINT_INVALID",          16>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::ascend_dpx";
}

def AscendDPX_StoreL2CachePolicy : I32EnumAttr<"AscendDPXStoreCachePolicy",
    "L2 Cache policy for David A5 store operations",
    [
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_FV",    0>,
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_LV",    1>,
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_PERS",  2>,
      I32EnumAttrCase<"L2_CACHE_HINT_NORMAL_RED",   3>,
      I32EnumAttrCase<"L2_CACHE_HINT_NOTALLOC_CI",  4>,
      I32EnumAttrCase<"L2_CACHE_HINT_NOTALLOC_PW",  5>,
      I32EnumAttrCase<"L2_CACHE_HINT_NOTALLOC_PI",  6>,
      I32EnumAttrCase<"L2_CACHE_HINT_NOTALLOC_RED", 7>,
      I32EnumAttrCase<"L2_CACHE_HINT_WBH_FV",       8>,
      I32EnumAttrCase<"L2_CACHE_HINT_WBH_LV",       9>,
      I32EnumAttrCase<"L2_CACHE_HINT_WBH_PERS",     10>,
      I32EnumAttrCase<"L2_CACHE_HINT_WBH_RED",      11>,
      I32EnumAttrCase<"L2_CACHE_HINT_WTS_FV",       12>,
      I32EnumAttrCase<"L2_CACHE_HINT_WTS_LV",       13>,
      I32EnumAttrCase<"L2_CACHE_HINT_WTS_PERS",     14>,
      I32EnumAttrCase<"L2_CACHE_HINT_WTS_RED",      15>,
      I32EnumAttrCase<"L2_CACHE_HINT_INVALID",      16>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::ascend_dpx";
}

def AscendDPX_CacheOption : I32EnumAttr<"AscendDPXCacheOption",
    "Cache Option for David A5 Memory Load/Store Operations",
    [
      I32EnumAttrCase<"LOADCACHEOPTION_NCA",  0>,
      I32EnumAttrCase<"LOADCACHEOPTION_CA",   1>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::ascend_dpx";
}

def AscendDPX_VolatileOption : I32EnumAttr<"AscendDPXVolatileOption",
    "Volatile Option for David A5 Load Operations",
    [
      I32EnumAttrCase<"NONVOLATILE", 0>,
      I32EnumAttrCase<"VOLATILE", 1>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::ascend_dpx";
}

def AscendDPX_LoadCachePolicyAttr : EnumAttr<AscendDPX_Dialect, AscendDPX_LoadCachePolicy, "dpx_load_cache"> {
  let assemblyFormat = "`<` $value `>`";
}

def AscendDPX_StoreCachePolicyAttr : EnumAttr<AscendDPX_Dialect, AscendDPX_StoreL2CachePolicy, "dpx_store_cache"> {
  let assemblyFormat = "`<` $value `>`";
}

def AscendDPX_CacheOptionAttr : EnumAttr<AscendDPX_Dialect, AscendDPX_CacheOption, "dpx_cache_option"> {
  let assemblyFormat = "`<` $value `>`";
}

def AscendDPX_VolatileOptionAttr : EnumAttr<AscendDPX_Dialect, AscendDPX_VolatileOption, "dpx_volatile_option"> {
  let assemblyFormat = "`<` $value `>`";
}

def AscendDPX_AddressSpace : I32EnumAttr<"AscendDPXAddressSpace",
    "Address Spaces",
    [ 
      I32EnumAttrCase<"GLOBAL_MEM", 1>,
      I32EnumAttrCase<"SHARED_MEM", 3>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::ascend_dpx";
}

def AscendDPX_CastKind : I32EnumAttr<"AscendDPXCastKind",
    "Type of cast operation",
    [
      I32EnumAttrCase<"SIGNED_TO_FLOAT", 0, "si_to_fp">,
      I32EnumAttrCase<"UNSIGNED_TO_FLOAT", 1, "ui_to_fp">,
      I32EnumAttrCase<"FLOAT_TO_SIGNED", 2, "fp_to_si">,
      I32EnumAttrCase<"FLOAT_TO_UNSIGNED", 3, "fp_to_ui">,
      I32EnumAttrCase<"FLOAT_TO_FLOAT", 4, "fp_to_fp">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::ascend_dpx";
}

def AscendDPX_CastKindAttr : EnumAttr<AscendDPX_Dialect, AscendDPX_CastKind, "dpx_cast_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ASCENDDPX_ATTRS
