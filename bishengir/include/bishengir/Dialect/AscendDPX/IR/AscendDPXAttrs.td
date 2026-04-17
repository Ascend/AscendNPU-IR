//===- AscendDPXAttrs.td - SIMT DPX attrs definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

def AscendDPX_LoadCachePolicyAttr : EnumAttr<AscendDPX_Dialect, AscendDPX_LoadCachePolicy, "dpx_load_cache"> {
  let assemblyFormat = "`<` $value `>`";
}

def AscendDPX_StoreCachePolicyAttr : EnumAttr<AscendDPX_Dialect, AscendDPX_StoreL2CachePolicy, "dpx_store_cache"> {
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
