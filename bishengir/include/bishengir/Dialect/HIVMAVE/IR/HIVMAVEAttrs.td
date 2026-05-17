//===- AVEAttrs.td - AVE regbased attrs definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR AVE dialect regbased attributes.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_VFATTRS
#define AVE_VFATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "bishengir/Dialect/HIVMAVE/IR/HIVMAVEBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"

//===----------------------------------------------------------------------===//
// AVE RegBased Attributes definitions
//===----------------------------------------------------------------------===//

/// V300 /clang/lib/Headers/__clang_cce_vector_intrinsics.h
/// Dist for VLD/PLD/PST and their variants
def AVE_LoadDist : I32EnumAttr<"LoadDist",
    "Post-processing patterns for AVE Load operation",
    [
      I32EnumAttrCase<"NORM",         0>,
      I32EnumAttrCase<"BRC_B8",       1>,
      I32EnumAttrCase<"BRC_B16",      2>,
      I32EnumAttrCase<"BRC_B32",      3>,
      I32EnumAttrCase<"US_B8",        6>,
      I32EnumAttrCase<"US_B16",       7>,
      I32EnumAttrCase<"DS_B8",        8>,
      I32EnumAttrCase<"DS_B16",       9>,
      I32EnumAttrCase<"BDINTLV",      10>,
      I32EnumAttrCase<"DINTLV_B8",    11>,
      I32EnumAttrCase<"DINTLV_B16",   12>,
      I32EnumAttrCase<"UNPK_B8",      13>,
      I32EnumAttrCase<"UNPK_B16",     14>,
      I32EnumAttrCase<"BLK",          15>,
      I32EnumAttrCase<"E2B_B16",      16>,
      I32EnumAttrCase<"E2B_B32",      17>,
      I32EnumAttrCase<"UNPK_B32",     18>,
      I32EnumAttrCase<"DINTLV_B32",   19>,
      I32EnumAttrCase<"UNPK4_B8",     20>,
      I32EnumAttrCase<"SPLT4CHN_B8",  21>,
      I32EnumAttrCase<"SPLT2CHN_B8",  22>,
      I32EnumAttrCase<"SPLT2CHN_B16", 23>,
      I32EnumAttrCase<"US",           24>,
      I32EnumAttrCase<"DS",           25>,
      I32EnumAttrCase<"PK",           26>,
      I32EnumAttrCase<"BRC_B64",      27>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_LoadDistAttr : EnumAttr<AVE_Dialect, AVE_LoadDist, "load_dist"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Dist for vst
def AVE_StoreDist : I32EnumAttr<"StoreDist",
    "Post-processing patterns for AVE Store operation",
    [
      I32EnumAttrCase<"NORM_B8",     0>,
      I32EnumAttrCase<"NORM_B16",    1>,
      I32EnumAttrCase<"NORM_B32",    2>,
      I32EnumAttrCase<"ONEPT_B8",    3>,
      I32EnumAttrCase<"ONEPT_B16",   4>,
      I32EnumAttrCase<"ONEPT_B32",   5>,
      I32EnumAttrCase<"PK_B16",      6>,
      I32EnumAttrCase<"PK_B32",      7>,
      I32EnumAttrCase<"INTLV_B8",    8>,
      I32EnumAttrCase<"INTLV_B16",   9>,
      I32EnumAttrCase<"PK_B64",      10>,
      I32EnumAttrCase<"INTLV_B32",   11>,
      I32EnumAttrCase<"PK4_B32",     12>,
      I32EnumAttrCase<"MRG4CHN_B8",  13>,
      I32EnumAttrCase<"MRG2CHN_B8",  14>,
      I32EnumAttrCase<"MRG2CHN_B16", 15>,
      I32EnumAttrCase<"NORM_B64",    16>,
      I32EnumAttrCase<"ONEPT_B64",   17>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_StoreDistAttr : EnumAttr<AVE_Dialect, AVE_StoreDist, "store_dist"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Mask for different element data type.
def AVE_B32_Mask : I32EnumAttrCase<"B32", 32, "b32">;
def AVE_B16_Mask : I32EnumAttrCase<"B16", 16, "b16">;
def AVE_B8_Mask : I32EnumAttrCase<"B8", 8, "b8">;
// for error handling
def AVE_UNDEF_Mask : I32EnumAttrCase<"UNDEF", 0, "undef">;

/// Enum attribute of the different mask
def AVE_MaskWidth : I32EnumAttr<"MaskWidth",
    "Different type masks for AVE generating predicate operation",
    [AVE_B32_Mask, AVE_B16_Mask, AVE_B8_Mask, AVE_UNDEF_Mask]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_MaskWidthAttr : EnumAttr<AVE_Dialect, AVE_MaskWidth, "mask_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def AVE_Layout_UnChanged : I32EnumAttrCase<"UNCHANGED", 0>;
def AVE_Layout_Sparse : I32EnumAttrCase<"SPARSE", 1>;
def AVE_Layout_Dense : I32EnumAttrCase<"DENSE", 2>;
def AVE_Layout_Change : I32EnumAttr<"Layout_Change", "Whether layout is changed.",
    [
      AVE_Layout_UnChanged,
      AVE_Layout_Sparse,
      AVE_Layout_Dense
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}
def AVE_LayoutChangeAttr : EnumAttr<AVE_Dialect, AVE_Layout_Change, "layout_change"> {
  let assemblyFormat = [{$value}];
}

def AVE_Layout_Unknown : I32EnumAttrCase<"UNKNOWN", 0>;
def AVE_Layout_Interleave : I32EnumAttrCase<"intlv", 1>;
def AVE_Layout_Planar : I32EnumAttrCase<"planar", 2>;
def AVE_Layout : I32EnumAttr<"Layout", "Layout of less than VL Vector.",
    [
      AVE_Layout_Unknown,
      AVE_Layout_Interleave,
      AVE_Layout_Planar
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}
def AVE_LayoutAttr : EnumAttr<AVE_Dialect, AVE_Layout, "layout"> {
  let assemblyFormat = [{$value}];
}

// V300 ISA Section 9.15.3
// All elements are set to True.
def AVE_PAT_ALL   : I32EnumAttrCase<"ALL", 0>;
// The lowest element.
def AVE_PAT_VL1   : I32EnumAttrCase<"VL1", 1>;
// The lowest 2 elements.
def AVE_PAT_VL2   : I32EnumAttrCase<"VL2", 2>;
// The lowest 3 elements.
def AVE_PAT_VL3   : I32EnumAttrCase<"VL3", 3>;
// The lowest 4 elements.
def AVE_PAT_VL4   : I32EnumAttrCase<"VL4", 4>;
// The lowest 8 elements.
def AVE_PAT_VL8 : I32EnumAttrCase<"VL8", 5>;
// The lowest 16 elements.
def AVE_PAT_VL16  : I32EnumAttrCase<"VL16", 6>;
// The lowest 32 elements.
def AVE_PAT_VL32  : I32EnumAttrCase<"VL32", 7>;
// The lowest 64 elements.
def AVE_PAT_VL64  : I32EnumAttrCase<"VL64", 8>;
// The lowest 128 elements.
def AVE_PAT_VL128 : I32EnumAttrCase<"VL128", 9>;
// Multiples of 3
def AVE_PAT_M3    : I32EnumAttrCase<"M3", 10>;
// Multiples of 4
def AVE_PAT_M4    : I32EnumAttrCase<"M4", 11>;
// The lowest half elements.
def AVE_PAT_H     : I32EnumAttrCase<"H", 12>;
// The lowest quarter elements.
def AVE_PAT_Q     : I32EnumAttrCase<"Q", 13>;
// All elements are set to False.
def AVE_PAT_ALLF  : I32EnumAttrCase<"ALLF", 15>;
def AVE_PgePattern : I32EnumAttr<"PgePattern",
    "Patterns for AVE Pge operation",
    [
      AVE_PAT_ALL,
      AVE_PAT_VL1,
      AVE_PAT_VL2,
      AVE_PAT_VL3,
      AVE_PAT_VL4,
      AVE_PAT_VL8,
      AVE_PAT_VL16,
      AVE_PAT_VL32,
      AVE_PAT_VL64,
      AVE_PAT_VL128,
      AVE_PAT_M3,
      AVE_PAT_M4,
      AVE_PAT_H,
      AVE_PAT_Q,
      AVE_PAT_ALLF
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_PgePatternAttr : EnumAttr<AVE_Dialect, AVE_PgePattern, "pge_pattern"> {
  let assemblyFormat = [{`<` $value `>`}];
}

def AVE_VCVT_PP0 : I32EnumAttrCase<"PP0", 0, "pp0">;
def AVE_VCVT_PP1 : I32EnumAttrCase<"PP1", 1, "pp1">;
def AVE_VCVT_PP2 : I32EnumAttrCase<"PP2", 2, "pp2">;
def AVE_VCVT_PP3 : I32EnumAttrCase<"PP3", 3, "pp3">;
def AVE_VCVT_PPType : I32EnumAttr<"VCVT_PPType",
    "PP patterns for AVE VCVT operation",
    [AVE_VCVT_PP0, AVE_VCVT_PP1, AVE_VCVT_PP2, AVE_VCVT_PP3]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_VCVT_PPTypeAttr : EnumAttr<AVE_Dialect, AVE_VCVT_PPType, "vcvt_pp_type"> {
  let assemblyFormat = [{`<` $value `>`}];
}

def AVE_VCVT_PART_EVEN : I32EnumAttrCase<"PART_EVEN", 0, "part_even">;
def AVE_VCVT_PART_ODD : I32EnumAttrCase<"PART_ODD", 1, "part_odd">;
def AVE_VCVT_PartType : I32EnumAttr<"VCVT_PartType",
    "Patterns for the #part option in AVE VCVT operation",
    [AVE_VCVT_PART_EVEN, AVE_VCVT_PART_ODD]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_VCVT_PartTypeAttr : EnumAttr<AVE_Dialect, AVE_VCVT_PartType, "vcvt_part_type"> {
  let assemblyFormat = [{`<` $value `>`}];
}

def AVE_VCI_INCREASE : I32EnumAttrCase<"INCREASE", 0>;
def AVE_VCI_DECREASE : I32EnumAttrCase<"DECREASE", 1>;
def AVE_VCIType : I32EnumAttr<"VCIType",
    "Patterns for AVE VCI operation",
    [AVE_VCI_INCREASE, AVE_VCI_DECREASE]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_VCITypeAttr : EnumAttr<AVE_Dialect, AVE_VCIType, "vci_type"> {
  let assemblyFormat = [{`<` $value `>`}];
}

def AVE_CMP_EQ : I32EnumAttrCase<"EQ", 0>; // equal to
def AVE_CMP_NE : I32EnumAttrCase<"NE", 1>; // not equal to
def AVE_CMP_GT : I32EnumAttrCase<"GT", 2>; // greater than
def AVE_CMP_GE : I32EnumAttrCase<"GE", 3>; // greater than or equal to
def AVE_CMP_LT : I32EnumAttrCase<"LT", 4>; // less than
def AVE_CMP_LE : I32EnumAttrCase<"LE", 5>; // less than or equal to
def AVE_CMP_ULE : I32EnumAttrCase<"ULE", 6>; // less than or equal to (unsigned)
def AVE_CMP_UGE : I32EnumAttrCase<"UGE", 7>; // greater than or equal to (unsigned)
def AVE_CMP_ULT : I32EnumAttrCase<"ULT", 8>; // less than (unsigned)
def AVE_CMP_UGT : I32EnumAttrCase<"UGT", 9>; // greater than (unsigned)

def AVE_CmpType : I32EnumAttr<"CmpType",
    "cmp op type for AVE Cmp operation",
    [
      AVE_CMP_EQ,
      AVE_CMP_NE,
      AVE_CMP_GT,
      AVE_CMP_GE,
      AVE_CMP_LT,
      AVE_CMP_LE,
      AVE_CMP_ULE,
      AVE_CMP_UGE,
      AVE_CMP_ULT,
      AVE_CMP_UGT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}

def AVE_CmpTypeAttr : EnumAttr<AVE_Dialect, AVE_CmpType, "cmp_type"> {
  let assemblyFormat = [{`<` $value `>`}];
}

// The "kind" of combining function for reductions.
def COMBINING_KIND_ADD : I32EnumAttrCase<"ADD", 0, "add">;
def COMBINING_KIND_MIN : I32EnumAttrCase<"MIN", 1, "min">;
def COMBINING_KIND_MAX : I32EnumAttrCase<"MAX", 2, "max">;
def COMBINING_KIND_XORI : I32EnumAttrCase<"XORI", 3, "xori">;
def COMBINING_KIND_UMIN : I32EnumAttrCase<"UMIN", 4, "umin">;
def COMBINING_KIND_UMAX : I32EnumAttrCase<"UMAX", 5, "umax">;


def CombiningKind : I32EnumAttr<
    "CombiningKind",
    "Kind of combining function for reductions",
    [COMBINING_KIND_ADD, COMBINING_KIND_MIN, COMBINING_KIND_MAX, COMBINING_KIND_XORI, COMBINING_KIND_UMIN, COMBINING_KIND_UMAX]> {
  let cppNamespace = "::mlir::hivmave";
  let genSpecializedAttr = 0;
}

/// An attribute that specifies the combining function for `ave.reduction`.
def AVE_CombiningKindAttr : EnumAttr<AVE_Dialect, CombiningKind, "kind"> {
  let assemblyFormat = "`<` $value `>`";
}

// The kind of type convert.
def TYPE_CVT_KIND_F322F16 : I32EnumAttrCase<"F322F16", 0>;
def TYPE_CVT_KIND_F162F32 : I32EnumAttrCase<"F162F32", 1>;

def TypeConvertKind : I32EnumAttr<
    "TypeConvertKind",
    "Kind of type convert",
    [TYPE_CVT_KIND_F322F16, TYPE_CVT_KIND_F162F32]> {
  let cppNamespace = "::mlir::hivmave";
  let genSpecializedAttr = 0;
}

def AVE_TypeConvertKindAttr : EnumAttr<AVE_Dialect, TypeConvertKind, "type_convert_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def AVE_UnalignedAttr : AttrDef<AVE_Dialect, "Unaligned"> {
  let mnemonic = "unaligned_ub_access";
  let description = [{Access UB with unaligned address}];
}

def AVE_VecMem_DENSE : I32EnumAttrCase<"DENSE", 0, "dense">;
def AVE_VecMem_ODD : I32EnumAttrCase<"ODD", 1, "odd">;
def AVE_VecMem_EVEN : I32EnumAttrCase<"EVEN", 2, "even">;
def AVE_VecMem_EVEN_4 : I32EnumAttrCase<"EVEN_4", 4, "even_4">;
def AVE_VecMemType : I32EnumAttr<"VecMemType", 
    "Vector memory layout", [AVE_VecMem_DENSE, AVE_VecMem_ODD, AVE_VecMem_EVEN, AVE_VecMem_EVEN_4]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave";
}
def AVE_VecMemTypeAttr : EnumAttr<AVE_Dialect, AVE_VecMemType, "vec_mem_type"> {
  let assemblyFormat = [{`<` $value `>`}];
}

def AVE_VecReg_N :I32EnumAttrCase<"N", 0, "n">;
def AVE_VecReg_2N0 : I32EnumAttrCase<"_2N0", 1, "2n0">;
def AVE_VecReg_2N1 : I32EnumAttrCase<"_2N1", 2, "2n1">;
def AVE_VecReg_4N0 : I32EnumAttrCase<"_4N0", 3, "4n0">;
def AVE_VecReg_4N1 : I32EnumAttrCase<"_4N1", 4, "4n1">;
def AVE_VecReg_4N2 : I32EnumAttrCase<"_4N2", 5, "4n2">;
def AVE_VecReg_4N3 : I32EnumAttrCase<"_4N3", 6, "4n3">;
def AVE_VecRegType : I32EnumAttr<"VecRegType",
    "Vector data offset",
    [AVE_VecReg_N, AVE_VecReg_2N0, AVE_VecReg_2N1, AVE_VecReg_4N0, AVE_VecReg_4N1, AVE_VecReg_4N2, AVE_VecReg_4N3]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::hivmave"; 
}
def AVE_VecRegTypeAttr : EnumAttr<AVE_Dialect, AVE_VecRegType, "vec_reg_type"> {
  let assemblyFormat = [{`<` $value `>`}];
}

def AVE_VectorLayoutAttr : AttrDef<AVE_Dialect, "VectorLayout"> {
  let mnemonic = "vector_layout";
  let description = [{Vector layout info}];

  let attrName = "ave.vector_layout";

  let parameters = (
    ins
    "Attribute": $mem,
    "Attribute": $reg
  );
  let cppNamespace = "mlir::hivmave";

  //let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "`<` `{` struct(params) `}` `>`";
}

#endif // AVE_VFATTRS
