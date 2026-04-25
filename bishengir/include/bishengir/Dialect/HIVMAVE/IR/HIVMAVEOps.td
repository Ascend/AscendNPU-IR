//===- AVEVFOps.td - AVE regbased operation definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the regbased operations for the AVE dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_VF_OPS
#define AVE_VF_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "bishengir/Dialect/HIVMAVE/IR/HIVMAVEBase.td"
include "bishengir/Dialect/HIVMAVE/IR/HIVMAVEAttrs.td"
include "bishengir/Dialect/HIVMAVE/IR/HIVMAVETypes.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

//===----------------------------------------------------------------------===//
// AVE RegBased Op definitions
//===----------------------------------------------------------------------===//

class AVE_LibFnOp<string mnemonic, list<Trait> traits = []>
    : AVE_Op<mnemonic,
      !listconcat([
        AVELibraryFunctionOpInterface,
        DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface>
      ], traits)>{
}

//class AVE_AveOp<string mnemonic, list<Trait> traits = []> :
    //AVE_Op<"ave." # mnemonic, traits>;

// Vector unary arithmetic op
class AVE_UnaryOp<string mnemonic, list<Trait> traits = []> :
    AVE_LibFnOp<mnemonic, !listconcat(traits,
      [Pure, AVEUnaryOpInterface, AllTypesMatch<["src", "res"]>])> {
  let summary = "AVE regbase unary op " # mnemonic # ".";
  let arguments = (ins
    AveExtValidVector:$src,
    AveMaskType:$mask,
    Optional<AveExtValidVector>:$pass_thru);
  let results = (outs AveExtValidVector:$res);
  let assemblyFormat = [{
    $src `,` $mask (`,` $pass_thru^)? attr-dict `:` type($src) `,` type($mask)
    (`,` type($pass_thru)^)?
  }];
}

// Vector binary arithmetic op
class AVE_BinaryOp<string mnemonic, list<Trait> traits = []> :
    AVE_LibFnOp<mnemonic, !listconcat(traits,
      [Pure, AVEBinaryOpInterface, AllTypesMatch<["lhs", "rhs", "out"]>])> {
  let summary = "Ave binary op " # mnemonic # ".";
  let arguments = (ins AveExtValidVector:$lhs,
                       AveExtValidVector:$rhs,
                       AveMaskType:$mask,
                       Optional<AveExtValidVector>:$pass_thru);
  let results = (outs AveExtValidVector:$out);

  // TODO: remove type($pass_thru) from assemblyFormat with InferTypeOpInterface.
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask ( `,` $pass_thru^)? attr-dict `:` type($lhs) `,`
    type($mask) ( `,` type($pass_thru)^)?
  }];
}

class AVE_VmullBinaryOp<string mnemonic, list<Trait> traits = []> :
    AVE_LibFnOp<mnemonic, !listconcat(traits,
      [Pure, AllTypesMatch<["src1", "src2", "res1", "res2"]>])> {
  let summary = "Ave binary op " # mnemonic # ".";
  let arguments = (ins AveExtValidVector:$src1,
                       AveExtValidVector:$src2,
                       AveMaskType:$mask,
                       Optional<AveExtValidVector>:$pass_thru);
  let results = (outs AveExtValidVector:$res1,
                      AveExtValidVector:$res2);

  // TODO: remove type($pass_thru) from assemblyFormat with InferTypeOpInterface.
  let assemblyFormat = [{
    $src1 `,` $src2 `,` $mask ( `,` $pass_thru^)? attr-dict `:` type($src1) `,`
    type($mask) ( `,` type($pass_thru)^)?
  }];
}

// Vector tenary arithmatic op
class AVE_TenaryOp<string mnemonic, list<Trait> traits = []> :
    AVE_Op<mnemonic, !listconcat(traits,
      [Pure, AVETenaryOpInterface, AllTypesMatch<["src1", "src2", "src3", "out"]>])> {
  let summary = "Ave tenary op " # mnemonic # ".";
  let arguments = (ins AveValidVector:$src1,
                       AveValidVector:$src2,
                       AveValidVector:$src3,
                       AveMaskType:$mask,
                       Optional<AveValidVector>:$pass_thru);
  let results = (outs AveValidVector:$out);

  //TODO: remove type($pass_thru) from assemblyFormat with InferTypeOpInterface.
  let assemblyFormat = [{
    $src1 `,` $src2 `,` $src3 `,` $mask ( `,` $pass_thru^)? attr-dict `:`
    type($src1) `,` type($mask) ( `,` type($pass_thru)^)?
  }];
}

class AVE_VecScalarOp<string mnemonic, list<Trait> traits = []> :
    AVE_LibFnOp<mnemonic, !listconcat(traits, [
      Pure,
      TypesMatchWith<"the scalar should have the same type with the vector's element type",
                     "vec", "scalar", "getElementType($_self)">,
      AllTypesMatch<["vec", "out"]>
    ])> {
  let summary = "Ave vector-scalar op " # mnemonic # ".";
  let arguments = (ins AveExtValidVector:$vec,
                       AnyTypeOf<[ I8, I16, I32, I64, F16, F32 ]>:$scalar,
                       AveMaskType:$mask,
                       Optional<AveExtValidVector>:$pass_thru);
  let results = (outs AveExtValidVector:$out);

  // TODO: remove type($pass_thru) from assemblyFormat with InferTypeOpInterface.
  let assemblyFormat = [{
    $vec `,` $scalar `,` $mask ( `,` $pass_thru^ )? attr-dict `:` type($vec)
    `,` type($scalar) `,` type($mask) ( `,` type($pass_thru)^ )?
  }];
}


class AVE_ShiftVecScalarOp<string mnemonic, list<Trait> traits = []> :
    AVE_Op<mnemonic, !listconcat(traits, [
      Pure,
      TypesMatchWith<"the scalar should have the same type with the vector's element type",
                     "vec", "scalar", "getElementType($_self)">,
      AllTypesMatch<["vec", "out"]>
    ])> {
  let summary = "Ave vector-scalar shift op " # mnemonic # ".";
  let arguments = (ins AveExtValidVector:$vec,
                       AnyTypeOf<[ I8, I16, I32, I64, F16, F32]>:$scalar,
                       AveMaskType:$mask,
                       I1:$sign,
                       Optional<AveValidVector>:$pass_thru);
  let results = (outs AveExtValidVector:$out);

  // TODO: remove type($pass_thru) from assemblyFormat with InferTypeOpInterface.
  let assemblyFormat = [{
    $vec `,` $scalar `,` $mask `,` $sign ( `,` $pass_thru^)? attr-dict`:` type($vec)
    `,` type($scalar) `,` type($mask) ( `,` type($pass_thru)^ )?
  }];
}

class AVE_PregUnaryOp<string mnemonic, list<Trait> traits = []> :
    AVE_Op<"preg." # mnemonic, !listconcat(traits,
      [Pure, AllTypesMatch<["src", "mask", "res"]>])> {
  let summary = "Ave preg " # mnemonic # ". dst = mask { " # mnemonic # " ( src ) }";
  let arguments = (ins AVE_MaskWidthAttr:$masked_type,
                       AveMaskType:$src,
                       AveMaskType:$mask);
  let results = (outs AveMaskType:$res);
  let assemblyFormat = "$masked_type $src `,` $mask attr-dict `:` type($src)";
}

class AVE_PregBinaryOp<string mnemonic, list<Trait> traits = []> :
    AVE_Op<"preg." # mnemonic, !listconcat(traits,
      [Pure, AllTypesMatch<["lhs", "rhs", "mask", "res"]>])> {
  let summary = "Ave preg " # mnemonic # ". dst = mask { " # mnemonic # " ( lhs, rhs ) }";
  let arguments = (ins AVE_MaskWidthAttr:$masked_type,
                       AveMaskType:$lhs,
                       AveMaskType:$rhs,
                       AveMaskType:$mask);
  let results = (outs AveMaskType:$res);
  let assemblyFormat = "$masked_type $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs)";
}

def AVE_VFGatherOp : AVE_LibFnOp<"vgather",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName", "getOpLibraryCallInputs"]>]>{
  let summary = "Gather operation for AVE.";
  let arguments = (ins Arg<AnyMemRef, "load base", [MemRead]>:$base,
                       Variadic<Index>:$indices,
                       // TODO: Add verifier for index_vec
                       FixedVectorOf<[AnyInteger, Index]>:$index_vec,
                       AveMaskType:$mask);
  let results = (outs AveExtValidVector:$res);
  let assemblyFormat = "$base `[` $indices `]` `[` $index_vec `]` `,` $mask attr-dict `:`"
                       "type($base) `,` type($index_vec) `,` type($mask) `into` type($res)";
  // let hasVerifier = 1;
  let extraClassDeclaration = opNameDeclaration # [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
  }];
}

def AVE_VFScatterOp : AVE_LibFnOp<"vscatter",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let summary = "Scatter operation for AVE.";
  let arguments = (ins Arg<AnyMemRef, "store base", [MemWrite]>:$base,
                       Variadic<Index>:$indices,
                       FixedVectorOf<[Index]>:$index_vec, // TODO: Add verifier for index_vec
                       AveMaskType:$mask,
                       AveExtValidVector:$val);
  let assemblyFormat = "$base `[` $indices `]` `[` $index_vec `]` `,` $mask `,` $val attr-dict `:`"
                       "type($base) `,` type($index_vec) `,` type($mask) `,` type($val)";
  // let hasVerifier = 1;
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFExtFOp : AVE_Op<"vextf", [Pure, AVEElementwiseOpInterface]> {
  let summary = "Extfloat operation for AVE.";
  let arguments = (ins AveVector<[BF16, F16, F8E4M3FN, F8E5M2]>:$src,
                       AveMaskType:$mask,
                       AVE_VCVT_PartTypeAttr:$part);
  let results = (outs AveVector<[F32, BF16, F16]>:$res);
  let assemblyFormat = "$src `,` $part `,` $mask attr-dict `:`"
                       "type($src) `,` type($res) `,` type($mask)";
}

def AVE_VFExtSIOp : AVE_LibFnOp<"vextsi", [Pure, AVEElementwiseOpInterface, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let summary = "Extinteger operation for AVE.";
  let arguments = (ins AveVector<[I8, I16, I32]>:$src,
                       AveMaskType:$mask,
                       OptionalAttr<AVE_VCVT_PartTypeAttr>:$part,
                       OptionalAttr<AVE_VCVT_PPTypeAttr>:$pp);
  let results = (outs AveVector<[I16, I32, I64]>:$res);
  let assemblyFormat = "$src `,` $mask attr-dict `:`"
                       "type($src) `,` type($res) `,` type($mask)";
  let hasVerifier = 1;
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFExtUIOp : AVE_LibFnOp<"vextui", [Pure, AVEElementwiseOpInterface, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let summary = "Extunsigned integer operation for AVE.";
  let arguments = (ins AveVector<[I8, I16, I32]>:$src,
                       AveMaskType:$mask,
                       OptionalAttr<AVE_VCVT_PartTypeAttr>:$part,
                       OptionalAttr<AVE_VCVT_PPTypeAttr>:$pp);
  let results = (outs AveVector<[I16, I32, I64]>:$res);
  let assemblyFormat = "$src `,` $mask attr-dict `:`"
                       "type($src) `,` type($res) `,` type($mask)";
  let hasVerifier = 1;
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFTruncIOp : AVE_LibFnOp<"vtrunci", [Pure, AVEElementwiseOpInterface, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let summary = "Truncate integer operation for AVE.";
  let arguments = (ins AveVector<[I64, I32, I16]>:$src,
                       AveMaskType:$mask,
                       BoolAttr:$sat,
                       OptionalAttr<AVE_VCVT_PartTypeAttr>:$part,
                       OptionalAttr<AVE_VCVT_PPTypeAttr>:$pp,
                       OptionalAttr<HIVM_UnsignedModeAttr>:$uni);
  let results = (outs AveVector<[I32, I16, I8]>:$res);
  let assemblyFormat = "$src `,` $sat `,` $mask attr-dict `:`"
                       "type($src) `,`  type($res) `,` type($mask)";
  let hasVerifier = 1;
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFTruncFOp : AVE_Op<"vtruncf", [Pure, AVEElementwiseOpInterface]> {
  let summary = "Truncate float operation for AVE.";
  let arguments = (ins AveVector<[BF16, F16, F32]>:$src,
                       AveMaskType:$mask,
                       HIVM_RoundModeAttr:$rnd,
                       BoolAttr:$sat,
                       AVE_VCVT_PartTypeAttr:$part);
  let results = (outs AveVector<[BF16, F16, F8E4M3FN, F8E5M2]>:$res);
  let assemblyFormat = "$src `,` $rnd `,` $sat `,` $part `,` $mask attr-dict `:`"
                       "type($src) `,` type($res) `,` type($mask)";
}

def AVE_VFFpToSIntOp : AVE_LibFnOp<"vfptosi", [Pure, AVEElementwiseOpInterface, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName","getOpLibraryCallInputs"]>]> {
  let summary = "Float to signed integer operation.";
  let arguments = (ins AveVector<[F32, F16, BF16]>:$src,
                       AveMaskType:$mask,
                       HIVM_RoundModeAttr:$rnd,
                       OptionalAttr<BoolAttr>:$sat,
                       OptionalAttr<AVE_VCVT_PartTypeAttr>:$part);
  let results = (outs AveVector<[I8, I16, I32, I64]>:$res);
  let assemblyFormat = "$src `,` $rnd `,` $mask attr-dict `:`"
                       "type($src) `,` type($mask) `,` type($res)";
  let extraClassDeclaration = opNameDeclaration;
  let hasVerifier = 1;
}

def AVE_VFFpToUIntOp : AVE_LibFnOp<"vfptoui", [Pure, AVEElementwiseOpInterface, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName","getOpLibraryCallInputs"]>]> {
  let summary = "Float to unsigned integer operation for AVE, only support f16->u8.";
  let arguments = (ins AveVector<[F16, F32]>:$src,
                       AveMaskType:$mask,
                       HIVM_RoundModeAttr:$rnd,
                       BoolAttr:$sat,
                       AVE_VCVT_PartTypeAttr:$part);
  let results = (outs AveVector<[I8, I16, I32, I64]>:$res);
  let assemblyFormat = "$src `,` $rnd `,` $sat `,` $part `,` $mask attr-dict `:`"
                       "type($src) `,` type($mask) `,` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFSIntToFpOp : AVE_LibFnOp<"vsitofp", [Pure, AVEElementwiseOpInterface, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName","getOpLibraryCallInputs"]>]> {
  let summary = "Signed integer to float operation for AVE.";
  let arguments = (ins AveVector<[I8, I16, I32, I64]>:$src,
                       AveMaskType:$mask,
                       OptionalAttr<HIVM_RoundModeAttr>:$rnd,
                       OptionalAttr<AVE_VCVT_PartTypeAttr>:$part);
  let results = (outs AveVector<[F16, F32]>:$res);
  let assemblyFormat = "$src `,` $mask attr-dict `:`"
                       "type($src) `,` type($mask) `,` type($res)";
  let extraClassDeclaration = opNameDeclaration;
  let hasVerifier = 1;
}

def AVE_VFUIntToFpOp : AVE_Op<"vuitofp", [
  Pure,  AVEElementwiseOpInterface, DeclareOpInterfaceMethods<
            AVELibraryFunctionOpInterface,
            [ "getOpLibraryCallName", "getOpLibraryCallInputs" ]>
]> {
  let summary = "Unsigned integer to float operation for AVE, only support u8->f16 / u64->f32.";
  let arguments = (ins AveVector<[I8, I64]>:$src,
                       AveMaskType:$mask,
                       OptionalAttr<HIVM_RoundModeAttr>:$rnd,
                       AVE_VCVT_PartTypeAttr:$part);
  let results = (outs AveVector<[F16, F32]>:$res);
  let assemblyFormat = "$src `,` $part `,` $mask attr-dict `:`"
                       "type($src) `,` type($mask) `,` type($res)";
  let extraClassDeclaration = opNameDeclaration;
  let hasVerifier = 1;
}

def AVE_VFVtrcOp : AVE_Op<"vtrc", [Pure, AVEElementwiseOpInterface]> {
  let summary = "Just rounding a op.";
  let arguments = (ins AveVector<[F16, F32, BF16]>:$src,
                       HIVM_RoundModeAttr:$rnd,
                       AveMaskType:$mask);
  let results = (outs AveVector<[F16, F32, BF16]>:$res);
  let assemblyFormat = "$src `,` $rnd `,` $mask attr-dict `:`"
                       "type($src) `,` type($mask) `,` type($res)";
}

def AVE_VFCmpOp : AVE_LibFnOp<"vcmp", [
    Pure, AVEElementwiseOpInterface, AllTypesMatch<["lhs", "rhs"]>,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let summary = "Compare operation for AVE.";
  let arguments = (ins AVE_CmpTypeAttr:$cmp,
                       AveExtValidVector:$lhs,
                       AveExtValidVector:$rhs,
                       AveMaskType:$mask);
  let results = (outs AveMaskType:$res);
  let assemblyFormat = "$cmp $lhs `,` $rhs `,` $mask attr-dict `:`"
                      "type($lhs) `,` type($mask) `->` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFCmpS : AVE_LibFnOp<"vcmps", [Pure,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let summary = "Compare vector with scalar operation for AVE.";
  let arguments = (ins AVE_CmpTypeAttr:$cmp,
                       AveExtValidVector:$vec,
                       AnyTypeOf<[I8, I16, I32, I64, BF16, F16, F32]>:$scalar,
                       AveMaskType:$mask);
  let results = (outs AveMaskType:$res);
  let assemblyFormat = "$cmp $vec `,` $scalar `,` $mask attr-dict `:`"
                       "type($vec) `,` type($scalar) `,` type($mask) `->` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

// TODO: Verify this.
def AVE_VFBroadcastScalarOp : AVE_LibFnOp<"scalar_broadcast", [Pure,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let arguments = (ins AnyTypeOf<[I1, I8, I16, I32, I64, BF16, F16, F32, F8E4M3FN, F8E5M2]>:$src);
  let results = (outs AveVector<[I1, I8, I16, I32, I64, BF16, F16, F32, F8E4M3FN, F8E5M2]>:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

def HIVM_VFBroadcastVectorOp : AVE_LibFnOp<"vector_broadcast", [Pure,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName","getOpLibraryCallInputs"]>]> {
  let arguments = (ins AveExtValidVector:$src,
                       AveMaskType:$mask,
                       BoolAttr: $low);
  let results = (outs AveExtValidVector:$res);
  let assemblyFormat = "$src `,` $mask `,` $low attr-dict `:` type($src) `,` type($mask) `->` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFBroadcastScalarMaskOp : AVE_LibFnOp<"broadcast", [Pure,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let arguments = (ins AnyTypeOf<[I8, I16, I32, I64, BF16, F16, F32, F8E4M3FN, F8E5M2]>:$src,
                       AveMaskType:$mask);
  let results = (outs AveVector<[I8, I16, I32, I64, BF16, F16, F32, F8E4M3FN, F8E5M2]>:$res);
  let assemblyFormat = "$src `,` $mask attr-dict `:` type($src) `,` type($mask) `->` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFVCIOp : AVE_LibFnOp<"vci", [Pure,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName","getOpLibraryCallInputs"]>]> {
  // index, increase_p
  let arguments = (ins AnyTypeOf<[I8, I16, I32, I64, F16, F32]>:$index, AVE_VCITypeAttr:$increase_p);
  let results = (outs AveVector<[I8, I16, I32, I64, F16, F32]>:$res);
  let assemblyFormat = "$index `,` $increase_p attr-dict `:` type($index) `,` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

// Preg type casting operation,
// This operation convert the mask type among b8, b16, or b32. To implement
// the conversion, Hardware can use interleave/deinterleave with mask type.
// TODO: Constraint the mask length should be the same between src and res.
def AVE_VFPregTypeCastOp : AVE_Op<"preg.cast", [Pure]> {
  let summary = "Preg type casting operation.";
  let arguments = (ins AveMaskType:$src);
  let results = (outs AveMaskType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
  let hasCanonicalizer = 1;
}

def AVE_VFSelectOp : AVE_LibFnOp<"vsel", [
    Pure,
    AllTypesMatch<["true_value", "false_value", "res"]>,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let arguments = (ins AveMaskType:$mask,
                       AveExtValidVector:$true_value,
                       AveExtValidVector:$false_value);
  let results = (outs AveExtValidVector:$res);
  let assemblyFormat = "$mask `,` $true_value `,` $false_value attr-dict `:`"
                       "type($mask) `,` type($res)";
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFPltOp : AVE_Op<"plt", [Pure]> {
  let summary = [{
  PLT operation on ISA.

  if true_shape >= VL/bit_width_of(mask_type), then result all true, otherwise the part
  of true_shape will be true and the rest will be false.

  the second result is the new true_shape, $new_true_shape = $true_shape - VL/bit_width_of($mask_type).
  }];
  let arguments = (ins Index:$true_shape);
  let results = (outs AveMaskType:$res,
                      Index:$new_true_shape);
  let assemblyFormat = "$true_shape attr-dict `:` type($res) `,` type($new_true_shape)";
}

def AVE_VFPltMOp : AVE_Op<"pltm", [Pure]> {
  let summary = [{
  PLTM operation on ISA. It takes two integers as input: offset and ub.

  For each bit b in res at index i, b = (i + offset < ub).
  Note this is different from the hardware instruction, where the above equation is b = (i + offset * VL_t < ub)
  This difference is fixed during conversion to AVE Intrin.
  }];
  let arguments = (ins Index:$offset, Index:$ub);
  let results = (outs AveMaskType:$res);
  let assemblyFormat = "$offset `,` $ub attr-dict `:` type($res)";
}

def AVE_VFPgeOp : AVE_Op<"pge", [Pure]> {
  let summary = [{
  PGE operation on ISA.

  the result is a mask vector set according to the $pattern.
  }];
  let arguments = (ins AVE_PgePatternAttr:$pattern);
  let results = (outs AveMaskType:$res);
  let assemblyFormat = "$pattern attr-dict `:` type($res)";
}

def AVE_VFLoadOp : AVE_LibFnOp<"vload", [Pure,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName","getOpLibraryCallInputs"]>]> {
  let summary = "Load operation for AVE.";
  let arguments = (ins AVE_LoadDistAttr:$pattern,
                       Arg<AnyMemRef, "load base", [MemRead]>:$base, Variadic<Index>:$indices);
  let results = (outs AveVector<[I1, I8, I16, I32, I64, BF16, F16, F32, F8E5M2, F8E4M3FN]>:$res,
                      Optional<AveVector<[I1, I8, I16, I32, I64, BF16, F16, F32, F8E5M2, F8E4M3FN]>>:$res1);
  let assemblyFormat = "$pattern $base `[` $indices `]` attr-dict `:`"
                       "type($base) `into` type($res) ( `,` type($res1)^)?";
  let extraClassDeclaration =  opNameDeclaration # [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getRes().getType());
    }
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$vec, "::mlir::Value":$base, "::mlir::Value":$offset), [{
      build($_builder, $_state, vec, base, ::mlir::ValueRange{offset});
    }]>,
    OpBuilder<(ins "::mlir::Type":$vec, "::mlir::Value":$base, "::mlir::ValueRange":$indices), [{
      auto loadPattern = symbolizeLoadDist("NORM").value();
      auto attr = LoadDistAttr::get($_builder.getContext(), loadPattern);

      $_state.getOrAddProperties<Properties>().pattern = attr;
      $_state.addOperands(base);
      $_state.addOperands(indices);
      $_state.addTypes(vec);
    }]>
  ];
}

def AVE_VFUnalignedMaskedStoreOp : AVE_Op<"unaligned.masked.store"> {
  let summary = "Unaligned masked store operation for AVE.";
  let arguments = (ins Arg<AnyMemRef, "store base", [MemWrite]>:$base,
                       Variadic<Index>:$indices,
                       AveMaskType:$mask,
                       AveVector<[I8, I16, I32, I64, BF16, F16, F32, F8E4M3FN]>:$val);
  let assemblyFormat = "$base `[` $indices `]` `,` $mask `,` $val attr-dict `:`"
                       "type($base) `,` type($mask) `,` type($val)";
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getVal().getType());
    }
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Value":$base, "::mlir::Value":$offset, "::mlir::Value":$mask, "::mlir::Value":$val), [{
      $_state.addOperands(base);
      $_state.addOperands({offset});
      $_state.addOperands(mask);
      $_state.addOperands(val);
    }]>
  ];
}

def AVE_VFStoreWithStrideOp : AVE_LibFnOp<"store_with_stride",[]> {
  let summary = "Masked store with stride operation for AVE.";
  let arguments = (ins Arg<AnyMemRef, "store base", [MemWrite]>:$base,
                       Variadic<Index>:$indices,
                       Index:$stride,
                       AveMaskType:$mask,
                       AveVector<[I8, I16, I32, BF16, F16, F32, F8E4M3FN, F8E5M2]>:$val);
  let assemblyFormat = "$base `[` $indices `]` `,` $stride `,` $mask `,` $val attr-dict `:`"
                       "type($base) `,` type($mask) `,` type($val)";
  let extraClassDeclaration = opNameDeclaration # [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getVal().getType());
    }
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Value":$base, "::mlir::ValueRange":$indices, "::mlir::Value":$stride, "::mlir::Value":$mask, "::mlir::Value":$val), [{
      $_state.addOperands(base);
      $_state.addOperands(indices);
      $_state.addOperands(stride);
      $_state.addOperands(mask);
      $_state.addOperands(val);
    }]>
  ];
}

def AVE_VFMaskedStoreOp : AVE_LibFnOp<"masked_store",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName","getOpLibraryCallInputs"]>]> {
  let summary = "Masked store operation for AVE.";
  let arguments = (ins AVE_StoreDistAttr:$pattern,
                       Arg<AnyMemRef, "store base", [MemWrite]>:$base,
                       Variadic<Index>:$indices,
                       AveMaskType:$mask,
                       AveVector<[I1, I8, I16, I32, I64, BF16, F16, F32, F8E4M3FN, F8E5M2]>:$val);
  let assemblyFormat = "$pattern $base `[` $indices `]` `,` $mask `,` $val attr-dict `:`"
                       "type($base) `,` type($mask) `,` type($val)";
  let extraClassDeclaration = opNameDeclaration # [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getVal().getType());
    }
  }];
  // let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Value":$base, "::mlir::ValueRange":$indices, "::mlir::Value":$mask, "::mlir::Value":$val), [{
      auto elementTypeBitWidth = mlir::dyn_cast<ShapedType>(base.getType()).getElementTypeBitWidth();

      std::string pattern = "NORM_B" + std::to_string(elementTypeBitWidth);
      auto storePattern = symbolizeStoreDist(pattern).value();
      auto attr = StoreDistAttr::get($_builder.getContext(), storePattern);

      $_state.getOrAddProperties<Properties>().pattern = attr;
      $_state.addOperands(base);
      $_state.addOperands(indices);
      $_state.addOperands(mask);
      $_state.addOperands(val);
    }]>,
    OpBuilder<(ins "::mlir::Value":$base, "::mlir::Value":$offset, "::mlir::Value":$mask, "::mlir::Value":$val), [{
      build($_builder, $_state, base, ::mlir::ValueRange{offset}, mask, val);
    }]>
  ];
}

def AVE_VFInterleaveOp : AVE_LibFnOp<"vintlv", [
    Pure, AllTypesMatch<["src1", "src2"]>, AllTypesMatch<["res1", "res2"]>, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let arguments = (ins AveExtValidVector:$src1,
                       AveExtValidVector:$src2);
  let results = (outs AveExtValidVector:$res1,
                      AveExtValidVector:$res2);
  let assemblyFormat = "$src1 `,` $src2 attr-dict `:` type($src1) `,` type($res1)";
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFDeInterleaveOp : AVE_LibFnOp<"vdintlv", [
    Pure, AllTypesMatch<["src1", "src2"]>, AllTypesMatch<["res1", "res2"]>, DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let arguments = (ins AveExtValidVector:$src1,
                       AveExtValidVector:$src2);
  let results = (outs AveExtValidVector:$res1,
                      AveExtValidVector:$res2);
  let assemblyFormat = "$src1 `,` $src2 attr-dict `:` type($src1) `,` type($res1)";
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFMemBarOp : AVE_Op<"membar", [AllTypesMatch<["src"]>]> {
  let arguments = (ins I32:$src);
  let summary = "Memory barrier operation for AVE.";
  let assemblyFormat = "$src attr-dict";
}

def AVE_ReductionOp :
  AVE_Op<"reduction", [Pure,
     PredOpTrait<"source operand vector and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>,
     AllShapesMatch<["vector", "result"]>,
     HasMatchingMaskTypeConstraint<"result", "mask">,
     DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>
     ]>,
     Arguments<(ins AVE_CombiningKindAttr:$kind,
                AnyVectorOfAnyRank:$vector,
                VectorOfRankAndType<[1], [I1]>:$mask)>,
     Results<(outs AnyVectorOfAnyRank:$result)> {
  let summary = "reduction operation";
  let description = [{
    Reduces an 1-d vector "horizontally" into a vector using the given
    operation.

    Not that these operations are restricted to 1-D vectors to remain close to
    the ave ops. Result will be placed at the first position of vector.

    Example:

    ```mlir
    %1 = ave.reduction <add>, %0, %mask : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getVector().getType());
    }

    VectorType getMaskVectorType() {
      return ::llvm::cast<VectorType>(getMask().getType());
    }

    bool isUnsigned() {
      switch(getKind()) {
        case CombiningKind::UMAX:
        case CombiningKind::UMIN:
          return true;
        default:
          return false;
      }
    }
  }];
  let builders = [
    // Builders that infers the type of `dest`.
    OpBuilder<(ins "CombiningKind":$kind, "Value":$vector, "Value":$acc)>,
    // Builder that infers the type of `dest` and has no accumulator.
    OpBuilder<(ins "CombiningKind":$kind, "Value":$vector)>
  ];

  let assemblyFormat = "$kind `,` $vector `,` $mask attr-dict"
                       " `:` type($vector) `,` type($mask) `->` type($result)";
}

// Base class for shift operations
class AVE_ShiftOp<string mnemonic, list<Trait> traits = []> :
    AVE_LibFnOp<mnemonic, !listconcat(traits,
      [Pure, AllTypesMatch<["lhs", "rhs", "out"]>])> {
  let summary = "Ave shift op " # mnemonic # ".";
  let arguments = (ins AveVector<[I8, I16, I32, I64]>:$lhs,
                       AveVector<[I8, I16, I32, I64]>:$rhs,
                       AveMaskType:$mask,
                       I1:$sign,
                       Optional<AveValidVector>:$pass_thru);
  let results = (outs AveVector<[I8, I16, I32, I64]>:$out);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask `,` $sign ( `,` $pass_thru^)? attr-dict `:` type($lhs) `,`
    type($mask) ( `,` type($pass_thru)^)?
  }];
}

// Left shift operation
def AVE_VFShlOp : AVE_ShiftOp<"vshl", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let description = [{
    Performs left shift operations. The `sign` attribute determines whether
    the operation treats values as signed (arithmetic shift) or unsigned (logical shift).

    Syntax: VSHL.type Vd, Vn, Vm, Pg, #sign
  }];
  let extraClassDeclaration = opNameDeclaration;
}

// Right shift operation
def AVE_VFShrOp : AVE_ShiftOp<"vshr", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let description = [{
    Performs right shift operations. The `sign` attribute determines whether
    the operation treats values as signed (arithmetic shift) or unsigned (logical shift).

    Syntax: VSHR.type Vd, Vn, Vm, Pg, #sign
  }];
  let extraClassDeclaration = opNameDeclaration;
}

//
//// Unary Ops
//

def AVE_VFAbsOp : AVE_UnaryOp<"vabs",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFNegOp : AVE_UnaryOp<"vneg",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFNotOp : AVE_UnaryOp<"vnot", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFExpOp : AVE_UnaryOp<"vexp">;
def AVE_VFSqrtOp : AVE_UnaryOp<"vsqrt">;
def AVE_VFRsqrtOp : AVE_UnaryOp<"vrsqrt">;
def AVE_VFLnOp : AVE_UnaryOp<"vln">;
def AVE_VFReluOp : AVE_UnaryOp<"vrelu">;
//
//// Binary Ops
//

def AVE_VFAddOp : AVE_BinaryOp<"vadd",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFMulOp : AVE_BinaryOp<"vmul",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFSubOp : AVE_BinaryOp<"vsub",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFDivOp : AVE_BinaryOp<"vdiv",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let arguments = (ins AveVector<[I16, I32, F16, F32, I64, UI64]>:$lhs,
                       AveVector<[I16, I32, F16, F32, I64, UI64]>:$rhs,
                       AveMaskType:$mask,
                       DefaultValuedAttr<HIVM_TypeFnAttr, "hivm::TypeFn::cast_signed">:$cast,
                       Optional<AveExtValidVector>:$pass_thru);
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFDivfOp : AVE_BinaryOp<"vdivf">{
  let extraClassDeclaration = opNameDeclaration;
} //fdiv

def AVE_VFMaxOp : AVE_BinaryOp<"vmax",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFMinOp : AVE_BinaryOp<"vmin",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VMaxSIOp : AVE_BinaryOp<"vsmax",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmax";
}

def AVE_VMinSIOp : AVE_BinaryOp<"vsmin",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmin";
}

def AVE_VMaxUIOp : AVE_BinaryOp<"vumax",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmax";
}

def AVE_VMinUIOp : AVE_BinaryOp<"vumin",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmin";
}

def AVE_VFDivFHPOp : AVE_BinaryOp<"vdivfhp",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFModOp : AVE_BinaryOp<"vmod",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFModUIOp : AVE_BinaryOp<"vmodui",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFAbsDiffOp : AVE_BinaryOp<"vabs_diff">;
def AVE_VFSaddOp : AVE_BinaryOp<"vsadd">;
def AVE_VFSsubOp : AVE_BinaryOp<"vssub">;
def AVE_VFRndOp : AVE_BinaryOp<"vrnd">;
def AVE_VFPReluOp : AVE_BinaryOp<"vprelu">;

def AVE_VFAndOp : AVE_BinaryOp<"vand",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFOrOp : AVE_BinaryOp<"vor",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFXorOp : AVE_BinaryOp<"vxor",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFExpdifOp : AVE_BinaryOp<"vexpdif">{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFVMULLOp : AVE_VmullBinaryOp<"mull", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let extraClassDeclaration = opNameDeclaration;
}

//
//// Tenary Ops
//
def AVE_VFMulaOp : AVE_TenaryOp<"vmula">;

//
//// Vector-Scalar Ops
//
//  vadds/vmuls/vlrelu/vmaxs/vmins/vshls/vshrs/vrnds/vsadds
def AVE_VFAddsOp : AVE_VecScalarOp<"vadds", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFMulsOp : AVE_VecScalarOp<"vmuls", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFLRelusOp : AVE_VecScalarOp<"vlrelus", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFMaxsOp : AVE_VecScalarOp<"vmaxs", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VFMinsOp : AVE_VecScalarOp<"vmins", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
}

def AVE_VMaxsSIOp : AVE_VecScalarOp<"vsmaxs", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmaxs";
}

def AVE_VMinsSIOp : AVE_VecScalarOp<"vsmins", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmins";
}

def AVE_VMaxsUIOp : AVE_VecScalarOp<"vumaxs", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmaxs";
}

def AVE_VMinsUIOp : AVE_VecScalarOp<"vumins", [DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let intrinsicName = "vmins";
}

def AVE_VFShlsOp : AVE_ShiftVecScalarOp<"vshls",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let description = [{
    Shifts each element of a vector by a scalar amount to the left.
  }];
}

def AVE_VFShrsOp : AVE_ShiftVecScalarOp<"vshrs",[DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]>{
  let extraClassDeclaration = opNameDeclaration;
  let description = [{
    Shifts each element of a vector by a scalar amount to the right.
  }];
}

//
//// Preg Ops
//
def AVE_PregNotOp : AVE_PregUnaryOp<"not">;
def AVE_PregAndOp : AVE_PregBinaryOp<"and">;
def AVE_PregOrOp : AVE_PregBinaryOp<"or">;
def AVE_PregXorOp : AVE_PregBinaryOp<"xor">;

def AVE_VFSlideOp : AVE_Op<"vslide", [Pure, AllTypesMatch<["src1", "src2", "result"]>,DeclareOpInterfaceMethods<AVELibraryFunctionOpInterface,["getOpLibraryCallName"]>]> {
  let summary = "Vector slide operation";
  let description = [{
    Performs a vector slide operation by a given amount. The source vector is held
    in two consecutive vector registers. The instruction slides a
    window of the same size as the vector length over the two source vectors to
    produce the destination vector.

    The slide amount is given by a shared register (Sm) which must be in the
    range S0-S95.
  }];

  let arguments = (ins  AveVector<[ I8, I16, I32, F16, F32, I64]>:$src1,
  AveVector<[ I8, I16, I32, F16, F32, I64]>:$src2,
  I16:$slide_amount
  );
  let results = (outs AveVector<[ I8, I16, I32, F16, F32, I64]>:$result);
  let extraClassDeclaration = opNameDeclaration;
  let assemblyFormat = [{
    $src1 `,` $src2 `,` $slide_amount attr-dict `:` type($src1) `,` type($slide_amount) `->` type($result)
  }];
}

def AVE_VectorLayoutCastOp : AVE_Op<"vector.layout_cast", [Pure]> {
  let summary = "Vector layout casting operation.";
  let arguments = (ins Builtin_Vector: $src);
  let results = (outs Builtin_Vector: $res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

#endif // AVE_VF_OPS
