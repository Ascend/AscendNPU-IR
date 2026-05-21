//===- HIVMRegbaseIntrins.td - Regbase intrinsic operation -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines intrinsic operations of the HIVMRegbaseIntrins dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HIVMREGBASEINTRINS_OPS
#define HIVMREGBASEINTRINS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/IR/DialectBase.td"

def HIVMRegbaseIntrins_Dialect : Dialect {
  let name = "hivm_regbaseintrins";

  let summary = [{
    This dialect provides an representation of hivm reg-based operations.
  }];

  let cppNamespace = "::mlir::hivm_regbaseintrins";

    let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// HIVM regbase Intrinsic op definitions
//===----------------------------------------------------------------------===//
class HIVMRegbaseIntrins_Op<string mnemonic, list<int> overloadedResults = [], list<int> overloadOperands = [],
                       list<Trait> traits = [], int numResults = 1>
    : LLVM_IntrOpBase<
          /*Dialect dialect=*/HIVMRegbaseIntrins_Dialect,
          /*string opName=*/"intr." #mnemonic,
          /*string enumName=*/"HIVMIntrinsics::" # !subst(".", "_", mnemonic),
          /*list<int> overloadedResults=*/overloadedResults,
          /*list<int> overloadedOperands=*/overloadOperands,
          /*list<Trait> traits=*/traits,
          /*int numResults=*/numResults>; 

// An representation of regbased architecturte operations.
// TODO: need to be split into a dedicated file.
// TODO: consider The VL length varies depending on different platform
// for example V300(256 Bytes) vs V310(128 Bytes).

// The different available pointers
def GmPtr   : LLVM_PointerInAddressSpace<1>;
def CbufPtr : LLVM_PointerInAddressSpace<2>;
def CaPtr   : LLVM_PointerInAddressSpace<3>;
def CbPtr   : LLVM_PointerInAddressSpace<4>;
def CcPtr   : LLVM_PointerInAddressSpace<5>;
def UbufPtr : LLVM_PointerInAddressSpace<6>;
def FbufPtr : LLVM_PointerInAddressSpace<7>;

def ValidVector : Type<And<[IsVectorTypePred,
                            CPred<"::mlir::cast<::mlir::ShapedType>($_self).getRank() == 1">,
                            Or<[VectorOfLengthAndType<[64], [I32]>.predicate,
                                VectorOfLengthAndType<[64], [SI32]>.predicate,
                                VectorOfLengthAndType<[64], [UI32]>.predicate,
                                VectorOfLengthAndType<[128], [I16]>.predicate,
                                VectorOfLengthAndType<[128], [SI16]>.predicate,
                                VectorOfLengthAndType<[128], [UI16]>.predicate,
                                VectorOfLengthAndType<[256], [I8]>.predicate,
                                VectorOfLengthAndType<[256], [SI8]>.predicate,
                                VectorOfLengthAndType<[256], [UI8]>.predicate,
                                VectorOfLengthAndType<[64], [F32]>.predicate,
                                VectorOfLengthAndType<[128], [F16]>.predicate,
                                VectorOfLengthAndType<[128], [BF16]>.predicate,
                                VectorOfLengthAndType<[256], [F8E4M3FN]>.predicate,
                                VectorOfLengthAndType<[256], [F8E5M2]>.predicate,
                                ]>]>,
    "one of the following types: vector<64xi32>, vector<64xui32>, vector<64xsi32>,"
    "vector<128xi16>, vector<128xui16>, vector<128xsi16>, vector<256xi8>, vector<256xui8>, vector<256xsi8>,"
    "vector<64xf32>, vector<128xf16>, vector<128xbf16>">;

def predicateVector : Type<And<[VectorOfLengthAndType<[256], [I1]>.predicate,
                           CPred<"::mlir::cast<::mlir::ShapedType>($_self).getRank() == 1">]>,
    "vector<256xi1>">;

def AlignedVector : Type<Or<[VectorOfLengthAndType<[32], [I8]>.predicate,
                           CPred<"$_self.cast<::mlir::ShapedType>().getRank() == 1">]>,
    "vector<32xi8>">;

def LDOrSTVector : Type<And<[IsVectorTypePred,
                             CPred<"::mlir::cast<::mlir::ShapedType>($_self).getRank() == 1">,
                             Or<[VectorOfLengthAndType<[256], [I8]>.predicate,
                                 VectorOfLengthAndType<[256], [SI8]>.predicate,
                                 VectorOfLengthAndType<[256], [UI8]>.predicate,
                                 VectorOfLengthAndType<[128], [I16]>.predicate,
                                 VectorOfLengthAndType<[128], [SI16]>.predicate,
                                 VectorOfLengthAndType<[128], [UI16]>.predicate,
                                 VectorOfLengthAndType<[64], [I32]>.predicate,
                                 VectorOfLengthAndType<[64], [SI32]>.predicate,
                                 VectorOfLengthAndType<[64], [UI32]>.predicate,
                                 VectorOfLengthAndType<[32], [I64]>.predicate,
                                 VectorOfLengthAndType<[32], [SI64]>.predicate,
                                 VectorOfLengthAndType<[32], [UI64]>.predicate,
                                 VectorOfLengthAndType<[256], [F8E4M3FN]>.predicate,
                                 VectorOfLengthAndType<[256], [F8E5M2]>.predicate,
                                 VectorOfLengthAndType<[128], [BF16]>.predicate,
                                 VectorOfLengthAndType<[128], [F16]>.predicate,
                                 VectorOfLengthAndType<[64], [F32]>.predicate]>]>,
    "one of the following types: vector<256xi8>, vector<256xui8>, vector<256xsi8>, vector<128xi16>"
    "vector<128xui16>, vector<128xsi16>, vector<64xi32>, vector<64xui32>, vector<64xsi32>, vector<32xi64>,"
    "vector<32xui64>, vector<32xsi64>, vector<128xbf16>, vector<64xf32>">;

//
// Generic vector binary arithmetic op with part.
//
class ElementwiseBinaryWithPartInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src1", "src2", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src1,
                       ValidVector:$src2,
                       predicateVector:$mask,
                       I32:$part);
  let results = (outs ValidVector:$res);
}

//
// Generic vector binary arithmetic op.
//
class ElementwiseBinaryInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src1", "src2", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src1,
                       ValidVector:$src2,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

//
// Generic vector binary arithmetic op.
//
class ElementwiseBinaryInstrV2Op<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src1", "src2", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src1,
                       ValidVector:$src2,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

class ElementwiseBinaryAnyInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src1", "src2", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src1,
                       ValidVector:$src2,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

//
// Generic vector unary arithmetic op
//
class ElementwiseUnaryInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

class ElementwiseUnaryAnyInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

//
// Generic vector tenary arithmetic op
//
class ElementwiseTenaryInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src1", "src2", "src3", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src1,
                       ValidVector:$src2,
                       ValidVector:$src3,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}


//
// Generic vector scalar binary arithmetic op.
//
class BinaryVecScalarOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["vec", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$vec,
                       AnyTypeOf<[ I8, I16, I32, F16, F32 ]>:$scalar,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

class LoadX1Op<string mnemonic, int numResults> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/[],
                /*int numResults=*/numResults>{
  let arguments = (ins UbufPtr:$base,
                   I32:$offset,
                   I32:$dist,
                   I32:$postUpdateEnable
                   );
  let results = (outs LDOrSTVector:$res);
}

class LoadX1PostOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],        
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins UbufPtr:$base,
                         I32:$offset,
                         I32:$dist,
                         I32:$postUpdateEnable)>;

class LoadX2Op<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins UbufPtr:$base,
                         I32:$offset,
                         I32:$dist,
                         I32:$postUpdateEnable)>;

class LoadX2PostOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins UbufPtr:$base,
                         I32:$offset,
                         I32:$dist,
                         I32:$postUpdateEnable)>;

class LoadUnAlignedOp<string mnemonic, int numResults> :
  HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins UbufPtr:$base,
                         AlignedVector:$alignData)>;

class LoadUnAlignedPostOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins UbufPtr:$base,
                         AlignedVector:$alignData,
                         I32:$cceVL)>;

class LoadAlignedOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins UbufPtr:$base)>;

class StoreOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins LDOrSTVector:$data,
                         UbufPtr:$base,
                         I32:$offset,
                         I32:$dist,
                         I32:$postUpdateEnable,
                         predicateVector:$mask)>;

class StoreWithStrideOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins LDOrSTVector:$data,
                         UbufPtr:$base,
                         I32:$strideConfig,
                         I32:$postUpdateEnable,
                         predicateVector:$mask)>;

class StoreUnAlignedOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins LDOrSTVector:$data,
                         UbufPtr:$base,
                         I32:$eleSize,
                         AlignedVector:$tmp)>;

class StoreUnAlignedPostOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins LDOrSTVector:$data,
                         UbufPtr:$base,
                         I32:$eleSize,
                         AlignedVector:$tmp)>;

class StoreAlignedOp<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins AlignedVector:$data,
                         UbufPtr:$base,
                         I32:$offset,
                         I32:$postUpdateEnable)>;

class HIVMRegbaseInstrs_Op<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let results = (outs AlignedVector:$res);
}

class StoreX2Op<string mnemonic, int numResults>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults>,
          Arguments<(ins LDOrSTVector:$data0,
                         LDOrSTVector:$data1,
                         UbufPtr:$base,
                         I32:$offset,
                         I32:$dist,
                         I32:$postUpdateEnable,
                         predicateVector:$mask)>;

//
// Generate true predicate for certain elements
//
class PLTOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1>,
          Arguments<(ins I32:$src)>;

//
// Generate true predicate for certain elements
//
class PLTMOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1>{
  let arguments = (ins I16:$src,
                       I32:$src2);
  let results = (outs predicateVector:$res);
}

//
// Predicate register generation
//
class PGEOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins I32:$src,
                       I32:$src2);
  let results = (outs predicateVector:$res);
}

//
// Predicate register set
//
class PSETOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins I32:$src);
  let results = (outs predicateVector:$res);
}

//
// V300 ISA section 11
//
def VaddSXInstrOp : ElementwiseBinaryInstrOp<"hivm.vadd.s.x"> {
  let summary = "A V300 vector addition instruction";
  let description = [{
    Adds the elements of two shared vector registers and stores the result into
    another vector register.
    A predicate (MASK) is provided to tell which elements are added in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector addition on two <128xf16> vectors
    %0 = "hivm.intr.hivm.vadd.s.x"(%src1, %src2, %preg) : (vector<128xf16>, vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}
def VaddUXInstrOp : ElementwiseBinaryInstrOp<"hivm.vadd.u.x">;
def VaddSMInstrOp : ElementwiseBinaryInstrOp<"hivm.vadd.s.m">;
def VaddUMInstrOp  : ElementwiseBinaryInstrOp<"hivm.vadd.u.m">;

def VmulSXInstrOp : ElementwiseBinaryInstrOp<"hivm.vmul.s.x"> {
  let summary = "A V300 vector multiplication instruction";
  let description = [{
    Multiply the two vectors together and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector multiplication on two <64xui32> vectors.
    %0 = "hivm.intr.hivm.vmul.s.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}
def VmulUXInstrOp : ElementwiseBinaryInstrOp<"hivm.vmul.u.x">;
def VmulSMInstrOp : ElementwiseBinaryInstrOp<"hivm.vmul.s.m">;
def VmulUMInstrOp : ElementwiseBinaryInstrOp<"hivm.vmul.u.m">;

def VsubSXInstrOp : ElementwiseBinaryInstrOp<"hivm.vsub.s.x"> {
  let summary = "A V300 vector subtraction instruction";
  let description = [{
    Subtract the elements of src2 vector register from src1 (so result is src1 - src2)
    and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector subtraction on two <64xi32> vectors
    %0 = "hivm.intr.hivm.vsub.s.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}
def VsubUXInstrOp : ElementwiseBinaryInstrOp<"hivm.vsub.u.x">;
def VsubSMInstrOp : ElementwiseBinaryInstrOp<"hivm.vsub.s.m">;
def VsubUMInstrOp : ElementwiseBinaryInstrOp<"hivm.vsub.u.m">;

def VdivSXInstrOp : ElementwiseBinaryInstrOp<"hivm.vdiv.s.x"> {
  let summary = "A V300 vector division instruction";
  let description = [{
    Divide the first vector by the second vector and stores the result
    into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector division on two <64xi32> vectors
    %0 = "hivm.intr.hivm.vdiv.s.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ``
  }];
}
def VdivUXInstrOp : ElementwiseBinaryInstrOp<"hivm.vdiv.u.x">;
def VdivSMInstrOp : ElementwiseBinaryInstrOp<"hivm.vdiv.s.m">;
def VdivUMInstrOp : ElementwiseBinaryInstrOp<"hivm.vdiv.u.m">;

def VdivfV128U16XInstrOp : ElementwiseBinaryInstrV2Op<"hivm.vdivf.v128u16.x"> {
  let summary = "A V300 vector unsigned 16bit fixed-point division instruction";
  let description = [{
    Perform unsigned 16bit fixed-point element-wise division for every
    active elements in first vector and second vector, and write the result into
    another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector divisionF on two <128xui16> vectors
    %0 = "hivm.intr.hivm.vdivf.v128u16.x"(%src1, %src2, %preg) : (vector<128xi16>, vector<128xi16>, vector<256xi1>)-> (vector<128xi16>)
    ```
  }];
}

def VmaxSXInstrOp : ElementwiseBinaryInstrOp<"hivm.vmax.s.x"> {
  let summary = "A V300 vector instruction to find element-wise max";
  let description = [{
    Perform element-wise maximum for the two vector registers and stores the result
    into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector element-wise max on two <256xi8> vectors
    %0 = "hivm.intr.hivm.vmax.v256s8.x"(%src1, %src2, %preg) : (vector<256xi8>, vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}
def VmaxUXInstrOp : ElementwiseBinaryInstrOp<"hivm.vmax.u.x">;
def VmaxSMInstrOp : ElementwiseBinaryInstrOp<"hivm.vmax.s.m">;
def VmaxUMInstrOp : ElementwiseBinaryInstrOp<"hivm.vmax.u.m">;

def VminSXInstrOp : ElementwiseBinaryInstrOp<"hivm.vmin.s.x"> {
  let summary = "A V300 vector instruction to find element-wise min";
  let description = [{
    Perform element-wise minimum for the two vector registers and stores the result
    into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector element-wise min on two <256xi8> vectors
    %0 = "hivm.intr.hivm.vmin.s.x"(%src1, %src2, %preg) : (vector<256xi8>, vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}
def VminUXInstrOp : ElementwiseBinaryInstrOp<"hivm.vmin.u.x">;
def VminSMInstrOp : ElementwiseBinaryInstrOp<"hivm.vmin.s.m">;
def VminUMInstrOp : ElementwiseBinaryInstrOp<"hivm.vmin.u.m">;

def VabsdifSXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vabsdif.s.x"> {
  let summary = "A V300 vector difference-and-absolute instruction";
  let description = [{
    Perform element-wise difference-and-absolute for the two vector registers and stores the result
    into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector element-wise min on two <64xi32> vectors
    %0 = "hivm.intr.hivm.vabsdif.s.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}
def VabsdifUXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vabsdif.u.x">;

def VsaddV128S16XInstrOp : ElementwiseBinaryInstrV2Op<"hivm.vsadd.v128s16.x"> {
  let summary = "A V300 vector addition instruction with satutation";
  let description = [{
    Adds the elements of two shared vector registers, do saturation and stores the result into
    another vector register.
    A predicate (MASK) is provided to tell which elements are added in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector addition on two <128xi16> vectors
    %0 = "hivm.intr.hivm.vsadd.v128s16.x"(%src1, %src2, %preg) : (vector<128xi16>, vector<128xi16>, vector<256xi1>)-> (vector<128xi16>)
    ```
  }];
}

def VssubV128S16XInstrOp : ElementwiseBinaryInstrV2Op<"hivm.vssub.v128s16.x"> {
  let summary = "A V300 vector subtraction instruction with saturation";
  let description = [{
    Subtract the elements of src2 vector register from src1 (so result is src1 - src2),
    do saturation and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector subtraction on two <128xi16> vectors
    %0 = "hivm.intr.hivm.vssub.v128s16.x"(%src1, %src2, %preg) : (vector<128xi16>, vector<128xi16>, vector<256xi1>)-> (vector<128xi16>)
    ```
  }];
}

def VshlSXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vshl.s.x"> {
  let summary = "A V300 vector logical/arithmetic shift left instruction";
  let description = [{
    Perform element-wise shift left for src1 by the amount in src2
    and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    For unsigned data, it's logical shift. For signed data, it's arithmetic shift.

    src2 is always treated as signed integer regardless of its type. i.e. if elements in
    src2 is u8, then it is treated as s8. Negative shift amount means shift in the
    other direction, so a left shift with negative amount becomes a right shift

    #### Examples:

    ```mlir
    // Perform vector left shift on the <128xi16> vector src1 by the amounts in src2
    %0 = "hivm.intr.hivm.vshl.s.x"(%src1, %src2, %preg) : (vector<128xi16>, vector<128xi16>, vector<256xi1>)-> (vector<128xi16>)
    ```
  }];
}
def VshlUXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vshl.u.x">;

def VshrSXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vshr.s.x"> {
  let summary = "A V300 vector logical/arithmetic shift right instruction";
  let description = [{
    Perform element-wise shift right for src1 by the amount in src2
    and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    For unsigned data, it's logical shift. For signed data, it's arithmetic shift.

    src2 is always treated as signed integer regardless of its type. i.e. if elements in
    src2 is u8, then it is treated as s8. Negative shift amount means shift in the
    other direction, so a right shift with negative amount becomes a left shift

    #### Examples:

    ```mlir
    // Perform vector right shift on the <128xi16> vector src1 by the amounts in src2
    %0 = "hivm.intr.hivm.vshr.s.x"(%src1, %src2, %preg) : (vector<128xi16>, vector<128xi16>, vector<256xi1>)-> (vector<128xi16>)
    ```
  }];
}
def VshrUXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vshr.u.x">;

def VrndXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vrnd.x"> {
  let summary = "A V300 vector round instruction";
  let description = [{
    Perform element-wise round for src1 according to corresponding unsigned
    shift amount in src2 and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    Shift amount with magnitude greater than the element width will result in a 0.

    #### Examples:

    ```mlir
    // Perform vector round on the <64xi32> vector src1 by the shift amounts in src2
    %0 = "hivm.intr.hivm.vrnd.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}

def VpreluXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vprelu.x"> {
  let summary = "A V300 vector pReLU instruction";
  let description = [{
    Perform element-wise prelu for the two vector registers and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector prelu on two <64xf32> vector
    %0 = "hivm.intr.hivm.vprelu.x"(%src1, %src2, %preg) : (vector<64xf32>, vector<64xf32>, vector<256xi1>)-> (vector<64xf32>)
    ```
  }];
}

def VandXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vand.x"> {
  let summary = "A V300 vector logical AND instruction";
  let description = [{
    Logical AND the two vector together and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector logical AND on two <64xi32> vectors
    %0 = "hivm.intr.hivm.vand.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}

def VorXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vor.x"> {
  let summary = "A V300 vector logical OR instruction";
  let description = [{
    Logical OR the two vector together and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector logical OR on two <64xi32> vectors
    %0 = "hivm.intr.hivm.vor.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}

def VxorXInstrOp : ElementwiseBinaryAnyInstrOp<"hivm.vxor.x"> {
  let summary = "A V300 vector logical XOR instruction";
  let description = [{
    Logical XOR the two vector together and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector logical XOR on two <64xi32> vectors
    %0 = "hivm.intr.hivm.vxor.x"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}
def VselInstrOp : ElementwiseBinaryInstrOp<"hivm.vsel"> {
    // add traits for type constraint
  let summary = "HIVM vector select operation";
  let description = [{
    Select elements from two source vector according to the cmpmask vector. If the corresponding bit of cmpmask is 1, select src0.

    Common Constraints:
      1. The input vectors and output vector must have the same ranks
      2. The element type of cmpmask must be bool
      3. The element type of src0, src1 and dst must be b16 or b32
  }];
}

//
//// Unary Ops
//
def VabsXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vabs.x"> {
  let summary = "A V300 vector absolute value instruction";
  let description = [{
    Perform element-wise absoulte value for the source vector and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector absolute value on a <64xf32> vector
    %0 = "hivm.intr.hivm.vabs.x"(%src, %preg) : (vector<256xi8>, vector<256 x i1>)-> (vector<256xi8>)
    ```
  }];
}

def VNegXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vneg.x"> {
  let summary = "A V300 vector nagation instruction";
  let description = [{
    Perform element-wise nagation for the source vector (dest = -src) and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector nagation on a <256xi8> vector
    %0 = "hivm.intr.hivm.vneg.x"(%src, %preg) : (vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}

def VNotXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vnot.x"> {
  let summary = "A V300 vector logic-not instruction";
  let description = [{
    Perform element-wise logic-not for the source vector (dest = !src) and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector logic-not on a <256xi8> vector
    %0 = "hivm.intr.hivm.vnot.x"(%src, %preg) : (vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}

def VReluXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vrelu.x"> {
  let summary = "A V300 vector relu instruction";
  let description = [{
    Perform element-wise relu for the source vector (dest = (src≥0)?src:0) and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector relu on a <128xf16> vector
    %0 = "hivm.intr.hivm.vrelu.x"(%src, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}

def VExpXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vexp.x"> {
  let summary = "A V300 vector base e exponential instruction";
  let description = [{
    Perform element-wise exponential with base e for the source vector and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector base e exponential on a <128xf16> vector
    %0 = "hivm.intr.hivm.vexp.x"(%src, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}

def VExpdifInstrOp : ElementwiseBinaryWithPartInstrOp<"hivm.vexpdif"> {
  let summary = "A V300 vector base e exponential instruction with subtraction";
  let description = [{
    Perform element-wise exponential with subtraction for the source vector v1 and v2 (so result is exp(v1 - v2)).
    The result will be stored into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector base e exponential on a <64xf32> vector
    %0 = "hivm.intr.hivm.vexpdif"(%src1, %src2, %preg, &part) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32)-> (vector<64xf32>)
    ```
  }];
}

def VSqrtXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vsqrt.x"> {
  let summary = "A V300 vector square root instruction";
  let description = [{
    Perform element-wise square root for the source vector and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector square root on a <128xf16> vector
    %0 = "hivm.intr.hivm.vsqrt.x"(%src, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}

def VRsqrtXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vrsqrt.x"> {
  let summary = "A V300 vector reciprocal square root instruction";
  let description = [{
    Perform element-wise reciprocal square root for the source vector and
    stores the result into another vector register. (dest = 1 / sqrt(src))
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector reciprocal square root on a <128xf16> vector
    %0 = "hivm.intr.hivm.vrsqrt.x"(%src, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}

def VRecXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vrec.x"> {
  let summary = "A V300 vector reciprocal estimation instruction";
  let description = [{
    Perform element-wise reciprocal estimation for the source vector and
    stores the result into another vector register. (dest = 1 / src)
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector reciprocal estimation on a <128xf16> vector
    %0 = "hivm.intr.hivm.vrec.x"(%src, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}

def VLnXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vln.x"> {
  let summary = "A V300 vector base e logarithmic instruction";
  let description = [{
    Perform element-wise logarithm with base e for the source vector and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Perform vector base e logarithm on a <128xf16> vector
    %0 = "hivm.intr.hivm.vln.x"(%src, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}

def VBcntXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vbcnt.x"> {
  let summary = "A V300 vector count set bits instruction";
  let description = [{
    Counts the set bits (bits of 1) of elements in the input vector and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    #### Examples:

    ```mlir
    // Count set bits of elements of a <256xi8> vector
    %0 = "hivm.intr.hivm.vbcnt.x"(%src, %preg) : (vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}

def VClsSXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vcls.s.x"> {
  let summary = "A V300 vector count leading sign bits instruction";
  let description = [{
    Counts the leading zero or sign bits of the elements in input vector and
    stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    For signed types, leading sign bits (0 or 1) are counted. For unsigned types
    leading zero is counted.

    #### Examples:

    ```mlir
    // Count leading zero of a <256xi8> vector
    %0 = "hivm.intr.hivm.vcls.s.x"(%src, %preg) : (vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}
def VClsUXInstrOp : ElementwiseUnaryAnyInstrOp<"hivm.vcls.u.x">;

def VunsupportedBinaryInstrOp : ElementwiseBinaryInstrV2Op<"hivm.binaryop.unsupported">;
def VunsupportedUnaryInstrOp : ElementwiseUnaryInstrOp<"hivm.unaryop.unsupported">;
def VunsupportedTenaryInstrOp : ElementwiseTenaryInstrOp<"hivm.tenaryop.unsupported">;

//
//// Tenary Ops
//
def VmulaSMInstrOp : ElementwiseTenaryInstrOp<"hivm.vmula.s.m"> {
  let summary = "A V300 vector multiply-and-addition instruction";
  let description = [{
    Multiply the two input vectors and add the dst vector,
    stores the result into dst vector register(dst = src1 * src2 + dst).
    A predicate (MASK) is provided to tell which elements are involved in the three vectors.

    #### Examples:

    ```mlir
    // Perform vector multiplication and addition on three <128xi16> vectors
    %0 = "hivm.intr.hivm.vmula.s.m"(%dst, %src1, %src2, %preg) : (vector<128xi16>, vector<128xi16>, vector<128xi16>, vector<256xi1>)-> (vector<128xi16>)
    ```
  }];
}
def VmulaUMInstrOp : ElementwiseTenaryInstrOp<"hivm.vmula.u.m">;

def Vadd3SInstrOp : ElementwiseTenaryInstrOp<"hivm.vadd3.s.m"> {
  let summary = "A V300 vector addition with 3 source operands instruction";
  let description = [{
    Add the three vectors together and stores the result into dst vector register(dst = dst + src1 + src2).
    A predicate (MASK) is provided to tell which elements are involved in the three vectors.

    #### Examples:

    ```mlir
    // Perform vector addition on three <64xi32> vectors
    %0 = "hivm.intr.hivm.vadd3.s.m"(%dst, %src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}
def Vadd3UInstrOp : ElementwiseTenaryInstrOp<"hivm.vadd3.u.m">;

def VadifSInstrOp : ElementwiseTenaryInstrOp<"hivm.vadif.s.m"> {
  let summary = "A V300 vector addition and difference instruction";
  let description = [{
    Vectore addition and difference with 3 source operands(dst = src2 - src3 + dst)
    A predicate (MASK) is provided to tell which elements are involved in the three vectors.

    #### Examples:

    ```mlir
    // Perform vector addition and difference on three <64xi32> vectors
    %0 = "hivm.intr.hivm.vadif.s.m"(%dst, %src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}
def VadifUInstrOp : ElementwiseTenaryInstrOp<"hivm.vadif.u.m">;

def VsadSInstrOp : ElementwiseTenaryInstrOp<"hivm.vsad.s.m"> {
  let summary = "A V300 vector summation of absolute difference instruction";
  let description = [{
    Vectore summation of absolute difference with 3 source operands(dst = |src1 - src2| + dst)
    A predicate (MASK) is provided to tell which elements are involved in the three vectors.

    #### Examples:

    ```mlir
    // Perform vector summation of absolute difference on three <64xi32> vectors
    %0 = "hivm.intr.hivm.vsad.s.m"(%dst, %src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<256xi1>)-> (vector<64xi32>)
    ```
  }];
}
def VsadUInstrOp : ElementwiseTenaryInstrOp<"hivm.vsad.u.m">;

//
/// VLDSx1, Pre mode
//
def Vldsx1V32U64InstrOp : LoadX1Op<"hivm.vldsx1.v32u64", 1> {
  let summary = "A V300 vlds instruction";
  let description = [{
    Loads a shared vector register with the data from a given ubuf pointer.

    The offset is the BYTE aligned offset into the pointer to begin moving the memory into
    the shared vector register.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    post update mode is 0

    dist decides the data distribution and broadcasting, takes value from an enum

    returns one destination vector.

    Examples:

    ```mlir
    // Load the shared vector registers
    %0 = "hivm.intr.hivm.vldsx1.v32u64"(%base, %offset, %dis, %postUpdateEnable) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
    ```
  }];
}
def Vldsx1V32S64InstrOp : LoadX1Op<"hivm.vldsx1.v32s64", 1>;
def Vldsx1V64S32InstrOp : LoadX1Op<"hivm.vldsx1.v64s32", 1>;
def Vldsx1V64U32InstrOp : LoadX1Op<"hivm.vldsx1.v64u32", 1>;
def Vldsx1V128S16InstrOp : LoadX1Op<"hivm.vldsx1.v128s16", 1>;
def Vldsx1V128U16InstrOp : LoadX1Op<"hivm.vldsx1.v128u16", 1>;
def Vldsx1V256S8InstrOp : LoadX1Op<"hivm.vldsx1.v256s8", 1>;
def Vldsx1V256U8InstrOp : LoadX1Op<"hivm.vldsx1.v256u8", 1>;
def Vldsx1V64F32InstrOp : LoadX1Op<"hivm.vldsx1.v64f32", 1>;
def Vldsx1V128F16InstrOp : LoadX1Op<"hivm.vldsx1.v128f16", 1>;
def Vldsx1V128BF16InstrOp : LoadX1Op<"hivm.vldsx1.v128bf16", 1>;
def Vldsx1V256F8E4InstrOp : LoadX1Op<"hivm.vldsx1.v256f8e4m3", 1>;
def Vldsx1V256F8E5InstrOp : LoadX1Op<"hivm.vldsx1.v256f8e5m2", 1>;

// VLDSx2, Pre mode
def Vldsx2V32U64InstrOp : LoadX2Op<"hivm.vldsx2.v32u64", 2> {
  let summary = "A V300 vlds instruction";
  let description = [{
    Loads a shared vector register with the data from a given ubuf pointer.

    The offset is the BYTE aligned offset into the pointer to begin moving the memory into
    the shared vector register.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    post update mode is 0

    dist decides the data distribution and broadcasting, takes value from an enum

    return two destination vectors

    Examples:

    ```mlir
    // Load the shared vector registers
    %0 = "hivm.intr.hivm.vldsx2.v32u64"(%base, %offset, %dis, %postUpdateEnable) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<32xi64>, vector<32xi64>)>
    ```
  }];
}
def Vldsx2V32S64InstrOp : LoadX2Op<"hivm.vldsx2.v32s64", 2>;
def Vldsx2V64S32InstrOp : LoadX2Op<"hivm.vldsx2.v64s32", 2>;
def Vldsx2V64U32InstrOp : LoadX2Op<"hivm.vldsx2.v64u32", 2>;
def Vldsx2V128S16InstrOp : LoadX2Op<"hivm.vldsx2.v128s16", 2>;
def Vldsx2V128U16InstrOp : LoadX2Op<"hivm.vldsx2.v128u16", 2>;
def Vldsx2V256S8InstrOp : LoadX2Op<"hivm.vldsx2.v256s8", 2>;
def Vldsx2V256U8InstrOp : LoadX2Op<"hivm.vldsx2.v256u8", 2>;
def Vldsx2V64F32InstrOp : LoadX2Op<"hivm.vldsx2.v64f32", 2>;
def Vldsx2V128F16InstrOp : LoadX2Op<"hivm.vldsx2.v128f16", 2>;
def Vldsx2V128BF16InstrOp : LoadX2Op<"hivm.vldsx2.v128bf16", 2>;

// VLDSx1, Post mode
def Vldsx1PostV32U64InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v32u64", 2> {
  let summary = "A V300 vlds instruction";
  let description = [{
    Loads a shared vector register with the data from a given ubuf pointer.

    The offset is the BYTE aligned offset into the pointer to begin moving the memory into
    the shared vector register.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    post update mode is 1

    dist decides the data distribution and broadcasting, takes value from an enum

    returns two destination vectors

    Examples:

    ```mlir
    // Load the shared vector registers
    %0 = "hivm.intr.hivm.vldsx1.post.v32u64"(%base, %offset, %dis, %postUpdateEnable) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<64xi64>, !llvm.ptr)>
    ```
  }];
}
def Vldsx1PostV32S64InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v32s64", 2>;
def Vldsx1PostV64S32InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v64s32", 2>;
def Vldsx1PostV64U32InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v64u32", 2>;
def Vldsx1PostV128S16InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v128s16", 2>;
def Vldsx1PostV128U16InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v128u16", 2>;
def Vldsx1PostV256S8InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v256s8", 2>;
def Vldsx1PostV256U8InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v256u8", 2>;
def Vldsx1PostV64F32InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v64f32", 2>;
def Vldsx1PostV128F16InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v128f16", 2>;
def Vldsx1PostV128BF16InstrOp : LoadX1PostOp<"hivm.vldsx1.post.v128bf16", 2>;

// VLDSx2, Post mode
def Vldsx2PostV32U64InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v32u64", 3> {
  let summary = "A V300 vlds instruction";
  let description = [{
    Loads a shared vector register with the data from a given ubuf pointer.

    The offset is the BYTE aligned offset into the pointer to begin moving the memory into
    the shared vector register.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    post update mode is 0

    dist decides the data distribution and broadcasting, takes value from an enum

    return three destination vectors

    Examples:

    ```mlir
    // Load the shared vector registers
    %0 = "hivm.intr.hivm.vldsx2.post.v32u64"(%base, %offset, %dis, %postUpdateEnable) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<32xi64>, vector<32xi64>, !llvm.ptr)>
    ```
  }];
}
def Vldsx2PostV32S64InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v32s64", 3>;
def Vldsx2PostV64S32InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v64s32", 3>;
def Vldsx2PostV64U32InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v64u32", 3>;
def Vldsx2PostV128S16InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v128s16", 3>;
def Vldsx2PostV128U16InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v128u16", 3>;
def Vldsx2PostV256S8InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v256s8", 3>;
def Vldsx2PostV256U8InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v256u8", 3>;
def Vldsx2PostV64F32InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v64f32", 3>;
def Vldsx2PostV128F16InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v128f16", 3>;
def Vldsx2PostV128BF16InstrOp : LoadX2PostOp<"hivm.vldsx2.post.v128bf16", 3>;

// VLDUS, Pre mode
def VldusS8InstrOp : LoadUnAlignedOp<"hivm.vldus.s8", 1> {
  let summary = "A V300 vldus instruction";
  let description = [{
    Loads a shared vector register with the data from a given ubuf pointer.

    returns three destination vector.

    Examples:

    ```mlir
    // Load the shared vector registers
    %0 = "hivm.intr.hivm.vldus.s8"(%base, %alignData) : (!llvm.ptr<6>, vector<32xi8>) -> !llvm.struct<(vector<256xi8>, vector<32xi8>, !llvm.ptr)>
    ```
  }];
}
def VldusU8InstrOp : LoadUnAlignedOp<"hivm.vldus.u8", 1>;
def VldusS16InstrOp : LoadUnAlignedOp<"hivm.vldus.s16", 1>;
def VldusU16InstrOp : LoadUnAlignedOp<"hivm.vldus.u16", 1>;
def VldusS32InstrOp : LoadUnAlignedOp<"hivm.vldus.s32", 1>;
def VldusU32InstrOp : LoadUnAlignedOp<"hivm.vldus.u32", 1>;
def VldusS64InstrOp : LoadUnAlignedOp<"hivm.vldus.s64", 1>;
def VldusF8E4InstrOp : LoadUnAlignedOp<"hivm.vldus.f8e4m3", 1>;
def VldusF8E5InstrOp : LoadUnAlignedOp<"hivm.vldus.f8e5m2", 1>;
def VldusBF16InstrOp : LoadUnAlignedOp<"hivm.vldus.bf16", 1>;
def VldusF16InstrOp : LoadUnAlignedOp<"hivm.vldus.f16", 1>;
def VldusF32InstrOp : LoadUnAlignedOp<"hivm.vldus.f32", 1>;

// VLDUS, Post mode
def VldusPostS8InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.s8", 1> {
  let summary = "A V300 vldus instruction";
  let description = [{
    Loads a shared vector register with the data from a given ubuf pointer.

    returns three destination vectors

    Examples:

    ```mlir
    // Load the shared vector registers
    %0 = "hivm.intr.hivm.vldus.post.s8"(%base, %alignData, %cceVL) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<256xi8>, vector<32xi8>, !llvm.ptr)>
    ```
  }];
}
def VldusPostU8InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.u8", 1>;
def VldusPostS16InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.s16", 1>;
def VldusPostU16InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.u16", 1>;
def VldusPostS32InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.s32", 1>;
def VldusPostU32InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.u32", 1>;
def VldusPostS64InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.s64", 1>;
def VldusPostF8E4InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.f8e4m3", 1>;
def VldusPostF8E5InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.f8e5m2", 1>;
def VldusPostBF16InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.bf16", 1>;
def VldusPostF16InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.f16", 1>;
def VldusPostF32InstrOp : LoadUnAlignedPostOp<"hivm.vldus.post.f32", 1>;

// VLDAS
def VldasInstrOp : LoadAlignedOp<"hivm.vldas", 1> {
  let summary = "A V300 vldas instruction";
  let description = [{
    Loads a shared vector register with the data from a given ubuf pointer.

    returns one destination vector.

    Examples:

    ```mlir
    // Load the shared vector registers
    %0 = "hivm.intr.hivm.vldas"(%base) : (!llvm.ptr<6>) -> (vector<32xi8>)
    ```
  }];
}

// VSTSx1, Pre mode
def Vstsx1V32U64InstrOp : StoreOp<"hivm.vstsx1.v32u64", 0> {
  let summary = "A V300 vsts instruction";
  let description = [{
    Stores the value of a shared vector register into memory defined by a ubuf
    pointer.

    Offset is the BYTE aligned offset into the destination ubuf pointer.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    postUpdateEnable is 0

    dist decides the data distribution and broadcasting, takes value from an enum

    predicateVector provides a mask

    Examples:

    ```mlir
    // Store the result from the shared vector register back into ubuf
    "hivm.intr.hivm.vstsx1.v32u64"(%data, %base, %offset, %dis, %postUpdateEnable, %mask) : (vector<32xi64>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    ```
  }];
}
def Vstsx1V32S64InstrOp : StoreOp<"hivm.vstsx1.v32s64", 0>;
def Vstsx1V64S32InstrOp : StoreOp<"hivm.vstsx1.v64s32", 0>;
def Vstsx1V64U32InstrOp : StoreOp<"hivm.vstsx1.v64u32", 0>;
def Vstsx1V128S16InstrOp : StoreOp<"hivm.vstsx1.v128s16", 0>;
def Vstsx1V128U16InstrOp : StoreOp<"hivm.vstsx1.v128u16", 0>;
def Vstsx1V256S8InstrOp : StoreOp<"hivm.vstsx1.v256s8", 0>;
def Vstsx1V256U8InstrOp : StoreOp<"hivm.vstsx1.v256u8", 0>;
def Vstsx1V64F32InstrOp : StoreOp<"hivm.vstsx1.v64f32", 0>;
def Vstsx1V128F16InstrOp : StoreOp<"hivm.vstsx1.v128f16", 0>;
def Vstsx1V128BF16InstrOp : StoreOp<"hivm.vstsx1.v128bf16", 0>;
def Vstsx1V256F8E4InstrOp : StoreOp<"hivm.vstsx1.v256f8e4m3", 0>;
def Vstsx1V256F8E5InstrOp : StoreOp<"hivm.vstsx1.v256f8e5m2", 0>;

// VSTSx1, Post mode
def Vstsx1PostV32U64InstrOp : StoreOp<"hivm.vstsx1.post.v32u64", 1> {
  let summary = "A V300 vsts instruction";
  let description = [{
    Stores the value of a shared vector register into memory defined by a ubuf
    pointer.

    Offset is the BYTE aligned offset into the destination ubuf pointer.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    postUpdateEnable is 1

    dist decides the data distribution and broadcasting, takes value from an enum

    predicateVector provides a mask

    Examples:

    ```mlir
    // Store the result from the shared vector register back into ubuf
    %0 = "hivm.intr.hivm.vstsx1.post.v32u64"(%data, %base, %offset, %dis, %postUpdateEnable, %mask) : (vector<32xi64>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> (!llvm.ptr<6>)
    ```
  }];
}
def Vstsx1PostV32S64InstrOp : StoreOp<"hivm.vstsx1.post.v32s64", 1>;
def Vstsx1PostV64S32InstrOp : StoreOp<"hivm.vstsx1.post.v64s32", 1>;
def Vstsx1PostV64U32InstrOp : StoreOp<"hivm.vstsx1.post.v64u32", 1>;
def Vstsx1PostV128S16InstrOp : StoreOp<"hivm.vstsx1.post.v128s16", 1>;
def Vstsx1PostV128U16InstrOp : StoreOp<"hivm.vstsx1.post.v128u16", 1>;
def Vstsx1PostV256S8InstrOp : StoreOp<"hivm.vstsx1.post.v256s8", 1>;
def Vstsx1PostV256U8InstrOp : StoreOp<"hivm.vstsx1.post.v256u8", 1>;
def Vstsx1PostV64F32InstrOp : StoreOp<"hivm.vstsx1.post.v64f32", 1>;
def Vstsx1PostV128F16InstrOp : StoreOp<"hivm.vstsx1.post.v128f16", 1>;
def Vstsx1PostV128BF16InstrOp : StoreOp<"hivm.vstsx1.post.v128bf16", 1>;

// VSSTB, Pre mode
def VsstbV32U64InstrOp : StoreWithStrideOp<"hivm.vsstb.v32u64", 0>;
def VsstbV32S64InstrOp : StoreWithStrideOp<"hivm.vsstb.v32s64", 0>;
def VsstbV64S32InstrOp : StoreWithStrideOp<"hivm.vsstb.v64s32", 0>;
def VsstbV64U32InstrOp : StoreWithStrideOp<"hivm.vsstb.v64u32", 0>;
def VsstbV128S16InstrOp : StoreWithStrideOp<"hivm.vsstb.v128s16", 0>;
def VsstbV128U16InstrOp : StoreWithStrideOp<"hivm.vsstb.v128u16", 0>;
def VsstbV256S8InstrOp : StoreWithStrideOp<"hivm.vsstb.v256s8", 0>;
def VsstbV256U8InstrOp : StoreWithStrideOp<"hivm.vsstb.v256u8", 0>;
def VsstbV64F32InstrOp : StoreWithStrideOp<"hivm.vsstb.v64f32", 0>;
def VsstbV128F16InstrOp : StoreWithStrideOp<"hivm.vsstb.v128f16", 0>;
def VsstbV128BF16InstrOp : StoreWithStrideOp<"hivm.vsstb.v128bf16", 0>;
def Vsstb1V256F8E4InstrOp : StoreWithStrideOp<"hivm.vsstb.v256f8e4m3", 0>;
def Vsstb1V256F8E5InstrOp : StoreWithStrideOp<"hivm.vsstb.v256f8e5m2", 0>;

// VSTSx2, Pre mode
def Vstsx2V256B8InstrOp : StoreX2Op<"hivm.vstsx2.v256b8", 0> {
  let summary = "A V300 vsts instruction";
  let description = [{
    Stores the value of a shared vector register into memory defined by a ubuf
    pointer.

    Offset is the BYTE aligned offset into the destination ubuf pointer.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    postUpdateEnable is 0

    dist decides the data distribution and broadcasting, takes value from an enum

    predicateVector provides a mask

    Examples:

    ```mlir
    // Store the result from the shared vector register back into ubuf
    "hivm.intr.hivm.vstsx2.v256b8"(%data0, %data1, %base, %offset, %dis, %postUpdateEnable, %mask) : (vector<256xi8>, vector<256xi8>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    ```
  }];
}
def Vstsx2V128B16InstrOp : StoreX2Op<"hivm.vstsx2.v128b16", 0>;
def Vstsx2V64B32InstrOp : StoreX2Op<"hivm.vstsx2.v64b32", 0>;

// VSTUS, Pre mode
def VstusS8InstrOp : StoreUnAlignedOp<"hivm.vstus.s8", 1> {
  let summary = "A V300 vstus instruction";
  let description = [{
    Stores the value of a shared vector register into memory defined by a ubuf pointer.

    returns one destination vector.

    Examples:

    ```mlir
    // Store the result from the shared vector register back into ubuf
    %0 = "hivm.intr.hivm.vstus.s8"(%data, %base, %eleSize, %tmp) : (vector<256xi8>, !llvm.ptr<6>, i32, vector<32i8>) -> (vector<32i8>)
    ```
  }];
}
def VstusU8InstrOp : StoreUnAlignedOp<"hivm.vstus.u8", 1>;
def VstusS16InstrOp : StoreUnAlignedOp<"hivm.vstus.s16", 1>;
def VstusU16InstrOp : StoreUnAlignedOp<"hivm.vstus.u16", 1>;
def VstusS32InstrOp : StoreUnAlignedOp<"hivm.vstus.s32", 1>;
def VstusS64InstrOp : StoreUnAlignedOp<"hivm.vstus.s64", 1>;
def VstusBF16InstrOp : StoreUnAlignedOp<"hivm.vstus.bf16", 1>;
def VstusF16InstrOp : StoreUnAlignedOp<"hivm.vstus.f16", 1>;
def VstusU32InstrOp : StoreUnAlignedOp<"hivm.vstus.u32", 1>;
def VstusF32InstrOp : StoreUnAlignedOp<"hivm.vstus.f32", 1>;


// VSTUS Post mode
def VstusPostS8InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.s8", 1> {
  let summary = "A V300 vstus instruction";
  let description = [{
    Stores the value of a shared vector register into memory defined by a ubuf pointer.

    returns two destination vector.

    Examples:

    ```mlir
    // Store the result from the shared vector register back into ubuf
    %0 = "hivm.intr.hivm.vstus.post.s8"(%data, %base, %eleSize, %tmp) : (vector<256xi8>, !llvm.ptr<6>, i32, vector<32i8>) -> !llvm.struct<(vector<32i8>, !llvm.ptr<6>)>
    ```
  }];
}
def VstusPostU8InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.u8", 1>;
def VstusPostS16InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.s16", 1>;
def VstusPostU16InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.u16", 1>;
def VstusPostS32InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.s32", 1>;
def VstusPostU32InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.u32", 1>;
def VstusPostS64InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.s64", 1>;
def VstusPostBF16InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.bf16", 1>;
def VstusPostF16InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.f16", 1>;
def VstusPostF32InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.f32", 1>;
def VstusPostF8E4InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.f8e4m3", 1>;
def VstusPostF8E5InstrOp : StoreUnAlignedPostOp<"hivm.vstus.post.f8e5m2", 1>;

def InitVectorAlignDataInstrOp : HIVMRegbaseInstrs_Op<"hivm.init.vector.align.data">;

// VSTAS, Pre mode
def VstasInstrOp : StoreAlignedOp<"hivm.vstas", 0> {
  let summary = "A V300 vstas instruction";
  let description = [{
    Stores the value of a shared vector register into memory defined by a ubuf pointer.

    Offset is the BYTE aligned offset into the destination ubuf pointer.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    postUpdateEnable is 0

    Examples:

    ```mlir
    // Store the result from the shared vector register back into ubuf
    "hivm.intr.hivm.vstas"(%data, %base, %offset, %postUpdateEnable) : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()
    ```
  }];
}

// VSTAS, Post mode
def VstasPostInstrOp : StoreAlignedOp<"hivm.vstas.post", 1> {
  let summary = "A V300 vstas instruction";
  let description = [{
    Stores the value of a shared vector register into memory defined by a ubuf pointer.

    Offset is the BYTE aligned offset into the destination ubuf pointer.

    postUpdateEnable decided if baseAddress should be modified to baseAddress + offset after each store
    postUpdateEnable is 1

    Examples:

    ```mlir
    // Store the result from the shared vector register back into ubuf
    %0 = "hivm.intr.hivm.vstas.post"(%data, %base, %offset, %postUpdateEnable) : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> (!llvm.ptr<6>)
    ```
  }];
}

def PltB8 : PLTOp<"hivm.plt.b8.v300"> {
  let summary = "A V300 predicate (MASK) flexible set instruction";
  let description = [{
    Vector operations in V300 require a mask which decides which elements of the
    vector will be operated on. This operation generates this mask with
    the given number of elements set as true bits (instead of using fixed patterns).

    The numElements argument decides how many elements are set to true.

    Examples:

    ```mlir
    // Get the predicate (MASK) with the first 18 elements set to true
    %cst = arith.constant 18 : i32
    %0:2 = "hivm.intr.hivm.plt.b8.v300"(%cst) : (i32) -> (vector<256 x i1>, i32)
    ```
  }];
}
def PltB16 : PLTOp<"hivm.plt.b16.v300">;
def PltB32 : PLTOp<"hivm.plt.b32.v300">;

def PltMB8 : PLTMOp<"hivm.pltm.b8.v300"> {
  let summary = "A V300 predicate (MASK) flexible set instruction";
  let description = [{
    Vector operations in V300 require a mask which decides which elements of the
    vector will be operated on. This operation generates this mask with
    the given number of elements set as true bits (instead of using fixed patterns).

    The numElements argument decides how many elements are set to true.

    Examples:

    ```mlir
    // Get the predicate (MASK) with the first 18 elements set to true
    %cst = arith.constant 18 : i32
    %0:2 = "hivm.intr.hivm.pltm.b8.v300"(%cst) : (i32) -> (vector<256 x i1>, i32)
    ```
  }];
}
def PltMB16 : PLTMOp<"hivm.pltm.b16.v300">;
def PltMB32 : PLTMOp<"hivm.pltm.b32.v300">;

def PgeB8 : PGEOp<"hivm.pge.b8"> {
  let summary = "A V300 predicate (MASK) generation instruction";
  let description = [{
    Vector operations in V300 require a mask which decides which elements of the
    vector will be operated on. This operation generates this mask.

    The dest predicate register name is a unique id for a predicate register.

    The imm pattern can be one of the following values:
    * ALL: All elements are set to true
    * VL1: The lowest element
    * VL2: The lowest 2 elements
    * VL3: The lowest 3 elements
    * VL4: The lowest 4 elements
    * VL8: The lowest 8 elements
    * VL16: The lowest 16 elements
    * VL32: The lowest 32 elements
    * VL64: The lowest 64 elements
    * VL128: The lowest 128 elements
    * M3: Multiples of 3
    * M4: Multiples of 4
    * H  The lowest half elements
    * Q:  The lowest quarter elements
    * ALLF: All elements set to false

    Examples:

    ```mlir
    // Get the predicate (MASK)
    %0 = "hivm.intr.hivm.pge.b8"(%scalar, ALL) : (i32, i32) -> (vector<256 x i1>)
    ```
  }];
}
def PgeB16 : PGEOp<"hivm.pge.b16">;
def PgeB32 : PGEOp<"hivm.pge.b32">;

def PsetB8 : PSETOp<"hivm.pset.b8"> {
  let summary = "A V300 predicate (MASK) set instruction";
  let description = [{
    Vector operations in V300 require a mask which decides which elements of the
    vector will be operated on. This operation generates this mask.

    The imm pattern can be one of the following values:
    * ALL: All elements are set to true
    * VL1: The lowest element
    * VL2: The lowest 2 elements
    * VL3: The lowest 3 elements
    * VL4: The lowest 4 elements
    * VL8: The lowest 8 elements
    * VL16: The lowest 16 elements
    * VL32: The lowest 32 elements
    * VL64: The lowest 64 elements
    * VL128: The lowest 128 elements
    * M3: Multiples of 3
    * M4: Multiples of 4
    * H  The lowest half elements
    * Q:  The lowest quarter elements
    * ALLF: All elements set to false

    Examples:

    ```mlir
    // Get the predicate (MASK)
    %0 = "hivm.intr.hivm.pset.b8"(ALL) : (i32) -> (vector<256 x i1>)
    ```
  }];
}
def PsetB16 : PSETOp<"hivm.pset.b16">;
def PsetB32 : PSETOp<"hivm.pset.b32">;

//
// Reduction ops
//
class ReductionInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

class ReductionAnyInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[0],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res);
}

//
// V300 ISA section 12
//
def VcaddSXInstrOp : ReductionAnyInstrOp<"hivm.vcadd.s.x"> {
  let summary = "A V300 vector addition reduction instruction";
  let description = [{
    Add all active elements together and write the result to the lowest element
    of the destination vector register.Other element positions are filled with all 0s.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector addition reduction on <128xf16> vectors
    %0 = "hivm.intr.hivm.vcadd.s.x"(%src1, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}
def VcaddUXInstrOp : ReductionAnyInstrOp<"hivm.vcadd.u.x">;

def VcmaxSXInstrOp : ReductionInstrOp<"hivm.vcmax.s.x"> {
  let summary = "A V300 vector maximum reduction instruction";
  let description = [{
    Find the maximum element in the source vector and write the result to the
    lowest element of the destination vector register.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector maximum reduction on <128xf16> vectors
    %0 = "hivm.intr.hivm.vcmax.s.x"(%src1, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}
def VcmaxUXInstrOp : ReductionInstrOp<"hivm.vcmax.u.x">;
def VcmaxSMInstrOp : ReductionInstrOp<"hivm.vcmax.s.m">;
def VcmaxUMInstrOp : ReductionInstrOp<"hivm.vcmax.u.m">;

def VcminSXInstrOp : ReductionInstrOp<"hivm.vcmin.s.x"> {
  let summary = "A V300 vector minimum reduction instruction";
  let description = [{
    Find the minimum element in the source vector and write the result to the
    lowest element of the destination vector register.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector minimum reduction on <128xf16> vectors
    %0 = "hivm.intr.hivm.vcmin.s.x"(%src1, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}
def VcminUXInstrOp : ReductionInstrOp<"hivm.vcmin.u.x">;
def VcminSMInstrOp : ReductionInstrOp<"hivm.vcmin.s.m">;
def VcminUMInstrOp : ReductionInstrOp<"hivm.vcmin.u.m">;

def VcgaddUXInstrOp : ReductionInstrOp<"hivm.vcgadd.u.x"> {
  let summary = "A V300 vector addition group reduction instruction";
  let description = [{
    Add all active elements in each 32B block together and write the result
    continuously to the LSBs of the destination vector register, other element
    positions are filled with all 0s.Inactive elements are treated as 0.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector addition group reduction on <256xi8> vectors
    %0 = "hivm.intr.hivm.vcgadd.u.x"(%src1, %preg) : (vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}
def VcgaddSXInstrOp : ReductionInstrOp<"hivm.vcgadd.s.x">;

def VcgmaxUInstrOp : ReductionInstrOp<"hivm.vcgmax.u.x"> {
  let summary = "A V300 vector maximum group reduction instruction";
  let description = [{
    Find all active elements in each 32B block of the source vector and write
    the result continuously to the LSBs of the destination vector register.
    other element positions are filled with all 0s.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector maximum reduction on <256xi8> vectors
    %0 = "hivm.intr.hivm.vcgmax.u.x"(%src1, %preg) : (vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}
def VcgmaxSInstrOp : ReductionInstrOp<"hivm.vcgmax.s.x">;

def VcgminUInstrOp : ReductionInstrOp<"hivm.vcgmin.u.x"> {
  let summary = "A V300 vector minimum group reduction instruction";
  let description = [{
    Find the minimum element in each block of the source vector and write the
    result continuously to the LSBs of the destination vector register.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector minimum reduction on <256xi8> vectors
    %0 = "hivm.intr.hivm.vcgmin.u.x"(%src1, %preg) : (vector<256xi8>, vector<256xi1>)-> (vector<256xi8>)
    ```
  }];
}
def VcgminSInstrOp : ReductionInstrOp<"hivm.vcgmin.s.x">;

def VcpaddInstrOp : ReductionInstrOp<"hivm.vcpadd.x"> {
  let summary = "A V300 vector addition pair-reduction instruction";
  let description = [{
    Add every 2 adjacent elements of the source vector register and write the
    result to the lower half of the destination vector register. Other element
    positions are filled with all 0s.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector addition pair-reduction on <128xf16> vectors
    %0 = "hivm.intr.hivm.vcpadd.x"(%src1, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>)
    ```
  }];
}

class ReductionVcbInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/2> {
  let arguments = (ins ValidVector:$src,
                       predicateVector:$mask);
  let results = (outs ValidVector:$res,
                      predicateVector:$res1);
}

def VcbmaxSXInstrOp : ReductionVcbInstrOp<"hivm.vcbmax.s.x"> {
  let summary = "A V300 vector maximum reduction instruction, with output predicate"
                "register matching maximum locations";
  let description = [{
    Find the maximum element in the source vector and write the result to the
    lowest element of the destination vector register.Other element positions
    are filled with all 0s, and generate predication for each matching maximum
    element locations.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector maximum reduction on <128xf16> vectors
    %0:2 = "hivm.intr.hivm.vcbmax.s.x"(%src1, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>, vector<256xi1>)
    ```
  }];
}

def VcbmaxUXInstrOp : ReductionVcbInstrOp<"hivm.vcbmax.u.x">;
def VcbmaxSMInstrOp : ReductionVcbInstrOp<"hivm.vcbmax.s.m">;
def VcbmaxUMInstrOp : ReductionVcbInstrOp<"hivm.vcbmax.u.m">;

def VcbminSXInstrOp : ReductionVcbInstrOp<"hivm.vcbmin.s.x"> {
  let summary = "A V300 vector minimum reduction instruction, with output predicate"
                "register matching maximum locations";
  let description = [{
    Find the minimum element in the source vector and write the result to the
    lowest element of the destination vector register.Other element positions
    are filled with all 0s, and generate predication for each matching maximum
    element locations.
    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    #### Examples:

    ```mlir
    // Perform vector minimum reduction on <128xf16> vectors
    %0:2 = "hivm.intr.hivm.vcbmin.s.x"(%src1, %preg) : (vector<128xf16>, vector<256xi1>)-> (vector<128xf16>, vector<256xi1>)
    ```
  }];
}
def VcbminUXInstrOp : ReductionVcbInstrOp<"hivm.vcbmin.u.x">;
def VcbminSMInstrOp : ReductionVcbInstrOp<"hivm.vcbmin.s.m">;
def VcbminUMInstrOp : ReductionVcbInstrOp<"hivm.vcbmin.u.m">;

//
// broadcast op
//
class BroadCastZmodeInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128,  128, 128, 128,  256, 256, 256,    256,      64,  64,  64],
                                             [BF16, F16, I16, UI16, I8,  UI8, F8E5M2, F8E4M3FN, F32, I32, UI32]>:$src,
                       VectorOfLengthAndType<[256], [I1]>:$mask,
                       I32:$predicateMode);
  let results = (outs VectorOfLengthAndType<[128,  128, 128, 128,  256, 256, 256,    256,      64,  64,  64],
                                            [BF16, F16, I16, UI16, I8,  UI8, F8E5M2, F8E4M3FN, F32, I32, UI32]>:$res);
}

class BroadCastMmodeInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128, 128, 128, 128, 256, 256, 64, 64, 64],
                                             [BF16, F16, I16, UI16, I8, UI8, F32, I32, UI32]>:$src1,
                       VectorOfLengthAndType<[128, 128, 128, 128, 256, 256, 64, 64, 64],
                                             [BF16, F16, I16, UI16, I8, UI8, F32, I32, UI32]>:$src2,
                       VectorOfLengthAndType<[256], [I1]>:$mask,
                       I32:$predicateMode);
  let results = (outs VectorOfLengthAndType<[128, 128, 128, 128, 256, 256, 64, 64, 64],
                                            [BF16, F16, I16, UI16, I8, UI8, F32, I32, UI32]>:$res);
}

//
// V300 ISA section 9
//
def VdupMInstrOp : BroadCastMmodeInstrOp<"hivm.vdup.m"> {
  let summary = "A V300 vector broadcasting instruction";
  let description = [{
    Broadcast the element in the lowest element position of source vector register
    to a destination vector register.

    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    two predication modes are provided:
    m:merging
    z:zeroing

    Examples:

    ```mlir
    // Broadcast the src into a vector
    %0 = "hivm.intr.hivm.vdup.m"(%src0, %src1, %mask, %predicateMode) : (vector<128 x bf16>, vector<128 x bf16>, vector<256 x i1>, i32)-> (vector<128 x bf16>)
    ```
  }];
}
def VdupZInstrOp : BroadCastZmodeInstrOp<"hivm.vdup.z">;

def VdupmMInstrOp : BroadCastMmodeInstrOp<"hivm.vdupm.m"> {
  let summary = "A V300 vector broadcasting instruction";
  let description = [{
    Broadcast the element in the highest element position of source vector register
    to a destination vector register.

    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    two predication modes are provided:
    m:merging
    z:zeroing

    Examples:

    ```mlir
    // Broadcast the src into a vector
    %0 = "hivm.intr.hivm.vdupm.m"(%src0, %src1, %mask, %predicateMode) : (vector<128 x bf16>, vector<128 x bf16>, vector<256 x i1>, i32)-> (vector<128 x bf16>)
    ```
  }];
}
def VdupmZInstrOp : BroadCastZmodeInstrOp<"hivm.vdupm.z">;

def Vdups_IntputType: AnyTypeOf<[I8, UI8, I16, UI16, I32, UI32, F16, F32, BF16, F8E4M3FN, F8E5M2]>;

class VdupsZmodeInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins Vdups_IntputType:$src,
                       VectorOfLengthAndType<[256], [I1]>:$mask,
                       I32:$mode);
  let results = (outs VectorOfLengthAndType<[128, 128, 128, 128, 256, 256, 64, 64, 64, 256, 256],
                                            [BF16, F16, I16, UI16, I8, UI8, F32, I32, UI32, F8E4M3FN, F8E5M2]>:$res);
}

class VdupsMmodeInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128, 128, 128, 128, 256, 256, 64, 64, 64],
                                             [BF16, F16, I16, UI16, I8, UI8, F32, I32, UI32]>:$src1,
                       Vdups_IntputType:$src2,
                       VectorOfLengthAndType<[256], [I1]>:$mask,
                       I32:$predicateMode);
  let results = (outs VectorOfLengthAndType<[128, 128, 128, 128, 256, 256, 64, 64, 64],
                                            [BF16, F16, I16, UI16, I8, UI8, F32, I32, UI32]>:$res);
}

def VdupsMInstrOp : VdupsMmodeInstrOp<"hivm.vdups.m"> {
  let summary = "A V300 vector broadcasting instruction";
  let description = [{
    Broadcast one single value in the source shared register to destination
    vector register.

    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    two predication modes are provided:
    m:merging
    z:zeroing

    Examples:

    ```mlir
    // Broadcast the src into a vector
    %0 = "hivm.intr.hivm.vdups.m"(%src0, %src1, %mask, %predicateMode) : (vector<128 x bf16>, bf16, vector<256 x i1>, i32)-> (vector<128 x bf16>)
    ```
  }];
}
def VdupsZInstrOp : VdupsZmodeInstrOp<"hivm.vdups.z">;

class VdupiB8InstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins I32:$scalar);
  let results = (outs VectorOfLengthAndType<[256, 256],
                                            [I8, UI8]>:$res);
}

def VdupiInstrOp : VdupiB8InstrOp<"hivm.vdupi"> {
  let summary = "A V300 vector broadcasting instruction";
  let description = [{
    Broadcast an immediate number to a destination vector register.

    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    Examples:

    ```mlir
    // Broadcast the src into a vector
    %0 = "hivm.intr.hivm.vdupi"(%src0) : (i32)-> (vector<256 x i8>)
    ```
  }];
}

class VdupiB16MmodeInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128, 128],
                                             [I16, UI16]>:$src,
                       I32:$scalar,
                       I32:$predicateMode);
  let results = (outs VectorOfLengthAndType<[128, 128],
                                            [I16, UI16]>:$res);
}
def VdupiMInstrOp : VdupiB16MmodeInstrOp<"hivm.vdupi.m"> {
  let summary = "A V300 vector broadcasting instruction";
  let description = [{
    Broadcast an immediate number to a destination vector register.

    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    Examples:

    ```mlir
    // Broadcast the src into a vector
    %0 = "hivm.intr.hivm.vdupi.m"(%src0, %src1, %src2) : (vector<128xi16>, i32, i32)-> (vector<128 x i16>)
    ```
  }];
}

class VdupiB16ZmodeInstrOp<string mnemonic>
    : HIVMRegbaseIntrins_Op<
          /*string opName=*/mnemonic,
          /*list<int> overloadedResults=*/[0],
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
  let arguments = (ins I32:$scalar,
                       I32:$predicateMode);
  let results = (outs VectorOfLengthAndType<[128, 128],
                                            [I16, UI16]>:$res);
}
def VdupiZInstrOp : VdupiB16ZmodeInstrOp<"hivm.vdupi.x"> {
  let summary = "A V300 vector broadcasting instruction";
  let description = [{
    Broadcast an immediate number to a destination vector register.

    A predicate (MASK) is provided to tell which elements are involved in the src vector.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    Examples:

    ```mlir
    // Broadcast the src into a vector
    %0 = "hivm.intr.hivm.vdupi.x"(%src0, %src1) : (i32, i32)-> (vector<128 x i16>)
    ```
  }];
}

def Vbr_IntputType: AnyTypeOf<[I8, UI8, I16, UI16, I32, UI32, F16, BF16, F32, F8E4M3FN, F8E5M2]>;

class VbrInstrBaseOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[0],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins Vbr_IntputType:$imm);
  let results = (outs VectorOfLengthAndType<[256, 256, 128, 128, 64, 64, 64, 128, 128, 256, 256],
                                            [I8, UI8, I16, UI16, I32, UI32, F32, F16, BF16, F8E4M3FN, F8E5M2]>:$res);
}
def VbrInstrOp : VbrInstrBaseOp<"hivm.vbr"> {
  let summary = "A V300 vector broadcasting instruction";
  let description = [{
    Broadcast the element to a vector.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    Examples:

    ```mlir
    // Broadcast the 8 bit integer into a vector
    %0 = "hivm.intr.hivm.vbr"(%src0, %src1) : (i16, i32)-> (vector<256 x i8>)
    ```
  }];
}

class VcvtifExtInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128, 256, 256], [I16, UI8, I8]>:$src,
                      predicateVector:$mask,
                      I32:$part);
  let results = (outs VectorOfLengthAndType<[64, 128], [F32, F16]>:$res);
}

def VcvtifU82F16InstrOp: VcvtifExtInstrOp<"hivm.vcvtif.u82f16.x">;
def VcvtifS82F16InstrOp: VcvtifExtInstrOp<"hivm.vcvtif.s82f16.x">;
def VcvtifS162F32InstrOp: VcvtifExtInstrOp<"hivm.vcvtif.s162f32.x">;

class VcvtifSameWidthInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[64, 128], [I16, I32]>:$src,
                      predicateVector:$mask,
                      I32:$round);
  let results = (outs VectorOfLengthAndType<[64, 128], [F32, F16]>:$res);
}

def VcvtifS162F16InstrOp: VcvtifSameWidthInstrOp<"hivm.vcvtif.s162f16.x">;
def VcvtifS322F32InstrOp: VcvtifSameWidthInstrOp<"hivm.vcvtif.s322f32.x">;

class VcvtifS4x2InstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[256], [I8]>:$src,
                      predicateVector:$mask,
                      I32:$pp);
  let results = (outs VectorOfLengthAndType<[128, 128], [F16, BF16]>:$res);
}

def VcvtifS4x22F16InstrOp: VcvtifS4x2InstrOp<"hivm.vcvtif.s4x22f16.x">;
def VcvtifS4x22BF16InstrOp: VcvtifS4x2InstrOp<"hivm.vcvtif.s4x22bf16.x">;

class VcvtifTruncInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[32], [I64]>:$src,
                      predicateVector:$mask,
                      I32:$round,
                      I32:$part);
  let results = (outs VectorOfLengthAndType<[64], [F32]>:$res);
}

def VcvtifS642F32InstrOp: VcvtifTruncInstrOp<"hivm.vcvtif.s642f32.x">;

class VcvtfiDiffWidthInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128, 128, 64, 64, 128], [F16, F16, F32, F32, BF16]>:$src,
                       predicateVector:$mask,
                       I32:$round,
                       I32:$sat,
                       I32:$part);
  let results = (outs VectorOfLengthAndType<[256, 256, 128, 32, 64],
                                            [UI8, I8, I16, I64, I32]>:$res);

}

def VcvtfiF162U8InstrOp: VcvtfiDiffWidthInstrOp<"hivm.vcvtfi.f162u8.x">;
def VcvtfiF162S8InstrOp: VcvtfiDiffWidthInstrOp<"hivm.vcvtfi.f162s8.x">;
def VcvtfiF322S16InstrOp: VcvtfiDiffWidthInstrOp<"hivm.vcvtfi.f322s16.x">;
def VcvtfiF322S64InstrOp: VcvtfiDiffWidthInstrOp<"hivm.vcvtfi.f322s64.x">;
def VcvtfiBF162S32InstrOp: VcvtfiDiffWidthInstrOp<"hivm.vcvtfi.bf162s32.x">;

class VcvtfiSameWidthInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[64, 128],
                                            [F32, F16]>:$src,
                       predicateVector:$mask,
                       I32:$round,
                       I32:$sat);
  let results = (outs VectorOfLengthAndType<[64, 128],
                                            [I32, I16]>:$res);

}

def VcvtfiF322S32InstrOp: VcvtfiSameWidthInstrOp<"hivm.vcvtfi.f322s32.x">;
def VcvtfiF162S16InstrOp: VcvtfiSameWidthInstrOp<"hivm.vcvtfi.f162s16.x">;

class VcvtfiF162S32InstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128],
                                            [F16]>:$src,
                       predicateVector:$mask,
                       I32:$round,
                       I32:$part);
  let results = (outs VectorOfLengthAndType<[64],
                                            [I32]>:$res);

}

def VcvtfiF162S32InstrOp: VcvtfiF162S32InstrOp<"hivm.vcvtfi.f162s32.x">;

class VcvtfiS4x2InstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128],
                                            [F16]>:$src,
                       predicateVector:$mask,
                       I32:$round,
                       I32:$sat,
                       I32:$pp);
  let results = (outs VectorOfLengthAndType<[256],
                                            [I8]>:$res);

}

def VcvtfiF162S4x2InstrOp: VcvtfiF162S32InstrOp<"hivm.vcvtfi.f162s4x2.x">;

class VcvtiiTruncInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128, 128, 64, 64, 64, 64, 32],
                                            [UI16, I16, UI32, UI32, I32, I32, I64]>:$src,
                       predicateVector:$mask,
                       I32:$sat,
                       I32:$part);
  let results = (outs VectorOfLengthAndType<[256, 256, 128, 128, 128, 128, 64],
                                            [UI8, I8, UI16, I16, UI16, I16, I32]>:$res);
}

def VcvtiiU162U8InstrOp: VcvtiiTruncInstrOp<"hivm.vcvtii.u162u8.x">;
def VcvtiiS162U8InstrOp: VcvtiiTruncInstrOp<"hivm.vcvtii.s162u8.x">;
def VcvtiiU322U16InstrOp: VcvtiiTruncInstrOp<"hivm.vcvtii.u322u16.x">;
def VcvtiiU322S16InstrOp: VcvtiiTruncInstrOp<"hivm.vcvtii.u322s16.x">;
def VcvtiiS322U16InstrOp: VcvtiiTruncInstrOp<"hivm.vcvtii.s322u16.x">;
def VcvtiiS322S16InstrOp: VcvtiiTruncInstrOp<"hivm.vcvtii.s322s16.x">;
def VcvtiiS642S32InstrOp: VcvtiiTruncInstrOp<"hivm.vcvtii.s642s32.x">;

class VcvtiiExtInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[256, 256, 128, 128, 128, 64],
                                            [UI8, I8, UI16, I16, I16, I32]>:$src,
                       predicateVector:$mask,
                       I32:$part);
  let results = (outs VectorOfLengthAndType<[128, 128, 64, 64, 64, 32],
                                            [UI16, I16, UI32, UI32, I32, I64]>:$res);
}

def VcvtiiU82U16InstrOp: VcvtiiExtInstrOp<"hivm.vcvtii.u82u16.x">;
def VcvtiiS82S16InstrOp: VcvtiiExtInstrOp<"hivm.vcvtii.s82s16.x">;
def VcvtiiU162U32InstrOp: VcvtiiExtInstrOp<"hivm.vcvtii.u162u32.x">;
def VcvtiiS162U32InstrOp: VcvtiiExtInstrOp<"hivm.vcvtii.s162u32.x">;
def VcvtiiS162S32InstrOp: VcvtiiExtInstrOp<"hivm.vcvtii.s162s32.x">;
def VcvtiiS322S64InstrOp: VcvtiiExtInstrOp<"hivm.vcvtii.s322s64.x">;

class VcvtiiOnlyPPInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[256, 256, 256],
                                            [UI8, I8, I8]>:$src,
                       predicateVector:$mask,
                       I32:$pp);
  let results = (outs VectorOfLengthAndType<[64, 6, 128],
                                            [UI32, I32, I16]>:$res);
}

def VcvtiiU82U32InstrOp: VcvtiiOnlyPPInstrOp<"hivm.vcvtii.u82u32.x">;
def VcvtiiS82S32InstrOp: VcvtiiOnlyPPInstrOp<"hivm.vcvtii.s82s32.x">;
def VcvtiiS4x22S16InstrOp: VcvtiiOnlyPPInstrOp<"hivm.vcvtii.s4x22s16.x">;

class VcvtiiSatPPInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[64, 64, 128],
                                            [UI32, I32, I16]>:$src,
                       predicateVector:$mask,
                       I32:$sat,
                       I32:$pp);
  let results = (outs VectorOfLengthAndType<[256, 256, 256],
                                            [UI8, I8, I8]>:$res);
}

def VcvtiiU322U8InstrOp: VcvtiiSatPPInstrOp<"hivm.vcvtii.u322u8.x">;
def VcvtiiS322U8InstrOp: VcvtiiSatPPInstrOp<"hivm.vcvtii.s322u8.x">;
def VcvtiiS162S4x2InstrOp: VcvtiiSatPPInstrOp<"hivm.vcvtii.s162s4x2.x">;

class VcvtffTruncInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[64, 64],
                                            [F32, F32]>:$src,
                       predicateVector:$mask,
                       I32:$round,
                       I32:$sat,
                       I32:$part);
  let results = (outs VectorOfLengthAndType<[128, 128, 256, 256],
                                            [F16, BF16, F8E5M2, F8E4M3FN]>:$res);

}

def VcvtffF322F16InstrOp: VcvtffTruncInstrOp<"hivm.vcvtff.f322f16.x">;
def VcvtffF322BF16InstrOp: VcvtffTruncInstrOp<"hivm.vcvtff.f322bf16.x">;
def VcvtffF322F8E4M3InstrOp: VcvtffTruncInstrOp<"hivm.vcvtff.f322f8e4m3.x">;
def VcvtffF322F8E5M2InstrOp: VcvtffTruncInstrOp<"hivm.vcvtff.f322f8e5m2.x">;

class VcvtffExtInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[256, 256, 128, 128],
                                            [F8E5M2, F8E4M3FN, F16, BF16]>:$src,
                       predicateVector:$mask,
                       I32:$part);
  let results = (outs VectorOfLengthAndType<[64],
                                            [F32]>:$res);

}

def VcvtffF162F32InstrOp: VcvtffExtInstrOp<"hivm.vcvtff.f162f32.x">;
def VcvtffBF162F32InstrOp: VcvtffExtInstrOp<"hivm.vcvtff.bf162f32.x">;
def VcvtffF8E4M32F32InstrOp: VcvtffExtInstrOp<"hivm.vcvtff.f8e4m32f32.x">;
def VcvtffF8E5M22F32InstrOp: VcvtffExtInstrOp<"hivm.vcvtff.f8e5m22f32.x">;

class VcvtffF16ToBF16InstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128],
                                            [F16]>:$src,
                       predicateVector:$mask,
                       I32:$round);
  let results = (outs VectorOfLengthAndType<[128],
                                            [BF16]>:$res);

}

def VcvtffF162BF16InstrOp: VcvtffF16ToBF16InstrOp<"hivm.vcvtff.f162bf16.x">;

class VcvtffBF16ToF16InstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins VectorOfLengthAndType<[128],
                                            [BF16]>:$src,
                       predicateVector:$mask,
                       I32:$sat,
                       I32:$round);
  let results = (outs VectorOfLengthAndType<[128],
                                            [F16]>:$res);

}

def VcvtffBF162F16InstrOp: VcvtffBF16ToF16InstrOp<"hivm.vcvtff.bf162f16.x">;

class VtrcInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                          /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/2> {
  let arguments = (ins VectorOfLengthAndType<[128, 128, 64],
                                            [BF16, F16, F32]>:$src,
                      I32:$round,
                      predicateVector:$mask);
  let results = (outs VectorOfLengthAndType<[128, 128, 64],
                                            [BF16, F16, F32]>:$res);
                                         }

def VtrcV64F32InstrOp: VtrcInstrOp<"hivm.vtrc.f32.x">;
def VtrcV128F16InstrOp: VtrcInstrOp<"hivm.vtrc.f16.x">;
def VtrcV128BF16InstrOp: VtrcInstrOp<"hivm.vtrc.bf16.x">;

class VintlvInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[0],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/2> {
  let arguments = (ins ValidVector:$src0,
                       ValidVector:$src1
                       );

}

def VintlvInstrOp: VintlvInstrOp<"hivm.vintlv">;

class VDintlvInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[0],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/2> {
  let arguments = (ins ValidVector:$src0,
                       ValidVector:$src1
                       );

}

def VDintlvInstrOp: VDintlvInstrOp<"hivm.vdintlv">;

class PintlvInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/2> {
  let arguments = (ins predicateVector:$src0,
                       predicateVector:$src1
                       );
}

def PintlvB32InstrOp: PintlvInstrOp<"hivm.pintlv.b32">;
def PintlvB16InstrOp: PintlvInstrOp<"hivm.pintlv.b16">;
def PintlvB8InstrOp: PintlvInstrOp<"hivm.pintlv.b8">;

class PDintlvInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/2> {
  let arguments = (ins predicateVector:$src0,
                       predicateVector:$src1
                       );
}

def PDintlvB32InstrOp: PDintlvInstrOp<"hivm.pdintlv.b32">;
def PDintlvB16InstrOp: PDintlvInstrOp<"hivm.pdintlv.b16">;
def PDintlvB8InstrOp: PDintlvInstrOp<"hivm.pdintlv.b8">;

class VpackInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[0],
                /*list<Trait> traits=*/[],
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src,
                       I32:$part);
  let results = (outs ValidVector:$res);
}

def VpackInstrOp: VpackInstrOp<"hivm.vpack.x">;

class UnpackInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[0],
                /*list<int> overloadedOperands=*/[0],
                /*list<Trait> traits=*/[],
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src,
                       I32:$part);
  let results = (outs ValidVector:$res);
}

def Vzunpack: UnpackInstrOp<"hivm.vzunpack">;
def Vsunpack: UnpackInstrOp<"hivm.vsunpack">;

class PackUnpackInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/1> {
  let arguments = (ins predicateVector:$src,
                       I32:$part);
}

def PpackInstrOp: PackUnpackInstrOp<"hivm.ppack.z">;
def PunpackInstrOp: PackUnpackInstrOp<"hivm.punpack">;

class VmullInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
                                         /*string opName=*/mnemonic,
                                         /*list<int> overloadedResults=*/[],
                                         /*list<int> overloadedOperands=*/[],
                                         /*list<Trait> traits=*/[],
                                         /*int numResults=*/2> {
  let arguments = (ins ValidVector:$src0,
                       ValidVector:$src1,
                       predicateVector:$mask);

}

def VmullV64S32InstrOp : VmullInstrOp<"hivm.vmull.v64s32">{
  let summary = "Widening Vector Multiplication Instruction";
  let description = [{
    Perform 32-bit element-wise extended multiplication for every active elements in Vn and Vm,
    and write the 64bit result into Vd and Vd+1, with lower 32bit in Vd and higher 32bit in Vd+1.
    note: This instruction preserves the complete results of the computation, hence no saturation
    or truncation will be applied to the output regardless of all the relative control in SPR.CTRL
    #### Examples:

    ```mlir
    // Widening vector multiplication on two <64xi32> vectors.
    %0 = "hivm.intr.hivm.vmull.v64s32"(%src1, %src2, %preg) : (vector<64xi32>, vector<64xi32>,
     vector<256xi1>)-> (vector<64xi32>, vector<64xi32>)
    ```
  }];
}
def VmullV64U32InstrOp : VmullInstrOp<"hivm.vmull.v64u32">;

class VciInstrBaseOp<string mnemonic> : HIVMRegbaseIntrins_Op<
  /*opName=*/ mnemonic,
  /*list<int> overloadedResults=*/[0],
  /*overloadedOperands=*/ [],
  /*traits=*/ [],
  /*numResults=*/1
> {
  let arguments = (ins
    AnyTypeOf<[I32, I16, I8, F32, F16]>:$src0,
    AnyTypeOf<[I32]>:$src1
  );

  let results = (outs
    VectorOfLengthAndType<[64, 128, 256], [I32, I16, I8, F32, F16]>:$res
  );
}

def VciInstrOp : VciInstrBaseOp<"hivm.vci"> {
  let summary = "A V64 vector conversion instruction";
  let description = [{
    Convert the elements of a vector to a different type.

    The result is a vector with the given type.
    Result size depends on input type bit width.

    Examples:

    ```mlir
    // Convert vector of 64 elements of 32-bit integers
    %0 = "hivm.intr.hivm.vci"(%src0, %src1) : (i32, i32)-> (vector<64 x i32>)
    ```
  }];
}


class VGatherInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
  /*string opName =*/ mnemonic,
  /*list<int> overloadedResults=*/[],
  /*list<int> overloadedOperands =*/ [],
  /*list<Trait> traits =*/ [],
  /*int numResults =*/ 1> {
  let arguments = (ins UbufPtr:$ptr,
                   VectorOfLengthAndType<[64, 128, 256],
                                         [I32, I16, I8, F32, F16, BF16, F8E4M3FN, F8E5M2]>:$indices,
                   VectorOfLengthAndType<[256],
                                         [I1]>:$mask);
  let results = (outs VectorOfLengthAndType<[64, 128, 256],
                                            [I32, I16, I8, F32, F16, BF16, F8E4M3FN, F8E5M2]>:$res);
}

def VGatherV64S32InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v64s32"> {
  let summary = "A V64 vector gather instruction";
  let description = [{
    Gather elements from memory using indices and a mask.
    The result is a vector with the gathered elements.
    Example:
    ```mlir
    // Gather 64 elements of 32-bit integers
    %0 = "hivm.intr.hivm.vgather2_v300.v64s32"(%ptr, %indices, %mask) : (ptr_ty6, v64i32, v256i1) -> (v64i32)
    ```
  }];
}

def VGatherV64U32InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v64u32">;
def VGatherV128S16InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v128s16">;
def VGatherV128U16InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v128u16">;
def VGatherV256S8InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v256s8">;
def VGatherV256U8InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v256u8">;
def VGatherV64F32InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v64f32">;
def VGatherV128F16InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v128f16">;
def VGatherV128BF16InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v128bf16">;
def VGatherV256F8E4InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v256f8e4m3">;
def VGatherV256F8E5InstrOp : VGatherInstrOp<"hivm.vgather2_v300.v256f8e5m2">;

class VScatterInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
  /*string opName =*/ mnemonic,
  /*list<int> overloadedResults=*/[],
  /*list<int> overloadedOperands =*/ [],
  /*list<Trait> traits =*/ [],
  /*int numResults =*/ 0> {
  let arguments = (ins VectorOfLengthAndType<[64, 128, 256], [I32, I16, I8, F32, F16, BF16]>:$vec,
                   UbufPtr:$ptr,
                   VectorOfLengthAndType<[64, 128, 256], [I32, I16, I8, F32, F16, BF16]>:$indices,
                   VectorOfLengthAndType<[256], [I1]>:$mask);
  let results = (outs);
}

def VScatterV64S32InstrOp : VScatterInstrOp<"hivm.vscatter_v64s32_v300"> {
  let summary = "A V64 vector scatter instruction";
  let description = [{
    Scatter elements to memory using indices and a mask.
    The input vector's elements are written to memory locations determined by the indices vector,
    with the mask vector controlling which elements are written.
    Example:
    ```mlir
    // Scatter 64 elements of 32-bit integers
    "hivm.intr.hivm.vscatter_v64s32_v300"(%vec, %ptr, %indices, %mask) : (v64i32, ptr_ty6, v64i32, v256i1) -> ()
    ```
  }];
}

def VScatterV64U32InstrOp : VScatterInstrOp<"hivm.vscatter.v64u32_v300">;
def VScatterV128S16InstrOp : VScatterInstrOp<"hivm.vscatter.v128s16_v300">;
def VScatterV128U16InstrOp : VScatterInstrOp<"hivm.vscatter.v128u16_v300">;
def VScatterV256S8InstrOp : VScatterInstrOp<"hivm.vscatter.v256s8_v300">;
def VScatterV256U8InstrOp : VScatterInstrOp<"hivm.vscatter.v256u8_v300">;
def VScatterV64F32InstrOp : VScatterInstrOp<"hivm.vscatter.v64f32_v300">;
def VScatterV128F16InstrOp : VScatterInstrOp<"hivm.vscatter.v128f16_v300">;
def VScatterV128BF16InstrOp : VScatterInstrOp<"hivm.vscatter.v128bf16_v300">;

class VCompareInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src1", "src2"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[0],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src1,
                       ValidVector:$src2,
                       predicateVector:$mask);
  let results = (outs predicateVector:$res);
}

def VCmpUEqInstrOp: VCompareInstrOp<"hivm.vcmp.eq.u.z">;
def VCmpSEqInstrOp: VCompareInstrOp<"hivm.vcmp.eq.s.z">;

def VCmpUGeInstrOp: VCompareInstrOp<"hivm.vcmp.ge.u.z">;
def VCmpSGeInstrOp: VCompareInstrOp<"hivm.vcmp.ge.s.z">;

def VCmpUGtInstrOp: VCompareInstrOp<"hivm.vcmp.gt.u.z">;
def VCmpSGtInstrOp: VCompareInstrOp<"hivm.vcmp.gt.s.z">;

def VCmpULeInstrOp: VCompareInstrOp<"hivm.vcmp.le.u.z">;
def VCmpSLeInstrOp: VCompareInstrOp<"hivm.vcmp.le.s.z">;

def VCmpULtInstrOp: VCompareInstrOp<"hivm.vcmp.lt.u.z">;
def VCmpSLtInstrOp: VCompareInstrOp<"hivm.vcmp.lt.s.z">;

def VCmpUNeInstrOp: VCompareInstrOp<"hivm.vcmp.ne.u.z">;
def VCmpSNeInstrOp: VCompareInstrOp<"hivm.vcmp.ne.s.z">;

def VCompareS_IntputType: AnyTypeOf<[I8, UI8, I16, UI16, I32, UI32, BF16, F16, F32]>;

class VCompareSInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[0],
                /*list<Trait> traits=*/[],
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src1,
                       VCompareS_IntputType:$src2,
                       predicateVector:$mask);
  let results = (outs predicateVector:$res);
}

def VCmpsUZEqInstrOp: VCompareSInstrOp<"hivm.vcmps.eq.u.z">;
def VCmpsSZEqInstrOp: VCompareSInstrOp<"hivm.vcmps.eq.s.z">;


def VCmpsUZGeInstrOp: VCompareSInstrOp<"hivm.vcmps.ge.u.z">;
def VCmpsSZGeInstrOp: VCompareSInstrOp<"hivm.vcmps.ge.s.z">;

def VCmpsUZGtInstrOp: VCompareSInstrOp<"hivm.vcmps.gt.u.z">;
def VCmpsSZGtInstrOp: VCompareSInstrOp<"hivm.vcmps.gt.s.z">;

def VCmpsUZLeInstrOp: VCompareSInstrOp<"hivm.vcmps.le.u.z">;
def VCmpsSZLeInstrOp: VCompareSInstrOp<"hivm.vcmps.le.s.z">;

def VCmpsUZLtInstrOp: VCompareSInstrOp<"hivm.vcmps.lt.u.z">;
def VCmpsSZLtInstrOp: VCompareSInstrOp<"hivm.vcmps.lt.s.z">;

def VCmpsUZNeInstrOp: VCompareSInstrOp<"hivm.vcmps.ne.u.z">;
def VCmpsSZNeInstrOp: VCompareSInstrOp<"hivm.vcmps.ne.s.z">;

class BinaryPredicateLogicInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src1", "src2", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins predicateVector:$src1,
                       predicateVector:$src2,
                       predicateVector:$mask);
  let results = (outs predicateVector:$res);
}

def PorB8InstrOp : BinaryPredicateLogicInstrOp<"hivm.por.z">;
def PandB8InstrOp : BinaryPredicateLogicInstrOp<"hivm.pand.z">;
def PxorB8InstrOp : BinaryPredicateLogicInstrOp<"hivm.pxor.z">;

class UnaryPredicateLogicInstrOp<string mnemonic,
  list<Trait> traits = [AllTypesMatch<["src", "res"]>]> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/traits,
                /*int numResults=*/1> {
  let arguments = (ins predicateVector:$src,
                       predicateVector:$mask);
  let results = (outs predicateVector:$res);
}
def PnotB8InstrOp : UnaryPredicateLogicInstrOp<"hivm.pnot.z">;
class VSlideInstrOp<string mnemonic> : HIVMRegbaseIntrins_Op<
  /*string opName =*/ mnemonic,
  /*list<int> overloadedResults=*/[0],
  /*list<int> overloadedOperands =*/ [],
  /*list<Trait> traits =*/ [],
  /*int numResults =*/ 1> {
  let arguments = (ins VectorOfLengthAndType<[64, 128, 256], [I32, I16, I8, F32, F16]>:$scr1,
                   VectorOfLengthAndType<[64, 128, 256], [I32, I16, I8, F32, F16]>:$scr2, I16:$offset);
  let results = (outs VectorOfLengthAndType<[64, 128, 256], [I32, I16, I8, F32, F16]>:$res);
}

def VSlideInstrOp : VSlideInstrOp<"hivm.vslide"> {
 let summary = "Vector slide operation";
  let description = [{
    Performs a vector slide operation by a given amount. The source vector is held
    in two consecutive vector registers. The instruction slides a
    window of the same size as the vector length over the two source vectors to
    produce the destination vector.
    
    The slide amount is given by a shared register (Sm) which must be in the
    range S0-S95.
  }];
}

class SIMT_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HIVMRegbaseIntrins_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let cppNamespace = "mlir::hivm_regbaseintrins";
}

def SIMT_EntryAttr : SIMT_Attr<"SIMT_Entry", "simt_entry"> {
  let parameters = (ins AttrParameter<"uint32_t", "">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def SIMT_CallableAttr : SIMT_Attr<"SIMT_Callable", "simt_callable">;

def SIMT_TargetAttr : SIMT_Attr<"SIMT_Target", "target"> {
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
}

//
//// AVE SIMT Intrinsics Ops
//

class GenericSIMT_Op<string mnemonic, list<Trait> traits = []>
    : Op<HIVMRegbaseIntrins_Dialect, !strconcat("intrins.", mnemonic), traits> {
  // C++ function that returns the op name.
  code opNameDeclaration = [{
    static StringRef getOpName() { return "}] # mnemonic # [{"; }
  }];
}

def LaunchFuncOp : GenericSIMT_Op<"launch_func"> {
  let arguments = (ins FlatSymbolRefAttr:$kernel, I64:$blockSizeX, I64:$blockSizeY,
      I64:$blockSizeZ, Variadic<AnyType>:$opnds);
  let assemblyFormat =
      [{$kernel `threads` `in` ` ` `(`$blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
      attr-dict (`args``(`$opnds^`)` `:` type($opnds))? }];
}

class HIVMRegbaseIntrinsSIMT_IntrOp<string mnemonic, string intr, int numResults = 0,
                     list<int> overloadedOperands = [], list<Trait> traits = []>
    : LLVM_IntrOpBase<HIVMRegbaseIntrins_Dialect, mnemonic,
                      /*enumName*/ "hivm_" # intr,
                      /*overloadedResult*/[], overloadedOperands, traits,
                      /*numResults*/ numResults>;

def VFSIMTInfoOp: HIVMRegbaseIntrinsSIMT_IntrOp<"store_vfsimt_info","store_vfsimt_info"> {
  let arguments = (ins I64: $config);
  let assemblyFormat = "$config attr-dict";
}

class HIVMRegbaseIntrinsSIMT_Ops<string mnemonic, string intr>
    : HIVMRegbaseIntrinsSIMT_IntrOp<mnemonic, intr, 1> {
  let results = (outs I32:$res);
  let assemblyFormat = "attr-dict";
}

// SIMT function intrinsics
// NOTE: Ideally the x/y/z should be a single op, but due to the way that MLIR
//       interfaces with LLVM intrinsics, they are defined as seperate ops.
def ThreadIdXOp : HIVMRegbaseIntrinsSIMT_Ops<"thread_id_x", "get_TID_X">;
def ThreadIdYOp : HIVMRegbaseIntrinsSIMT_Ops<"thread_id_y", "get_TID_Y">;
def ThreadIdZOp : HIVMRegbaseIntrinsSIMT_Ops<"thread_id_z", "get_TID_Z">;
def BlockIdxXOp : HIVMRegbaseIntrinsSIMT_Ops<"block_idx_x", "get_BLOCK_IDX_X">;
def BlockIdxYOp : HIVMRegbaseIntrinsSIMT_Ops<"block_idx_y", "get_BLOCK_IDX_Y">;
def BlockIdxZOp : HIVMRegbaseIntrinsSIMT_Ops<"block_idx_z", "get_BLOCK_IDX_Z">;
def BlockDimXOp : HIVMRegbaseIntrinsSIMT_Ops<"block_dim_x", "get_BLOCK_DIM_X">;
def BlockDimYOp : HIVMRegbaseIntrinsSIMT_Ops<"block_dim_y", "get_BLOCK_DIM_Y">;
def BlockDimZOp : HIVMRegbaseIntrinsSIMT_Ops<"block_dim_z", "get_BLOCK_DIM_Z">;
def BlockIdxOp : HIVMRegbaseIntrinsSIMT_Ops<"block_idx", "tpe_get_BLOCK_IDX">;
def BlockNumOp : HIVMRegbaseIntrinsSIMT_Ops<"block_num", "tpe_get_BLOCK_NUM">;

def Clock32Op : HIVMRegbaseIntrinsSIMT_Ops<"clock32", "get_CLOCK32">;
def Clock64Op : HIVMRegbaseIntrinsSIMT_Ops<"clock64", "get_CLOCK64"> {
  let results = (outs I64:$res);
}

def CoreIdOp : HIVMRegbaseIntrinsSIMT_Ops<"core_id", "tpe_get_COREID">;

def SyncThreadsOp : HIVMRegbaseIntrinsSIMT_IntrOp<"sync_threads", "sync_workitems", 0> {
  let assemblyFormat = "attr-dict";
}

class ReduceOp<string mnemonic, Type opType> : 
  HIVMRegbaseIntrins_Op<mnemonic, [], [], [], 1> {
  let arguments = (ins opType:$src);
  let results = (outs opType:$res);
  let assemblyFormat = "$src attr-dict";
}

def ReduceAddOpU32 : ReduceOp<"hivm.redux.add.u32", I32>;
def ReduceAddOpS32 : ReduceOp<"hivm.redux.add.s32", I32>;
def ReduceAddOpF32 : ReduceOp<"hivm.redux.add.f32", F32>;
def ReduceAddOpF16 : ReduceOp<"hivm.redux.add.f16", F16>;
def ReduceMaxOpU32 : ReduceOp<"hivm.redux.max.u32", I32>;
def ReduceMaxOpS32 : ReduceOp<"hivm.redux.max.s32", I32>;
def ReduceMaxOpF32 : ReduceOp<"hivm.redux.max.f32", F32>;
def ReduceMaxOpF16 : ReduceOp<"hivm.redux.max.f16", F16>;
def ReduceMinOpU32 : ReduceOp<"hivm.redux.min.u32", I32>;
def ReduceMinOpS32 : ReduceOp<"hivm.redux.min.s32", I32>;
def ReduceMinOpF32 : ReduceOp<"hivm.redux.min.f32", F32>;
def ReduceMinOpF16 : ReduceOp<"hivm.redux.min.f16", F16>;

class ShflOp<string mnemonic> : 
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/[],
                /*int numResults=*/1> {
}

class ShflI32Op<string mnemonic> : ShflOp<mnemonic> {
  let arguments = (ins I32:$src,
                       I32:$ctrl);
  let results = (outs I32:$res);
}

class ShflI64Op<string mnemonic> : ShflOp<mnemonic> {
  let arguments = (ins I64:$src,
                       I32:$ctrl);
  let results = (outs I64:$res);
}

class ShflF32Op<string mnemonic> : ShflOp<mnemonic> {
  let arguments = (ins F32:$src,
                       I32:$ctrl);
  let results = (outs F32:$res);
}

class ShflF16Op<string mnemonic> : ShflOp<mnemonic> {
  let arguments = (ins F16:$src,
                       I32:$ctrl);
  let results = (outs F16:$res);
}

class ShflV2F16Op<string mnemonic> : ShflOp<mnemonic> {
  let arguments = (ins VectorOfLengthAndType<[2], [F16]>:$src,
                       I32:$ctrl);
  let results = (outs VectorOfLengthAndType<[2], [F16]>:$res);
}

def ShflUpOpI32 : ShflI32Op<"hivm.shfl.up.i32">;
def ShflUpOpI32p : ShflI32Op<"hivm.shfl.up.i32p">;
def ShflUpOpI64 : ShflI64Op<"hivm.shfl.up.i64">;
def ShflUpOpI64p : ShflI64Op<"hivm.shfl.up.i64p">;
def ShflUpOpF32 : ShflF32Op<"hivm.shfl.up.f32">;
def ShflUpOpF32p : ShflF32Op<"hivm.shfl.up.f32p">;
def ShflUpOpF16 : ShflF16Op<"hivm.shfl.up.f16">;
def ShflUpOpF16p : ShflF16Op<"hivm.shfl.up.f16p">;
def ShflUpOpV2F16 : ShflV2F16Op<"hivm.shfl.up.v2f16">;
def ShflUpOpV2F16p : ShflV2F16Op<"hivm.shfl.up.v2f16p">;

def ShflDownOpI32 : ShflI32Op<"hivm.shfl.down.i32">;
def ShflDownOpI32p : ShflI32Op<"hivm.shfl.down.i32p">;
def ShflDownOpI64 : ShflI64Op<"hivm.shfl.down.i64">;
def ShflDownOpI64p : ShflI64Op<"hivm.shfl.down.i64p">;
def ShflDownOpF32 : ShflF32Op<"hivm.shfl.down.f32">;
def ShflDownOpF32p : ShflF32Op<"hivm.shfl.down.f32p">;
def ShflDownOpF16 : ShflF16Op<"hivm.shfl.down.f16">;
def ShflDownOpF16p : ShflF16Op<"hivm.shfl.down.f16p">;
def ShflDownOpV2F16 : ShflV2F16Op<"hivm.shfl.down.v2f16">;
def ShflDownOpV2F16p : ShflV2F16Op<"hivm.shfl.down.v2f16p">;

def ShflBflyOpI32 : ShflI32Op<"hivm.shfl.bfly.i32">;
def ShflBflyOpI32p : ShflI32Op<"hivm.shfl.bfly.i32p">;
def ShflBflyOpI64 : ShflI64Op<"hivm.shfl.bfly.i64">;
def ShflBflyOpI64p : ShflI64Op<"hivm.shfl.bfly.i64p">;
def ShflBflyOpF32 : ShflF32Op<"hivm.shfl.bfly.f32">;
def ShflBflyOpF32p : ShflF32Op<"hivm.shfl.bfly.f32p">;
def ShflBflyOpF16 : ShflF16Op<"hivm.shfl.bfly.f16">;
def ShflBflyOpF16p : ShflF16Op<"hivm.shfl.bfly.f16p">;
def ShflBflyOpV2F16 : ShflV2F16Op<"hivm.shfl.bfly.v2f16">;
def ShflBflyOpV2F16p : ShflV2F16Op<"hivm.shfl.bfly.v2f16p">;

def ShflIdxOpI32 : ShflI32Op<"hivm.shfl.idx.i32">;
def ShflIdxOpI32p : ShflI32Op<"hivm.shfl.idx.i32p">;
def ShflIdxOpI64 : ShflI64Op<"hivm.shfl.idx.i64">;
def ShflIdxOpI64p : ShflI64Op<"hivm.shfl.idx.i64p">;
def ShflIdxOpF32 : ShflF32Op<"hivm.shfl.idx.f32">;
def ShflIdxOpF32p : ShflF32Op<"hivm.shfl.idx.f32p">;
def ShflIdxOpF16 : ShflF16Op<"hivm.shfl.idx.f16">;
def ShflIdxOpF16p : ShflF16Op<"hivm.shfl.idx.f16p">;
def ShflIdxOpV2F16 : ShflV2F16Op<"hivm.shfl.idx.v2f16">;
def ShflIdxOpV2F16p : ShflV2F16Op<"hivm.shfl.idx.v2f16p">;

class HIVMRegbaseIntrinsSIMTUnary_Ops<string mnemonic, string intr, Type insTy, Type outsTy>
    : HIVMRegbaseIntrinsSIMT_IntrOp<mnemonic, intr, 1> {
  let arguments = (ins insTy:$ins);
  let results = (outs outsTy:$res);
  let assemblyFormat = "$ins attr-dict";
}

def RintOpF16 : HIVMRegbaseIntrinsSIMTUnary_Ops<"rinth", "rint_h", F16, F16>;
def RintOpF32 : HIVMRegbaseIntrinsSIMTUnary_Ops<"rintf", "rint_f", F32, F32>;
def RintOpBF16 : HIVMRegbaseIntrinsSIMTUnary_Ops<"rinty", "rint_y", BF16, BF16>;

def FloorOpF16 : HIVMRegbaseIntrinsSIMTUnary_Ops<"floorh", "floor_h", F16, F16>;
def FloorOpF32 : HIVMRegbaseIntrinsSIMTUnary_Ops<"floorf", "floor_f", F32, F32>;
def FloorOpBF16 : HIVMRegbaseIntrinsSIMTUnary_Ops<"floory", "floor_y", BF16, BF16>;

def CeilOpF16 : HIVMRegbaseIntrinsSIMTUnary_Ops<"ceilh", "ceil_h", F16, F16>;
def CeilOpF32 : HIVMRegbaseIntrinsSIMTUnary_Ops<"ceilf", "ceil_f", F32, F32>;
def CeilOpBF16 : HIVMRegbaseIntrinsSIMTUnary_Ops<"ceily", "ceil_y", BF16, BF16>;

def SqrtOpF32 : HIVMRegbaseIntrinsSIMTUnary_Ops<"sqrtf", "sqrt_f", F32, F32>; 

def LogOpF32 : HIVMRegbaseIntrinsSIMTUnary_Ops<"logf", "log_f", F32, F32>;

def UmulhiOp : HIVMRegbaseIntrinsSIMT_IntrOp<"umulhi", "umulhi", 1> {
  let arguments = (ins I32:$ins1, I32:$ins2);
  let results = (outs I32:$res);
  let assemblyFormat = "$ins1 `,` $ins2 attr-dict";
}

class PredicateStoreInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/[],
                /*int numResults=*/0> {
  let summary = "A V300 Predicate Store Instruction";
  let arguments = (ins predicateVector:$src,
                         UbufPtr:$dst,
                         I32:$offset,
                         I32:$dist,
                         I32:$postUpdateEnable);
}
def PStoreB8InstOp : PredicateStoreInstrOp<"hivm.psts.b8">;

class MovvpInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[0],
                /*list<Trait> traits=*/[],
                /*int numResults=*/1> {
  let arguments = (ins ValidVector:$src,
                       I32:$part);
  let results = (outs predicateVector:$res);
}

def MovvpInstrOp : MovvpInstrOp<"hivm.movvp">{
  let summary = "A V300 extracted instruction";
  let description = [{
    This instruction will extract VL_16 bit/VL_32 bit data from Vn and expand to VL_8 bit then write to Pd.
    Examples:
    ```mlir
    %0= "hivm_regbaseintrins.intr.hivm.movvp"(%src, %part) : (vector<64xi32>, i32) -> vector<256xi1>
    ```
  }];
}

class PredicatePSTUStoreInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/[],
                /*int numResults=*/1> {
  let summary = "A V300 Predicate Store Instruction";
  let arguments = (ins predicateVector:$pd,
                       UbufPtr:$src,
                       AlignedVector:$ustd);
}
def PSTUStoreS16InstOp : PredicatePSTUStoreInstrOp<"hivm.pstu.b16">;
def PSTUStoreS32InstOp : PredicatePSTUStoreInstrOp<"hivm.pstu.b32">;

class PredicateLoadInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/[],
                /*int numResults=*/1> {
  let summary = "A V300 Predicate Load Instruction";
  let arguments = (ins UbufPtr:$src,
                       I32:$offset,
                       I32:$dist,
                       I32:$postUpdateEnable);
  let results = (outs predicateVector:$res);
}
def PLoadB8InstOp : PredicateLoadInstrOp<"hivm.plds.b8">;
def VaddsUXInstrOp : BinaryVecScalarOp<"hivm.vadds.u.x"> {
  let summary = "A V300 vector-scalar addition instruction";
  let description = [{
    Adds the elements of a vector registers and scalar stores the result into
    another vector register.
    A predicate (MASK) is provided to tell which elements are added in the two vectors.  
  }];
}
def VaddsSXInstrOp : BinaryVecScalarOp<"hivm.vadds.s.x">;
def VaddsUMInstrOp : BinaryVecScalarOp<"hivm.vadds.u.m">;
def VaddsSMInstrOp  : BinaryVecScalarOp<"hivm.vadds.s.m">;

def VmulsSMInstrOp : BinaryVecScalarOp<"hivm.vmuls.s.m"> {
  let summary = "A V300 vector-scalar multiplication instruction";
  let description = [{
    Multiply the elements of a vector registers and scalar stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.
  }];
}
def VmulsUMInstrOp : BinaryVecScalarOp<"hivm.vmuls.u.m">;
def VmulsSXInstrOp : BinaryVecScalarOp<"hivm.vmuls.s.x">;
def VmulsUXInstrOp : BinaryVecScalarOp<"hivm.vmuls.u.x">;

def VlreluMInstrOp : BinaryVecScalarOp<"hivm.vlrelu.m"> {
  let summary = "A V300 vector instruction to find element-wise min";
  let description = [{
    Vd = (Vn ≥0) ? Vn : Vn*Sm
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.
  }];
}
def VlreluXInstrOp : BinaryVecScalarOp<"hivm.vlrelu.x">;

def VmaxsUXInstrOp : BinaryVecScalarOp<"hivm.vmaxs.u.x"> {
  let summary = "A V300 vector instruction to find element-wise max";
  let description = [{
    Perform element-wise maximum of a vector registers and scalar and stores the result
    into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.
  }];
}
def VmaxsSXInstrOp : BinaryVecScalarOp<"hivm.vmaxs.s.x">;
def VmaxsUMInstrOp : BinaryVecScalarOp<"hivm.vmaxs.u.m">;
def VmaxsSMInstrOp : BinaryVecScalarOp<"hivm.vmaxs.s.m">;

def VminsUXInstrOp : BinaryVecScalarOp<"hivm.vmins.u.x"> {
  let summary = "A V300 vector instruction to find element-wise min";
  let description = [{
    Perform element-wise minimum of a vector registers and scalar and stores the result
    into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.
  }];
}
def VminsSXInstrOp : BinaryVecScalarOp<"hivm.vmins.s.x">;
def VminsUMInstrOp : BinaryVecScalarOp<"hivm.vmins.u.m">;
def VminsSMInstrOp : BinaryVecScalarOp<"hivm.vmins.s.m">;

def VshlsSXInstrOp : BinaryVecScalarOp<"hivm.vshls.s.x"> {
  let summary = "A V300 vector logical/arithmetic shift left instruction";
  let description = [{
    Perform element-wise shift left for src1 by the amount defined as a scalar
    and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    For unsigned data, it's logical shift. For signed data, it's arithmetic shift.

    src2 is always treated as signed integer regardless of its type. i.e. if elements in
    src2 is u8, then it is treated as s8. Negative shift amount means shift in the
    other direction, so a left shift with negative amount becomes a right shift
  }];
}
def VshlsUXInstrOp : BinaryVecScalarOp<"hivm.vshls.u.x">;

def VshrsSXInstrOp : BinaryVecScalarOp<"hivm.vshrs.s.x"> {
  let summary = "A V300 vector logical/arithmetic shift right instruction";
  let description = [{
    Perform element-wise shift right for src1 by the amount defined as a scalar
    and stores the result into another vector register.
    A predicate (MASK) is provided to tell which elements are involved in the two vectors.

    For unsigned data, it's logical shift. For signed data, it's arithmetic shift.

    src2 is always treated as signed integer regardless of its type. i.e. if elements in
    src2 is u8, then it is treated as s8. Negative shift amount means shift in the
    other direction, so a right shift with negative amount becomes a left shift
  }];
}
def VshrsUXInstrOp : BinaryVecScalarOp<"hivm.vshrs.u.x">;

class MemBarInstrOp<string mnemonic> :
  HIVMRegbaseIntrins_Op</*string opName=*/mnemonic,
                /*list<int> overloadedResults=*/[],
                /*list<int> overloadedOperands=*/[],
                /*list<Trait> traits=*/[],
                /*int numResults=*/0> {
  let summary = "A V300 MemBar Instruction";  
}
 
def MemBarVVALLInstrOp: MemBarInstrOp<"hivm.mem.bar.vv.all">;
def MemBarVSTVLDInstrOp: MemBarInstrOp<"hivm.mem.bar.vst.vld">;
def MemBarVLDVSTInstrOp: MemBarInstrOp<"hivm.mem.bar.vld.vst">;
def MemBarVSTVSTInstrOp: MemBarInstrOp<"hivm.mem.bar.vst.vst">;
def MemBarVSALLInstrOp: MemBarInstrOp<"hivm.mem.bar.vs.all">;
def MemBarVSTLDInstrOp: MemBarInstrOp<"hivm.mem.bar.vst.ld">;
def MemBarVLDSTInstrOp: MemBarInstrOp<"hivm.mem.bar.vld.st">;
def MemBarVSTSTInstrOp: MemBarInstrOp<"hivm.mem.bar.vst.st">;
def MemBarSVALLInstrOp: MemBarInstrOp<"hivm.mem.bar.sv.all">;
def MemBarSTVLDInstrOp: MemBarInstrOp<"hivm.mem.bar.st.vld">;
def MemBarLDVSTInstrOp: MemBarInstrOp<"hivm.mem.bar.ld.vst">;
def MemBarSTVSTInstrOp: MemBarInstrOp<"hivm.mem.bar.st.vst">;
def MemBarSSALLTInstrOp: MemBarInstrOp<"hivm.mem.bar.ss.all">;
def MemBarSTLDInstrOp: MemBarInstrOp<"hivm.mem.bar.st.ld">;
def MemBarLDSTInstrOp: MemBarInstrOp<"hivm.mem.bar.ld.st">;
def MemBarSTSTInstrOp: MemBarInstrOp<"hivm.mem.bar.st.st">;

#endif // HIVMREGBASEINTRINS_OPS

