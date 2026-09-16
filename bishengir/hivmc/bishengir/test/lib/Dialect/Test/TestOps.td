//===--- TestOps.td - Test dialect operation definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TESTOPS
#define TEST_TESTOPS

include "TestDialect.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"

class BishengIRTest_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, !listconcat([], traits)>;

class BaseStructuredOp<string mnemonic, list<Trait> traits = []> :
  BishengIRTest_Op<mnemonic, !listconcat([
    AttrSizedOperandSegments, HIVMStructuredOpInterface,
    NoLibraryFunctionTrait],
  traits)> {
  let arguments = (ins Variadic<AnyType>:$src,
                       Variadic<AnyType>:$dst);
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
    (`outs` `(` $dst^  `:` type($dst) `)`)?
    (`->` type($result)^)?
  }];
  code baseClassDeclaration = [{
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
  code baseIteratorTypeClassDeclaration = [{
    // Implement functions necessary for HIVMStructuredOpInterface.
    mlir::SmallVector<mlir::hivm::IteratorType> getIteratorTypesArray() {
      return mlir::hivm::detail::getIteratorTypesArrayForElemwiseOp(getOperation());
    }

    mlir::LogicalResult
    setIteratorTypesArray(const mlir::hivm::IteratorType iteratorTypes,
                          const mlir::DenseI64ArrayAttr &arrayAttr) {
      return mlir::hivm::detail::setIteratorTypesArrayForElemwiseOp(
                               getOperation(), iteratorTypes, arrayAttr);
    }

    mlir::ArrayAttr getIndexingMaps() {
      return mlir::ArrayAttr();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Test ElementwiseNaryOpTrait.
//===----------------------------------------------------------------------===//

class BaseElemwiseOp<string mnemonic, list<Trait> traits = []> :
  BaseStructuredOp<mnemonic,
    !listconcat([HIVMOpSameOperandsAndResultRank], traits)> {
  let extraClassDeclaration = baseClassDeclaration #
                              baseIteratorTypeClassDeclaration;
}

def TestElementwiseUnaryOp :
  BaseElemwiseOp<"elementwise_unary", [ElementwiseNaryOpTrait<1>]>;
def TestElementwiseBinaryOp :
  BaseElemwiseOp<"elementwise_binary", [ElementwiseNaryOpTrait<2>]>;
def TestElementwiseTernaryOp :
  BaseElemwiseOp<"elementwise_ternary", [ElementwiseNaryOpTrait<3>]>;

//===----------------------------------------------------------------------===//
// Test VectorOnlyTrait and ScalarOnlyTrait Trait.
//===----------------------------------------------------------------------===//
def TestVectorScalarOnlyOp :
  BaseStructuredOp<"vector_scalar_only", [
    VectorOnlyTrait<0>, ScalarOnlyTrait<1>
  ]> {
  let extraClassDeclaration = baseClassDeclaration #
                              baseIteratorTypeClassDeclaration;
}

//===----------------------------------------------------------------------===//
// Test BroadcastableOTF Trait.
//===----------------------------------------------------------------------===//

def TestBroadcastableOTFOp : BaseStructuredOp<"broadcastableOTF",
  [HIVMOpSameOperandsAndResultRank, BroadcastableOTF]> {
  let arguments = (ins Variadic<AnyType>:$src,
                       Variadic<AnyType>:$dst,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let extraClassDeclaration = baseClassDeclaration #
                              baseIteratorTypeClassDeclaration;
  let assemblyFormat = [{
    attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
    (`outs` `(` $dst^  `:` type($dst) `)`)?
    (`broadcast` `=` $broadcast^)?
    (`->` type($result)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Test TransposableOTF Trait.
//===----------------------------------------------------------------------===//
def TestTransposableOTFOp : BaseStructuredOp<"transposableOTF",
  [HIVMOpSameOperandsAndResultRank, TransposableOTF]> {
  let arguments = (ins Variadic<AnyType>:$src,
                       Variadic<AnyType>:$dst,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose);
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let extraClassDeclaration = baseClassDeclaration #
                              baseIteratorTypeClassDeclaration;
  let assemblyFormat = [{
    attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
    (`outs` `(` $dst^  `:` type($dst) `)`)?
    (`transpose` `=` $transpose^)?
    (`->` type($result)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Test OperElemTypeConstraints Trait.
//===----------------------------------------------------------------------===//

class TestElementTypeOnly_BaseOP<string mnemonic, list<Trait> traits = []> :
  BishengIRTest_Op<mnemonic, traits> {
  let arguments = (ins Variadic<AnyType>:$input);
  let assemblyFormat = [{
    attr-dict (`ins` `(` $input^ `:` type($input) `)`)?
  }];
}

def TestElementTypeOnly_F16_I32Op :
  TestElementTypeOnly_BaseOP<"element_type_only_f16_i32",
    [OperElemTypeConstraints<[0], [F16]>,
     OperElemTypeConstraints<[1], [I32]>]>;

def TestElementTypeOnly_F16F32_I8I32Op :
  TestElementTypeOnly_BaseOP<"element_type_only_f16f32_i8i32",
    [OperElemTypeConstraints<[0], [F16, F32]>,
     OperElemTypeConstraints<[1], [I8, I32]>]>;

def TestElementTypeOnly_Int_FloatOp :
  TestElementTypeOnly_BaseOP<"element_type_only_int_float",
    [OperElemTypeConstraints<[0], [AnyInteger]>,
     OperElemTypeConstraints<[1], [AnyFloat]>]>;

#endif // TEST_TESTOPS
