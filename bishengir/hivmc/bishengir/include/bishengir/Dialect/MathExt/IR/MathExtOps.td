//===- MathExtOps.td - Math ext op definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Math ext dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MATHEXT_OPS
#define MATHEXT_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/Math/IR/MathBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"

include "bishengir/Dialect/MathExt/IR/MathExtBase.td"

// Base class for math dialect ops. Ops in this dialect have no side effects and
// can be applied element-wise to vectors and tensors.
class MathExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MathExt_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for unary math operations on floating point types. Require an
// operand and result of the same type. This type can be a floating point type,
// vector or tensor thereof.
class MathExt_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

// Base class for binary math operations on floating point types. Require two
// operands and one result of the same type. This type can be a floating point
// type, vector or tensor thereof.
class MathExt_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<Arith_FastMathAttr,
                        "::mlir::arith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

//===----------------------------------------------------------------------===//
// IlogbOp
//===----------------------------------------------------------------------===//

def MathExt_IlogbOp : MathExt_FloatUnaryOp<"ilogb"> {
  let summary = "an exponent components of a floating-point number";
  let description = [{
    ilogb(x) := floor(log2(abs(x)))

    The `ilogb` operation takes one operand of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. It has no
    standard attributes.

  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// LdexpOp
//===----------------------------------------------------------------------===//

def MathExt_LdexpOp : MathExt_FloatBinaryOp<"ldep"> {
  let summary = "a fractional components of a floating-point number";
  let description = [{
    ldep(x) = x * (ilogb(x) + 1)^(-1)

    The `ldexp` operation takes two operands of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. Operands
    must have the same type.

  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DivFHPOp
//===----------------------------------------------------------------------===//

def MathExt_DivFHPOp : MathExt_FloatBinaryOp<"divfhp"> {
  let summary = "a a high precision division";
  let description = [{
    Computes division recovering last ULPs
  }];
  let hasFolder = 1;
}

#endif // MATHEXT_OPS
