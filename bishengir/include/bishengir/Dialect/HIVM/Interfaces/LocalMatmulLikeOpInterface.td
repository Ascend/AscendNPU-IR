//===- LocalMatmulLikeOpInterface.td -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the LocalMatmulLikeOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def HIVMLocalMatmulLikeOpInterface
    : OpInterface<"LocalMatmulLikeOpInterface"> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
    Interface for local matmul-like ops that share A/B/C, init-condition,
    real-MKN, bias-mode, and layout behavior.

    Ops implementing this interface should use `HIVMLocalMatmulLikeOpTraits`,
    which bundles the common traits required alongside this interface.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the local matmul A operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulA",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getA(); }]>,
    InterfaceMethod<
      /*desc=*/"Return the local matmul B operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulB",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getB(); }]>,
    InterfaceMethod<
      /*desc=*/"Return the local matmul C/init operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulC",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getC(); }]>,
    InterfaceMethod<
      /*desc=*/"Return the optional per-channel bias operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulPerChannelBias",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getPerChannelBias(); }]>,
    InterfaceMethod<
      /*desc=*/"Return the local matmul init condition operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulInitCondition",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getInitCondition(); }]>,
    InterfaceMethod<
      /*desc=*/"Return the local matmul real-M operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulRealM",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getRealM(); }]>,
    InterfaceMethod<
      /*desc=*/"Return the local matmul real-K operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulRealK",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getRealK(); }]>,
    InterfaceMethod<
      /*desc=*/"Return the local matmul real-N operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulRealN",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getRealN(); }]>,
    InterfaceMethod<
      /*desc=*/"Return whether A is transposed.",
      /*retTy=*/"bool",
      /*methodName=*/"isMatmulATransposed",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getATranspose().has_value(); }]>,
    InterfaceMethod<
      /*desc=*/"Return whether B is transposed.",
      /*retTy=*/"bool",
      /*methodName=*/"isMatmulBTransposed",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getBTranspose().has_value(); }]>,
    InterfaceMethod<
      /*desc=*/"Return whether init condition is a matching constant.",
      /*retTy=*/"bool",
      /*methodName=*/"isInitConstant",
      /*args=*/(ins "::std::optional<bool>":$cst)>,
    InterfaceMethod<
      /*desc=*/"Set the init condition operand.",
      /*retTy=*/"void",
      /*methodName=*/"setInitCondition",
      /*args=*/(ins "::mlir::Value":$init)>,
    InterfaceMethod<
      /*desc=*/"Set the local matmul C/init operand.",
      /*retTy=*/"void",
      /*methodName=*/"setMatmulC",
      /*args=*/(ins "::mlir::Value":$c),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getCMutable().assign(c); }]>,
    InterfaceMethod<
      /*desc=*/"Set the optional per-channel bias operand.",
      /*retTy=*/"void",
      /*methodName=*/"setPerChannelBias",
      /*args=*/(ins "::mlir::Value":$bias),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getPerChannelBiasMutable().assign(bias); }]>,
    InterfaceMethod<
      /*desc=*/"Set the real-M/K/N operands.",
      /*retTy=*/"void",
      /*methodName=*/"setMatmulRealMKN",
      /*args=*/(ins "::mlir::Value":$realM, "::mlir::Value":$realK,
                    "::mlir::Value":$realN),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        $_op.getRealMMutable().assign(realM);
        $_op.getRealKMutable().assign(realK);
        $_op.getRealNMutable().assign(realN);
      }]>,
    InterfaceMethod<
      /*desc=*/"Return the detected matmul bias mode.",
      /*retTy=*/"::mlir::hivm::MatmulBiasMode",
      /*methodName=*/"getMatmulBiasMode",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return whether bias should be decomposed by elementwise add.",
      /*retTy=*/"bool",
      /*methodName=*/"shouldDecomposeBiasByElementAdd",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return whether scale operands are present.",
      /*retTy=*/"bool",
      /*methodName=*/"supportsScaleOperands",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return whether target layouts use BoolAttr as memory-space marker.",
      /*retTy=*/"bool",
      /*methodName=*/"usesBoolAttrForTargetLayout",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return A/B block sizes for target layouts.",
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getMatmulBlockSizesTile",
      /*args=*/(ins "::mlir::Value":$oper, "bool":$isTranspose, "bool":$isA)>,
    InterfaceMethod<
      /*desc=*/"Return A operand current layout.",
      /*retTy=*/"::mlir::FailureOr<::mlir::hivm::DataLayoutAttr>",
      /*methodName=*/"getOperandALayout",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getLocalMatmulOperandALayoutImpl($_op.getOperation());
      }]>,
    InterfaceMethod<
      /*desc=*/"Return B operand current layout.",
      /*retTy=*/"::mlir::FailureOr<::mlir::hivm::DataLayoutAttr>",
      /*methodName=*/"getOperandBLayout",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getLocalMatmulOperandBLayoutImpl($_op.getOperation());
      }]>,
    InterfaceMethod<
      /*desc=*/"Return C operand current layout.",
      /*retTy=*/"::mlir::FailureOr<::mlir::hivm::DataLayoutAttr>",
      /*methodName=*/"getOperandCLayout",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getLocalMatmulOperandCLayoutImpl($_op.getOperation());
      }]>,
    InterfaceMethod<
      /*desc=*/"Return common A/B/C current layouts.",
      /*retTy=*/"::llvm::SmallDenseMap<::mlir::Value, ::mlir::hivm::DataLayoutAttr>",
      /*methodName=*/"getMatmulOperandsCurrentLayout",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getLocalMatmulOperandsCurrentLayoutImpl($_op.getOperation());
      }]>,
    InterfaceMethod<
      /*desc=*/"Return common A/B/C target layouts.",
      /*retTy=*/"::llvm::SmallDenseMap<::mlir::Value, ::mlir::hivm::DataLayoutAttr>",
      /*methodName=*/"getMatmulOperandsTargetLayout",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getLocalMatmulOperandsTargetLayoutImpl($_op.getOperation());
      }]>
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_TD
