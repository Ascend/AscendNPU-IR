//===- LocalMatmulLikeOpInterface.td -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the LocalMatmulLikeOpInterface (L0 and L1).
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Base Interface - Common matmul operations
//===----------------------------------------------------------------------===//

def HIVMLocalMatmulBaseOpInterface : OpInterface<"LocalMatmulBaseOpInterface"> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
    Base interface for all local matmul-like operations.
    Provides core operand access (A/B/C), real dimensions (M/K/N),
    init condition, basic bias mode queries, and HF32/I4 enable queries.

    This interface defines the minimal common functionality shared by
    both L0 (computation-only) and L1 (DMA + computation) operations.
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
      /*desc=*/"Set the local matmul A operand.",
      /*retTy=*/"void",
      /*methodName=*/"setMatmulA",
      /*args=*/(ins "::mlir::Value":$a),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getAMutable().assign(a); }]>,

    InterfaceMethod<
      /*desc=*/"Set the local matmul B operand.",
      /*retTy=*/"void",
      /*methodName=*/"setMatmulB",
      /*args=*/(ins "::mlir::Value":$b),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getBMutable().assign(b); }]>,

    InterfaceMethod<
      /*desc=*/"Set the local matmul C/init operand.",
      /*retTy=*/"void",
      /*methodName=*/"setMatmulC",
      /*args=*/(ins "::mlir::Value":$c),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getCMutable().assign(c); }]>,

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
      /*desc=*/"Return whether scale operands are present.",
      /*retTy=*/"bool",
      /*methodName=*/"supportsScaleOperands",
      /*args=*/(ins)>,

    InterfaceMethod<
      /*desc=*/"Return whether HF32 mode is enabled.",
      /*retTy=*/"bool",
      /*methodName=*/"isHF32Enabled",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getEnable_HF32().has_value(); }]>,

    InterfaceMethod<
      /*desc=*/"Return whether I4 conversion is enabled.",
      /*retTy=*/"bool",
      /*methodName=*/"isI4Enabled",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getEnable_I4().has_value(); }]>
  ];
}
//===----------------------------------------------------------------------===//
// L0 Interface - Core computation with hardware controls
//===----------------------------------------------------------------------===//

def HIVMLocalMatmulL0OpInterface
    : OpInterface<"LocalMatmulL0OpInterface", [HIVMLocalMatmulBaseOpInterface]> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
    Interface for local matmul-like ops at L0 level.
    Inherits operand access and HF32/I4 queries from the base interface and
    adds L0-specific hardware controls: K-direction alignment and is-with-bias.

    Ops at L0 level operate on data already present in local memory
    (registers, scratchpad) and expose low-level hardware configuration knobs.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the K-direction alignment flag.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getKDirectionAlign",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getKDirectionAlign(); }]>,

    InterfaceMethod<
      /*desc=*/"Return the is-with-bias flag.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getIsWithBias",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getIsWithBias(); }]>,

    InterfaceMethod<
      /*desc=*/"Set the K-direction alignment flag.",
      /*retTy=*/"void",
      /*methodName=*/"setKDirectionAlign",
      /*args=*/(ins "::mlir::Value":$align),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getKDirectionAlignMutable().assign(align); }]>,

    InterfaceMethod<
      /*desc=*/"Set the is-with-bias flag.",
      /*retTy=*/"void",
      /*methodName=*/"setIsWithBias",
      /*args=*/(ins "::mlir::Value":$bias),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getIsWithBiasMutable().assign(bias); }]>
  ];



}

//===----------------------------------------------------------------------===//
// L1 Interface - DMA + computation abstraction
//===----------------------------------------------------------------------===//

def HIVMLocalMatmulL1OpInterface
    : OpInterface<"LocalMatmulL1OpInterface", [HIVMLocalMatmulBaseOpInterface]> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
    Interface for local matmul-like ops at L1 level.
    Inherits basic operand access from base interface and adds DMA-related
    features including transpose attributes, per-channel bias operand,
    and layout transformation methods.

    Ops at L1 level handle data movement from global memory (DRAM) to
    local memory, potentially with layout transformations and transpose
    operations during the transfer.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the optional per-channel bias operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMatmulPerChannelBias",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return $_op.getPerChannelBias(); }]>,

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
      /*desc=*/"Set the local matmul A transpose attribute.",
      /*retTy=*/"void",
      /*methodName=*/"setMatmulATranspose",
      /*args=*/(ins "::mlir::UnitAttr":$transpose),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.setATransposeAttr(transpose); }]>,

    InterfaceMethod<
      /*desc=*/"Set the local matmul B transpose attribute.",
      /*retTy=*/"void",
      /*methodName=*/"setMatmulBTranspose",
      /*args=*/(ins "::mlir::UnitAttr":$transpose),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.setBTransposeAttr(transpose); }]>,

    InterfaceMethod<
      /*desc=*/"Set the optional per-channel bias operand.",
      /*retTy=*/"void",
      /*methodName=*/"setPerChannelBias",
      /*args=*/(ins "::mlir::Value":$bias),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ $_op.getPerChannelBiasMutable().assign(bias); }]>,

    InterfaceMethod<
      /*desc=*/"Return whether bias should be decomposed by elementwise add.",
      /*retTy=*/"bool",
      /*methodName=*/"shouldDecomposeBiasByElementAdd",
      /*args=*/(ins)>,

    InterfaceMethod<
      /*desc=*/[{
        Return whether a loop-carried bias should be decomposed by hoisting a
        zero fill out of the loop and adding the bias back after it. Only the
        mem-based path detects this mode, so ops that never see it (MmadMx)
        keep the default.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"shouldDecomposeBiasByCrossLoopElementAdd",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return false; }]>,

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

//===----------------------------------------------------------------------===//
// Backward Compatibility Alias
//===----------------------------------------------------------------------===//
def HIVMLocalMatmulLikeOpInterface
    : OpInterface<"LocalMatmulLikeOpInterface", [HIVMLocalMatmulL1OpInterface]> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
    Interface for local matmul-like ops that share A/B/C, init-condition,
    real-MKN, bias-mode, and layout behavior.

    Ops implementing this interface should use `HIVMLocalMatmulLikeOpTraits`,
    which bundles the common traits required alongside this interface.

    NOTE: This interface is now implemented as an alias to HIVMLocalMatmulL1OpInterface.
    New L0-level operations should use HIVMLocalMatmulL0OpInterface instead.
  }];

  let methods = [];  // All methods inherited from L1 interface
}
//I am not sure if this method is needed


#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_LOCALMATMULLIKEOPINTERFACE_TD
