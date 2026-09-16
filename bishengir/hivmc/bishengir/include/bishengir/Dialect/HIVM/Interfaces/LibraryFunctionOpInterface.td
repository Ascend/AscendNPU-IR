//===- LibraryFunctionOpInterface.td - Op w/. Library Fun. -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the LibraryFunctionOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_LIBRARYFUNCTIONOPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_LIBRARYFUNCTIONOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def LibraryFunctionOpInterface : OpInterface<"OpWithLibraryFunction"> {
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the name registered for this op when lowering to an external library call.
      }],
      /*retTy=*/"std::string",
      /*methodName=*/"getOpLibraryCallName",
      /*args=*/(ins "std::optional<bool>":$isOpsAligned),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the max op rank to generate the corresponding library call.
      }],
      /*retTy=*/"std::optional<int>",
      /*methodName=*/"getOpLibraryMaxRank",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if constexpr(ConcreteOp::template hasTrait<::mlir::OpTrait::InferMaxRankTrait::Impl>()) {
          return $_op.inferOpLibraryMaxRank();
        } else if constexpr(ConcreteOp::template hasTrait<::mlir::OpTrait::NoMaxRankTrait::Impl>()) {
          return std::nullopt;
        } else {
          return $_op.getOpLibraryMaxRankImpl();
        }
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the op rank to generate the corresponding library call.
      }],
      /*retTy=*/"int",
      /*methodName=*/"getOpLibraryCallRank",
      /*args=*/(ins "int":$rank),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(rank > 0 && "Invalid operand rank.");
        std::optional<int> maxOpRank = $_op.getOpLibraryMaxRank();
        assert(maxOpRank.has_value() && "Invalid max operand rank.");
        return std::min(rank, maxOpRank.value());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Infer the non-constant max op rank to generate the corresponding library call.
      }],
      /*retTy=*/"int",
      /*methodName=*/"inferOpLibraryMaxRank",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("inferOpLibraryMaxRank not implemented");
      }]
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_LIBRARYFUNCTIONOPINTERFACE_TD
