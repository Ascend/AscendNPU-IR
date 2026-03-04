//===- HACCInterfaces.td -----------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCINTERFACES_TD
#define BISHENGIR_DIALECT_HACC_IR_HACCINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

def HACCFunctionInterface :
  OpInterface<"HACCFunction", [FunctionOpInterface]> {
  let description = [{
    This interfaces provides support for interacting with Heterogeneous
    functions.
  }];
  let cppNamespace = "::mlir::hacc";

  let methods = [
    //===------------------------------------------------------------------===//
    // Query HACC Function attributes.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the HACC function type. If the function has unknown HACC function
        type, return `std::nullopt`.
      }],
      /*retTy=*/"std::optional<hacc::HACCFuncType>",
      /*methodName=*/"getHACCFuncType",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return if the function is a host function.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isHost",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return if the function is a device function.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDevice",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return if the function is a device entry function.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDeviceEntry",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the host function type. If the function is not a host
        function, return `std::nullopt`.
      }],
      /*retTy=*/"std::optional<hacc::HostFuncType>",
      /*methodName=*/"getHostFuncType",
      /*args=*/(ins)
    >,
    //===------------------------------------------------------------------===//
    // Set HACC Function attributes.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Set the function to be a device function.
        Note: disallowed attributes will be automatically dropped.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setDevice",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Set the function to be a device entry function.
        Note: disallowed attributes will be automatically dropped.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setDeviceEntry",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Set the function to be a host function.
        Note: disallowed attributes will be automatically dropped.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setHost",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Set the host function function type.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setHostFuncType",
      /*args=*/(ins "::mlir::hacc::HostFuncType":$funcType)
    >,
    //===------------------------------------------------------------------===//
    // Query HACC Function argument attributes.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return if the function argument at `argIdx` has
        `hacc.arg_type` of `argType`.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isKernelArg",
      /*args=*/(ins "int":$argIdx,
                    "::mlir::hacc::KernelArgType":$argType)
    >,
  ];

  // TODO: verifier does not work with external models, need to find another way.
  let verify = [{
    return detail::verifyHACCFunctionOpInterface($_op);
  }];
}

def HACCTargetDeviceSpecInterface :
  AttrInterface<"HACCTargetDeviceSpecInterface", [TargetDeviceSpecInterface]> {
  let cppNamespace = "::mlir::hacc";
  let description = [{
    This interface provide some methods to query device spec.
  }];
  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the entry related to the given identifier enum"
                      " value, if present.",
      /*retTy=*/"::mlir::DataLayoutEntryInterface",
      /*methodName=*/"getSpecForIdentifierEnum",
      /*args=*/(ins "::mlir::hacc::DeviceSpec":$identifier),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hacc::detail::getSpecImpl($_attr, identifier);
      }]
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HACC_IR_HACCINTERFACES_TD