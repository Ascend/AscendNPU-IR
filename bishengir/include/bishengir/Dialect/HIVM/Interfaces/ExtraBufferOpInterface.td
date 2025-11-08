//===- ExtraBufferOpInterface.td - Ops with Temp. Buffer ---*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the ExtraBufferOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ExtraBufferOpInterface : OpInterface<"ExtraBufferOpInterface"> {
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Allocate extra temp buffers for the current operation.
      }],
      /*retTy=*/"llvm::LogicalResult",
      /*methodName=*/"allocExtraBuffersIfPossible",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the size of the extra buffer. Returns std::nullopt if the op doesn't
        require extra buffer.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getExtraBufferSize",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the extra temp buffers.
      }],
      /*retTy=*/"mlir::OperandRange",
      /*methodName=*/"getExtraBuffers",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Should allocate extra buffer for scalar or OTF Brc.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"shouldAllocExtraBufferForScalarOrOTFBrc",
      /*args=*/(ins)
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
