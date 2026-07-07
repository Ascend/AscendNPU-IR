//===- FlattenInterface.td ---------------------------------*- tablegen -*-===//
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
// This is the definition file for the FlattenInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_TD

include "mlir/IR/OpBase.td"

def FlattenInterface : OpInterface<"FlattenInterface"> {
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the HIVM Layout reassociation after flattening.
        Return identity reassociation if not flattenable.
      }],
      /*retTy=*/"llvm::FailureOr<::mlir::hivm::FlattenResult>",
      /*methodName=*/"getFlattened",
      /*args=*/(ins "::mlir::hivm::FlattenOptions":$flattenOptions),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getFlattenedImpl($_op, flattenOptions);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        This returns the axes that cannot be collapsed with its
        adjacent dimension, even though it's a unit dimension.
      }],
      /*retTy=*/"llvm::SmallVector<int64_t>",
      /*methodName=*/"getLimitedAxes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("get limited axes is not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        This function adjust target dimensions which semantics changed after collapsing.
      }],
      /*retTy=*/"void",
      /*methodName=*/"adjustTargetDimensions",
      /*args=*/(ins "::mlir::OpBuilder&":$builder, "const FlattenResult&":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return;
      }]
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_TD
