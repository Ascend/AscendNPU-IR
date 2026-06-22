//===- LibraryFunctionOpInterface.td - Op w/. Library Fun. -*- tablegen -*-===//
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
        llvm::report_fatal_error("getOpLibraryMaxRank not implemented");
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
        llvm::report_fatal_error("inferOpLibraryMaxRank not implemented");
      }]
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_LIBRARYFUNCTIONOPINTERFACE_TD
