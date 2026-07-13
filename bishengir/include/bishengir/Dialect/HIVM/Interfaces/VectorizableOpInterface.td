//===- VectorizableOpInterface.td - Ops to vectorize -------*- tablegen -*-===//
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
// This is the definition file for the VectorizableOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def VectorizableOpInterface : OpInterface<"VectorizableOpInterface"> {
  let description = [{
    Interface for operations that needs to be vectorized.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to vectorize the operation.
      }],
      /*retType=*/"::llvm::LogicalResult",
      /*methodName=*/"vectorize",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter, "::llvm::ArrayRef<int64_t>":$vectorSizes),
      /*methodBody=*/""
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_TD
