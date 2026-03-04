//===- OpLayoutInterface.td - Ops with Layout Information --*- tablegen -*-===//
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
// This is the definition file for the OpLayoutInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD

include "mlir/IR/OpBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

def OpLayoutInterface : OpInterface<"OpWithLayoutInterface"> {
  let description = [{
     Interface to query characteristics of HIVM Ops that have specific
     data layout requirements.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the target data layouts for memref-typed operands.",
    /*retTy=*/"llvm::SmallDenseMap<mlir::Value, mlir::hivm::DataLayoutAttr>",
    /*methodName=*/"getOperandsTargetLayout",
    /*args=*/(ins)
    >,
    InterfaceMethod<
    /*desc=*/"Returns the current data layouts for memref-typed operands.",
    /*retTy=*/"llvm::SmallDenseMap<mlir::Value, mlir::hivm::DataLayoutAttr>",
    /*methodName=*/"getOperandsCurrentLayout",
    /*args=*/(ins)
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD
