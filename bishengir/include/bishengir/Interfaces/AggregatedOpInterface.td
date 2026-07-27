//===- AggregatedOpInterface.td - Ops to decompose ---------*- tablegen -*-===//
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
// This is the definition file for the BiShengIR custom AggregatedOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD
#define BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def BiShengIRAggregatedOpInterface : OpInterface<"BiShengIRAggregatedOpInterface"> {
  let description = [{
    Interface for decomposing aggregated operations into a sequence of simpler
    ops.
  }];
  let cppNamespace = "::bishengir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to decompose the operation into simpler operations.
      }],
      /*retType=*/"mlir::FailureOr<llvm::SmallVector<mlir::Value>>",
      /*methodName=*/"decomposeOperation",
      /*args=*/(ins
          "mlir::PatternRewriter &":$b),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the phase for decomposing the operation.
      }],
      /*retType=*/"::bishengir::DecomposePhase",
      /*methodName=*/"getDecomposePhase",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::bishengir::DecomposePhase::NO_CONSTRAINT;
      }]
    >
  ];
}

#endif // BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD
