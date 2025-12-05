//===- ScopeOps.td - Scope dialect operation definitions ---*- tablegen -*-===//
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
// This is the definition file for Scope dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SCOPE_IR_SCOPEOPS_TD
#define BISHENGIR_DIALECT_SCOPE_IR_SCOPEOPS_TD

include "bishengir/Dialect/Scope/IR/ScopeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Scope Dialect operations.
//===----------------------------------------------------------------------===//

def ScopeOp : Scope_Op<"scope", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"scope::ReturnOp">,
    SingleBlock]> {
  let summary = "Represents scope of a region";
  let description = [{
    The "scope.scope" operation represents a scope of the
    operations inside the region.

    Example:
    ```
    scope.scope : () -> () {
        scope.return
    } {tcore_type = #hivm.tcore_type<CUBE>, ...}

    scope.scope : () -> () {
        scope.return
    }
    ```
  }];
  let arguments = (ins
    UnitAttr:$no_inline
  );
  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

def ReturnOp : Scope_Op<"return", [
    HasParent<"ScopeOp">,
    Pure,
    ReturnLike,
    Terminator]> {
  let summary = "Return operation inside of scope.scope operation";
  let description = [{
    The `scope.return` operation as a terminator of scope.scope region.

    Example:
    ```
    scope.return
    ```
  }];

  let assemblyFormat =
      [{ attr-dict }];
}

#endif // BISHENGIR_DIALECT_SCOPE_IR_SCOPEOPS_TD
