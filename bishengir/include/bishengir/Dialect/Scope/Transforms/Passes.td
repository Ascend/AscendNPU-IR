//===- Passes.td - Scope pass definition file --------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def OutlineScope : Pass<"outline-scope", "mlir::ModuleOp"> {
  let summary = "Outline scope region within ScopeOp.";
  let description = [{
    Convert `scope.scope` into a `func.func`.

    Example:

    ```mlir
    module {
      func.func @test() {
        scope.scope : () -> () {
          ...
          scope.return
        } {tcore_type = #hivm.tcore_type<CUBE>, ...}
        scope.scope : () -> () {
          ...
          scope.return
        } {tcore_type = #hivm.tcore_type<VECTOR>, ...}
        return
      }
    }
    ```

    will be transformed into:

    ```mlir
    module {
      func.func @test_scope_0() attributes {tcore_type = #hivm.tcore_type<CUBE>, ...} {
        ...
        return
      }
      func.func @test_scope_1() attributes {tcore_type = #hivm.tcore_type<VECTOR>, ...} {
        ...
        return
      }
      func.func @test() {
        call @test_scope_scope_scope_0() : () -> ()
        call @test_scope_scope_scope_1() : () -> ()
        return
      }
    }
    ```
  }];
  let constructor = "mlir::scope::createOutlineScopePass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def InlineScope : Pass<"inline-scope", "mlir::ModuleOp"> {
  let summary = "Inline scope region within ScopeOp.";
  let description = [{
    Move out operations inside the scope region to its parent
    region if `scope.scope` doesn't have the `no_inline` attribute.
    Example:

    ```mlir
    func.func @test() {
      scope.scope : () -> () {
        ...
        scope.return
      } {no_inline}
      scope.scope : () -> () {
        <inlinable_operations>
        scope.return
      }
      return
    }
    ```

    will be transformed into:

    ```mlir
    func.func @test() {
      scope.scope : () -> () {
        ...
        scope.return
      } {no_inline}
      <inlinable_operations>
      return
    }
    ```
  }];
  let constructor = "mlir::scope::createInlineScopePass()";
  let dependentDialects = [];
}

#endif // BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_TD
