//===- ScopeAttrs.td - Scope dialect attributes definitions -*- tablegen -*-===//
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
// This is the definition file for Scope dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SCOPE_IR_SCOPEATTRS_TD
#define BISHENGIR_DIALECT_SCOPE_IR_SCOPEATTRS_TD

include "bishengir/Dialect/Scope/IR/ScopeBase.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Scope Attributes.
//===----------------------------------------------------------------------===//

class Scope_Attr<string attrName, string attrMnemonic,
                 list<Trait> traits = []>
    : AttrDef<Scope_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def Scope_OutlineAttr : Scope_Attr<"Outline", "outline"> {
  let description = [{
    Unit attribute marking a `scope.scope` that should be outlined into a
    function by `outline-scope{outline-marked-scopes-only}`. It is set by
    `auto-scope` and removed by `outline-scope`.
  }];
}

#endif // BISHENGIR_DIALECT_SCOPE_IR_SCOPEATTRS_TD
