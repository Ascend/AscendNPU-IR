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
    `auto-scope` and `inline-func-to-scope` and removed by `outline-scope`.
  }];
}

def Scope_SymFuncNameAttr : Scope_Attr<"SymFuncName", "sym_func_name"> {
  let parameters = (ins "::mlir::StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
  let description = [{
    Records the symbol name of the function a `scope.scope` was created from
    (e.g. by `inline-func-to-scope`). `outline-scope` uses it as the name of the
    outlined function when it does not clash with an existing symbol.
  }];
}

def Scope_CallAttrsVecAttr : Scope_Attr<"CallAttrsVec", "call_attrs_vec"> {
  let parameters = (ins "::mlir::DictionaryAttr":$attrs);
  let assemblyFormat = "`<` $attrs `>`";
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<::mlir::NamedAttribute>":$attrs), [{
      return $_get($_ctxt, ::mlir::DictionaryAttr::get($_ctxt, attrs));
    }]>
  ];
  let description = [{
    Records the call-site attributes of the `func.call` a `scope.scope` was
    created from (e.g. by `inline-func-to-scope`). `outline-scope` restores them
    on the recreated call.
  }];
}

#endif // BISHENGIR_DIALECT_SCOPE_IR_SCOPEATTRS_TD
