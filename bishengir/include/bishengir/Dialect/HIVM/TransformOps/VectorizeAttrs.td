//===- VectorizeAttrs.td - Vectorizer metadata definitions --*- tablegen-*-===//
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

#ifndef BISHENGIR_HIVM_VECTORIZE_ATTRS_TD
#define BISHENGIR_HIVM_VECTORIZE_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def HIVMVectorize_Dialect : Dialect {
  let name = "hivm_vectorize";
  let cppNamespace = "::mlir::hivm";
  let useDefaultAttributePrinterParser = 1;
}

class HIVMVectorize_Attr<string attrName, string attrMnemonic>
    : AttrDef<HIVMVectorize_Dialect, attrName> {
  let mnemonic = attrMnemonic;
}

class HIVMVectorize_I32Enum<string name, string description,
                            list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hivm";
}

def HIVM_GroupsAttr : HIVMVectorize_Attr<"Groups", "groups"> {
  let parameters = (ins ArrayRefParameter<"DenseI64ArrayAttr">:$groups);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def HIVM_PlanAttr : HIVMVectorize_Attr<"Plan", "plan"> {
  let parameters = (ins "FloatAttr":$cost, "GroupsAttr":$groups);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def HIVM_TransactionOutcome_Succeeded
    : I32EnumAttrCase<"succeeded", 0, "succeeded">;
def HIVM_TransactionOutcome_Rejected
    : I32EnumAttrCase<"rejected", 1, "rejected">;
def HIVM_TransactionOutcome_Error : I32EnumAttrCase<"error", 2, "error">;

def HIVM_TransactionOutcomeEnum
    : HIVMVectorize_I32Enum<"TransactionOutcome", "transaction attempt outcome",
                            [HIVM_TransactionOutcome_Succeeded,
                             HIVM_TransactionOutcome_Rejected,
                             HIVM_TransactionOutcome_Error]>;

def HIVM_TransactionVerdictState_Applied
    : I32EnumAttrCase<"applied", 0, "applied">;
def HIVM_TransactionVerdictState_Eligible
    : I32EnumAttrCase<"eligible", 1, "eligible">;
def HIVM_TransactionVerdictState_Rejected
    : I32EnumAttrCase<"rejected", 2, "rejected">;
def HIVM_TransactionVerdictState_Error : I32EnumAttrCase<"error", 3, "error">;
def HIVM_TransactionVerdictState_Invalid
    : I32EnumAttrCase<"invalid", 4, "invalid">;

def HIVM_TransactionVerdictStateEnum
    : HIVMVectorize_I32Enum<"TransactionVerdictState",
                            "transaction candidate verdict state",
                            [HIVM_TransactionVerdictState_Applied,
                             HIVM_TransactionVerdictState_Eligible,
                             HIVM_TransactionVerdictState_Rejected,
                             HIVM_TransactionVerdictState_Error,
                             HIVM_TransactionVerdictState_Invalid]>;

def HIVM_TransactionTicketAttr
    : HIVMVectorize_Attr<"TransactionTicket", "transaction_ticket"> {
  let parameters = (ins "uint64_t":$session, "uint64_t":$ticket);
  let assemblyFormat = "`<` struct(params) `>`";
}

def HIVM_TransactionResultAttr
    : HIVMVectorize_Attr<"TransactionResult", "transaction_result"> {
  let parameters = (ins "uint64_t":$session, "uint64_t":$ticket,
      EnumParameter<HIVM_TransactionOutcomeEnum>:$outcome,
      OptionalParameter<"Attribute">:$options,
      OptionalParameter<"Attribute">:$result,
      OptionalParameter<"StringAttr">:$diagnostics);
  let assemblyFormat = "`<` struct(params) `>`";
}

def HIVM_TransactionVerdictAttr
    : HIVMVectorize_Attr<"TransactionVerdict", "transaction_verdict"> {
  let parameters = (ins "uint64_t":$ticket,
      EnumParameter<HIVM_TransactionVerdictStateEnum>:$state,
      OptionalParameter<"FloatAttr">:$cost,
      OptionalParameter<"StringAttr">:$reason);
  let assemblyFormat = "`<` struct(params) `>`";
}

def HIVM_TransactionSelectionAttr
    : HIVMVectorize_Attr<"TransactionSelection", "transaction_selection"> {
  let parameters = (ins "BoolAttr":$applied,
      OptionalParameter<"uint64_t">:$ticket,
      OptionalParameter<"FloatAttr">:$cost,
      OptionalParameter<"ArrayAttr">:$verdicts,
      OptionalParameter<"StringAttr">:$diagnostics);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif
