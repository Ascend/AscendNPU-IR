//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LegalizeLoopIterArgs : Pass<"hivm-legalize-loop-iter-arg"> {
  let summary = "Legalize vector typed iter args.";
  let constructor = "mlir::scf::createLegalizeLoopIterArgsPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "scf::SCFDialect"];
}

def MapForToForall
    : Pass<"map-for-to-forall", "func::FuncOp"> {
  let constructor = "mlir::scf::createMapForToForallPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
  let options = [
    Option<"simtMode", "simt-mode", "bool", "false",
      "Activate for SIMT mode compilation">,
  ];
}

def RemoveRedundantLoopInit
    : Pass<"scf-remove-redundant-loop-init"> {
  let constructor = "mlir::scf::createRemoveRedundantLoopInitPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def CanonicalizeIterArg
    : Pass<"scf-canonicalize-iter-arg", "func::FuncOp"> {
  let constructor = "mlir::scf::createCanonicalizeIterArgPass()";
  let options = [Option<"enableRemoveDeadIterArgBackwardPatterns",
                        "enable-remove-dead-iter-arg-bwd-patterns", "bool",
                        "true", "Enable RemoveDeadIterArgBackward patterns">];
  let dependentDialects = [
    "scf::SCFDialect"
  ];
}

#endif // BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_TD
