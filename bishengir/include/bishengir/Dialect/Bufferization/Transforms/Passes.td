//===- Passes.td - Bufferization pass definition file ------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def NormalizeToTensorOp : Pass<"normalize-to-tensor", "mlir::func::FuncOp"> {
  let summary = "Normalize bufferization.to_tensor ops.";
  let description = [{
    Normalize `bufferization.to_tensor` ops before bufferization:

    1. Isolate the memory-space information carried by the memref operand
       by inserting a `memref.memory_space_cast` before the `to_tensor` when
       the memref has a memory space, keeping the memory-space attr out of
       the tensor bufferization path.

    2. Expose memref-level writes (e.g., hivm.hir.load) to tensor-level
       analysis by inserting a `bufferization.alloc_tensor(copy...)` for the
       toTensorOp and replacing all users of the toTensorOp. This makes
       memref-level writes visible to One-Shot Bufferization analysis,
       enabling RAW conflict detection and correct copy insertion.
  }];
  let constructor = "bishengir::bufferization::createNormalizeToTensorOpPass()";
  let dependentDialects = ["mlir::bufferization::BufferizationDialect",
                           "mlir::memref::MemRefDialect"];
}

def FlipIf : Pass<"bufferization-flip-if", "mlir::func::FuncOp"> {
  let summary = "Temporarily flip tensor state updates around bufferization";
  let constructor = "bishengir::bufferization::createFlipIfPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect"];
  let options = [Option<"restoreOnly", "restore-only", "bool", "false",
      "Restore marked branches instead of finding new tensor state updates">];
}

def TensorCopyInsertion : Pass<"tensor-copy-insertion", "mlir::ModuleOp"> {
  let summary = "Run One-Shot analysis and insert tensor copies to resolve "
                "bufferization conflicts";
  let description = [{
    This pass runs One-Shot Bufferize analysis (without bufferizing) and
    inserts `bufferization.alloc_tensor` copies wherever a conflict (e.g.,
    RaW) was detected. It is equivalent to the analyze + insertTensorCopies
    phase of `one-shot-bufferize`, but without the final bufferization step.

    Running this pass before `one-shot-bufferize` (with `copy-before-write`
    or the default analyze+copy path) allows a second round of analysis to
    catch conflicts that were exposed by the first round's copy insertion.
  }];
  let constructor = "bishengir::bufferization::createTensorCopyInsertionPass()";
  let dependentDialects = ["mlir::bufferization::BufferizationDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::tensor::TensorDialect",
  ];
  let options =
      [Option<"allowReturnAllocsFromLoops", "allow-return-allocs-from-loops",
              "bool", /*default=*/"false",
              "Allows returning/yielding new allocations from a loop.">,
       Option<"allowUnknownOps", "allow-unknown-ops", "bool",
              /*default=*/"false",
              "Allows unknown (not bufferizable) ops in the input IR.">,
       Option<"analysisHeuristic", "analysis-heuristic", "std::string",
              /*default=*/"\"bottom-up\"",
              "Heuristic that control the IR traversal during analysis">,
       Option<"bufferizeFunctionBoundaries", "bufferize-function-boundaries",
              "bool", /*default=*/"false",
              "Bufferize function boundaries (experimental).">,
  ];
}

#endif // BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_TD
