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
                           "mlir::func::FuncDialect"];
  let options =
      [Option<"allowReturnAllocsFromLoops", "allow-return-allocs-from-loops",
              "bool", /*default=*/"false",
              "Allows returning/yielding new allocations from a loop.">,
       Option<"allowUnknownOps", "allow-unknown-ops", "bool",
              /*default=*/"false",
              "Allows unknown (not bufferizable) ops in the input IR.">,
       Option<
           "analysisFuzzerSeed", "analysis-fuzzer-seed", "unsigned",
           /*default=*/"0",
           "Test only: Analyze ops in random order with a given seed (fuzzer)">,
       Option<"analysisHeuristic", "analysis-heuristic", "std::string",
              /*default=*/"\"bottom-up\"",
              "Heuristic that control the IR traversal during analysis">,
       Option<"bufferizeFunctionBoundaries", "bufferize-function-boundaries",
              "bool", /*default=*/"false",
              "Bufferize function boundaries (experimental).">,
       Option<"checkParallelRegions", "check-parallel-regions", "bool",
              /*default=*/"true",
              "Account for parallel regions in RaW analysis.">,
       Option<"copyBeforeWrite", "copy-before-write", "bool",
              /*default=*/"false",
              "Skip the analysis. Make a buffer copy on every write.">,
       ListOption<"dialectFilter", "dialect-filter", "std::string",
                  "Restrict bufferization to ops from these dialects.">,
       Option<"dumpAliasSets", "dump-alias-sets", "bool", /*default=*/"false",
              "Test only: Annotate tensor IR with alias sets">,
       ListOption<"noAnalysisFuncFilter", "no-analysis-func-filter",
                  "std::string",
                  "Skip analysis of functions with these symbol names."
                  "Set copyBeforeWrite to true when bufferizing them.">,
       Option<"mustInferMemorySpace", "must-infer-memory-space", "bool",
              /*default=*/"false",
              "The memory space of an memref types must always be inferred. If "
              "unset, a default memory space of 0 is used otherwise.">,
       Option<"testAnalysisOnly", "test-analysis-only", "bool",
              /*default=*/"false",
              "Test only: Only run inplaceability analysis and annotate IR">,
       Option<"printConflicts", "print-conflicts", "bool",
              /*default=*/"false",
              "Test only: Annotate IR with RaW conflicts. Requires "
              "test-analysis-only.">,
  ];

  let statistics = [Statistic<"numTensorInPlace", "num-tensor-in-place",
                              "Number of in-place tensor OpOperands">,
                    Statistic<"numTensorOutOfPlace", "num-tensor-out-of-place",
                              "Number of out-of-place tensor OpOperands">,
  ];
}

def BufferizeModuleOp : Pass<"bufferize-module-op", "mlir::ModuleOp"> {
  let summary = "Bufferize a module and its function boundaries";
  let description = [{
    This pass rewrites tensor IR to memref IR, bufferizing the module and
    its function boundaries. It is the bufferization (rewrite) phase of
    `one-shot-bufferize` and typically runs after `tensor-copy-insertion`,
    which performs the inplaceability analysis and inserts the required
    `bufferization.alloc_tensor` copies. Running the two passes separately
    avoids re-running the analysis.

    This pass always bufferizes function boundaries (the equivalent of
    `bufferize-function-boundaries` in `one-shot-bufferize`).
  }];
  let constructor = "bishengir::bufferization::createBufferizeModuleOpPass()";
  let dependentDialects = ["mlir::bufferization::BufferizationDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::func::FuncDialect"];
  let options =
      [Option<"allowUnknownOps", "allow-unknown-ops", "bool",
              /*default=*/"false",
              "Allows unknown (not bufferizable) ops in the input IR.">,
       Option<"copyBeforeWrite", "copy-before-write", "bool",
              /*default=*/"false",
              "Skip the analysis. Make a buffer copy on every write.">,
       ListOption<"dialectFilter", "dialect-filter", "std::string",
                  "Restrict bufferization to ops from these dialects.">,
       Option<"functionBoundaryTypeConversion",
              "function-boundary-type-conversion", "std::string",
              /*default=*/"\"infer-layout-map\"",
              "Controls layout maps when bufferizing function signatures.">,
       Option<"mustInferMemorySpace", "must-infer-memory-space", "bool",
              /*default=*/"false",
              "The memory space of an memref types must always be inferred. If "
              "unset, a default memory space of 0 is used otherwise.">,
       ListOption<"noAnalysisFuncFilter", "no-analysis-func-filter",
                  "std::string",
                  "Skip analysis of functions with these symbol names."
                  "Set copyBeforeWrite to true when bufferizing them.">,
       Option<"unknownTypeConversion", "unknown-type-conversion", "std::string",
              /*default=*/"\"fully-dynamic-layout-map\"",
              "Controls layout maps for non-inferrable memref types.">,
  ];

  let statistics = [Statistic<"numBufferAlloc", "num-buffer-alloc",
                              "Number of buffer allocations">,
  ];
}

#endif // BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_TD
