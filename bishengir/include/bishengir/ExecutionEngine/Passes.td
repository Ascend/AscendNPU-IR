//===-- Passes.td - ExecutionEngine pass definition file ------*-tablegen-*-==//
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

#ifndef BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES
#define BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ExecutionEngineHostMainCreator : Pass<"execution-engine-create-host-main", "ModuleOp"> {
  let summary = "Wrap the only host entry function with a main function";
  let description = [{
    Wrap the only host entry function with another function to allow for testing
    the kernel function with the use of different runner utils functions.

    Modifications made:
      * Create a wrapper function to initialize the inputs to the function, and
      print inputs and generated outputs to files.
      * Create declarations for the functions being used during wrapping.

    Constraints/Assumptions:
      * There has to be exactly one function with
      `hacc.function_kind = #hacc.function_kind<HOST>` and
      `hacc.host_func_type = #hacc.host_func_type<host_entry>` attributes.
      * That single host entry function shouldn't be named with the wrapper
      function's name.
      * Arguments to that function and return results have to be tensors or
      memrefs.

    Input

    ```mlir
    func.func @kernel(...) -> (...) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
      // do something
      %result = ...
      return %result
    }
    ```

    Output

    ```mlir
    func.func @kernel(...)
    func.func @main(...) {
      // prepare arguments
      %input = ...
      // invoke kernel
      %result = call @kernel(%input, ...)
      // dump output
    }
   ```
  }];
  let constructor = "mlir::execution_engine::createCreateHostMainPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];
  let options =
        [Option<"wrapperName", "wrapper-name", "std::string", "\"main\"",
        "Name of the wrapper function to be generated for the single host entry"
        " function provided.">];
}

def ExecutionEngineHIVMToUpstreamConversion : Pass<"execution-engine-convert-hivm-to-upstream"> {
  let summary = "Convert HIVM operations to upstream dialect's equivalent";
  let description = [{
    Convert selected HIVM operations into upstream dialects mostly using Linalg.
    The main goal is to use dialects that can be lowered to LLVMIR using
    upstream passes instead of internal ones.

    Modifications made:
      * Remove HIVM attributes (e.g. memory spaces) from types.
      * Convert HIVM operations to their upstream dialect's equivalent.

    Constraints/Assumptions:
      * The operations have to have either tensor or buffer semantics.
      * Synchronization operations are considered to be NoOp.
      * PointerCastOp is not supported.
  }];
  let constructor = "mlir::execution_engine::createConvertHIVMToUpstreamPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "hfusion::HFusionDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];
}
def ExecutionEngineHFusionToUpstreamConversion : Pass<"execution-engine-convert-hfusion-to-upstream"> {
  let summary = "Convert HFusion operations to upstream dialect's equivalent for execution engine";
  let description = [{
    Convert selected HFusion operations (e.g., HistogramOp) into upstream dialects 
    to allow for CPU execution and precision debugging.

    Modifications made:
      * Decompose complex HFusion operations into standard Linalg, SCF, Arith, and Tensor ops.
      * Support tensor shape collapsing, dynamic shape resolution, and mask semantics.

    Constraints/Assumptions:
      * Operations are currently expected to have pure tensor semantics
  }];
  let constructor = "mlir::execution_engine::createConvertHFusionToUpstreamPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "hfusion::HFusionDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}
#endif // BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES
