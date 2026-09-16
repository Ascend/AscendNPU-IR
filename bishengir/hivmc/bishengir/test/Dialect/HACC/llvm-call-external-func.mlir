// RUN: bishengir-opt %s | FileCheck %s

// CHECK: external_tiling_function
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // External functions must have external/extern_weak linkage, which conflicts with private; also must be marked as host
  llvm.func @external_tiling_function(%arg0: i64) -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.external_function_path = "test.cpp"}
  llvm.func @external_tiling_function2(%arg0: i64) -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.external_function_path = "test2.cpp"}
}
