// RUN: bishengir-opt %s -hivm-infer-func-core-type | FileCheck %s
// RUN: bishengir-opt %s -hivm-infer-func-core-type="resolve-op-placement=false" | FileCheck %s --check-prefix=DEFER

// Preserve the coupled architecture's implicit AIV convention.
module attributes {hacc.target = #hacc.target<"Ascend310B1">} {
  // DEFER-LABEL: func.func @implicit_vector(
  // DEFER-NOT: hivm.func_core_type
  // DEFER: hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
  // CHECK-LABEL: func.func @implicit_vector(
  // CHECK-NOT: hivm.func_core_type
  // CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>
  func.func @implicit_vector() {
    hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "query"} "query"
    return
  }
  // CHECK-LABEL: func.func @explicit_cube(
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
  // CHECK: hivm.tcore_type = #hivm.tcore_type<CUBE>
  func.func @explicit_cube() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "query"} "query"
    return
  }
}
