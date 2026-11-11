// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -hivm-infer-func-core-type -hivm-split-mix-kernel -convert-hivm-to-std -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-infer-func-core-type -hivm-split-mix-kernel -convert-hivm-to-std -split-input-file | FileCheck %s

// Non-distributed calls must reach both standard lowering implementations with
// a concrete core type; function inference defaults this kernel to AIV.
// CHECK: func.func private @_mlir_ciface_vector_query() -> i32 attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @vector(
// CHECK: call @_mlir_ciface_vector_query()
module {
  func.func @vector() -> i32 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %x = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "vector_query"} "vector_query" -> i32
    return %x : i32
  }
}

// -----
// CHECK: func.func private @_mlir_ciface_cube_query() -> i32 attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-LABEL: func.func @cube(
// CHECK: call @_mlir_ciface_cube_query()
module {
  func.func @cube() -> i32 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %x = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "cube_query"} "cube_query" -> i32
    return %x : i32
  }
}
