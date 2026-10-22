// RUN: bishengir-opt %s | FileCheck %s
// RUN: bishengir-opt %s --cse | FileCheck %s

// 4095 is the last 12-bit region id.
// CHECK-LABEL: func.func @max_region_id
// CHECK: %[[REGION_ID:.*]] = arith.constant 4095 : i32
// CHECK: hivm.hir.custom
// CHECK-SAME: "__builtin_proton_scope_boundary"
// CHECK-SAME: ins(%[[REGION_ID]] : i32)
func.func @max_region_id() {
  %region_id = arith.constant 4095 : i32
  hivm.hir.custom
      {hivm.inline_mode = #hivm.inline_mode<always_inline>,
       hivm.pipe = #hivm.pipe<PIPE_S>,
       hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>}
      "__builtin_proton_scope_boundary" ins(%region_id : i32)
  return
}

// Two identical boundaries have no results. Memory effects keep both of them.
// CHECK-LABEL: func.func @duplicate_boundaries_survive_cse
// CHECK-COUNT-2: "__builtin_proton_scope_boundary"
func.func @duplicate_boundaries_survive_cse() {
  %region_id = arith.constant 0 : i32
  hivm.hir.custom
      {hivm.inline_mode = #hivm.inline_mode<always_inline>,
       hivm.pipe = #hivm.pipe<PIPE_S>,
       hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>}
      "__builtin_proton_scope_boundary" ins(%region_id : i32)
  hivm.hir.custom
      {hivm.inline_mode = #hivm.inline_mode<always_inline>,
       hivm.pipe = #hivm.pipe<PIPE_S>,
       hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>}
      "__builtin_proton_scope_boundary" ins(%region_id : i32)
  return
}
