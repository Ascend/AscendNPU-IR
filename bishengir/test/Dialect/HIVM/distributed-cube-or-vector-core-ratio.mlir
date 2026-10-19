// RUN: bishengir-opt %s -hivm-infer-func-core-type | FileCheck %s

// Every core ratio requires MIX function and module annotations. Resolve
// flexible distributed operations using that MIX annotation, without selecting
// a single core from the ratio here. Splitting retains its existing semantics.
// CHECK: hivm.module_core_type = #hivm.module_core_type<MIX>
module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">} {
  // CHECK-LABEL: func.func @ratio_0_1(
  // CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<0, 1>
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  // CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>{{.*}}symbol = "arbitrary_query"
  // CHECK: return %[[RANK]] : i32
  func.func @ratio_0_1(%team: i32) -> i32 attributes {hivm.core_ratio = #hivm.core_ratio<0, 1>} {
    %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, pure = true, symbol = "arbitrary_query"} "dist.arbitrary_query" ins(%team : i32) -> i32
    return %rank : i32
  }
  // CHECK-LABEL: func.func @ratio_1_0(
  // CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 0>
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  // CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>{{.*}}symbol = "arbitrary_query"
  // CHECK: return %[[RANK]] : i32
  func.func @ratio_1_0(%team: i32) -> i32 attributes {hivm.core_ratio = #hivm.core_ratio<1, 0>} {
    %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, pure = true, symbol = "arbitrary_query"} "dist.arbitrary_query" ins(%team : i32) -> i32
    return %rank : i32
  }
  // CHECK-LABEL: func.func @ratio_1_1(
  // CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 1>
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  // CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>{{.*}}symbol = "arbitrary_query"
  // CHECK: return %[[RANK]] : i32
  func.func @ratio_1_1(%team: i32) -> i32 attributes {hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, pure = true, symbol = "arbitrary_query"} "dist.arbitrary_query" ins(%team : i32) -> i32
    return %rank : i32
  }
  // CHECK-LABEL: func.func @ratio_1_2(
  // CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 2>
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  // CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>{{.*}}symbol = "arbitrary_query"
  // CHECK: return %[[RANK]] : i32
  func.func @ratio_1_2(%team: i32) -> i32 attributes {hivm.core_ratio = #hivm.core_ratio<1, 2>} {
    %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, pure = true, symbol = "arbitrary_query"} "dist.arbitrary_query" ins(%team : i32) -> i32
    return %rank : i32
  }
}
