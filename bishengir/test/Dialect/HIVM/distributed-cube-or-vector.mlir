// RUN: bishengir-opt %s -hivm-infer-func-core-type | FileCheck %s --check-prefix=INFER
// RUN: bishengir-opt %s -hivm-infer-func-core-type -hivm-split-mix-kernel | FileCheck %s --check-prefix=SPLIT
// RUN: bishengir-opt %s -hivm-split-mix-kernel | FileCheck %s --check-prefix=DIRECT

// Flexible distributed ops follow the inferred function without forcing MIX.
// Explicit core types and non-distributed custom ops remain unchanged.

// INFER-LABEL: func.func @scalar(
// INFER-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_barrier_all"
// SPLIT-LABEL: func.func @scalar(
// SPLIT: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_barrier_all"
func.func @scalar() {
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
  return
}

// INFER-LABEL: func.func @cube(
// INFER-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// INFER: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "aclshmem_barrier_all"
// SPLIT-LABEL: func.func @cube(
// SPLIT: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "aclshmem_barrier_all"
func.func @cube(%a: memref<64x64xf16>, %b: memref<64x64xf16>, %c: memref<64x64xf16>) {
  hivm.hir.matmul ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>) outs(%c : memref<64x64xf16>)
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
  return
}

// INFER-LABEL: func.func @vector(
// INFER-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_barrier_all"
// SPLIT-LABEL: func.func @vector(
// SPLIT: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_barrier_all"
func.func @vector(%a: memref<64x64xf16>, %b: memref<64x64xf16>, %c: memref<64x64xf16>) {
  hivm.hir.vadd ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>) outs(%c : memref<64x64xf16>)
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
  return
}

// INFER-LABEL: func.func @mixed(
// INFER-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
// INFER: hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "aclshmem_barrier_all"
// SPLIT-LABEL: func.func @mixed_mix_aic(
// SPLIT: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "aclshmem_barrier_all"
// SPLIT-LABEL: func.func @mixed_mix_aiv(
// SPLIT: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_barrier_all"
func.func @mixed(%a: memref<64x64xf16>, %b: memref<64x64xf16>, %c: memref<64x64xf16>) {
  hivm.hir.matmul ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>) outs(%c : memref<64x64xf16>)
  hivm.hir.vadd ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>) outs(%c : memref<64x64xf16>)
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
  return
}

// INFER-LABEL: func.func @explicit_and_unrelated(
// INFER-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
// INFER: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "cube"
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "vector"
// INFER: hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "both"
// INFER: hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "unrelated"
func.func @explicit_and_unrelated() {
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "cube"} "dist.cube"
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "vector"} "dist.vector"
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "both"} "dist.both"
  hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "unrelated"} "dist.unrelated"
  return
}

// Standalone splitting also resolves flexible ops in annotated functions.
// DIRECT-LABEL: func.func @annotated_cube(
// DIRECT: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "aclshmem_barrier_all"
func.func @annotated_cube() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
  return
}

// Standalone splitting also resolves flexible ops in annotated functions.
// DIRECT-LABEL: func.func @annotated_vector(
// DIRECT: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_barrier_all"
func.func @annotated_vector() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
  return
}

// Standalone splitting also resolves flexible ops in annotated functions.
// DIRECT-LABEL: func.func @annotated_mix_mix_aic(
// DIRECT: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "aclshmem_barrier_all"
// DIRECT-LABEL: func.func @annotated_mix_mix_aiv(
// DIRECT: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_barrier_all"
func.func @annotated_mix() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
  return
}

// INFER-LABEL: func.func @nested_macro(
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// DIRECT-LABEL: func.func @nested_macro(
// DIRECT: hivm.tcore_type = #hivm.tcore_type<VECTOR>
func.func @nested_macro(%cond: i1) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  scf.if %cond {
    hivm.hir.custom_macro {hivm.is_distributed, hivm.pipe_in = #hivm.pipe<PIPE_S>, hivm.pipe_out = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "macro"} "dist.macro"
  }
  return
}

// Host functions must not be rectified.
// INFER-LABEL: func.func @host(
// INFER: hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
// DIRECT-LABEL: func.func @host(
// DIRECT: hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
func.func @host() attributes {hacc.function_kind = #hacc.function_kind<HOST>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "host_op"} "dist.host_op"
  return
}

// INFER-LABEL: func.func @flexible_result(
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_my_pe"
// SPLIT-LABEL: func.func @flexible_result(
// SPLIT: %[[PE:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_my_pe"
// SPLIT: return %[[PE]] : i32
func.func @flexible_result() -> i32 {
  %pe = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_my_pe"} "dist.aclshmem_my_pe" -> i32
  return %pe : i32
}

// Notify inference must also be written to the attribute read by cross-core
// synchronization. It follows preceding work instead of the MIX function.
// INFER-LABEL: func.func @notify(
// INFER-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_int64_p"
// INFER: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "aclshmem_int64_p"
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_int64_p"
func.func @notify(%a: memref<64x64xf16>, %b: memref<64x64xf16>, %c: memref<64x64xf16>, %ptr: memref<1xi64>, %value: i64, %pe: i32) {
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%ptr, %value, %pe : memref<1xi64>, i64, i32)
  hivm.hir.matmul ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>) outs(%c : memref<64x64xf16>)
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%ptr, %value, %pe : memref<1xi64>, i64, i32)
  hivm.hir.vadd ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>) outs(%c : memref<64x64xf16>)
  hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%ptr, %value, %pe : memref<1xi64>, i64, i32)
  return
}

// An arbitrary scalar query guards a two-core operation. Its symbol has no
// special meaning to inference: the flexible annotation alone keeps the real
// query and predicate on both sides. Matching i32 operands are not rank stubs.
// INFER-LABEL: func.func @query_guard(
// INFER: %[[RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>{{.*}}symbol = "arbitrary_query"
// INFER: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// INFER: scf.if %[[MEMBER]] {
// SPLIT-LABEL: func.func @query_guard_mix_aic(
// SPLIT: %[[RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE>{{.*}}symbol = "arbitrary_query"
// SPLIT: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// SPLIT: scf.if %[[MEMBER]] {
// SPLIT: symbol = "arbitrary_barrier"
// SPLIT-LABEL: func.func @query_guard_mix_aiv(
// SPLIT: %[[RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<VECTOR>{{.*}}symbol = "arbitrary_query"
// SPLIT: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// SPLIT: scf.if %[[MEMBER]] {
// SPLIT: symbol = "arbitrary_barrier"
func.func @query_guard(%team: i32) {
  %c0 = arith.constant 0 : i32
  %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "arbitrary_query"} "dist.arbitrary_query" ins(%team : i32) -> i32
  %member = arith.cmpi sge, %rank, %c0 : i32
  scf.if %member {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, pure = false, symbol = "arbitrary_barrier"} "dist.arbitrary_barrier" ins(%team : i32)
  }
  return
}
