// RUN: sed "s/hivm.is_distributed, //g" %s | bishengir-opt -hivm-infer-func-core-type -hivm-insert-anchors-and-backup -hivm-split-mix-kernel -canonicalize -hivm-delayed-cross-core-gss="force-is-reg-based=true solver-version=v1" -o /dev/null
// RUN: sed "s/hivm.is_distributed, //g" %s | bishengir-opt -hivm-infer-func-core-type -hivm-insert-anchors-and-backup -hivm-split-mix-kernel -canonicalize -hivm-delayed-cross-core-gss="force-is-reg-based=true solver-version=v2" -o /dev/null
// RUN: bishengir-opt %s -hivm-infer-func-core-type -hivm-insert-anchors-and-backup -hivm-split-mix-kernel -canonicalize | FileCheck %s
// RUN: bishengir-opt %s -hivm-infer-func-core-type -hivm-insert-anchors-and-backup -hivm-split-mix-kernel -canonicalize -hivm-delayed-cross-core-gss="force-is-reg-based=true solver-version=v1" -o /dev/null
// RUN: bishengir-opt %s -hivm-infer-func-core-type -hivm-insert-anchors-and-backup -hivm-split-mix-kernel -canonicalize -hivm-delayed-cross-core-gss="force-is-reg-based=true solver-version=v2" -o /dev/null

// Flexible scalar SHMEM queries must remain real queries on both cores:
// forwarding the team handle as the rank changes barrier participation.
// For a constant handle, canonicalization
// folds away the AIC branch and breaks the delayed solver's anchor skeleton.
// Check both constant and runtime handles, including the complete chain of
// scalar queries used to compute a barrier predicate.

// CHECK-LABEL: func.func @team_queries_constant_mix_aic(
// CHECK: %[[WORLD_RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE>{{.*}}symbol = "aclshmem_my_pe"
// CHECK: %[[WORLD_SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_n_pes"
// CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_my_pe"
// CHECK: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// CHECK: scf.if %[[MEMBER]] {
// CHECK: %[[SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_n_pes"
// CHECK: %[[NEXT:.*]] = arith.remsi %[[WORLD_RANK]], %[[SIZE]] : i32
// CHECK: %[[PE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_translate_pe"{{.*}}ins(%{{.*}}, %[[NEXT]], %{{.*}} : i32, i32, i32)
// CHECK: %[[VALID:.*]] = arith.cmpi slt, %[[PE]], %[[WORLD_SIZE]] : i32
// CHECK: scf.if %[[VALID]] {
// CHECK: symbol = "aclshmem_barrier"
// CHECK-LABEL: func.func @team_queries_constant_mix_aiv(
// CHECK: %[[WORLD_RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<VECTOR>{{.*}}symbol = "aclshmem_my_pe"
// CHECK: %[[WORLD_SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_n_pes"
// CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_my_pe"
// CHECK: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// CHECK: scf.if %[[MEMBER]] {
// CHECK: %[[SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_n_pes"
// CHECK: %[[NEXT:.*]] = arith.remsi %[[WORLD_RANK]], %[[SIZE]] : i32
// CHECK: %[[PE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_translate_pe"{{.*}}ins(%{{.*}}, %[[NEXT]], %{{.*}} : i32, i32, i32)
// CHECK: %[[VALID:.*]] = arith.cmpi slt, %[[PE]], %[[WORLD_SIZE]] : i32
// CHECK: scf.if %[[VALID]] {
// CHECK: symbol = "aclshmem_barrier"
func.func @team_queries_constant(%team: i32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world_rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_my_pe"} "dist.aclshmem_my_pe" -> i32
  %world_size = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_n_pes"} "dist.aclshmem_n_pes" -> i32
  %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_team_my_pe"} "dist.aclshmem_team_my_pe" ins(%c1 : i32) -> i32
  %member = arith.cmpi sge, %rank, %c0 : i32
  scf.if %member {
    %size = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_team_n_pes"} "dist.aclshmem_team_n_pes" ins(%c1 : i32) -> i32
    %next = arith.remsi %world_rank, %size : i32
    %pe = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_team_translate_pe"} "dist.aclshmem_team_translate_pe" ins(%c1, %next, %c0 : i32, i32, i32) -> i32
    %valid = arith.cmpi slt, %pe, %world_size : i32
    scf.if %valid {
      hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, pure = false, symbol = "aclshmem_barrier"} "dist.aclshmem_barrier" ins(%c1 : i32)
    }
  }
  return
}

// CHECK-LABEL: func.func @team_queries_runtime_mix_aic(
// CHECK: %[[WORLD_RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE>{{.*}}symbol = "aclshmem_my_pe"
// CHECK: %[[WORLD_SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_n_pes"
// CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_my_pe"
// CHECK: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// CHECK: scf.if %[[MEMBER]] {
// CHECK: %[[SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_n_pes"
// CHECK: %[[NEXT:.*]] = arith.remsi %[[WORLD_RANK]], %[[SIZE]] : i32
// CHECK: %[[PE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_translate_pe"{{.*}}ins(%{{.*}}, %[[NEXT]], %{{.*}} : i32, i32, i32)
// CHECK: %[[VALID:.*]] = arith.cmpi slt, %[[PE]], %[[WORLD_SIZE]] : i32
// CHECK: scf.if %[[VALID]] {
// CHECK: symbol = "aclshmem_barrier"
// CHECK-LABEL: func.func @team_queries_runtime_mix_aiv(
// CHECK: %[[WORLD_RANK:.*]] = hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<VECTOR>{{.*}}symbol = "aclshmem_my_pe"
// CHECK: %[[WORLD_SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_n_pes"
// CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_my_pe"
// CHECK: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// CHECK: scf.if %[[MEMBER]] {
// CHECK: %[[SIZE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_n_pes"
// CHECK: %[[NEXT:.*]] = arith.remsi %[[WORLD_RANK]], %[[SIZE]] : i32
// CHECK: %[[PE:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_translate_pe"{{.*}}ins(%{{.*}}, %[[NEXT]], %{{.*}} : i32, i32, i32)
// CHECK: %[[VALID:.*]] = arith.cmpi slt, %[[PE]], %[[WORLD_SIZE]] : i32
// CHECK: scf.if %[[VALID]] {
// CHECK: symbol = "aclshmem_barrier"
func.func @team_queries_runtime(%team: i32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world_rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_my_pe"} "dist.aclshmem_my_pe" -> i32
  %world_size = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_n_pes"} "dist.aclshmem_n_pes" -> i32
  %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_team_my_pe"} "dist.aclshmem_team_my_pe" ins(%team : i32) -> i32
  %member = arith.cmpi sge, %rank, %c0 : i32
  scf.if %member {
    %size = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_team_n_pes"} "dist.aclshmem_team_n_pes" ins(%team : i32) -> i32
    %next = arith.remsi %world_rank, %size : i32
    %pe = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, no_side_effect, pure = true, symbol = "aclshmem_team_translate_pe"} "dist.aclshmem_team_translate_pe" ins(%team, %next, %c0 : i32, i32, i32) -> i32
    %valid = arith.cmpi slt, %pe, %world_size : i32
    scf.if %valid {
      hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, pure = false, symbol = "aclshmem_barrier"} "dist.aclshmem_barrier" ins(%team : i32)
    }
  }
  return
}
