// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=true --enable-auto-bind-sub-block=false --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 --mlir-disable-threading --mlir-print-ir-after=hivm-delayed-cross-core-gss -o %t 2>&1 | FileCheck %s

// Mix splitting used to replace the AIC rank query with its constant subteam
// handle. This folded away its membership branch, invalidating the anchor
// skeleton and crashing delayed cross-core synchronization. Keep the actual
// query and membership branch on both cores through the compilation pipeline.
// Query annotations match the frontend contract: CUBE_OR_VECTOR is preserved
// until NPU-IR resolves it using the function core type.

// CHECK-LABEL: func.func @_increase_with_barrier_kernel_mix_aic(
// CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_my_pe"
// CHECK: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// CHECK: scf.if %[[MEMBER]] {
// CHECK: symbol = "aclshmem_barrier"
// CHECK: symbol = "aclshmem_barrier"
// CHECK-LABEL: func.func @_increase_with_barrier_kernel_mix_aiv(
// CHECK: %[[RANK:.*]] = hivm.hir.custom {{.*}}symbol = "aclshmem_team_my_pe"
// CHECK: %[[MEMBER:.*]] = arith.cmpi sge, %[[RANK]], %{{.*}} : i32
// CHECK: scf.if %[[MEMBER]] {
// CHECK: symbol = "aclshmem_barrier"
// CHECK: symbol = "aclshmem_barrier"

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">, hivm.disable_auto_tile_and_bind_subblock, ssbuffer.insertionOptimization} {
  func.func @_increase_with_barrier_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi64> {tt.tensor_kind = 2 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", no_side_effect, pure = true, symbol = "aclshmem_team_my_pe"} "dist.aclshmem_team_my_pe" ins(%c1_i32 : i32) -> i32
    %1 = arith.cmpi sge, %0, %c0_i32 : i32
    scf.if %1 {
      %2 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", no_side_effect, pure = true, symbol = "aclshmem_team_n_pes"} "dist.aclshmem_team_n_pes" ins(%c1_i32 : i32) -> i32
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
      %3 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1]>>
      hivm.hir.custom {already_sub_block_bound, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier"} "dist.aclshmem_barrier" ins(%c1_i32 : i32)
      %4 = hivm.hir.get_sub_block_idx -> i64
      %5 = arith.cmpi eq, %4, %c0_i64 : i64
      scf.if %5 {
        %6 = arith.addi %0, %c1_i32 : i32
        %7 = arith.remsi %6, %2 : i32
        %8 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", no_side_effect, pure = true, symbol = "aclshmem_team_translate_pe"} "dist.aclshmem_team_translate_pe" ins(%c1_i32, %7, %c0_i32 : i32, i32, i32) -> i32
        %9 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", no_side_effect, pure = true, symbol = "aclshmem_ptr_int64"} "dist.aclshmem_ptr_int64" ins(%arg2, %8 : memref<?xi64>, i32) -> memref<?xi64>
        %10 = arith.addi %3, %c1_i64 : i64
        %reinterpret_cast_0 = memref.reinterpret_cast %9 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
        %11 = tensor.empty() : tensor<1xi64>
        %inserted = tensor.insert %10 into %11[%c0] : tensor<1xi64>
        bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_0 : (tensor<1xi64>, memref<1xi64, strided<[1]>>) -> ()
      }
      hivm.hir.custom {already_sub_block_bound, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier"} "dist.aclshmem_barrier" ins(%c1_i32 : i32)
    }
    return
  }
}
