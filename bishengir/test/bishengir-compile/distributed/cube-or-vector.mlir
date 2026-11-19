// REQUIRES: hivmc, shmem
// RUN: split-file %s %t
// RUN: bishengir-compile %t/vector.mlir --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=False --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile -o %t/vector.o
// RUN: bishengir-compile %t/cube.mlir --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=False --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile -o %t/cube.o
// RUN: bishengir-compile %t/mixed.mlir --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=False --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile -o %t/mixed.o

// Flexible SHMEM calls must reach template linking in standalone and mixed kernels.

//--- vector.mlir
module attributes {hacc.target = #hacc.target<"Ascend910B4">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_vector(%workspace: memref<?xi8>, %sync: memref<?xi8>, %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    return
  }
}

//--- cube.mlir
module attributes {hacc.target = #hacc.target<"Ascend910B4">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_cube(%workspace: memref<?xi8>, %sync: memref<?xi8>, %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    return
  }
}

//--- mixed.mlir
module attributes {hacc.target = #hacc.target<"Ascend910B4">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_mixed(%workspace: memref<?xi8>, %sync: memref<?xi8>, %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    return
  }
}
