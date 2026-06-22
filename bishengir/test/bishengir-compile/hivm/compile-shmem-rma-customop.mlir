// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
//
// Exercise distributed custom ops that lower to the SHMEM RMA C wrappers in
// bishengir/lib/Template/lib/Shmem/Shmem.cpp (aclshmem_{getmem,putmem[_nbi]}).

// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// CHECK-DAG: declare {{.*}}@_mlir_ciface_aclshmem_half_getmem
// CHECK-DAG: declare {{.*}}@_mlir_ciface_aclshmem_half_putmem_nbi
// CHECK-DAG: declare {{.*}}@_mlir_ciface_aclshmem_barrier_all

module attributes {hacc.target = #hacc.target<"Ascend910B4">, hivm.disable_auto_tile_and_bind_subblock} {
func.func @shmem_rma_smoke (
    %arg0: memref<?xf16>,
    %arg1: memref<?xf16>,
    %arg2: memref<?xi8>,
    %arg3: memref<?xi8>,
    %grid_dim_x: i32,
    %grid_dim_y: i32,
    %grid_dim_z: i32,
    %program_id_x: i32,
    %program_id_y: i32,
    %program_id_z: i32
    )
    attributes {
      hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } 
  {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_half_getmem"} "dist.aclshmem_half_getmem" ins(%arg0, %arg1, %c64_i32, %c0_i32 : memref<?xf16>, memref<?xf16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_half_putmem_nbi"} "dist.aclshmem_half_putmem_nbi" ins(%arg0, %arg1, %c64_i32, %c0_i32 : memref<?xf16>, memref<?xf16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    return
  }
}
