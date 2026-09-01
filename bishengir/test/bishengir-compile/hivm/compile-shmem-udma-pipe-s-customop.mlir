// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
//
// Exercise each typed PIPE_S UDMA put and put-with-signal wrapper, and quiet,
// in its own mixed AIC/AIV kernel.

// RUN: split-file %s %t
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_half.mlir -o %t/put_nbi_half.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_float.mlir -o %t/put_nbi_float.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_int8.mlir -o %t/put_nbi_int8.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_int16.mlir -o %t/put_nbi_int16.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_int32.mlir -o %t/put_nbi_int32.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_int64.mlir -o %t/put_nbi_int64.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_uint8.mlir -o %t/put_nbi_uint8.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_uint16.mlir -o %t/put_nbi_uint16.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_uint32.mlir -o %t/put_nbi_uint32.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_uint64.mlir -o %t/put_nbi_uint64.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_nbi_bfloat16.mlir -o %t/put_nbi_bfloat16.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_half.mlir -o %t/put_signal_nbi_half.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_float.mlir -o %t/put_signal_nbi_float.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_int8.mlir -o %t/put_signal_nbi_int8.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_int16.mlir -o %t/put_signal_nbi_int16.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_int32.mlir -o %t/put_signal_nbi_int32.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_int64.mlir -o %t/put_signal_nbi_int64.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_uint8.mlir -o %t/put_signal_nbi_uint8.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_uint16.mlir -o %t/put_signal_nbi_uint16.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_uint32.mlir -o %t/put_signal_nbi_uint32.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_uint64.mlir -o %t/put_signal_nbi_uint64.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/put_signal_nbi_bfloat16.mlir -o %t/put_signal_nbi_bfloat16.ll
// RUN: bishengir-compile --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=false %t/quiet.mlir -o %t/quiet.ll
// RUN: cat %t/put_nbi_*.ll %t/put_signal_nbi_*.ll %t/quiet.ll | FileCheck %s

//--- put_nbi_half.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_half

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_half(
      %dst: memref<?xf16>,
      %src: memref<?xf16>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_half"} "dist.aclshmemi_udma_put_nbi_half" ins(%dst, %src, %num_elements, %pe : memref<?xf16>, memref<?xf16>, i32, i32)
    return
  }
}

//--- put_nbi_float.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_float

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_float(
      %dst: memref<?xf32>,
      %src: memref<?xf32>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_float"} "dist.aclshmemi_udma_put_nbi_float" ins(%dst, %src, %num_elements, %pe : memref<?xf32>, memref<?xf32>, i32, i32)
    return
  }
}

//--- put_nbi_int8.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_int8

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_int8(
      %dst: memref<?xi8>,
      %src: memref<?xi8>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_int8"} "dist.aclshmemi_udma_put_nbi_int8" ins(%dst, %src, %num_elements, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    return
  }
}

//--- put_nbi_int16.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_int16

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_int16(
      %dst: memref<?xi16>,
      %src: memref<?xi16>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_int16"} "dist.aclshmemi_udma_put_nbi_int16" ins(%dst, %src, %num_elements, %pe : memref<?xi16>, memref<?xi16>, i32, i32)
    return
  }
}

//--- put_nbi_int32.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_int32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_int32(
      %dst: memref<?xi32>,
      %src: memref<?xi32>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_int32"} "dist.aclshmemi_udma_put_nbi_int32" ins(%dst, %src, %num_elements, %pe : memref<?xi32>, memref<?xi32>, i32, i32)
    return
  }
}

//--- put_nbi_int64.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_int64

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_int64(
      %dst: memref<?xi64>,
      %src: memref<?xi64>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_int64"} "dist.aclshmemi_udma_put_nbi_int64" ins(%dst, %src, %num_elements, %pe : memref<?xi64>, memref<?xi64>, i32, i32)
    return
  }
}

//--- put_nbi_uint8.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_uint8

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_uint8(
      %dst: memref<?xi8>,
      %src: memref<?xi8>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_uint8"} "dist.aclshmemi_udma_put_nbi_uint8" ins(%dst, %src, %num_elements, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    return
  }
}

//--- put_nbi_uint16.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_uint16

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_uint16(
      %dst: memref<?xi16>,
      %src: memref<?xi16>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_uint16"} "dist.aclshmemi_udma_put_nbi_uint16" ins(%dst, %src, %num_elements, %pe : memref<?xi16>, memref<?xi16>, i32, i32)
    return
  }
}

//--- put_nbi_uint32.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_uint32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_uint32(
      %dst: memref<?xi32>,
      %src: memref<?xi32>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_uint32"} "dist.aclshmemi_udma_put_nbi_uint32" ins(%dst, %src, %num_elements, %pe : memref<?xi32>, memref<?xi32>, i32, i32)
    return
  }
}

//--- put_nbi_uint64.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_uint64

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_uint64(
      %dst: memref<?xi64>,
      %src: memref<?xi64>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_uint64"} "dist.aclshmemi_udma_put_nbi_uint64" ins(%dst, %src, %num_elements, %pe : memref<?xi64>, memref<?xi64>, i32, i32)
    return
  }
}

//--- put_nbi_bfloat16.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemi_udma_put_nbi_bfloat16

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_nbi_bfloat16(
      %dst: memref<?xbf16>,
      %src: memref<?xbf16>,
      %num_elements: i32,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemi_udma_put_nbi_bfloat16"} "dist.aclshmemi_udma_put_nbi_bfloat16" ins(%dst, %src, %num_elements, %pe : memref<?xbf16>, memref<?xbf16>, i32, i32)
    return
  }
}

//--- put_signal_nbi_half.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_half

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_half(
      %dst: memref<?xf16>,
      %src: memref<?xf16>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_half"} "dist.aclshmemx_udma_put_signal_nbi_half" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xf16>, memref<?xf16>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_float.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_float

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_float(
      %dst: memref<?xf32>,
      %src: memref<?xf32>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_float"} "dist.aclshmemx_udma_put_signal_nbi_float" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xf32>, memref<?xf32>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_int8.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_int8

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_int8(
      %dst: memref<?xi8>,
      %src: memref<?xi8>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_int8"} "dist.aclshmemx_udma_put_signal_nbi_int8" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_int16.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_int16

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_int16(
      %dst: memref<?xi16>,
      %src: memref<?xi16>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_int16"} "dist.aclshmemx_udma_put_signal_nbi_int16" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi16>, memref<?xi16>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_int32.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_int32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_int32(
      %dst: memref<?xi32>,
      %src: memref<?xi32>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_int32"} "dist.aclshmemx_udma_put_signal_nbi_int32" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi32>, memref<?xi32>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_int64.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_int64

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_int64(
      %dst: memref<?xi64>,
      %src: memref<?xi64>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_int64"} "dist.aclshmemx_udma_put_signal_nbi_int64" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi64>, memref<?xi64>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_uint8.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_uint8

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_uint8(
      %dst: memref<?xi8>,
      %src: memref<?xi8>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_uint8"} "dist.aclshmemx_udma_put_signal_nbi_uint8" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_uint16.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_uint16

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_uint16(
      %dst: memref<?xi16>,
      %src: memref<?xi16>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_uint16"} "dist.aclshmemx_udma_put_signal_nbi_uint16" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi16>, memref<?xi16>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_uint32.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_uint32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_uint32(
      %dst: memref<?xi32>,
      %src: memref<?xi32>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_uint32"} "dist.aclshmemx_udma_put_signal_nbi_uint32" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi32>, memref<?xi32>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_uint64.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_uint64

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_uint64(
      %dst: memref<?xi64>,
      %src: memref<?xi64>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_uint64"} "dist.aclshmemx_udma_put_signal_nbi_uint64" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xi64>, memref<?xi64>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- put_signal_nbi_bfloat16.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_put_signal_nbi_bfloat16

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_put_signal_nbi_bfloat16(
      %dst: memref<?xbf16>,
      %src: memref<?xbf16>,
      %num_elements: i32,
      %sig_addr: memref<?xui64>,
      %signal: ui64,
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_signal_nbi_bfloat16"} "dist.aclshmemx_udma_put_signal_nbi_bfloat16" ins(%dst, %src, %num_elements, %sig_addr, %signal, %pe : memref<?xbf16>, memref<?xbf16>, i32, memref<?xui64>, ui64, i32)
    return
  }
}

//--- quiet.mlir

// CHECK-DAG: {{.*}}@_mlir_ciface_aclshmemx_udma_quiet

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_quiet(
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_quiet"} "dist.aclshmemx_udma_quiet" ins(%pe : i32)
    return
  }
}
