// REQUIRES: hivmc, shmem
// UNSUPPORTED: bishengir_published
//
// Compile and link the current byte-counted PIPE_S UDMA wrappers, including
// ordinary and QP get/put/put-with-signal for every exported payload type.
// The signal is uint64, the count is bytes, and QP zero is an explicit QP.
// Each payload type gets a mixed AIC/AIV kernel. Both completion APIs are
// covered separately. Relay wrappers require a relay-enabled template build
// and are covered by the transport-wrappers ABI test instead.
// Check the saved pre-link LLVM IR for the exact current wrapper symbols.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/half.mlir -o %t/half.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/float.mlir -o %t/float.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/int8.mlir -o %t/int8.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/int16.mlir -o %t/int16.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/int32.mlir -o %t/int32.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/int64.mlir -o %t/int64.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/uint8.mlir -o %t/uint8.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/uint16.mlir -o %t/uint16.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/uint32.mlir -o %t/uint32.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/uint64.mlir -o %t/uint64.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/char.mlir -o %t/char.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/bfloat16.mlir -o %t/bfloat16.o
// RUN: bishengir-compile --target=Ascend950PR_9589 --mlir-disable-threading --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-triton-kernel-compile --enable-hfusion-compile --enable-hivm-compile --enable-lir-compile=true --save-temps=%t %t/quiet.mlir -o %t/quiet.o
// RUN: cat %t/*.ll | FileCheck %s

//--- half.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_half(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_half(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_half(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_half(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_half(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_half(
// Seven bytes must remain seven bytes for a wider-than-byte payload.
// CHECK-DAG: call void @aclshmemx_udma_put_nbi_half({{.*}}i32 7, i32 {{[^,)]+}})
// CHECK-DAG: call void @aclshmemx_udma_qp_put_nbi_half({{.*}}i32 7, i32 {{[^,)]+}}, i32 0)

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_half(
      %dst: memref<?xf16>, %src: memref<?xf16>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_half"} "dist.aclshmemx_udma_put_nbi_half" ins(%dst, %src, %nbytes, %pe : memref<?xf16>, memref<?xf16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_half"} "dist.aclshmemx_udma_get_nbi_half" ins(%dst, %src, %nbytes, %pe : memref<?xf16>, memref<?xf16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_half"} "dist.aclshmemx_udma_putmem_signal_nbi_half" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xf16>, memref<?xf16>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_half"} "dist.aclshmemx_udma_qp_put_nbi_half" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xf16>, memref<?xf16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_half"} "dist.aclshmemx_udma_qp_get_nbi_half" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xf16>, memref<?xf16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_half"} "dist.aclshmemx_udma_qp_put_signal_nbi_half" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xf16>, memref<?xf16>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- float.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_float(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_float(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_float(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_float(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_float(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_float(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_float(
      %dst: memref<?xf32>, %src: memref<?xf32>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_float"} "dist.aclshmemx_udma_put_nbi_float" ins(%dst, %src, %nbytes, %pe : memref<?xf32>, memref<?xf32>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_float"} "dist.aclshmemx_udma_get_nbi_float" ins(%dst, %src, %nbytes, %pe : memref<?xf32>, memref<?xf32>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_float"} "dist.aclshmemx_udma_putmem_signal_nbi_float" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xf32>, memref<?xf32>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_float"} "dist.aclshmemx_udma_qp_put_nbi_float" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xf32>, memref<?xf32>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_float"} "dist.aclshmemx_udma_qp_get_nbi_float" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xf32>, memref<?xf32>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_float"} "dist.aclshmemx_udma_qp_put_signal_nbi_float" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xf32>, memref<?xf32>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- int8.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_int8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_int8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_int8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_int8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_int8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_int8(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_int8(
      %dst: memref<?xi8>, %src: memref<?xi8>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_int8"} "dist.aclshmemx_udma_put_nbi_int8" ins(%dst, %src, %nbytes, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_int8"} "dist.aclshmemx_udma_get_nbi_int8" ins(%dst, %src, %nbytes, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_int8"} "dist.aclshmemx_udma_putmem_signal_nbi_int8" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_int8"} "dist.aclshmemx_udma_qp_put_nbi_int8" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_int8"} "dist.aclshmemx_udma_qp_get_nbi_int8" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_int8"} "dist.aclshmemx_udma_qp_put_signal_nbi_int8" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- int16.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_int16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_int16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_int16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_int16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_int16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_int16(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_int16(
      %dst: memref<?xi16>, %src: memref<?xi16>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_int16"} "dist.aclshmemx_udma_put_nbi_int16" ins(%dst, %src, %nbytes, %pe : memref<?xi16>, memref<?xi16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_int16"} "dist.aclshmemx_udma_get_nbi_int16" ins(%dst, %src, %nbytes, %pe : memref<?xi16>, memref<?xi16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_int16"} "dist.aclshmemx_udma_putmem_signal_nbi_int16" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi16>, memref<?xi16>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_int16"} "dist.aclshmemx_udma_qp_put_nbi_int16" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi16>, memref<?xi16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_int16"} "dist.aclshmemx_udma_qp_get_nbi_int16" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi16>, memref<?xi16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_int16"} "dist.aclshmemx_udma_qp_put_signal_nbi_int16" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi16>, memref<?xi16>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- int32.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_int32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_int32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_int32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_int32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_int32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_int32(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_int32(
      %dst: memref<?xi32>, %src: memref<?xi32>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_int32"} "dist.aclshmemx_udma_put_nbi_int32" ins(%dst, %src, %nbytes, %pe : memref<?xi32>, memref<?xi32>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_int32"} "dist.aclshmemx_udma_get_nbi_int32" ins(%dst, %src, %nbytes, %pe : memref<?xi32>, memref<?xi32>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_int32"} "dist.aclshmemx_udma_putmem_signal_nbi_int32" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi32>, memref<?xi32>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_int32"} "dist.aclshmemx_udma_qp_put_nbi_int32" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi32>, memref<?xi32>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_int32"} "dist.aclshmemx_udma_qp_get_nbi_int32" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi32>, memref<?xi32>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_int32"} "dist.aclshmemx_udma_qp_put_signal_nbi_int32" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi32>, memref<?xi32>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- int64.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_int64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_int64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_int64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_int64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_int64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_int64(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_int64(
      %dst: memref<?xi64>, %src: memref<?xi64>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_int64"} "dist.aclshmemx_udma_put_nbi_int64" ins(%dst, %src, %nbytes, %pe : memref<?xi64>, memref<?xi64>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_int64"} "dist.aclshmemx_udma_get_nbi_int64" ins(%dst, %src, %nbytes, %pe : memref<?xi64>, memref<?xi64>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_int64"} "dist.aclshmemx_udma_putmem_signal_nbi_int64" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi64>, memref<?xi64>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_int64"} "dist.aclshmemx_udma_qp_put_nbi_int64" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi64>, memref<?xi64>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_int64"} "dist.aclshmemx_udma_qp_get_nbi_int64" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi64>, memref<?xi64>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_int64"} "dist.aclshmemx_udma_qp_put_signal_nbi_int64" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi64>, memref<?xi64>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- uint8.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_uint8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_uint8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_uint8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_uint8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_uint8(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_uint8(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_uint8(
      %dst: memref<?xi8>, %src: memref<?xi8>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_uint8"} "dist.aclshmemx_udma_put_nbi_uint8" ins(%dst, %src, %nbytes, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_uint8"} "dist.aclshmemx_udma_get_nbi_uint8" ins(%dst, %src, %nbytes, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_uint8"} "dist.aclshmemx_udma_putmem_signal_nbi_uint8" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_uint8"} "dist.aclshmemx_udma_qp_put_nbi_uint8" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_uint8"} "dist.aclshmemx_udma_qp_get_nbi_uint8" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_uint8"} "dist.aclshmemx_udma_qp_put_signal_nbi_uint8" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- uint16.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_uint16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_uint16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_uint16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_uint16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_uint16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_uint16(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_uint16(
      %dst: memref<?xi16>, %src: memref<?xi16>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_uint16"} "dist.aclshmemx_udma_put_nbi_uint16" ins(%dst, %src, %nbytes, %pe : memref<?xi16>, memref<?xi16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_uint16"} "dist.aclshmemx_udma_get_nbi_uint16" ins(%dst, %src, %nbytes, %pe : memref<?xi16>, memref<?xi16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_uint16"} "dist.aclshmemx_udma_putmem_signal_nbi_uint16" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi16>, memref<?xi16>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_uint16"} "dist.aclshmemx_udma_qp_put_nbi_uint16" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi16>, memref<?xi16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_uint16"} "dist.aclshmemx_udma_qp_get_nbi_uint16" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi16>, memref<?xi16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_uint16"} "dist.aclshmemx_udma_qp_put_signal_nbi_uint16" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi16>, memref<?xi16>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- uint32.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_uint32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_uint32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_uint32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_uint32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_uint32(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_uint32(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_uint32(
      %dst: memref<?xi32>, %src: memref<?xi32>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_uint32"} "dist.aclshmemx_udma_put_nbi_uint32" ins(%dst, %src, %nbytes, %pe : memref<?xi32>, memref<?xi32>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_uint32"} "dist.aclshmemx_udma_get_nbi_uint32" ins(%dst, %src, %nbytes, %pe : memref<?xi32>, memref<?xi32>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_uint32"} "dist.aclshmemx_udma_putmem_signal_nbi_uint32" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi32>, memref<?xi32>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_uint32"} "dist.aclshmemx_udma_qp_put_nbi_uint32" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi32>, memref<?xi32>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_uint32"} "dist.aclshmemx_udma_qp_get_nbi_uint32" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi32>, memref<?xi32>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_uint32"} "dist.aclshmemx_udma_qp_put_signal_nbi_uint32" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi32>, memref<?xi32>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- uint64.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_uint64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_uint64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_uint64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_uint64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_uint64(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_uint64(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_uint64(
      %dst: memref<?xi64>, %src: memref<?xi64>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_uint64"} "dist.aclshmemx_udma_put_nbi_uint64" ins(%dst, %src, %nbytes, %pe : memref<?xi64>, memref<?xi64>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_uint64"} "dist.aclshmemx_udma_get_nbi_uint64" ins(%dst, %src, %nbytes, %pe : memref<?xi64>, memref<?xi64>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_uint64"} "dist.aclshmemx_udma_putmem_signal_nbi_uint64" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi64>, memref<?xi64>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_uint64"} "dist.aclshmemx_udma_qp_put_nbi_uint64" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi64>, memref<?xi64>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_uint64"} "dist.aclshmemx_udma_qp_get_nbi_uint64" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi64>, memref<?xi64>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_uint64"} "dist.aclshmemx_udma_qp_put_signal_nbi_uint64" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi64>, memref<?xi64>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- char.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_char(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_char(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_char(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_char(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_char(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_char(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_char(
      %dst: memref<?xi8>, %src: memref<?xi8>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_char"} "dist.aclshmemx_udma_put_nbi_char" ins(%dst, %src, %nbytes, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_char"} "dist.aclshmemx_udma_get_nbi_char" ins(%dst, %src, %nbytes, %pe : memref<?xi8>, memref<?xi8>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_char"} "dist.aclshmemx_udma_putmem_signal_nbi_char" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_char"} "dist.aclshmemx_udma_qp_put_nbi_char" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_char"} "dist.aclshmemx_udma_qp_get_nbi_char" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_char"} "dist.aclshmemx_udma_qp_put_signal_nbi_char" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xi8>, memref<?xi8>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- bfloat16.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_put_nbi_bfloat16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_get_nbi_bfloat16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_putmem_signal_nbi_bfloat16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_nbi_bfloat16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_get_nbi_bfloat16(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_bfloat16(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_bfloat16(
      %dst: memref<?xbf16>, %src: memref<?xbf16>, %sig_addr: memref<?xui64>, %signal: ui64, %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %nbytes = arith.constant 7 : i32
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_put_nbi_bfloat16"} "dist.aclshmemx_udma_put_nbi_bfloat16" ins(%dst, %src, %nbytes, %pe : memref<?xbf16>, memref<?xbf16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_get_nbi_bfloat16"} "dist.aclshmemx_udma_get_nbi_bfloat16" ins(%dst, %src, %nbytes, %pe : memref<?xbf16>, memref<?xbf16>, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_putmem_signal_nbi_bfloat16"} "dist.aclshmemx_udma_putmem_signal_nbi_bfloat16" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe : memref<?xbf16>, memref<?xbf16>, i32, memref<?xui64>, ui64, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_nbi_bfloat16"} "dist.aclshmemx_udma_qp_put_nbi_bfloat16" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xbf16>, memref<?xbf16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_get_nbi_bfloat16"} "dist.aclshmemx_udma_qp_get_nbi_bfloat16" ins(%dst, %src, %nbytes, %pe, %qp_idx : memref<?xbf16>, memref<?xbf16>, i32, i32, i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_put_signal_nbi_bfloat16"} "dist.aclshmemx_udma_qp_put_signal_nbi_bfloat16" ins(%dst, %src, %nbytes, %sig_addr, %signal, %pe, %qp_idx : memref<?xbf16>, memref<?xbf16>, i32, memref<?xui64>, ui64, i32, i32)
    return
  }
}

//--- quiet.mlir

// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_quiet(
// CHECK-DAG: call void @_mlir_ciface_aclshmemx_udma_qp_quiet(

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @shmem_udma_quiet(
      %pe: i32,
      %grid_dim_x: i32, %grid_dim_y: i32, %grid_dim_z: i32,
      %program_id_x: i32, %program_id_y: i32, %program_id_z: i32)
      attributes {
        hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
      } {
    %qp_idx = arith.constant 0 : i32
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_quiet"} "dist.aclshmemx_udma_quiet" ins(%pe : i32)
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_udma_qp_quiet"} "dist.aclshmemx_udma_qp_quiet" ins(%pe, %qp_idx : i32, i32)
    return
  }
}
