// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s
module attributes {hacc.target = #hacc.target<"Ascend960">} {
// CHECK-LABEL: func.func @i8(
// CHECK: call @load_gm_to_ubuf_2d_int8_t(
// CHECK-NOT: _broadcast
func.func @i8(%src: memref<4x1xi8, #hivm.address_space<gm>>, %dst: memref<4x128xi8, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xi8, #hivm.address_space<gm>>) outs(%dst : memref<4x128xi8, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @i8_nocache(
// CHECK: call @load_gm_to_ubuf_2d_int8_t_nocache(
// CHECK-NOT: _broadcast
func.func @i8_nocache(%src: memref<4x1xi8, #hivm.address_space<gm>>, %dst: memref<4x128xi8, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xi8, #hivm.address_space<gm>>) outs(%dst : memref<4x128xi8, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @i16(
// CHECK: call @load_gm_to_ubuf_2d_int16_t(
// CHECK-NOT: _broadcast
func.func @i16(%src: memref<4x1xi16, #hivm.address_space<gm>>, %dst: memref<4x64xi16, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xi16, #hivm.address_space<gm>>) outs(%dst : memref<4x64xi16, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @i16_nocache(
// CHECK: call @load_gm_to_ubuf_2d_int16_t_nocache(
// CHECK-NOT: _broadcast
func.func @i16_nocache(%src: memref<4x1xi16, #hivm.address_space<gm>>, %dst: memref<4x64xi16, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xi16, #hivm.address_space<gm>>) outs(%dst : memref<4x64xi16, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @i32(
// CHECK: call @load_gm_to_ubuf_2d_int32_t(
// CHECK-NOT: _broadcast
func.func @i32(%src: memref<4x1xi32, #hivm.address_space<gm>>, %dst: memref<4x32xi32, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xi32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xi32, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @i32_nocache(
// CHECK: call @load_gm_to_ubuf_2d_int32_t_nocache(
// CHECK-NOT: _broadcast
func.func @i32_nocache(%src: memref<4x1xi32, #hivm.address_space<gm>>, %dst: memref<4x32xi32, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xi32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xi32, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f16(
// CHECK: call @load_gm_to_ubuf_2d_half(
// CHECK-NOT: _broadcast
func.func @f16(%src: memref<4x1xf16, #hivm.address_space<gm>>, %dst: memref<4x64xf16, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf16, #hivm.address_space<gm>>) outs(%dst : memref<4x64xf16, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f16_nocache(
// CHECK: call @load_gm_to_ubuf_2d_half_nocache(
// CHECK-NOT: _broadcast
func.func @f16_nocache(%src: memref<4x1xf16, #hivm.address_space<gm>>, %dst: memref<4x64xf16, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf16, #hivm.address_space<gm>>) outs(%dst : memref<4x64xf16, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @bf16(
// CHECK: call @load_gm_to_ubuf_2d_bfloat16_t(
// CHECK-NOT: _broadcast
func.func @bf16(%src: memref<4x1xbf16, #hivm.address_space<gm>>, %dst: memref<4x64xbf16, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xbf16, #hivm.address_space<gm>>) outs(%dst : memref<4x64xbf16, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @bf16_nocache(
// CHECK: call @load_gm_to_ubuf_2d_bfloat16_t_nocache(
// CHECK-NOT: _broadcast
func.func @bf16_nocache(%src: memref<4x1xbf16, #hivm.address_space<gm>>, %dst: memref<4x64xbf16, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xbf16, #hivm.address_space<gm>>) outs(%dst : memref<4x64xbf16, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f32(
// CHECK: call @load_gm_to_ubuf_2d_float(
// CHECK-NOT: _broadcast
func.func @f32(%src: memref<4x1xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f32_nocache(
// CHECK: call @load_gm_to_ubuf_2d_float_nocache(
// CHECK-NOT: _broadcast
func.func @f32_nocache(%src: memref<4x1xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f8E4M3FN(
// CHECK: call @load_gm_to_ubuf_2d_float8_e4m3_t(
// CHECK-NOT: _broadcast
func.func @f8E4M3FN(%src: memref<4x1xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<4x128xf8E4M3FN, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<4x128xf8E4M3FN, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f8E4M3FN_nocache(
// CHECK: call @load_gm_to_ubuf_2d_float8_e4m3_t_nocache(
// CHECK-NOT: _broadcast
func.func @f8E4M3FN_nocache(%src: memref<4x1xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<4x128xf8E4M3FN, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<4x128xf8E4M3FN, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f8E5M2(
// CHECK: call @load_gm_to_ubuf_2d_float8_e5m2_t(
// CHECK-NOT: _broadcast
func.func @f8E5M2(%src: memref<4x1xf8E5M2, #hivm.address_space<gm>>, %dst: memref<4x128xf8E5M2, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf8E5M2, #hivm.address_space<gm>>) outs(%dst : memref<4x128xf8E5M2, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
// CHECK-LABEL: func.func @f8E5M2_nocache(
// CHECK: call @load_gm_to_ubuf_2d_float8_e5m2_t_nocache(
// CHECK-NOT: _broadcast
func.func @f8E5M2_nocache(%src: memref<4x1xf8E5M2, #hivm.address_space<gm>>, %dst: memref<4x128xf8E5M2, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<4x1xf8E5M2, #hivm.address_space<gm>>) outs(%dst : memref<4x128xf8E5M2, #hivm.address_space<ub>>) {disableDCache} broadcast_axes = [1]
  return
}
}
