// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s
// Explicit scalar PRE_QUANT must survive type-based conversion inference.
// A full QK512 tile is sent to two contiguous row-partitioned outputs.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @row_split
  // CHECK-SAME: %[[S:[a-zA-Z0-9_]+]]: f32
  // CHECK: %[[Q:.*]] = arith.constant 32 : i64
  // CHECK: %[[ROW:.*]] = arith.constant 1 : i8
  // CHECK-NOT: memref.alloc
  // CHECK: call @fixpipe_normal_dual_float_to_half_4d_to_4d_ubuf({{.*}}, %[[Q]], %[[S]], {{.*}}, %[[ROW]])
  func.func @row_split(%src: memref<32x8x16x16xf32, #hivm.address_space<cc>>, %dst: memref<32x4x16x16xf16, #hivm.address_space<ub>>, %scale: f32) {
    hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>} ins(%src : memref<32x8x16x16xf32, #hivm.address_space<cc>>) outs(%dst : memref<32x4x16x16xf16, #hivm.address_space<ub>>) quant_scale = %scale : f32 dual_dst_mode = <ROW_SPLIT>
    return
  }
  // CHECK-LABEL: func.func @single
  // CHECK-SAME: %[[S:[a-zA-Z0-9_]+]]: f32
  // CHECK: %[[Q:.*]] = arith.constant 32 : i64
  // CHECK: call @fixpipe_normal_float_to_half_4d_to_4d_ubuf({{.*}}, %[[Q]], %[[S]],
  func.func @single(%src: memref<16x4x16x16xf32, #hivm.address_space<cc>>, %dst: memref<16x4x16x16xf16, #hivm.address_space<ub>>, %scale: f32) {
    hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F16_PRE>} ins(%src : memref<16x4x16x16xf32, #hivm.address_space<cc>>) outs(%dst : memref<16x4x16x16xf16, #hivm.address_space<ub>>) quant_scale = %scale : f32
    return
  }
}
