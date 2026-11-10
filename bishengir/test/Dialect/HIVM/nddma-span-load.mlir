// RUN: bishengir-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hfusion-flatten-ops{flatten-mode=tidy}),canonicalize)' | FileCheck %s --check-prefix=FLATTEN
// FLATTEN-LABEL: func.func @span3_axis0(
// FLATTEN: hivm.hir.load ins({{.*}} : memref<1x3x32xf32
// FLATTEN-SAME: outs({{.*}} : memref<2x3x40xf32
// RUN: bishengir-opt %s -split-input-file -verify-diagnostics
// RUN: bishengir-opt %s -split-input-file -canonicalize -hivm-aggregated-decompose-op | FileCheck %s
// CHECK-LABEL: func.func @dynamic(
// CHECK: hivm.hir.load
// CHECK-SAME: left_padding_num = %arg2
// CHECK-SAME: right_padding_num = %arg3
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span2(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span3_axis0(%src: memref<1x3x32xf32, #hivm.address_space<gm>>, %dst: memref<2x3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 2 : index
  %rp = arith.constant 6 : index
  hivm.hir.load ins(%src : memref<1x3x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x3x40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span3_axis1(%src: memref<2x1x32xf32, #hivm.address_space<gm>>, %dst: memref<2x3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 2 : index
  %rp = arith.constant 6 : index
  hivm.hir.load ins(%src : memref<2x1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x3x40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [1]
  return
 }
}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @all_pad(%src: memref<2x3x0xf32, #hivm.address_space<gm>>, %dst: memref<2x3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 2 : index
  %rp = arith.constant 38 : index
  hivm.hir.load ins(%src : memref<2x3x0xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x3x40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @wide_pad(%src: memref<3x32xf32, #hivm.address_space<gm>>, %dst: memref<3x340xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 300 : index
  %rp = arith.constant 8 : index
  hivm.hir.load ins(%src : memref<3x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x340xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @dynamic(%src: memref<1x?xf32, strided<[32, 1]>, #hivm.address_space<gm>>, %dst: memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>, %lp: index, %rp: index, %p: f32) {
  hivm.hir.load ins(%src : memref<1x?xf32, strided<[32, 1]>, #hivm.address_space<gm>>) outs(%dst : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @dynamic3(%src: memref<1x1x?xf32, strided<[?, ?, 1]>, #hivm.address_space<gm>>, %dst: memref<?x?x?xf32, strided<[?, ?, 1]>, #hivm.address_space<ub>>, %lp: index, %rp: index, %p: f32) {
  hivm.hir.load ins(%src : memref<1x1x?xf32, strided<[?, ?, 1]>, #hivm.address_space<gm>>) outs(%dst : memref<?x?x?xf32, strided<[?, ?, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0, 1]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @dynamic_lp_const_rp(%src: memref<2x8xf32, #hivm.address_space<gm>>, %dst: memref<2x12xf32, #hivm.address_space<ub>>, %lp: index) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  hivm.hir.load ins(%src : memref<2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x12xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
// A dynamic last-axis stride is left to the template, which re-checks it at
// run time; the other span checks still apply.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @dynamic_last_stride(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<1x12xf32, strided<[?, ?]>, #hivm.address_space<ub>>, %p: f32) {
  %lp = arith.constant 2 : index
  %rp = arith.constant 2 : index
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x12xf32, strided<[?, ?]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

