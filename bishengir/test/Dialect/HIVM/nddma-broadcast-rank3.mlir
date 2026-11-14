// RUN: bishengir-opt %s -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -split-input-file -hivm-aggregated-decompose-op | FileCheck %s
// RUN: bishengir-opt %s -split-input-file -convert-hivm-to-std | FileCheck %s --check-prefix=CALL
// CHECK-LABEL: func.func @axes1(
// CHECK: broadcast_axes = [0]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @axes1(%src: memref<1x3x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x3x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x3x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x3x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}
}
// -----
// CHECK-LABEL: func.func @axes2(
// CHECK: broadcast_axes = [1]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @axes2(%src: memref<2x1x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x3x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<2x1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x3x32xf32, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
}
// -----
// CHECK-LABEL: func.func @axes3(
// CHECK: broadcast_axes = [0, 1]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @axes3(%src: memref<1x1x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x3x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x3x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
  return
}
}
// -----
// CHECK-LABEL: func.func @axes4(
// CHECK: broadcast_axes = [2]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @axes4(%src: memref<2x3x1xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x3x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<2x3x1xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x3x32xf32, #hivm.address_space<ub>>) broadcast_axes = [2]
  return
}
}
// -----
// CHECK-LABEL: func.func @axes5(
// CHECK: broadcast_axes = [0, 2]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @axes5(%src: memref<1x3x1xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x3x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x3x1xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x3x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 2]
  return
}
}
// -----
// CHECK-LABEL: func.func @axes6(
// CHECK: broadcast_axes = [1, 2]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @axes6(%src: memref<2x1x1xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x3x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<2x1x1xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x3x32xf32, #hivm.address_space<ub>>) broadcast_axes = [1, 2]
  return
}
}
// -----
// CHECK-LABEL: func.func @axes7(
// CHECK: broadcast_axes = [0, 1, 2]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @axes7(%src: memref<1x1x1xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<2x3x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x1x1xf32, #hivm.address_space<gm>>) outs(%ub : memref<2x3x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1, 2]
  return
}
}
// -----
// CHECK-LABEL: func.func @rank2_all(
// CHECK: broadcast_axes = [0, 1]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @rank2_all(%src: memref<1x1xi8, #hivm.address_space<gm>>) {
 %ub = memref.alloc() : memref<3x7xi8, #hivm.address_space<ub>>
 hivm.hir.load ins(%src : memref<1x1xi8, #hivm.address_space<gm>>) outs(%ub : memref<3x7xi8, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
 return
}
}

// CALL-LABEL: func.func @axes1(
// CALL: call @load_gm_to_ubuf_3d_float(
// CALL-LABEL: func.func @axes2(
// CALL: call @load_gm_to_ubuf_3d_float(
// CALL-LABEL: func.func @axes3(
// CALL: call @load_gm_to_ubuf_3d_float(
// CALL-LABEL: func.func @axes4(
// CALL: call @load_gm_to_ubuf_3d_float(
// CALL-LABEL: func.func @axes5(
// CALL: call @load_gm_to_ubuf_3d_float(
// CALL-LABEL: func.func @axes6(
// CALL: call @load_gm_to_ubuf_3d_float(
// CALL-LABEL: func.func @axes7(
// CALL: call @load_gm_to_ubuf_3d_float(
// CALL-LABEL: func.func @rank2_all(
// CALL: call @load_gm_to_ubuf_2d_int8_t(
