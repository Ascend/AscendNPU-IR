// RUN: bishengir-opt %s --split-input-file --hfusion-flatten-ops="flatten-mode=tidy multi-dynamic-shape=false register-based=true" | FileCheck %s
// Legacy targets and absent targets retain ordinary load/copy flattening.



// CHECK-LABEL: func.func @a5_rp0(
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-SAME: right_padding_num = %{{.*}} : index
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @a5_rp0(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    %pad = arith.constant 7.0 : f32
    %zero = arith.constant 0 : index
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f32 right_padding_num = %zero : index
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @a3_rp0(
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-SAME: right_padding_num = %{{.*}} : index
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_rp0(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    %pad = arith.constant 7.0 : f32
    %zero = arith.constant 0 : index
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f32 right_padding_num = %zero : index
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @no_target_rp0(
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-SAME: right_padding_num = %{{.*}} : index
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
module {
  func.func @no_target_rp0(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    %pad = arith.constant 7.0 : f32
    %zero = arith.constant 0 : index
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f32 right_padding_num = %zero : index
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}
