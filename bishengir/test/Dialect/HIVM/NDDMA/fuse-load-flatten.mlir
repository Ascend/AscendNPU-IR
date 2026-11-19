// RUN: bishengir-opt %s --hivm-fuse-op-into-load --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" --hivm-flatten-ops --split-input-file | FileCheck %s
// Preserve the axis/rank contracts of fused A6 broadcast loads.
// Ordinary loads still flatten; A3/A5 do not acquire the A6 fused form.
// Bufferize before FlattenOps, including the unfused A3/A5 tensor broadcasts.

// CHECK-LABEL: func.func @fused_fa_row(
// CHECK: hivm.hir.load ins({{.*}} : memref<1x32xf32, {{.*}}>) outs({{.*}} : memref<32x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @fused_fa_row(%gm: memref<32xf32, #hivm.address_space<gm>>) -> tensor<32x32xf32> {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%gm : memref<32xf32, #hivm.address_space<gm>>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %out = tensor.empty() : tensor<32x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%out : tensor<32x32xf32>) dimensions = [0]
  return %b : tensor<32x32xf32>
 }
}

// -----

// CHECK-LABEL: func.func @fused_gemm_scale(
// CHECK: hivm.hir.load ins({{.*}} : memref<1x64xf32, {{.*}}>) outs({{.*}} : memref<64x64xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @fused_gemm_scale(%gm: memref<1x64xf32, #hivm.address_space<gm>>) -> tensor<64x64xf32> {
  %row = memref.alloc() : memref<1x64xf32>
  hivm.hir.load ins(%gm : memref<1x64xf32, #hivm.address_space<gm>>) outs(%row : memref<1x64xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<1x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %c = tensor.collapse_shape %t [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
  %b = linalg.broadcast ins(%c : tensor<64xf32>) outs(%out : tensor<64x64xf32>) dimensions = [0]
  return %b : tensor<64x64xf32>
 }
}

// -----

// CHECK-LABEL: func.func @ordinary_a6(
// CHECK: memref.collapse_shape %arg0 {{.*}} : memref<4x32xf32, #hivm.address_space<gm>> into memref<128xf32, #hivm.address_space<gm>>
// CHECK: memref.collapse_shape %arg1 {{.*}} : memref<4x32xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-NOT: broadcast_axes
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @ordinary_a6(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @legacy_right_a3(
// CHECK: memref.collapse_shape %arg0 {{.*}} : memref<4x32xf32, #hivm.address_space<gm>> into memref<128xf32, #hivm.address_space<gm>>
// CHECK: memref.collapse_shape %arg1 {{.*}} : memref<4x32xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-SAME: pad_mode = <PadValue>
// CHECK-NOT: broadcast_axes
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
  func.func @legacy_right_a3(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
    %pad = arith.constant 7.0 : f32
    %right = arith.constant 8 : index
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f32 right_padding_num = %right : index
    return
  }
}

// -----

// CHECK-LABEL: func.func @legacy_right_a5(
// CHECK: memref.collapse_shape %arg0 {{.*}} : memref<4x32xf32, #hivm.address_space<gm>> into memref<128xf32, #hivm.address_space<gm>>
// CHECK: memref.collapse_shape %arg1 {{.*}} : memref<4x32xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-SAME: pad_mode = <PadValue>
// CHECK-NOT: broadcast_axes
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @legacy_right_a5(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
    %pad = arith.constant 7.0 : f32
    %right = arith.constant 8 : index
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f32 right_padding_num = %right : index
    return
  }
}


// -----

// CHECK-LABEL: func.func @fused_inner_axis(
// CHECK: hivm.hir.load ins({{.*}} : memref<32x1xf32, {{.*}}>) outs({{.*}} : memref<32x32xf32, #hivm.address_space<ub>>) broadcast_axes = [1]
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: linalg.broadcast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @fused_inner_axis(%gm: memref<32xf32, #hivm.address_space<gm>>) -> tensor<32x32xf32> {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%gm : memref<32xf32, #hivm.address_space<gm>>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %out = tensor.empty() : tensor<32x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%out : tensor<32x32xf32>) dimensions = [1]
  return %b : tensor<32x32xf32>
 }
}


// -----

// CHECK-LABEL: func.func @rank3_broadcast(
// CHECK: hivm.hir.load ins(%arg0 : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%arg1 : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @rank3_broadcast(%src: memref<1x1x32xf32, #hivm.address_space<gm>>, %dst: memref<2x4x32xf32, #hivm.address_space<ub>>) {
    hivm.hir.load ins(%src : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
    return
  }
}


// -----

// CHECK-LABEL: func.func @a3(
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: linalg.broadcast
// CHECK-NOT: broadcast_axes
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
 func.func @a3(%gm: memref<32xf32, #hivm.address_space<gm>>) -> tensor<4x32xf32> {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%gm : memref<32xf32, #hivm.address_space<gm>>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %out = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%out : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
 }
}


// -----

// CHECK-LABEL: func.func @a5(
// CHECK: hivm.hir.load
// CHECK-NOT: broadcast_axes
// CHECK: linalg.broadcast
// CHECK-NOT: broadcast_axes
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950">} {
 func.func @a5(%gm: memref<32xf32, #hivm.address_space<gm>>) -> tensor<4x32xf32> {
  %row = memref.alloc() : memref<32xf32>
  hivm.hir.load ins(%gm : memref<32xf32, #hivm.address_space<gm>>) outs(%row : memref<32xf32>)
  %t = bufferization.to_tensor %row restrict writable : memref<32xf32>
  %out = tensor.empty() : tensor<4x32xf32>
  %b = linalg.broadcast ins(%t : tensor<32xf32>) outs(%out : tensor<4x32xf32>) dimensions = [0]
  return %b : tensor<4x32xf32>
 }
}


// -----

// CHECK-LABEL: func.func @ordinary_a3(
// CHECK: memref.collapse_shape %arg0 {{.*}} : memref<4x32xf32, #hivm.address_space<gm>> into memref<128xf32, #hivm.address_space<gm>>
// CHECK: memref.collapse_shape %arg1 {{.*}} : memref<4x32xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-NOT: broadcast_axes
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
  func.func @ordinary_a3(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>)
    return
  }
}


// -----

// CHECK-LABEL: func.func @ordinary_a5(
// CHECK: memref.collapse_shape %arg0 {{.*}} : memref<4x32xf32, #hivm.address_space<gm>> into memref<128xf32, #hivm.address_space<gm>>
// CHECK: memref.collapse_shape %arg1 {{.*}} : memref<4x32xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-NOT: broadcast_axes
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @ordinary_a5(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>)
    return
  }
}
