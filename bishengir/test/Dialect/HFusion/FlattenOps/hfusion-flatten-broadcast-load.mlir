// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy multi-dynamic-shape=false register-based=true" --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy multi-dynamic-shape=false register-based=true" --hivm-flatten-ops --split-input-file | FileCheck %s --check-prefix=ORD
// Broadcast axes are a rank-preserving contract, including through a memory
// space cast. Constrain only participating dimensions, not the whole function.

// CHECK-LABEL: func.func @outer_axis(
// CHECK: memref.expand_shape {{.*}} into memref<1x32xf32>
// CHECK: memref.memory_space_cast {{.*}} : memref<1x32xf32> to memref<1x32xf32, #hivm.address_space<gm>>
// CHECK: hivm.hir.load ins({{.*}} : memref<1x32xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<16x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @outer_axis(%src: memref<32xf32>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    %expanded = memref.expand_shape %src [[0, 1]] output_shape [1, 32] : memref<32xf32> into memref<1x32xf32>
    %cast = memref.memory_space_cast %expanded : memref<1x32xf32> to memref<1x32xf32, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%cast : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<16x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @inner_axis(
// CHECK: memref.expand_shape {{.*}} into memref<32x1xf32>
// CHECK: memref.memory_space_cast {{.*}} : memref<32x1xf32> to memref<32x1xf32, #hivm.address_space<gm>>
// CHECK: hivm.hir.load ins({{.*}} : memref<32x1xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<32x16xf32, #hivm.address_space<ub>>) broadcast_axes = [1]
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @inner_axis(%src: memref<32xf32>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    %expanded = memref.expand_shape %src [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
    %cast = memref.memory_space_cast %expanded : memref<32x1xf32> to memref<32x1xf32, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<32x16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%cast : memref<32x1xf32, #hivm.address_space<gm>>) outs(%dst : memref<32x16xf32, #hivm.address_space<ub>>) broadcast_axes = [1]
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @rank3_axes(
// CHECK: memref.expand_shape {{.*}} into memref<1x1x32xf32>
// CHECK: memref.memory_space_cast {{.*}} : memref<1x1x32xf32> to memref<1x1x32xf32, #hivm.address_space<gm>>
// CHECK: hivm.hir.load ins({{.*}} : memref<1x1x32xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @rank3_axes(%src: memref<32xf32>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    %expanded = memref.expand_shape %src [[0, 1, 2]] output_shape [1, 1, 32] : memref<32xf32> into memref<1x1x32xf32>
    %cast = memref.memory_space_cast %expanded : memref<1x1x32xf32> to memref<1x1x32xf32, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<2x4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%cast : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @direct_rank3(
// CHECK: hivm.hir.load ins(%arg0 : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%arg1 : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @direct_rank3(%src: memref<1x1x32xf32, #hivm.address_space<gm>>, %dst: memref<2x4x32xf32, #hivm.address_space<ub>>) {
    hivm.hir.load ins(%src : memref<1x1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0, 1]
    return
  }
}

// -----

// CHECK-LABEL: func.func @ordinary_a6(
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-NOT: broadcast_axes
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
// ORD-LABEL: func.func @ordinary_a6(
// ORD: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// ORD-NOT: broadcast_axes
// ORD: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @ordinary_a6(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>)
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @ordinary_a3(
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-NOT: broadcast_axes
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
// ORD-LABEL: func.func @ordinary_a3(
// ORD: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// ORD-NOT: broadcast_axes
// ORD: return
module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
  func.func @ordinary_a3(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>)
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @ordinary_a5(
// CHECK: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-NOT: broadcast_axes
// CHECK: memref.copy {{.*}} : memref<128xf32> to memref<128xf32>
// CHECK: return
// ORD-LABEL: func.func @ordinary_a5(
// ORD: hivm.hir.load ins({{.*}} : memref<128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<128xf32, #hivm.address_space<ub>>)
// ORD-NOT: broadcast_axes
// ORD: return
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @ordinary_a5(%src: memref<4x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>, %other: memref<4x32xf32>, %out: memref<4x32xf32>) {
    hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>)
    memref.copy %other, %out : memref<4x32xf32> to memref<4x32xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @source_only_rank_loss(
// CHECK: memref.memory_space_cast {{.*}} : memref<1x32xf32> to memref<1x32xf32, #hivm.address_space<gm>>
// CHECK: hivm.hir.load ins({{.*}} : memref<1x32xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<16x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @source_only_rank_loss(%arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 32], strides: [128, 1] : memref<?xf32> to memref<16x32xf32, strided<[128, 1]>>
    %alloc = memref.alloc() : memref<16x32xf32>
    memref.copy %reinterpret_cast, %alloc : memref<16x32xf32, strided<[128, 1]>> to memref<16x32xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
    %expand_shape = memref.expand_shape %reinterpret_cast_0 [[0, 1]] output_shape [1, 32] : memref<32xf32, strided<[1]>> into memref<1x32xf32>
    %alloc_1 = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %expand_shape : memref<1x32xf32> to memref<1x32xf32, #hivm.address_space<gm>>
    hivm.hir.load ins(%memspacecast : memref<1x32xf32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    %1 = tensor.empty() : tensor<16x32xf32>
    %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x32xf32, #hivm.address_space<ub>>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %2 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%1 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 32], strides: [128, 1] : memref<?xf32> to memref<16x32xf32, strided<[128, 1]>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<16x32xf32>, memref<16x32xf32, strided<[128, 1]>>) -> ()
    return
  }
}
