// RUN: bishengir-opt -test-bubble-up-bufferization %s -split-input-file | FileCheck %s


// CHECK: Successfully bubble up bufferization
func.func @bubble_up_alloc_load_to_tensor(%arg0: memref<64x32xf32>) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32>
  hivm.hir.load ins(%arg0 : memref<64x32xf32>) outs(%alloc : memref<64x32xf32>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %slice = tensor.extract_slice %tensor[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %slice : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_alloc_dim0(%arg0: memref<64xf32>) -> tensor<32xf32> {
  %alloc = memref.alloc() : memref<64xf32>
  hivm.hir.load ins(%arg0 : memref<64xf32>) outs(%alloc : memref<64xf32>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
  %slice = tensor.extract_slice %tensor[32] [32] [1] {to_be_bubbled_slice}
      : tensor<64xf32> to tensor<32xf32>
  return %slice : tensor<32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_alloc_dim1(%arg0: memref<64x32xf32>) -> tensor<64x16xf32> {
  %alloc = memref.alloc() : memref<64x32xf32>
  hivm.hir.load ins(%arg0 : memref<64x32xf32>) outs(%alloc : memref<64x32xf32>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %slice = tensor.extract_slice %tensor[0, 16] [64, 16] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<64x16xf32>
  return %slice : tensor<64x16xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_memory_space_cast(%arg0: memref<64x32xf32>) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  %cast = memref.memory_space_cast %alloc
      : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
  annotation.mark %cast {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
      : memref<64x32xf32>
  hivm.hir.load ins(%arg0 : memref<64x32xf32>) outs(%cast : memref<64x32xf32>)
  %tensor = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
  %slice = tensor.extract_slice %tensor[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %slice : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_subview_alloc(%arg0: memref<64x32xf32>) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32>
  %dst = memref.subview %alloc[32, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: 1024>>
  %src = memref.subview %arg0[32, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: 1024>>
  hivm.hir.load ins(%src : memref<32x32xf32, strided<[32, 1], offset: 1024>>)
      outs(%dst : memref<32x32xf32, strided<[32, 1], offset: 1024>>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %slice = tensor.extract_slice %tensor[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %slice : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_reinterpret_cast(%arg0: memref<?xf32>) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32>
  %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 32], strides: [32, 1]
      : memref<?xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<64x32xf32, strided<[32, 1], offset: ?>>)
      outs(%alloc : memref<64x32xf32>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %slice = tensor.extract_slice %tensor[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %slice : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_alloc_odd_buffer_size(
    %arg0: memref<?xf16>, %arg1: index) -> tensor<?x80xf16> {
  %alloc = memref.alloc() : memref<115x80xf16>
  %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [115, 80], strides: [80, 1]
      : memref<?xf16> to memref<115x80xf16, strided<[80, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<115x80xf16, strided<[80, 1], offset: ?>>)
      outs(%alloc : memref<115x80xf16>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<115x80xf16>
  %slice = tensor.extract_slice %tensor[0, 0] [%arg1, 80] [1, 1] {to_be_bubbled_slice}
      : tensor<115x80xf16> to tensor<?x80xf16>
  return %slice : tensor<?x80xf16>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_subview_alloc_odd_buffer_size(
    %arg0: memref<?xf16>, %arg1: index) -> tensor<?x80xf16> {
  %alloc = memref.alloc() : memref<115x80xf16>
  %dst = memref.subview %alloc[0, 0] [115, 80] [1, 1]
      : memref<115x80xf16> to memref<115x80xf16, strided<[80, 1]>>
  %src_base = memref.reinterpret_cast %arg0 to offset: [0], sizes: [115, 80], strides: [80, 1]
      : memref<?xf16> to memref<115x80xf16, strided<[80, 1], offset: ?>>
  %src = memref.subview %src_base[0, 0] [115, 80] [1, 1]
      : memref<115x80xf16, strided<[80, 1], offset: ?>> to memref<115x80xf16, strided<[80, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<115x80xf16, strided<[80, 1], offset: ?>>)
      outs(%dst : memref<115x80xf16, strided<[80, 1]>>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<115x80xf16>
  %slice = tensor.extract_slice %tensor[0, 0] [%arg1, 80] [1, 1] {to_be_bubbled_slice}
      : tensor<115x80xf16> to tensor<?x80xf16>
  return %slice : tensor<?x80xf16>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @no_transform_without_to_tensor(%arg0: tensor<64x32xf32>) -> tensor<32x32xf32> {
  %slice = tensor.extract_slice %arg0[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %slice : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_subview_dropped_dim(%arg0: memref<2x8x16x16xf16>) -> tensor<1x8x1x16x16xf16> {
  %alloc = memref.alloc() : memref<2x8x1x16x16xf16>
  %dst = memref.subview %alloc[0, 0, 0, 0, 0] [2, 8, 1, 16, 16] [1, 1, 1, 1, 1]
      : memref<2x8x1x16x16xf16> to memref<2x8x16x16xf16, strided<[2048, 256, 16, 1]>>
  hivm.hir.load ins(%arg0 : memref<2x8x16x16xf16>) outs(%dst : memref<2x8x16x16xf16, strided<[2048, 256, 16, 1]>>)
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x8x1x16x16xf16>
  %slice = tensor.extract_slice %tensor[0, 0, 0, 0, 0] [1, 8, 1, 16, 16] [1, 1, 1, 1, 1] {to_be_bubbled_slice}
      : tensor<2x8x1x16x16xf16> to tensor<1x8x1x16x16xf16>
  return %slice : tensor<1x8x1x16x16xf16>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_to_tensor_same_alloc(%arg0: memref<64x32xf32>)
    -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %alloc = memref.alloc() : memref<64x32xf32>
  hivm.hir.load ins(%arg0 : memref<64x32xf32>) outs(%alloc : memref<64x32xf32>)
  %t0 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %t1 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %s0 = tensor.extract_slice %t0[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %s1 = tensor.extract_slice %t1[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %s0, %s1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_alloc_dynamic_offset(
    %arg0: memref<64x32xf32>, %arg1: memref<64x32xf32>, %offset: index)
    -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %alloc0 = memref.alloc() : memref<64x32xf32>
  %alloc1 = memref.alloc() : memref<64x32xf32>
  %src0 = memref.subview %arg0[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  %dst0 = memref.subview %alloc0[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%src0 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      outs(%dst0 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %src1 = memref.subview %arg1[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  %dst1 = memref.subview %alloc1[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%src1 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      outs(%dst1 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<64x32xf32>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<64x32xf32>
  %s0 = tensor.extract_slice %t0[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %s1 = tensor.extract_slice %t1[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %s0, %s1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_to_tensor_elementwise(
    %arg0: memref<64x32xf32>, %offset: index) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32>
  %src = memref.subview %arg0[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  %dst = memref.subview %alloc[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      outs(%dst : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %t0 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %t1 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %s0 = tensor.extract_slice %t0[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %s1 = tensor.extract_slice %t1[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %sum = arith.addf %s0, %s1 : tensor<32x32xf32>
  return %sum : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_to_tensor_dropped_dim_static(%arg0: memref<2x8x16x16xf16>)
    -> (tensor<1x8x1x16x16xf16>, tensor<1x8x1x16x16xf16>) {
  %alloc = memref.alloc() : memref<2x8x1x16x16xf16>
  %dst = memref.subview %alloc[0, 0, 0, 0, 0] [2, 8, 1, 16, 16] [1, 1, 1, 1, 1]
      : memref<2x8x1x16x16xf16> to memref<2x8x16x16xf16, strided<[2048, 256, 16, 1]>>
  hivm.hir.load ins(%arg0 : memref<2x8x16x16xf16>) outs(%dst : memref<2x8x16x16xf16, strided<[2048, 256, 16, 1]>>)
  %t0 = bufferization.to_tensor %alloc restrict writable : memref<2x8x1x16x16xf16>
  %t1 = bufferization.to_tensor %alloc restrict writable : memref<2x8x1x16x16xf16>
  %s0 = tensor.extract_slice %t0[0, 0, 0, 0, 0] [1, 8, 1, 16, 16] [1, 1, 1, 1, 1] {to_be_bubbled_slice}
      : tensor<2x8x1x16x16xf16> to tensor<1x8x1x16x16xf16>
  %s1 = tensor.extract_slice %t1[0, 0, 0, 0, 0] [1, 8, 1, 16, 16] [1, 1, 1, 1, 1] {to_be_bubbled_slice}
      : tensor<2x8x1x16x16xf16> to tensor<1x8x1x16x16xf16>
  return %s0, %s1 : tensor<1x8x1x16x16xf16>, tensor<1x8x1x16x16xf16>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_to_tensor_dropped_dim_dynamic(
    %arg0: memref<2x8x16x16xf16>, %offset: index)
    -> (tensor<1x8x1x16x16xf16>, tensor<1x8x1x16x16xf16>) {
  %alloc = memref.alloc() : memref<2x8x1x16x16xf16>
  %dst = memref.subview %alloc[0, 0, 0, 0, 0] [2, 8, 1, 16, 16] [1, 1, 1, 1, 1]
      : memref<2x8x1x16x16xf16> to memref<2x8x16x16xf16, strided<[2048, 256, 16, 1]>>
  hivm.hir.load ins(%arg0 : memref<2x8x16x16xf16>) outs(%dst : memref<2x8x16x16xf16, strided<[2048, 256, 16, 1]>>)
  %t0 = bufferization.to_tensor %alloc restrict writable : memref<2x8x1x16x16xf16>
  %t1 = bufferization.to_tensor %alloc restrict writable : memref<2x8x1x16x16xf16>
  %s0 = tensor.extract_slice %t0[%offset, 0, 0, 0, 0] [1, 8, 1, 16, 16] [1, 1, 1, 1, 1] {to_be_bubbled_slice}
      : tensor<2x8x1x16x16xf16> to tensor<1x8x1x16x16xf16>
  %s1 = tensor.extract_slice %t1[%offset, 0, 0, 0, 0] [1, 8, 1, 16, 16] [1, 1, 1, 1, 1] {to_be_bubbled_slice}
      : tensor<2x8x1x16x16xf16> to tensor<1x8x1x16x16xf16>
  return %s0, %s1 : tensor<1x8x1x16x16xf16>, tensor<1x8x1x16x16xf16>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_subview_multi_load(%arg0: memref<64x32xf32>) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32>
  %tile = memref.subview %alloc[32, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: 1024>>
  %dst0 = memref.subview %tile[0, 0] [16, 32] [1, 1]
      : memref<32x32xf32, strided<[32, 1], offset: 1024>> to memref<16x32xf32, strided<[32, 1], offset: 1024>>
  %dst1 = memref.subview %tile[16, 0] [16, 32] [1, 1]
      : memref<32x32xf32, strided<[32, 1], offset: 1024>> to memref<16x32xf32, strided<[32, 1], offset: 1536>>
  %src0 = memref.subview %arg0[32, 0] [16, 32] [1, 1]
      : memref<64x32xf32> to memref<16x32xf32, strided<[32, 1], offset: 1024>>
  %src1 = memref.subview %arg0[48, 0] [16, 32] [1, 1]
      : memref<64x32xf32> to memref<16x32xf32, strided<[32, 1], offset: 1536>>
  hivm.hir.load ins(%src0 : memref<16x32xf32, strided<[32, 1], offset: 1024>>) outs(%dst0 : memref<16x32xf32, strided<[32, 1], offset: 1024>>)
  hivm.hir.load ins(%src1 : memref<16x32xf32, strided<[32, 1], offset: 1536>>) outs(%dst1 : memref<16x32xf32, strided<[32, 1], offset: 1536>>)
  %t0 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %t1 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %s0 = tensor.extract_slice %t0[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %s1 = tensor.extract_slice %t1[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %sum = arith.addf %s0, %s1 : tensor<32x32xf32>
  return %sum : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_shared_subview_users(%arg0: memref<64x32xf32>) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32>
  %shared = memref.subview %alloc[32, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: 1024>>
  %inner = memref.subview %shared[0, 0] [32, 32] [1, 1]
      : memref<32x32xf32, strided<[32, 1], offset: 1024>> to memref<32x32xf32, strided<[32, 1], offset: 1024>>
  %src = memref.subview %arg0[32, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: 1024>>
  hivm.hir.load ins(%src : memref<32x32xf32, strided<[32, 1], offset: 1024>>) outs(%inner : memref<32x32xf32, strided<[32, 1], offset: 1024>>)
  %t = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
  %s = tensor.extract_slice %t[32, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %s : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_memspace_cast_users(
    %arg0: memref<64x32xf32>, %arg1: memref<64x32xf32>, %offset: index)
    -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  %cast = memref.memory_space_cast %alloc
      : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
  annotation.mark %cast {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
      : memref<64x32xf32>
  %dst0 = memref.subview %cast[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  %dst1 = memref.subview %cast[0, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  %src0 = memref.subview %arg0[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  %src1 = memref.subview %arg1[0, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  hivm.hir.load ins(%src0 : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%dst0 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  hivm.hir.load ins(%src1 : memref<32x32xf32, strided<[32, 1]>>) outs(%dst1 : memref<32x32xf32, strided<[32, 1]>>)
  %t0 = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
  %t1 = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
  %s0 = tensor.extract_slice %t0[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %s1 = tensor.extract_slice %t1[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %s0, %s1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_dual_memspace_cast_users(
    %arg0: memref<64x32xf32>, %offset: index) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  %cast0 = memref.memory_space_cast %alloc
      : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
  %cast1 = memref.memory_space_cast %alloc
      : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
  annotation.mark %cast0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
      : memref<64x32xf32>
  %dst = memref.subview %cast1[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  %src = memref.subview %arg0[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%dst : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %t0 = bufferization.to_tensor %cast0 restrict writable : memref<64x32xf32>
  %t1 = bufferization.to_tensor %cast1 restrict writable : memref<64x32xf32>
  %s0 = tensor.extract_slice %t0[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %s1 = tensor.extract_slice %t1[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %s0, %s1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_multi_load_subview_cast(
    %arg0: memref<64x32xf32>, %arg1: memref<64x32xf32>, %offset: index) -> tensor<32x32xf32> {
  %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  %cast = memref.memory_space_cast %alloc
      : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
  %tile = memref.subview %cast[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  %dst0 = memref.subview %tile[0, 0] [32, 16] [1, 1]
      : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<32x16xf32, strided<[32, 1], offset: ?>>
  %dst1 = memref.subview %tile[0, 16] [32, 16] [1, 1]
      : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<32x16xf32, strided<[32, 1], offset: ?>>
  %src0 = memref.subview %arg0[%offset, 0] [32, 16] [1, 1]
      : memref<64x32xf32> to memref<32x16xf32, strided<[32, 1], offset: ?>>
  %src1 = memref.subview %arg1[%offset, 16] [32, 16] [1, 1]
      : memref<64x32xf32> to memref<32x16xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%src0 : memref<32x16xf32, strided<[32, 1], offset: ?>>) outs(%dst0 : memref<32x16xf32, strided<[32, 1], offset: ?>>)
  hivm.hir.load ins(%src1 : memref<32x16xf32, strided<[32, 1], offset: ?>>) outs(%dst1 : memref<32x16xf32, strided<[32, 1], offset: ?>>)
  %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
  %s = tensor.extract_slice %t[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %s : tensor<32x32xf32>
}

// -----
// CHECK: Successfully bubble up bufferization
func.func @bubble_up_subview_shared_by_mark_and_load(
    %arg0: memref<64x32xf32>, %offset: index) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  %cast = memref.memory_space_cast %alloc
      : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
  %tile = memref.subview %cast[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  annotation.mark %tile {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
      : memref<32x32xf32, strided<[32, 1], offset: ?>>
  %src = memref.subview %arg0[%offset, 0] [32, 32] [1, 1]
      : memref<64x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  hivm.hir.load ins(%src : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%tile : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %t0 = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
  %t1 = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
  %s0 = tensor.extract_slice %t0[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  %s1 = tensor.extract_slice %t1[%offset, 0] [32, 32] [1, 1] {to_be_bubbled_slice}
      : tensor<64x32xf32> to tensor<32x32xf32>
  return %s0, %s1 : tensor<32x32xf32>, tensor<32x32xf32>
}
