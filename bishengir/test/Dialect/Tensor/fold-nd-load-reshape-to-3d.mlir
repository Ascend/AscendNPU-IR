//===----------------------------------------------------------------------===
// Fold nD memref load + reshape/collapse -> direct 3D load
//===----------------------------------------------------------------------===

// RUN: bishengir-opt --canonicalize-tensor-reshape -split-input-file %s | FileCheck %s

// Fold 4D load + reshape into a direct 3D load for batch_matmul.
// CHECK-LABEL: func.func @fold_4d_load_reshape_to_3d
// CHECK-NOT: memref.alloc() : memref<1x1x16x32xf16>
// CHECK-NOT: tensor.reshape
// CHECK: %[[RC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1, 16, 32], strides: [512, 32, 1]
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x16x32xf16>
// CHECK: memref.copy %[[RC]], %[[ALLOC]]
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<1x16x32xf16>
// CHECK: linalg.batch_matmul ins(%[[T]],
func.func @fold_4d_load_reshape_to_3d(%x_ptr: memref<?xf16>, %y_ptr: memref<?xf16>, %out: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> {
  %shape_x = arith.constant dense<[1, 16, 32]> : tensor<3xi64>
  %shape_y = arith.constant dense<[1, 32, 16]> : tensor<3xi64>
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [512, 512, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>>
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>> to memref<1x1x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16>
  %Y = memref.reinterpret_cast %y_ptr to offset: [0], sizes: [1, 1, 32, 16], strides: [512, 512, 16, 1] : memref<?xf16> to memref<1x1x32x16xf16, strided<[512, 512, 16, 1]>>
  %Y_alloc = memref.alloc() : memref<1x1x32x16xf16>
  memref.copy %Y, %Y_alloc : memref<1x1x32x16xf16, strided<[512, 512, 16, 1]>> to memref<1x1x32x16xf16>
  %Y_t = bufferization.to_tensor %Y_alloc restrict writable : memref<1x1x32x16xf16>
  %X3 = tensor.reshape %X_t(%shape_x) : (tensor<1x1x16x32xf16>, tensor<3xi64>) -> tensor<1x16x32xf16>
  %Y3 = tensor.reshape %Y_t(%shape_y) : (tensor<1x1x32x16xf16>, tensor<3xi64>) -> tensor<1x32x16xf16>
  %r = linalg.batch_matmul ins(%X3, %Y3 : tensor<1x16x32xf16>, tensor<1x32x16xf16>) outs(%out : tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
  return %r : tensor<1x16x16xf32>
}

// -----

// Same fold via collapse_shape (uses the op's reassociation).
// CHECK-LABEL: func.func @fold_4d_load_collapse_to_3d
// CHECK-NOT: memref.alloc() : memref<1x1x16x32xf16>
// CHECK-NOT: tensor.collapse_shape
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1, 16, 32], strides: [512, 32, 1]
// CHECK: memref.alloc() : memref<1x16x32xf16>
func.func @fold_4d_load_collapse_to_3d(%x_ptr: memref<?xf16>) -> tensor<1x16x32xf16> {
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [512, 512, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>>
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>> to memref<1x1x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16>
  %X3 = tensor.collapse_shape %X_t [[0, 1], [2], [3]] : tensor<1x1x16x32xf16> into tensor<1x16x32xf16>
  return %X3 : tensor<1x16x32xf16>
}

// -----

// Negative: to_tensor has another consumer — must NOT fold / must NOT erase copy.
// CHECK-LABEL: func.func @neg_multi_use_to_tensor
// CHECK: memref.alloc() : memref<1x1x16x32xf16>
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: tensor.collapse_shape
func.func @neg_multi_use_to_tensor(%x_ptr: memref<?xf16>) -> (tensor<1x16x32xf16>, tensor<1x1x16x32xf16>) {
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [512, 512, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>>
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>> to memref<1x1x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16>
  %X3 = tensor.collapse_shape %X_t [[0, 1], [2], [3]] : tensor<1x1x16x32xf16> into tensor<1x16x32xf16>
  return %X3, %X_t : tensor<1x16x32xf16>, tensor<1x1x16x32xf16>
}

// -----

// Negative: a real gap inside a collapsed group cannot be represented by the
// direct 3D view.
// CHECK-LABEL: func.func @neg_source_gap_in_collapsed_group
// CHECK: memref.alloc() : memref<2x2x16x32xf16>
// CHECK: tensor.collapse_shape
func.func @neg_source_gap_in_collapsed_group(%x_ptr: memref<?xf16>) -> tensor<4x16x32xf16> {
  // stride[0]=2048 != 2*768, so collapsing [0,1] crosses a real gap.
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [2, 2, 16, 32], strides: [2048, 768, 32, 1] : memref<?xf16> to memref<2x2x16x32xf16, strided<[2048, 768, 32, 1]>>
  %X_alloc = memref.alloc() : memref<2x2x16x32xf16>
  memref.copy %X, %X_alloc : memref<2x2x16x32xf16, strided<[2048, 768, 32, 1]>> to memref<2x2x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<2x2x16x32xf16>
  %X3 = tensor.collapse_shape %X_t [[0, 1], [2], [3]] : tensor<2x2x16x32xf16> into tensor<4x16x32xf16>
  return %X3 : tensor<4x16x32xf16>
}

// -----

// Unit dimensions do not impose a stride relation: arbitrary strides on the
// collapsed singleton group are address-equivalent.
// CHECK-LABEL: func.func @fold_unit_dims_arbitrary_strides
// CHECK-NOT: memref.alloc() : memref<1x1x16x32xf16>
// CHECK-NOT: tensor.collapse_shape
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1, 16, 32], strides: [2048, 32, 1]
// CHECK: memref.alloc() : memref<1x16x32xf16>
func.func @fold_unit_dims_arbitrary_strides(%x_ptr: memref<?xf16>) -> tensor<1x16x32xf16> {
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [2048, 1024, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[2048, 1024, 32, 1]>>
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[2048, 1024, 32, 1]>> to memref<1x1x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16>
  %X3 = tensor.collapse_shape %X_t [[0, 1], [2], [3]] : tensor<1x1x16x32xf16> into tensor<1x16x32xf16>
  return %X3 : tensor<1x16x32xf16>
}

// -----

// Alloc before reinterpret_cast: fold must still dominate (copy inserted at original copy).
// CHECK-LABEL: func.func @fold_alloc_before_reinterp
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x16x32xf16>
// CHECK: %[[RC:.*]] = memref.reinterpret_cast
// CHECK: memref.copy %[[RC]], %[[ALLOC]]
// CHECK: bufferization.to_tensor %[[ALLOC]]
func.func @fold_alloc_before_reinterp(%x_ptr: memref<?xf16>) -> tensor<1x16x32xf16> {
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [512, 512, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>> to memref<1x1x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16>
  %X3 = tensor.collapse_shape %X_t [[0, 1], [2], [3]] : tensor<1x1x16x32xf16> into tensor<1x16x32xf16>
  return %X3 : tensor<1x16x32xf16>
}

// -----

// Intervening side-effecting op between alloc and copy: new copy must stay after it.
// CHECK-LABEL: func.func @fold_preserves_copy_program_point
// CHECK: memref.reinterpret_cast {{.*}} sizes: [1, 16, 32]
// CHECK: memref.alloc() : memref<1x16x32xf16>
// CHECK: memref.store
// CHECK: memref.copy
func.func @fold_preserves_copy_program_point(%x_ptr: memref<?xf16>, %side: memref<f16>) -> tensor<1x16x32xf16> {
  %c0 = arith.constant 0.0 : f16
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [512, 512, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>>
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  memref.store %c0, %side[] : memref<f16>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>> to memref<1x1x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16>
  %X3 = tensor.collapse_shape %X_t [[0, 1], [2], [3]] : tensor<1x1x16x32xf16> into tensor<1x16x32xf16>
  return %X3 : tensor<1x16x32xf16>
}

// -----

// Preserve memory space and allocation alignment.
// CHECK-LABEL: func.func @fold_preserves_memspace_alignment
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1, 16, 32], strides: [512, 32, 1] : memref<?xf16, 3> to memref<1x16x32xf16, strided<[512, 32, 1]>, 3>
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<1x16x32xf16, 3>
func.func @fold_preserves_memspace_alignment(%x_ptr: memref<?xf16, 3>) -> tensor<1x16x32xf16> {
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [512, 512, 32, 1] : memref<?xf16, 3> to memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>, 3>
  %X_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x16x32xf16, 3>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>, 3> to memref<1x1x16x32xf16, 3>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16, 3>
  %X3 = tensor.collapse_shape %X_t [[0, 1], [2], [3]] : tensor<1x1x16x32xf16> into tensor<1x16x32xf16>
  return %X3 : tensor<1x16x32xf16>
}

// -----

// Preserve tensor encoding on the folded result type via tensor.cast.
// CHECK-LABEL: func.func @fold_preserves_tensor_encoding
// CHECK-NOT: tensor.reshape
// CHECK: %[[T:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<1x16x32xf16>
// CHECK: %[[C:.*]] = tensor.cast %[[T]] : tensor<1x16x32xf16> to tensor<1x16x32xf16, "enc">
// CHECK: return %[[C]]
func.func @fold_preserves_tensor_encoding(%x_ptr: memref<?xf16>) -> tensor<1x16x32xf16, "enc"> {
  %shape = arith.constant dense<[1, 16, 32]> : tensor<3xi64>
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [512, 512, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>>
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[512, 512, 32, 1]>> to memref<1x1x16x32xf16>
  %X_t = bufferization.to_tensor %X_alloc restrict writable : memref<1x1x16x32xf16>
  %X3 = tensor.reshape %X_t(%shape) : (tensor<1x1x16x32xf16>, tensor<3xi64>) -> tensor<1x16x32xf16, "enc">
  return %X3 : tensor<1x16x32xf16, "enc">
}

// -----

// Exact frontend shape: a contiguous batch-2 load followed by a same-rank
// linear-order reshape. Retarget the one copy, preserve offset/alignment and
// to_tensor flags, and allow multiple consumers of the final tensor.
// CHECK-LABEL: func.func @fold_batch2_same_rank_reshape
// CHECK-NOT: memref.alloc() {{.*}} : memref<2x64x32xf32>
// CHECK-NOT: tensor.reshape
// CHECK: %[[SRC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [13], sizes: [2, 32, 64], strides: [2048, 64, 1]
// CHECK: %[[DST:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2x32x64xf32>
// CHECK-COUNT-1: memref.copy %[[SRC]], %[[DST]]
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[DST]] restrict writable : memref<2x32x64xf32>
// CHECK: return %[[T]], %[[T]] : tensor<2x32x64xf32>, tensor<2x32x64xf32>
func.func @fold_batch2_same_rank_reshape(%base: memref<?xf32>) -> (tensor<2x32x64xf32>, tensor<2x32x64xf32>) {
  %shape = arith.constant dense<[2, 32, 64]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [13], sizes: [2, 64, 32], strides: [2048, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[2048, 32, 1], offset: 13>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x64x32xf32>
  memref.copy %src, %alloc : memref<2x64x32xf32, strided<[2048, 32, 1], offset: 13>> to memref<2x64x32xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x64x32xf32>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<2x64x32xf32>, tensor<3xi64>) -> tensor<2x32x64xf32>
  return %reshaped, %reshaped : tensor<2x32x64xf32>, tensor<2x32x64xf32>
}

// -----

// A gap between batches is legal when the reshape is wholly within each
// batch. Preserve that outer stride instead of requiring global contiguity.
// CHECK-LABEL: func.func @fold_same_rank_with_batch_gap
// CHECK-NOT: memref.alloc() {{.*}} : memref<2x64x32xf32>
// CHECK-NOT: tensor.reshape
// CHECK: %[[SRC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [2, 32, 64], strides: [4096, 64, 1]
// CHECK: %[[DST:.*]] = memref.alloc() : memref<2x32x64xf32>
// CHECK-COUNT-1: memref.copy %[[SRC]], %[[DST]]
func.func @fold_same_rank_with_batch_gap(%base: memref<?xf32>) -> tensor<2x32x64xf32> {
  %shape = arith.constant dense<[2, 32, 64]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [2, 64, 32], strides: [4096, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[4096, 32, 1]>>
  %alloc = memref.alloc() : memref<2x64x32xf32>
  memref.copy %src, %alloc : memref<2x64x32xf32, strided<[4096, 32, 1]>> to memref<2x64x32xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x64x32xf32>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<2x64x32xf32>, tensor<3xi64>) -> tensor<2x32x64xf32>
  return %reshaped : tensor<2x32x64xf32>
}

// -----

// Coprime dimensions need the strict contiguous flatten fallback because a
// loose expand/collapse reassociation cannot factor 6x10 directly into 4x15.
// Also preserve the absence of both to_tensor flags.
// CHECK-LABEL: func.func @fold_coprime_same_rank
// CHECK-NOT: memref.alloc() {{.*}} : memref<6x10xf32>
// CHECK-NOT: tensor.reshape
// CHECK: %[[SRC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [4, 15], strides: [15, 1]
// CHECK: %[[DST:.*]] = memref.alloc() : memref<4x15xf32>
// CHECK-COUNT-1: memref.copy %[[SRC]], %[[DST]]
// CHECK: bufferization.to_tensor %[[DST]] : memref<4x15xf32>
func.func @fold_coprime_same_rank(%base: memref<?xf32>) -> tensor<4x15xf32> {
  %shape = arith.constant dense<[4, 15]> : tensor<2xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [6, 10], strides: [10, 1] : memref<?xf32> to memref<6x10xf32, strided<[10, 1]>>
  %alloc = memref.alloc() : memref<6x10xf32>
  memref.copy %src, %alloc : memref<6x10xf32, strided<[10, 1]>> to memref<6x10xf32>
  %tensor = bufferization.to_tensor %alloc : memref<6x10xf32>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<6x10xf32>, tensor<2xi64>) -> tensor<4x15xf32>
  return %reshaped : tensor<4x15xf32>
}

// -----

// The replay is shape- and element-type-generic: this rank-changing f16 case
// needs both collapse and expansion, rather than a pure collapse to rank 3.
// CHECK-LABEL: func.func @fold_other_shape_rank_f16
// CHECK-NOT: memref.alloc() {{.*}} : memref<3x4x5x2xf16>
// CHECK-NOT: tensor.reshape
// CHECK: %[[SRC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [3, 10, 4], strides: [40, 4, 1]
// CHECK: %[[DST:.*]] = memref.alloc() : memref<3x10x4xf16>
// CHECK-COUNT-1: memref.copy %[[SRC]], %[[DST]]
// CHECK: bufferization.to_tensor %[[DST]] restrict : memref<3x10x4xf16>
func.func @fold_other_shape_rank_f16(%base: memref<?xf16>) -> tensor<3x10x4xf16> {
  %shape = arith.constant dense<[3, 10, 4]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [3, 4, 5, 2], strides: [40, 10, 2, 1] : memref<?xf16> to memref<3x4x5x2xf16, strided<[40, 10, 2, 1]>>
  %alloc = memref.alloc() : memref<3x4x5x2xf16>
  memref.copy %src, %alloc : memref<3x4x5x2xf16, strided<[40, 10, 2, 1]>> to memref<3x4x5x2xf16>
  %tensor = bufferization.to_tensor %alloc restrict : memref<3x4x5x2xf16>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<3x4x5x2xf16>, tensor<3xi64>) -> tensor<3x10x4xf16>
  return %reshaped : tensor<3x10x4xf16>
}

// -----

// Negative: a non-identity destination allocation is part of the old transfer
// contract and cannot be replaced by an identity-layout allocation.
// CHECK-LABEL: func.func @neg_nonidentity_alloc
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x64x32xf32, strided<[4096, 64, 2]>>
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK: bufferization.to_tensor %[[ALLOC]] restrict writable
func.func @neg_nonidentity_alloc(%base: memref<?xf32>) -> tensor<2x32x64xf32> {
  %shape = arith.constant dense<[2, 32, 64]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [2, 64, 32], strides: [2048, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[2048, 32, 1]>>
  %alloc = memref.alloc() : memref<2x64x32xf32, strided<[4096, 64, 2]>>
  memref.copy %src, %alloc : memref<2x64x32xf32, strided<[2048, 32, 1]>> to memref<2x64x32xf32, strided<[4096, 64, 2]>>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x64x32xf32, strided<[4096, 64, 2]>>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<2x64x32xf32>, tensor<3xi64>) -> tensor<2x32x64xf32>
  return %reshaped : tensor<2x32x64xf32>
}

// -----

// Negative: the copy does not initialize the buffer before to_tensor. Do not
// treat a later transfer as the initializing load.
// CHECK-LABEL: func.func @neg_copy_after_to_tensor
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x64x32xf32>
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
func.func @neg_copy_after_to_tensor(%base: memref<?xf32>) -> tensor<2x32x64xf32> {
  %shape = arith.constant dense<[2, 32, 64]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [2, 64, 32], strides: [2048, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[2048, 32, 1]>>
  %alloc = memref.alloc() : memref<2x64x32xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x64x32xf32>
  memref.copy %src, %alloc : memref<2x64x32xf32, strided<[2048, 32, 1]>> to memref<2x64x32xf32>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<2x64x32xf32>, tensor<3xi64>) -> tensor<2x32x64xf32>
  return %reshaped : tensor<2x32x64xf32>
}

// -----

// Negative: unknown transfer metadata may alter copy semantics. Decline
// instead of silently dropping it from the replacement copy.
// CHECK-LABEL: func.func @neg_unknown_copy_metadata
// CHECK: memref.copy %{{.*}}, %{{.*}} {test.transfer_contract = "keep"}
func.func @neg_unknown_copy_metadata(%base: memref<?xf32>) -> tensor<2x32x64xf32> {
  %shape = arith.constant dense<[2, 32, 64]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [2, 64, 32], strides: [2048, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[2048, 32, 1]>>
  %alloc = memref.alloc() : memref<2x64x32xf32>
  memref.copy %src, %alloc {"test.transfer_contract" = "keep"} : memref<2x64x32xf32, strided<[2048, 32, 1]>> to memref<2x64x32xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x64x32xf32>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<2x64x32xf32>, tensor<3xi64>) -> tensor<2x32x64xf32>
  return %reshaped : tensor<2x32x64xf32>
}

// -----

// Keep the retargeted copy at the original program point. A later source write
// must not change the snapshot observed through the tensor.
// CHECK-LABEL: func.func @fold_preserves_source_snapshot
// CHECK: %[[SRC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [2, 32, 64], strides: [2048, 64, 1]
// CHECK: %[[DST:.*]] = memref.alloc() : memref<2x32x64xf32>
// CHECK: memref.copy %[[SRC]], %[[DST]]
// CHECK: memref.store
// CHECK: bufferization.to_tensor %[[DST]] restrict writable
func.func @fold_preserves_source_snapshot(%base: memref<?xf32>, %value: f32) -> tensor<2x32x64xf32> {
  %c0 = arith.constant 0 : index
  %shape = arith.constant dense<[2, 32, 64]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [2, 64, 32], strides: [2048, 32, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[2048, 32, 1]>>
  %alloc = memref.alloc() : memref<2x64x32xf32>
  memref.copy %src, %alloc : memref<2x64x32xf32, strided<[2048, 32, 1]>> to memref<2x64x32xf32>
  memref.store %value, %base[%c0] : memref<?xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x64x32xf32>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<2x64x32xf32>, tensor<3xi64>) -> tensor<2x32x64xf32>
  return %reshaped : tensor<2x32x64xf32>
}

// -----

// Negative: reshaping across padded source rows is not an affine strided view,
// even though source and destination have the same element count.
// CHECK-LABEL: func.func @neg_same_rank_reshape_crosses_row_gaps
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x64x32xf32>
// CHECK: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK: bufferization.to_tensor %[[ALLOC]] restrict writable
func.func @neg_same_rank_reshape_crosses_row_gaps(%base: memref<?xf32>) -> tensor<2x32x64xf32> {
  %shape = arith.constant dense<[2, 32, 64]> : tensor<3xi64>
  %src = memref.reinterpret_cast %base to offset: [0], sizes: [2, 64, 32], strides: [4096, 64, 1] : memref<?xf32> to memref<2x64x32xf32, strided<[4096, 64, 1]>>
  %alloc = memref.alloc() : memref<2x64x32xf32>
  memref.copy %src, %alloc : memref<2x64x32xf32, strided<[4096, 64, 1]>> to memref<2x64x32xf32>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<2x64x32xf32>
  %reshaped = tensor.reshape %tensor(%shape) : (tensor<2x64x32xf32>, tensor<3xi64>) -> tensor<2x32x64xf32>
  return %reshaped : tensor<2x32x64xf32>
}
