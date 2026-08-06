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

// Negative: non-contiguous strides cannot be collapsed.
// CHECK-LABEL: func.func @neg_noncontiguous_strides
// CHECK: memref.alloc() : memref<1x1x16x32xf16>
// CHECK: tensor.collapse_shape
func.func @neg_noncontiguous_strides(%x_ptr: memref<?xf16>) -> tensor<1x16x32xf16> {
  // stride[0]=1024 != 1*512, so collapsing [0,1] is non-contiguous.
  %X = memref.reinterpret_cast %x_ptr to offset: [0], sizes: [1, 1, 16, 32], strides: [1024, 512, 32, 1] : memref<?xf16> to memref<1x1x16x32xf16, strided<[1024, 512, 32, 1]>>
  %X_alloc = memref.alloc() : memref<1x1x16x32xf16>
  memref.copy %X, %X_alloc : memref<1x1x16x32xf16, strided<[1024, 512, 32, 1]>> to memref<1x1x16x32xf16>
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
