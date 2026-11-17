// RUN: bishengir-opt %s --hivm-propagate-convert-layout="allow-agnostic-ops=false" --canonicalize --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --hivm-propagate-convert-layout="allow-agnostic-ops=false" --canonicalize --hivm-combine-optimized-convert-layout --hivm-convert-layout-to-transpose --split-input-file | FileCheck %s --check-prefix=LOWER

// Reduced from GEN-1091: the loop-carried vector is copied to L1 before its
// 1D-to-2D reshape. Fractalization must read the UB value before that copy.
// CHECK-LABEL: func.func @while_copy_reshape(
// CHECK:       ^bb0(%[[ITER:.*]]: tensor<1024xf32>):
// CHECK-NOT:   memref<1024xf32, #hivm.address_space<cbuf>>
// CHECK:       %[[L1:.*]] = memref.alloc() : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
// CHECK:       %[[TENSOR:.*]] = bufferization.to_tensor %[[L1]]
// CHECK:       %[[VIEW:.*]] = tensor.expand_shape %[[ITER]]
// CHECK:       %[[CONVERTED:.*]] = hivm.hir.convert_layout %[[VIEW]]
// CHECK:       hivm.hir.copy ins(%[[CONVERTED]] : tensor<4x2x16x8xf32>) outs(%[[L1]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>)
// CHECK-NOT:   hivm.hir.copy
// CHECK:       hivm.hir.mmadL1 ins(%[[TENSOR]],
// LOWER-LABEL: func.func @while_copy_reshape(
// LOWER:       ^bb0(%[[ITER:.*]]: tensor<1024xf32>):
// LOWER-NOT:   memref<1024xf32, #hivm.address_space<cbuf>>
// LOWER:       %[[VIEW:.*]] = tensor.expand_shape %[[ITER]]
// LOWER:       %[[EXPANDED:.*]] = tensor.expand_shape %[[VIEW]]
// LOWER:       hivm.hir.vtranspose ins(%[[EXPANDED]]
// LOWER:       hivm.hir.copy
// LOWER:       hivm.hir.mmadL1
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @while_copy_reshape(%input: tensor<1024xf32>, %rhs: tensor<4x2x16x8xf32>, %run: i1) -> tensor<1024xf32> {
  %c32 = arith.constant 32 : index
  %true = arith.constant true
  %false = arith.constant false
  %result = scf.while (%value = %input, %again = %run) : (tensor<1024xf32>, i1) -> tensor<1024xf32> {
    scf.condition(%again) %value : tensor<1024xf32>
  } do {
  ^bb0(%value: tensor<1024xf32>):
    %alloc = memref.alloc() : memref<1024xf32, #hivm.address_space<cbuf>>
    %tensor = bufferization.to_tensor %alloc restrict writable : memref<1024xf32, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%value : tensor<1024xf32>) outs(%alloc : memref<1024xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %matrix = tensor.expand_shape %tensor [[0, 1]] output_shape [32, 32] : tensor<1024xf32> into tensor<32x32xf32>
    %fractal = hivm.hir.convert_layout %matrix output_shape [4, 2, 16, 8]
        {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
        : (tensor<32x32xf32>) -> tensor<4x2x16x8xf32>
    %empty = tensor.empty() : tensor<2x2x16x16xf32>
    %dot = hivm.hir.mmadL1 ins(%fractal, %rhs, %true, %c32, %c32, %c32 : tensor<4x2x16x8xf32>, tensor<4x2x16x8xf32>, i1, index, index, index) outs(%empty : tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32>
    %ub = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
    %out = bufferization.to_tensor %ub restrict writable : memref<32x32xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<2x2x16x16xf32>) outs(%ub : memref<32x32xf32, #hivm.address_space<ub>>)
    %flat = tensor.collapse_shape %out [[0, 1]] : tensor<32x32xf32> into tensor<1024xf32>
    scf.yield %flat, %false : tensor<1024xf32>, i1
  }
  return %result : tensor<1024xf32>
}
}

// -----

// A second reader of the reshape still needs the original ND buffer.
// CHECK-LABEL: func.func @shared_reshape(
// CHECK:       %[[L1:.*]] = memref.alloc() : memref<1024xf32, #hivm.address_space<cbuf>>
// CHECK:       %[[TENSOR:.*]] = bufferization.to_tensor %[[L1]]
// CHECK:       hivm.hir.copy ins(%{{.*}} : tensor<1024xf32>) outs(%[[L1]]
// CHECK:       %[[VIEW:.*]] = tensor.expand_shape %[[TENSOR]]
// CHECK:       %[[CONVERTED:.*]] = hivm.hir.convert_layout %[[VIEW]]
// CHECK:       %[[DOT:.*]] = hivm.hir.mmadL1 ins(%[[CONVERTED]],
// CHECK:       return %[[DOT]], %[[VIEW]]
func.func @shared_reshape(%input: tensor<1024xf32>, %rhs: tensor<4x2x16x8xf32>) -> (tensor<2x2x16x16xf32>, tensor<32x32xf32>) {
  %c32 = arith.constant 32 : index
  %true = arith.constant true
  %alloc = memref.alloc() : memref<1024xf32, #hivm.address_space<cbuf>>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<1024xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%input : tensor<1024xf32>) outs(%alloc : memref<1024xf32, #hivm.address_space<cbuf>>)
  %matrix = tensor.expand_shape %tensor [[0, 1]] output_shape [32, 32] : tensor<1024xf32> into tensor<32x32xf32>
  %fractal = hivm.hir.convert_layout %matrix output_shape [4, 2, 16, 8]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
      : (tensor<32x32xf32>) -> tensor<4x2x16x8xf32>
  %empty = tensor.empty() : tensor<2x2x16x16xf32>
  %dot = hivm.hir.mmadL1 ins(%fractal, %rhs, %true, %c32, %c32, %c32 : tensor<4x2x16x8xf32>, tensor<4x2x16x8xf32>, i1, index, index, index) outs(%empty : tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32>
  return %dot, %matrix : tensor<2x2x16x16xf32>, tensor<32x32xf32>
}

// -----

// The reshape/layout extents are computed after the copy. Do not move either
// operation before its operands.
// CHECK-LABEL: func.func @late_reshape_extent(
// CHECK:       %[[L1:.*]] = memref.alloc(%{{.*}}) : memref<?xf32, #hivm.address_space<cbuf>>
// CHECK:       %[[TENSOR:.*]] = bufferization.to_tensor %[[L1]]
// CHECK:       hivm.hir.copy
// CHECK:       %[[COLS:.*]] = arith.divui
// CHECK:       %[[VIEW:.*]] = tensor.expand_shape %[[TENSOR]] {{.*}} output_shape [32, %[[COLS]]]
// CHECK:       hivm.hir.convert_layout %[[VIEW]]
func.func @late_reshape_extent(%input: tensor<?xf32>, %size: index, %rhs: tensor<4x2x16x8xf32>) -> tensor<2x2x16x16xf32> {
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %true = arith.constant true
  %alloc = memref.alloc(%size) : memref<?xf32, #hivm.address_space<cbuf>>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<?xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%input : tensor<?xf32>) outs(%alloc : memref<?xf32, #hivm.address_space<cbuf>>)
  %cols = arith.divui %size, %c32 : index
  %tiles = arith.ceildivui %cols, %c8 : index
  %matrix = tensor.expand_shape %tensor [[0, 1]] output_shape [32, %cols] : tensor<?xf32> into tensor<32x?xf32>
  %fractal = hivm.hir.convert_layout %matrix output_shape [%tiles, 2, 16, 8]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
      : (tensor<32x?xf32>) -> tensor<?x2x16x8xf32>
  %empty = tensor.empty() : tensor<2x2x16x16xf32>
  %dot = hivm.hir.mmadL1 ins(%fractal, %rhs, %true, %c32, %c32, %cols : tensor<?x2x16x8xf32>, tensor<4x2x16x8xf32>, i1, index, index, index) outs(%empty : tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32>
  return %dot : tensor<2x2x16x16xf32>
}

// -----

// A static result type does not guarantee that output extents are constants.
// CHECK-LABEL: func.func @static_result_late_extent(
// CHECK:       %[[L1:.*]] = memref.alloc() : memref<1024xf32, #hivm.address_space<cbuf>>
// CHECK:       %[[TENSOR:.*]] = bufferization.to_tensor %[[L1]]
// CHECK:       hivm.hir.copy
// CHECK:       %[[LATE:.*]] = arith.divui
// CHECK:       %[[VIEW:.*]] = tensor.expand_shape %[[TENSOR]] {{.*}} output_shape [32, %[[LATE]]]
// CHECK:       hivm.hir.convert_layout %[[VIEW]]
func.func @static_result_late_extent(%input: tensor<1024xf32>, %rows: index, %cols: index) -> tensor<4x2x16x8xf32> {
  %alloc = memref.alloc() : memref<1024xf32, #hivm.address_space<cbuf>>
  %tensor = bufferization.to_tensor %alloc restrict writable : memref<1024xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%input : tensor<1024xf32>) outs(%alloc : memref<1024xf32, #hivm.address_space<cbuf>>)
  %late = arith.divui %cols, %rows : index
  %matrix = tensor.expand_shape %tensor [[0, 1]] output_shape [32, %late] : tensor<1024xf32> into tensor<32x32xf32>
  %fractal = hivm.hir.convert_layout %matrix output_shape [4, 2, 16, 8]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
      : (tensor<32x32xf32>) -> tensor<4x2x16x8xf32>
  return %fractal : tensor<4x2x16x8xf32>
}
