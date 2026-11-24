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

// -----

// GEN-1749 just before transpose lowering. A's conversion has reached the
// live column. B's conversion still applies to the completed matrix.
// CHECK-LABEL: func.func @packed_column_and_row(
// CHECK-SAME:    %[[COLS:[^:]+]]: tensor<16x16xf32>, %[[ROWS:[^:]+]]: tensor<16x16xf32>
// CHECK-DAG:   %[[EMPTY:.*]] = tensor.empty() : tensor<2x1x16x8xf32>
// CHECK-DAG:   %[[AZ:.*]] = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} {{.*}}outs(%[[EMPTY]] : tensor<2x1x16x8xf32>)
// CHECK-DAG:   %[[BZ:.*]] = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} {{.*}} -> tensor<16x16xf32>
// CHECK-DAG:   %[[ABUF:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK-DAG:   %[[A:.*]] = bufferization.to_tensor %[[ABUF]]
// CHECK:       %[[COL:.*]] = tensor.extract_slice %[[COLS]][0, 0] [16, 1]
// CHECK:       %[[LIVE:.*]] = hivm.hir.convert_layout %[[COL]] output_shape [1, 1, 16, 1]
// CHECK-SAME:  not_to_propagate_up = true
// CHECK:       %[[AP:.*]] = tensor.insert_slice %[[LIVE]] into %[[AZ]][0, 0, 0, 0] [1, 1, 16, 1]
// CHECK:       hivm.hir.copy ins(%[[AP]] : tensor<2x1x16x8xf32>) outs(%[[ABUF]]
// CHECK-DAG:   %[[BBUF:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK-DAG:   %[[B:.*]] = bufferization.to_tensor %[[BBUF]]
// CHECK:       %[[ROW:.*]] = tensor.extract_slice %[[ROWS]][0, 0] [1, 16]
// CHECK:       %[[BM:.*]] = tensor.insert_slice %[[ROW]] into %[[BZ]]
// CHECK:       %[[BP:.*]] = hivm.hir.convert_layout %[[BM]] output_shape [2, 1, 16, 8]
// CHECK:       hivm.hir.copy ins(%[[BP]] : tensor<2x1x16x8xf32>) outs(%[[BBUF]]
// CHECK:       hivm.hir.mmadL1 ins(%[[A]], %[[B]],
// LOWER-LABEL: func.func @packed_column_and_row(
// LOWER-NOT:   tensor<16x8xf32>
// LOWER:       hivm.hir.copy ins(%{{.*}} : tensor<2x1x16x8xf32>)
// LOWER:       hivm.hir.vtranspose
// LOWER:       hivm.hir.copy ins(%{{.*}} : tensor<2x1x16x8xf32>)
// LOWER:       hivm.hir.mmadL1
func.func @packed_column_and_row(%cols: tensor<16x16xf32>, %rows: tensor<16x16xf32>) -> tensor<1x1x16x16xf32> {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %empty_a = tensor.empty() {hivm.address_space = #hivm.address_space<cbuf>} : tensor<2x1x16x8xf32>
  %az = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<CUBE>} ins(%zero : f32) outs(%empty_a : tensor<2x1x16x8xf32>) -> tensor<2x1x16x8xf32>
  %empty_b = tensor.empty() : tensor<16x16xf32>
  %bz = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<CUBE>} ins(%zero : f32) outs(%empty_b : tensor<16x16xf32>) -> tensor<16x16xf32>
  %abuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
  %av = bufferization.to_tensor %abuf restrict writable : memref<16x16xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%cols : tensor<16x16xf32>) outs(%abuf : memref<16x16xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
  %col = tensor.extract_slice %av[0, 0] [16, 1] [1, 1] : tensor<16x16xf32> to tensor<16x1xf32>
  %live = hivm.hir.convert_layout %col output_shape [1, 1, 16, 1] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>} : (tensor<16x1xf32>) -> tensor<1x1x16x1xf32>
  %a = tensor.insert_slice %live into %az[0, 0, 0, 0] [1, 1, 16, 1] [1, 1, 1, 1] : tensor<1x1x16x1xf32> into tensor<2x1x16x8xf32>
  %bbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
  %bv = bufferization.to_tensor %bbuf restrict writable : memref<16x16xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%rows : tensor<16x16xf32>) outs(%bbuf : memref<16x16xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
  %row = tensor.extract_slice %bv[0, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
  %bm = tensor.insert_slice %row into %bz[0, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
  %b = hivm.hir.convert_layout %bm output_shape [2, 1, 16, 8] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>} : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
  %acc = tensor.empty() : tensor<1x1x16x16xf32>
  %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
  return %dot : tensor<1x1x16x16xf32>
}

// -----

// Both matmuls share one packed A. The returned ND matrix still needs its copy.
// CHECK-LABEL: func.func @shared_sliced_matmul_input(
// CHECK-SAME:    %[[SRC:[^:]+]]: tensor<16x16xf32>
// CHECK:       %[[ND:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
// CHECK:       %[[NV:.*]] = bufferization.to_tensor %[[ND]]
// CHECK:       %[[BUF:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK:       %[[PV:.*]] = bufferization.to_tensor %[[BUF]]
// CHECK:       %[[S:.*]] = tensor.extract_slice %[[SRC]]
// CHECK:       %[[M:.*]] = tensor.insert_slice %[[S]]
// CHECK:       %[[P:.*]] = hivm.hir.convert_layout %[[M]]
// CHECK:       hivm.hir.copy ins(%[[P]] : tensor<2x1x16x8xf32>) outs(%[[BUF]]
// CHECK:       hivm.hir.copy ins(%[[SRC]] : tensor<16x16xf32>) outs(%[[ND]]
// CHECK:       %[[COL:.*]] = tensor.extract_slice %[[NV]]
// CHECK:       %[[MATRIX:.*]] = tensor.insert_slice %[[COL]]
// CHECK:       scf.if
// CHECK:       hivm.hir.mmadL1 ins(%[[PV]],
// CHECK:       } else {
// CHECK:       hivm.hir.mmadL1 ins(%[[PV]],
// CHECK:       return %{{.*}}, %[[MATRIX]]
func.func @shared_sliced_matmul_input(%cols: tensor<16x16xf32>, %rhs1: tensor<2x1x16x8xf32>, %rhs2: tensor<2x1x16x8xf32>, %cond: i1) -> (tensor<1x1x16x16xf32>, tensor<16x16xf32>) {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %zeros = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<CUBE>} ins(%zero : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %buf = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
  %v = bufferization.to_tensor %buf restrict writable : memref<16x16xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%cols : tensor<16x16xf32>) outs(%buf : memref<16x16xf32, #hivm.address_space<cbuf>>)
  %col = tensor.extract_slice %v[0, 0] [16, 1] [1, 1] : tensor<16x16xf32> to tensor<16x1xf32>
  %m = tensor.insert_slice %col into %zeros[0, 0] [16, 1] [1, 1] : tensor<16x1xf32> into tensor<16x16xf32>
  %acc = tensor.empty() : tensor<1x1x16x16xf32>
  %r = scf.if %cond -> tensor<1x1x16x16xf32> {
    %p = hivm.hir.convert_layout %m output_shape [2, 1, 16, 8] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>} : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
    %d = hivm.hir.mmadL1 ins(%p, %rhs1, %true, %c16, %c16, %c16 : tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    scf.yield %d : tensor<1x1x16x16xf32>
  } else {
    %p = hivm.hir.convert_layout %m output_shape [2, 1, 16, 8] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>} : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
    %d = hivm.hir.mmadL1 ins(%p, %rhs2, %true, %c16, %c16, %c16 : tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    scf.yield %d : tensor<1x1x16x16xf32>
  }
  return %r, %m : tensor<1x1x16x16xf32>, tensor<16x16xf32>
}
