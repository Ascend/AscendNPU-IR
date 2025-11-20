// RUN: bishengir-opt -hivm-reduce-rank-subview -split-input-file %s | FileCheck %s

// -----
// CHECK-LABEL: simple_brc_reduce_rank
// CHECK: %[[ARG0:.*]]: memref<1x1xf32>, %[[ARG1:.*]]: memref<1x192xf32, strided<[3072, 1]>>)
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[ARG0]][0, 0] [1, 1] [1, 1] : memref<1x1xf32> to memref<1xf32, strided<[1]>>
// CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[ARG1]][0, 0] [1, 192] [1, 1] : memref<1x192xf32, strided<[3072, 1]>> to memref<192xf32, strided<[1]>>
// CHECK: hivm.hir.vbrc ins(%[[SUBVIEW_SRC]] : memref<1xf32, strided<[1]>>) outs(%[[SUBVIEW_DST]] : memref<192xf32, strided<[1]>>) broadcast_dims = [0]
func.func @simple_brc_reduce_rank(%arg0: memref<1x1xf32>, %arg1: memref<1x192xf32, strided<[3072, 1]>>) {
  hivm.hir.vbrc ins(%arg0 : memref<1x1xf32>) outs(%arg1 : memref<1x192xf32, strided<[3072, 1]>>) broadcast_dims = [1]
  return
}

// -----
// CHECK-LABEL: simple_brc_reduce_rank_multi_brc_dims
// CHECK: %[[ARG0:.*]]: memref<1x1x1x1x8x1xf32>, %[[ARG1:.*]]: memref<1x192x1x192x8x192xf32>)
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[ARG0]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 1] [1, 1, 1, 1, 1, 1] : memref<1x1x1x1x8x1xf32> to memref<1x1x8x1xf32, strided<[8, 8, 1, 1]>>
// CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[ARG1]][0, 0, 0, 0, 0, 0] [1, 192, 1, 192, 8, 192] [1, 1, 1, 1, 1, 1] : memref<1x192x1x192x8x192xf32> to memref<192x192x8x192xf32, strided<[294912, 1536, 192, 1]>>
// CHECK: hivm.hir.vbrc ins(%[[SUBVIEW_SRC]] : memref<1x1x8x1xf32, strided<[8, 8, 1, 1]>>) outs(%[[SUBVIEW_DST]] : memref<192x192x8x192xf32, strided<[294912, 1536, 192, 1]>>) broadcast_dims = [0, 1, 3]
func.func @simple_brc_reduce_rank_multi_brc_dims(%arg0: memref<1x1x1x1x8x1xf32>, %arg1: memref<1x192x1x192x8x192xf32>) {
  hivm.hir.vbrc ins(%arg0 : memref<1x1x1x1x8x1xf32>) outs(%arg1 : memref<1x192x1x192x8x192xf32>) broadcast_dims = [1, 3, 5]
  return
}

// -----
// CHECK-LABEL: simple_vreduce
// CHECK: %[[SRC:.*]]: memref<1x10x1x10xf32>, %[[DST:.*]]: memref<1x1x1x1xf32>)
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[SRC]][0, 0, 0, 0] [1, 10, 1, 10] [1, 1, 1, 1] : memref<1x10x1x10xf32> to memref<10x10xf32, strided<[10, 1]>>
// CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[DST]][0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1xf32> to memref<1x1xf32, strided<[1, 1]>>
// CHECK: hivm.hir.vreduce <sum> ins(%[[SUBVIEW_SRC]] : memref<10x10xf32, strided<[10, 1]>>) outs(%[[SUBVIEW_DST]] : memref<1x1xf32, strided<[1, 1]>>) reduce_dims = [0, 1]
func.func @simple_vreduce(%src: memref<1x10x1x10xf32>, %dst: memref<1x1x1x1xf32>) {
  hivm.hir.vreduce <sum> ins(%src: memref<1x10x1x10xf32>)
                         outs(%dst: memref<1x1x1x1xf32>)
                         reduce_dims = [1, 3]
  return
}

// -----
// CHECK-LABEL: multiple_init_vreduce
// CHECK: %[[SRC:.*]]: memref<1x10x1x10xf32>, %[[DST:.*]]: memref<1x1x1x10xf32>, %[[DST2:.*]]: memref<1x1x1x10xi32>)
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[SRC]][0, 0, 0, 0] [1, 10, 1, 10] [1, 1, 1, 1] : memref<1x10x1x10xf32> to memref<10x10xf32, strided<[10, 1]>>
// CHECK: %[[SUBVIEW_DST1:.*]] = memref.subview %[[DST]][0, 0, 0, 0] [1, 1, 1, 10] [1, 1, 1, 1] : memref<1x1x1x10xf32> to memref<1x10xf32, strided<[10, 1]>>
// CHECK: %[[SUBVIEW_DST2:.*]] = memref.subview %[[DST2]][0, 0, 0, 0] [1, 1, 1, 10] [1, 1, 1, 1] : memref<1x1x1x10xi32> to memref<1x10xi32, strided<[10, 1]>>
// CHECK: hivm.hir.vreduce <max_with_index_left> ins(%[[SUBVIEW_SRC]] : memref<10x10xf32, strided<[10, 1]>>) outs(%[[SUBVIEW_DST1]], %[[SUBVIEW_DST2]] : memref<1x10xf32, strided<[10, 1]>>, memref<1x10xi32, strided<[10, 1]>>) reduce_dims = [0]
func.func @multiple_init_vreduce(%src: memref<1x10x1x10xf32>, %dst: memref<1x1x1x10xf32>, %dst2 : memref<1x1x1x10xi32>) {
  hivm.hir.vreduce <max_with_index_left> ins(%src: memref<1x10x1x10xf32>)
                         outs(%dst, %dst2: memref<1x1x1x10xf32>, memref<1x1x1x10xi32>)
                         reduce_dims = [1]
  return
}

// -----
// CHECK-LABEL: test_brc_drop_correct_dim
// CHECK: %[[SRC:.*]]: memref<?x1x1xf32, strided<[8, 8, 1]>>, %[[DST:.*]]: memref<?x1x?xf32, strided<[?, ?, 1], offset: ?>>)
// CHECK: %[[DIM:.*]] = memref.dim %[[SRC]]
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[SRC]][0, 0, 0] [%[[DIM]], 1, 1] [1, 1, 1] : memref<?x1x1xf32, strided<[8, 8, 1]>> to memref<?x1xf32, strided<[8, 1]>>
// CHECK: %[[DIM_0:.*]] = memref.dim %[[DST]]
// CHECK: %[[DIM_1:.*]] = memref.dim %[[DST]]
// CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[DST]][0, 0, 0] [%[[DIM_0]], 1, %[[DIM_1]]] [1, 1, 1] : memref<?x1x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK: hivm.hir.vbrc ins(%[[SUBVIEW_SRC:.*]] : memref<?x1xf32, strided<[8, 1]>>) outs(%[[SUBVIEW_DST:.*]] : memref<?x?xf32, strided<[?, 1], offset: ?>>) broadcast_dims = [1]
func.func @test_brc_drop_correct_dim(
    %src: memref<?x1x1xf32, strided<[8, 8, 1]>>, 
    %dst: memref<?x1x?xf32, strided<[?, ?, 1], offset: ?>>) {
    hivm.hir.vbrc ins(%src : memref<?x1x1xf32, strided<[8, 8, 1]>>) 
                  outs(%dst : memref<?x1x?xf32, strided<[?, ?, 1], offset: ?>>) broadcast_dims = [2]
  return
}

// -----

// CHECK-LABEL: func @simple_elem_reduce_rank
// CHECK: hivm.hir.vadd
// CHECK-SAME: memref<2xf32, strided<[1]>>
func.func @simple_elem_reduce_rank(%arg0: memref<1x2xf32>, %arg1: memref<1x2xf32>) {
  hivm.hir.vadd ins(%arg0, %arg0 : memref<1x2xf32>, memref<1x2xf32>) outs(%arg1 : memref<1x2xf32>)
  return
}

// -----

// CHECK-LABEL: func @simple_elem_all_one_reduce_rank
// CHECK: hivm.hir.vadd
// CHECK-SAME: memref<1xf32, strided<[1]>>
func.func @simple_elem_all_one_reduce_rank(%arg0: memref<1x1xf32>, %arg1: memref<1x1xf32>) {
  hivm.hir.vadd ins(%arg0, %arg0 : memref<1x1xf32>, memref<1x1xf32>) outs(%arg1 : memref<1x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @simple_elem_copy
// CHECK: hivm.hir.copy
// CHECK-SAME: memref<1xf32, strided<[1]>>
func.func @simple_elem_copy(%src: memref<1x1xf32>) {
  %dst = memref.alloc() : memref<1x1xf32>
  hivm.hir.copy ins(%src : memref<1x1xf32>)
                outs(%dst : memref<1x1xf32>)
  return
}
