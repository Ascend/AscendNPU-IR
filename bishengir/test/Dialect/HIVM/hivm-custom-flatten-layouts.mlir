// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file | FileCheck %s

// A non-contiguous layout cannot be collapsed at all.
// CHECK-LABEL: func.func @custom_non_contiguous
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
func.func @custom_non_contiguous(%src: memref<16x16xf32, strided<[16, 2], offset: 0>>,
                                 %dst: memref<16x16xf32, strided<[16, 2], offset: 0>>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_strided",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_strided_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<16x16xf32, strided<[16, 2], offset: 0>>)
      outs(%dst : memref<16x16xf32, strided<[16, 2], offset: 0>>)
  return
}

// -----

// Only the contiguous tail of a strided layout is merged.
// CHECK-LABEL: func.func @custom_partially_contiguous
// CHECK-SAME:      %[[SRC:.*]]: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>, %[[DST:.*]]: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0], [1], [2, 3]] {{.*}} into memref<8x?x8xf32, strided<[?, ?, 1]>>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0], [1], [2, 3]] {{.*}} into memref<8x?x8xf32, strided<[?, ?, 1]>>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_partial_3d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<8x?x8xf32, strided<[?, ?, 1]>>) outs(%[[FDST]] : memref<8x?x8xf32, strided<[?, ?, 1]>>)
func.func @custom_partially_contiguous(%src: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>,
                                       %dst: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_partial",
       flatten_ranks = [3 : i64], flatten_symbols = ["k_partial_3d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                         #hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>)
      outs(%dst : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>)
  return
}

// -----

// Dynamic leading dimension: still collapsible because the layout stays
// contiguous.
// CHECK-LABEL: func.func @custom_dynamic_dim
// CHECK-SAME:      %[[SRC:.*]]: memref<?x7xf32>, %[[DST:.*]]: memref<?x7xf32>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]] {{.*}} into memref<?xf32>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1]] {{.*}} into memref<?xf32>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_dynamic_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<?xf32>) outs(%[[FDST]] : memref<?xf32>)
func.func @custom_dynamic_dim(%src: memref<?x7xf32>, %dst: memref<?x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_dynamic",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_dynamic_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<?x7xf32>)
      outs(%dst : memref<?x7xf32>)
  return
}

// -----

// Rank 1 has nothing to merge.
// CHECK-LABEL: func.func @custom_rank1_unchanged
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<128xf32>
func.func @custom_rank1_unchanged(%src: memref<128xf32>, %dst: memref<128xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_rank1",
       iterator_types = [#hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<128xf32>)
      outs(%dst : memref<128xf32>)
  return
}

// -----

// The contiguous counterpart of the layout above does collapse.
#contig = affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>
// CHECK-LABEL: func.func @custom_affine_layout_contiguous
// CHECK-SAME:      %[[SRC:.*]]: memref<4x16xf32, #{{.*}}>, %[[DST:.*]]: memref<4x16xf32, #{{.*}}>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]] {{.*}} into memref<64xf32, strided<[1], offset: ?>>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1]] {{.*}} into memref<64xf32, strided<[1], offset: ?>>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      iterator_types = [#hivm.iterator_type<parallel>]
// CHECK-SAME:      symbol = "k_affine_contig_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<64xf32, strided<[1], offset: ?>>) outs(%[[FDST]] : memref<64xf32, strided<[1], offset: ?>>)
func.func @custom_affine_layout_contiguous(%src: memref<4x16xf32, #contig>,
                                           %dst: memref<4x16xf32, #contig>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_affine_contig",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_affine_contig_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.affine_contig"
      ins(%src : memref<4x16xf32, #contig>)
      outs(%dst : memref<4x16xf32, #contig>)
  return
}
