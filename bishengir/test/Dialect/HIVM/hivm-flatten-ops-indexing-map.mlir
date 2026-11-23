// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file | FileCheck %s
// Second run prints affine maps inline instead of through `#map` aliases, so
// that the `indexing_map` of a custom op can be checked before and after
// flattening.
// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file -mlir-print-local-scope | FileCheck %s --check-prefix=LOCAL

// An identity `indexing_map` is rewritten to the identity of the collapsed
// rank.
#idmap2d = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @custom_identity_indexing_map
// CHECK-SAME:      %[[SRC:.*]]: memref<32x7xf32>, %[[DST:.*]]: memref<32x7xf32>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_identity_map_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<224xf32>) outs(%[[FDST]] : memref<224xf32>)
// LOCAL-LABEL: func.func @custom_identity_indexing_map
// LOCAL:         hivm.hir.custom
// LOCAL-SAME:      indexing_map = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
func.func @custom_identity_indexing_map(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_identity_map",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_identity_map_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       indexing_map = [#idmap2d, #idmap2d]}
      "user.elemwise"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// A permuting `indexing_map` cannot be replayed on the collapsed iteration
// space, so flattening is declined.
#permmap2d = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @custom_non_identity_indexing_map
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<32x32xf32>
// LOCAL-LABEL: func.func @custom_non_identity_indexing_map
// LOCAL:         hivm.hir.custom
// LOCAL-SAME:      indexing_map = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>]
func.func @custom_non_identity_indexing_map(%src: memref<32x32xf32>, %dst: memref<32x32xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_perm_map",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_perm_map_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       indexing_map = [#permmap2d, #permmap2d]}
      "user.transposed"
      ins(%src : memref<32x32xf32>)
      outs(%dst : memref<32x32xf32>)
  return
}
