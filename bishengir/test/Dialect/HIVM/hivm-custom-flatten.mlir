// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// CustomOp flattening.
//
// `hivm.hir.custom` / `hivm.hir.custom_macro` are opaque to the compiler, so
// they are flattened only when the author declared the loop semantics through
// `iterator_types` and provided, through `flatten_ranks` / `flatten_symbols`,
// the implementation to call at the collapsed rank. A non-`parallel` iterator
// or a missing or inconsistent declaration keeps the op at its original rank
// and its original `symbol`.
//===----------------------------------------------------------------------===//

// All dimensions are parallel and contiguous: collapse down to rank 1.
// CHECK-LABEL: func.func @custom_2d_all_parallel
// CHECK-SAME:      %[[SRC:.*]]: memref<32x7xf32>, %[[DST:.*]]: memref<32x7xf32>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      iterator_types = [#hivm.iterator_type<parallel>]
// CHECK-SAME:      symbol = "k_2d_all_parallel_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<224xf32>)
// CHECK-SAME:      outs(%[[FDST]] : memref<224xf32>)
func.func @custom_2d_all_parallel(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_2d_all_parallel",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_2d_all_parallel_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// Rank 3, all parallel: the whole iteration space becomes a single dimension.
// CHECK-LABEL: func.func @custom_3d_all_parallel
// CHECK-SAME:      %[[SRC:.*]]: memref<4x8x16xf16>, %[[DST:.*]]: memref<4x8x16xf16>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]] {{.*}} into memref<512xf16>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1, 2]] {{.*}} into memref<512xf16>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      iterator_types = [#hivm.iterator_type<parallel>]
// CHECK-SAME:      symbol = "k_3d_all_parallel_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<512xf16>) outs(%[[FDST]] : memref<512xf16>)
func.func @custom_3d_all_parallel(%src: memref<4x8x16xf16>, %dst: memref<4x8x16xf16>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_3d_all_parallel",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_3d_all_parallel_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                         #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<4x8x16xf16>)
      outs(%dst : memref<4x8x16xf16>)
  return
}

// -----

// Unit dimensions are absorbed by their neighbours before the contiguous
// collapse, so [1, 64, 1, 128] ends up as a single dimension of 8192.
// CHECK-LABEL: func.func @custom_unit_dims_absorbed
// CHECK-SAME:      %[[SRC:.*]]: memref<1x64x1x128xi8>, %[[DST:.*]]: memref<1x64x1x128xi8>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2, 3]] {{.*}} into memref<8192xi8>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1, 2, 3]] {{.*}} into memref<8192xi8>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      iterator_types = [#hivm.iterator_type<parallel>]
// CHECK-SAME:      symbol = "k_unit_dims_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<8192xi8>) outs(%[[FDST]] : memref<8192xi8>)
func.func @custom_unit_dims_absorbed(%src: memref<1x64x1x128xi8>, %dst: memref<1x64x1x128xi8>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_unit_dims",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_unit_dims_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                         #hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<1x64x1x128xi8>)
      outs(%dst : memref<1x64x1x128xi8>)
  return
}

// -----

// The table may list several ranks; the entry for the rank actually produced
// is selected. Here the non-contiguous tail leaves rank 2, so the rank-2 kernel
// is chosen over the rank-1 one, and the table itself is carried along.
// CHECK-LABEL: func.func @custom_flatten_symbols_selects_produced_rank
// CHECK-SAME:      %[[SRC:.*]]: memref<4x8x16xf32, strided<[320, 40, 1]>>, %[[DST:.*]]: memref<4x8x16xf32, strided<[320, 40, 1]>>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1], [2]] {{.*}} into memref<32x16xf32, strided<[40, 1]>>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1], [2]] {{.*}} into memref<32x16xf32, strided<[40, 1]>>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      flatten_ranks = [1, 2], flatten_symbols = ["k_select_1d", "k_select_2d"]
// CHECK-SAME:      symbol = "k_select_2d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<32x16xf32, strided<[40, 1]>>) outs(%[[FDST]] : memref<32x16xf32, strided<[40, 1]>>)
func.func @custom_flatten_symbols_selects_produced_rank(%src: memref<4x8x16xf32, strided<[320, 40, 1]>>, %dst: memref<4x8x16xf32, strided<[320, 40, 1]>>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_select_3d",
       flatten_ranks = [1 : i64, 2 : i64], flatten_symbols = ["k_select_1d", "k_select_2d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                         #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<4x8x16xf32, strided<[320, 40, 1]>>)
      outs(%dst : memref<4x8x16xf32, strided<[320, 40, 1]>>)
  return
}

// -----

// The collapse would produce rank 2, but only a rank-1 implementation is
// declared. There is nothing the collapsed op could be lowered to, so the op
// keeps its rank and its original `symbol`.
// CHECK-LABEL: func.func @custom_flatten_symbols_without_produced_rank
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_missing_rank_3d"
// CHECK-SAME:      memref<4x8x16xf32, strided<[320, 40, 1]>>
func.func @custom_flatten_symbols_without_produced_rank(%src: memref<4x8x16xf32, strided<[320, 40, 1]>>, %dst: memref<4x8x16xf32, strided<[320, 40, 1]>>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_missing_rank_3d",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_missing_rank_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                         #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<4x8x16xf32, strided<[320, 40, 1]>>)
      outs(%dst : memref<4x8x16xf32, strided<[320, 40, 1]>>)
  return
}

// -----

// No `flatten_symbols` at all: the kernel behind `symbol` was written against
// rank-2 descriptors, so the op must not be handed rank-1 ones.
// CHECK-LABEL: func.func @custom_without_flatten_symbols
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_no_table"
// CHECK-SAME:      memref<32x7xf32>
func.func @custom_without_flatten_symbols(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_no_table",
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// Temp buffers are scratch memory outside the iteration space: the ins/outs
// operands are collapsed while the temp buffer keeps its own shape.
// CHECK-LABEL: func.func @custom_temp_buffer_not_collapsed
// CHECK-SAME:      %[[SRC:.*]]: memref<32x7xf32>, %[[DST:.*]]: memref<32x7xf32>
// CHECK:         %[[TMP:.*]] = memref.alloc() : memref<256xf32>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_with_tmp_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<224xf32>) outs(%[[FDST]] : memref<224xf32>) tmps(%[[TMP]] : memref<256xf32>)
func.func @custom_temp_buffer_not_collapsed(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  %tmp = memref.alloc() : memref<256xf32>
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_with_tmp",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_with_tmp_1d"],
       extra_buffers_types = [f32],
       extra_buffers_sizes = [256 : i64],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.with_tmp"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
      tmps(%tmp : memref<256xf32>)
  return
}

// -----

// Scalar operands take no part in the iteration space and are passed through.
// CHECK-LABEL: func.func @custom_with_scalar_operand
// CHECK-SAME:      %[[SRC:.*]]: memref<32x7xf32>, %[[SCALE:.*]]: f32, %[[DST:.*]]: memref<32x7xf32>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_scalar_1d"
// CHECK-SAME:      ins(%[[FSRC]], %[[SCALE]] : memref<224xf32>, f32) outs(%[[FDST]] : memref<224xf32>)
func.func @custom_with_scalar_operand(%src: memref<32x7xf32>, %scale: f32, %dst: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_scalar",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_scalar_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.scale"
      ins(%src, %scale : memref<32x7xf32>, f32)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// `hivm.hir.custom_macro` follows the very same rules.
// CHECK-LABEL: func.func @custom_macro_2d_all_parallel
// CHECK-SAME:      %[[SRC:.*]]: memref<32x7xf32>, %[[DST:.*]]: memref<32x7xf32>
// CHECK-DAG:     %[[FSRC:.*]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK-DAG:     %[[FDST:.*]] = memref.collapse_shape %[[DST]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK:         hivm.hir.custom_macro
// CHECK-SAME:      iterator_types = [#hivm.iterator_type<parallel>]
// CHECK-SAME:      symbol = "k_macro_2d_1d"
// CHECK-SAME:      ins(%[[FSRC]] : memref<224xf32>) outs(%[[FDST]] : memref<224xf32>)
func.func @custom_macro_2d_all_parallel(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_macro_2d",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_macro_2d_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.macro_elemwise"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// `inplace_outs` requires the mapped ins/outs to stay the same SSA value; the
// rewrite must therefore collapse that value once and reuse it.
// CHECK-LABEL: func.func @custom_inplace_outs_single_collapse
// CHECK-SAME:      %[[BUF:.*]]: memref<32x7xf32>
// CHECK:         %[[FLAT:.*]] = memref.collapse_shape %[[BUF]] {{\[}}[0, 1]] {{.*}} into memref<224xf32>
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      symbol = "k_inplace_1d"
// CHECK-SAME:      ins(%[[FLAT]] : memref<224xf32>
// CHECK-SAME:      outs(%[[FLAT]] : memref<224xf32>)
func.func @custom_inplace_outs_single_collapse(%buf: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_inplace",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_inplace_1d"],
       arg_attrs = [{inplace_outs = 0 : i64}],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.inplace"
      ins(%buf : memref<32x7xf32>)
      outs(%buf : memref<32x7xf32>)
  return
}

// -----

// A wide integer is accepted when its value fits the rank representation.
// CHECK-LABEL: func.func @custom_wide_rank
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.custom
// CHECK-SAME: symbol = "k_1d"
// CHECK-SAME: ins(%{{.*}} : memref<4xf32>) outs(%{{.*}} : memref<4xf32>)
func.func @custom_wide_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       flatten_ranks = [1 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// A wide integer is accepted when its value fits the rank representation.
// CHECK-LABEL: func.func @custom_macro_wide_rank
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: symbol = "k_1d"
// CHECK-SAME: ins(%{{.*}} : memref<4xf32>) outs(%{{.*}} : memref<4xf32>)
func.func @custom_macro_wide_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       flatten_ranks = [1 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}
