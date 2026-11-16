// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file | FileCheck %s

// A rank-to-symbol table cannot express the position of a reduction axis
// after every partial collapse. Keep the original call even with a rank-2 entry.
// CHECK-LABEL: func.func @custom_reduction_last_dim
// CHECK-SAME:      %[[SRC:.*]]: memref<4x8x16xf32>, %[[DST:.*]]: memref<4x8x16xf32>
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>, #hivm.iterator_type<reduction>]
// CHECK-SAME:      symbol = "k_reduce_last"
// CHECK-SAME:      ins(%[[SRC]] : memref<4x8x16xf32>) outs(%[[DST]] : memref<4x8x16xf32>)
func.func @custom_reduction_last_dim(%src: memref<4x8x16xf32>, %dst: memref<4x8x16xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_reduce_last",
       flatten_ranks = [2 : i64], flatten_symbols = ["k_reduce_last_2d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                         #hivm.iterator_type<reduction>]}
      "user.reduce"
      ins(%src : memref<4x8x16xf32>)
      outs(%dst : memref<4x8x16xf32>)
  return
}

// -----

// A reduction in the middle separates both parallel dimensions, so nothing can
// be merged and the op is left untouched.
// CHECK-LABEL: func.func @custom_reduction_middle_dim
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<4x8x16xf32>
func.func @custom_reduction_middle_dim(%src: memref<4x8x16xf32>, %dst: memref<4x8x16xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_reduce_middle",
       flatten_ranks = [1 : i64, 2 : i64], flatten_symbols = ["k_reduce_middle_1d", "k_reduce_middle_2d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<reduction>,
                         #hivm.iterator_type<parallel>]}
      "user.reduce"
      ins(%src : memref<4x8x16xf32>)
      outs(%dst : memref<4x8x16xf32>)
  return
}

// -----

// No `iterator_types`: the loop semantics are unknown, so the op keeps its rank.
// CHECK-LABEL: func.func @custom_without_iterator_types
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<32x7xf32>
func.func @custom_without_iterator_types(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_no_iter",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_no_iter_1d"]}
      "user.opaque"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// `iterator_types` shorter than the operand rank is an inconsistent
// declaration and must not be interpreted.
// CHECK-LABEL: func.func @custom_iterator_types_rank_mismatch
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<4x8x16xf32>
func.func @custom_iterator_types_rank_mismatch(%src: memref<4x8x16xf32>, %dst: memref<4x8x16xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_iter_mismatch",
       flatten_ranks = [1 : i64, 2 : i64], flatten_symbols = ["k_iter_mismatch_1d", "k_iter_mismatch_2d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<4x8x16xf32>)
      outs(%dst : memref<4x8x16xf32>)
  return
}

// -----

// An `opaque` iterator is a barrier just like any other non-parallel kind.
// CHECK-LABEL: func.func @custom_opaque_iterator_is_barrier
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<4x8xf32>
func.func @custom_opaque_iterator_is_barrier(%src: memref<4x8xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_opaque",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_opaque_1d"],
       iterator_types = [#hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>]}
      "user.opaque_iter"
      ins(%src : memref<4x8xf32>)
      outs(%dst : memref<4x8xf32>)
  return
}

// -----

// Builtins are bound to a fixed template signature, so their rank is frozen
// even though all dimensions are parallel.
// CHECK-LABEL: func.func @custom_builtin_not_flattened
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<4x8xf32>
func.func @custom_builtin_not_flattened(%src: memref<4x8xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMT>,
       symbol = "k_builtin",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_builtin_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "__builtin_gather_load"
      ins(%src : memref<4x8xf32>)
      outs(%dst : memref<4x8xf32>)
  return
}

// -----

// Operands of different ranks do not describe a single iteration space.
// CHECK-LABEL: func.func @custom_rank_mismatch_between_operands
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
func.func @custom_rank_mismatch_between_operands(%src: memref<4x8x16xf32>, %dst: memref<32x16xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_rank_mismatch",
       flatten_ranks = [1 : i64, 2 : i64], flatten_symbols = ["k_rank_mismatch_1d", "k_rank_mismatch_2d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                         #hivm.iterator_type<parallel>]}
      "user.reshape"
      ins(%src : memref<4x8x16xf32>)
      outs(%dst : memref<32x16xf32>)
  return
}

// -----

// `align_dim` pins an alignment requirement on one dimension of an operand,
// which a collapse would silently drop.
// CHECK-LABEL: func.func @custom_align_dim_not_flattened
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<32x7xf32>
func.func @custom_align_dim_not_flattened(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_align_dim",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_align_dim_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.aligned"
      ins(%src : memref<32x7xf32> {align_dim = 1 : i64})
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// Only operands are rewritten, so a shaped result would keep the original rank
// while the operands and `iterator_types` move to the collapsed one, handing
// the user kernel a rank-1-in / rank-2-out signature.
// CHECK-LABEL: func.func @custom_shaped_result_not_flattened
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      ins(%{{.*}} : memref<4x8xf32>)
// CHECK-SAME:      -> memref<4x8xf32>
func.func @custom_shaped_result_not_flattened(%src: memref<4x8xf32>) -> memref<4x8xf32> {
  %res = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_shaped_result",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_shaped_result_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.return_memref"
      ins(%src : memref<4x8xf32>) -> memref<4x8xf32>
  return %res : memref<4x8xf32>
}

// -----

// A value passed both as payload and as scratch cannot be collapsed for one
// role only: the rewrite maps values, not operand positions, so collapsing the
// payload occurrence would rewrite the `tmps` occurrence too and silently
// change the scratch buffer ABI.
// CHECK-LABEL: func.func @custom_temp_buffer_aliasing_payload
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      ins(%[[BUF:[a-zA-Z0-9_]+]] : memref<4x8xf32>)
// CHECK-SAME:      tmps(%[[BUF]] : memref<4x8xf32>)
func.func @custom_temp_buffer_aliasing_payload(%buf: memref<4x8xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_alias_tmp",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_alias_tmp_1d"],
       extra_buffers_types = [f32],
       extra_buffers_sizes = [32 : i64],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.alias_tmp"
      ins(%buf : memref<4x8xf32>)
      outs(%dst : memref<4x8xf32>)
      tmps(%buf : memref<4x8xf32>)
  return
}

// -----

// With identity indexing every payload operand walks the same domain, so one
// loop dimension must have one extent. `2x6` and `3x4` would both collapse to
// `12`, hiding the inconsistency instead of preserving it.
// CHECK-LABEL: func.func @custom_extent_mismatch_not_flattened
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      ins(%{{.*}} : memref<2x6xf32>)
// CHECK-SAME:      outs(%{{.*}} : memref<3x4xf32>)
func.func @custom_extent_mismatch_not_flattened(%src: memref<2x6xf32>, %dst: memref<3x4xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_extent_mismatch",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_extent_mismatch_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.extent_mismatch"
      ins(%src : memref<2x6xf32>)
      outs(%dst : memref<3x4xf32>)
  return
}

// -----

// A malformed `iterator_types` states nothing about the loops. No verifier
// rejects it today and the structured-op accessor hard-casts every element, so
// it must be read defensively and treated like a missing declaration.
// CHECK-LABEL: func.func @custom_malformed_iterator_types
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<4x8xf32>
func.func @custom_malformed_iterator_types(%src: memref<4x8xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_malformed_iter",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_malformed_iter_1d"],
       iterator_types = [0 : i64, 0 : i64]}
      "user.malformed_iter"
      ins(%src : memref<4x8xf32>)
      outs(%dst : memref<4x8xf32>)
  return
}

// -----

// Likewise for a malformed `indexing_map`: the accessor turns it into an empty
// list, which must not read as "no remapping declared".
// CHECK-LABEL: func.func @custom_malformed_indexing_map
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      memref<4x8xf32>
func.func @custom_malformed_indexing_map(%src: memref<4x8xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_malformed_map",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_malformed_map_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       indexing_map = "not-an-array"}
      "user.malformed_map"
      ins(%src : memref<4x8xf32>)
      outs(%dst : memref<4x8xf32>)
  return
}

// -----

// Two operands whose dynamic extents sit on *different* axes describe two
// different iteration domains: `?x6` and `3x?` only agree at run time if the
// unknowns happen to be `3` and `6`. Collapsing both to `?` erases the
// disagreement instead of preserving it, so the merge must be refused.
// CHECK-LABEL: func.func @custom_dynamic_cross_extent_not_flattened
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      ins(%{{.*}} : memref<?x6xf32>)
// CHECK-SAME:      outs(%{{.*}} : memref<3x?xf32>)
func.func @custom_dynamic_cross_extent_not_flattened(%src: memref<?x6xf32>, %dst: memref<3x?xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_dynamic_cross",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_dynamic_cross_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.dynamic_cross"
      ins(%src : memref<?x6xf32>)
      outs(%dst : memref<3x?xf32>)
  return
}

// -----

// `indexing_map` carries one map per structured operand. A shorter list leaves
// the remaining operands undescribed, so "every map present is identity" says
// nothing about them; the declaration is incomplete and must be rejected.
#idmap2d = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @custom_indexing_map_count_mismatch
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
// CHECK-SAME:      ins(%[[SRC:[a-zA-Z0-9_]+]] : memref<4x8xf32>) outs(%[[DST:[a-zA-Z0-9_]+]] : memref<4x8xf32>)
func.func @custom_indexing_map_count_mismatch(%src: memref<4x8xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_short_map",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_short_map_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       indexing_map = [#idmap2d]}
      "user.short_map"
      ins(%src : memref<4x8xf32>)
      outs(%dst : memref<4x8xf32>)
  return
}

// -----

// A layout with no stride form at all. `memref.collapse_shape` does not accept
// such an operand, and the collapse pipeline builds the collapsed type before
// any verifier runs, so letting this through crashes the compiler rather than
// producing a diagnostic. The operand is refused while the iteration space is
// being read.
#unstridable = affine_map<(d0, d1) -> (d0 floordiv 2, d0 mod 2, d1)>
// CHECK-LABEL: func.func @custom_unstridable_layout_not_flattened
// CHECK-NOT:     memref.collapse_shape
// CHECK:         hivm.hir.custom
func.func @custom_unstridable_layout_not_flattened(
    %src: memref<4x16xf32, #unstridable>,
    %dst: memref<4x16xf32, #unstridable>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_unstridable",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_unstridable_1d"],
       iterator_types = [#hivm.iterator_type<parallel>,
                         #hivm.iterator_type<parallel>]}
      "user.unstridable"
      ins(%src : memref<4x16xf32, #unstridable>)
      outs(%dst : memref<4x16xf32, #unstridable>)
  return
}
