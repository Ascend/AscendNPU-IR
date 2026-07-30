// RUN: bishengir-opt %s -propagate-reshape="for-regbased=false" -allow-unregistered-dialect -split-input-file | FileCheck %s --check-prefixes=COMMON,A3
// RUN: bishengir-opt %s -propagate-reshape="for-hivm=true for-regbased=false" -allow-unregistered-dialect -split-input-file | FileCheck %s --check-prefixes=COMMON,A3
// RUN: bishengir-opt %s -propagate-reshape="for-regbased=true" -allow-unregistered-dialect -split-input-file | FileCheck %s --check-prefixes=COMMON,REGBASE

// A3 keeps rank-4 unit-only expansions outside allocs; RegBase may absorb them.
// A3-LABEL: func.func @rank4_unit_alloc_guard(
// A3: %[[ALLOC:.*]] = memref.alloc() : memref<2x3x4xf32>
// A3: %[[EXPANDED:.*]] = memref.expand_shape %[[ALLOC]] {{\[\[}}0], [1], [2, 3]]
// A3-SAME: into memref<2x3x4x1xf32>
// A3: return %[[EXPANDED]] : memref<2x3x4x1xf32>
// REGBASE-LABEL: func.func @rank4_unit_alloc_guard(
// REGBASE: %[[ALLOC:.*]] = memref.alloc() : memref<2x3x4x1xf32>
// REGBASE-NOT: memref.expand_shape
// REGBASE: return %[[ALLOC]] : memref<2x3x4x1xf32>
func.func @rank4_unit_alloc_guard() -> memref<2x3x4x1xf32> {
  %alloc = memref.alloc() : memref<2x3x4xf32>
  %expanded = memref.expand_shape %alloc [[0], [1], [2, 3]]
      output_shape [2, 3, 4, 1] :
      memref<2x3x4xf32> into memref<2x3x4x1xf32>
  return %expanded : memref<2x3x4x1xf32>
}

// -----

// Rank 3 is below the A3 workaround boundary, even for a unit-only expansion.
// COMMON-LABEL: func.func @rank3_unit_alloc_near_miss(
// COMMON: %[[ALLOC:.*]] = memref.alloc() : memref<2x3x1xf32>
// COMMON-NOT: memref.expand_shape
// COMMON: return %[[ALLOC]] : memref<2x3x1xf32>
func.func @rank3_unit_alloc_near_miss() -> memref<2x3x1xf32> {
  %alloc = memref.alloc() : memref<2x3xf32>
  %expanded = memref.expand_shape %alloc [[0], [1, 2]]
      output_shape [2, 3, 1] :
      memref<2x3xf32> into memref<2x3x1xf32>
  return %expanded : memref<2x3x1xf32>
}

// -----

// A non-unit split is not protected by the rank-4 unit-dimension workaround.
// COMMON-LABEL: func.func @rank4_non_unit_alloc_near_miss(
// COMMON: %[[ALLOC:.*]] = memref.alloc() : memref<2x3x2x4xf32>
// COMMON-NOT: memref.expand_shape
// COMMON: return %[[ALLOC]] : memref<2x3x2x4xf32>
func.func @rank4_non_unit_alloc_near_miss() -> memref<2x3x2x4xf32> {
  %alloc = memref.alloc() : memref<2x3x8xf32>
  %expanded = memref.expand_shape %alloc [[0], [1], [2, 3]]
      output_shape [2, 3, 2, 4] :
      memref<2x3x8xf32> into memref<2x3x2x4xf32>
  return %expanded : memref<2x3x2x4xf32>
}

// -----

// A same-parent HIVM load consumes the uncollapsed destination and an expanded
// source; the now-dead collapse is removed.
// COMMON-LABEL: func.func @collapse_into_direct_load(
// COMMON: %[[SRC:.*]] = memref.expand_shape %arg0 {{\[\[}}0, 1]]
// COMMON-SAME: into memref<4x4xf32>
// COMMON: %[[DST:.*]] = "test.destination"() : () -> memref<4x4xf32>
// COMMON-NOT: memref.collapse_shape
// COMMON: hivm.hir.load ins(%[[SRC]] : memref<4x4xf32>) outs(%[[DST]] : memref<4x4xf32>)
// COMMON-NOT: memref.collapse_shape
// COMMON: return
func.func @collapse_into_direct_load(%src: memref<16xf32>) {
  %dst = "test.destination"() : () -> memref<4x4xf32>
  %flat = memref.collapse_shape %dst [[0, 1]] :
      memref<4x4xf32> into memref<16xf32>
  hivm.hir.load ins(%src : memref<16xf32>) outs(%flat : memref<16xf32>)
  return
}

// -----

// Greedy reapplication must process every same-parent user, despite each
// pattern invocation rewriting only one user.
// COMMON-LABEL: func.func @collapse_with_two_direct_loads(
// COMMON-DAG: %[[SRC0:.*]] = memref.expand_shape %arg0 {{\[\[}}0, 1]]
// COMMON-DAG: %[[SRC1:.*]] = memref.expand_shape %arg1 {{\[\[}}0, 1]]
// COMMON-DAG: %[[DST:.*]] = "test.destination"() : () -> memref<4x4xf32>
// COMMON-NOT: memref.collapse_shape
// COMMON: hivm.hir.load ins(%[[SRC0]] : memref<4x4xf32>) outs(%[[DST]] : memref<4x4xf32>)
// COMMON: hivm.hir.load ins(%[[SRC1]] : memref<4x4xf32>) outs(%[[DST]] : memref<4x4xf32>)
// COMMON-NOT: memref.collapse_shape
// COMMON: return
func.func @collapse_with_two_direct_loads(
    %src0: memref<16xf32>, %src1: memref<16xf32>) {
  %dst = "test.destination"() : () -> memref<4x4xf32>
  %flat = memref.collapse_shape %dst [[0, 1]] :
      memref<4x4xf32> into memref<16xf32>
  hivm.hir.load ins(%src0 : memref<16xf32>) outs(%flat : memref<16xf32>)
  hivm.hir.load ins(%src1 : memref<16xf32>) outs(%flat : memref<16xf32>)
  return
}

// -----

// A3's same-parent boundary keeps the nested use collapsed. RegBase follows
// A5: after rewriting the direct use, the remaining single nested use may
// cross the region boundary on the next greedy iteration.
// A3-LABEL: func.func @collapse_with_direct_and_nested_load(
// A3: %[[SRC2D:.*]] = memref.expand_shape %arg0 {{\[\[}}0, 1]]
// A3: %[[DST:.*]] = "test.destination"() : () -> memref<4x4xf32>
// A3: %[[FLAT:.*]] = memref.collapse_shape %[[DST]] {{\[\[}}0, 1]]
// A3: hivm.hir.load ins(%[[SRC2D]] : memref<4x4xf32>) outs(%[[DST]] : memref<4x4xf32>)
// A3: scf.if %arg1 {
// A3: hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%[[FLAT]] : memref<16xf32>)
// REGBASE-LABEL: func.func @collapse_with_direct_and_nested_load(
// REGBASE-DAG: %[[SRC0:.*]] = memref.expand_shape %arg0 {{\[\[}}0, 1]]
// REGBASE-DAG: %[[SRC1:.*]] = memref.expand_shape %arg0 {{\[\[}}0, 1]]
// REGBASE: %[[DST:.*]] = "test.destination"() : () -> memref<4x4xf32>
// REGBASE-NOT: memref.collapse_shape
// REGBASE: hivm.hir.load ins(%{{.*}} : memref<4x4xf32>) outs(%[[DST]] : memref<4x4xf32>)
// REGBASE: scf.if %arg1 {
// REGBASE: hivm.hir.load ins(%{{.*}} : memref<4x4xf32>) outs(%[[DST]] : memref<4x4xf32>)
func.func @collapse_with_direct_and_nested_load(
    %src: memref<16xf32>, %condition: i1) {
  %dst = "test.destination"() : () -> memref<4x4xf32>
  %flat = memref.collapse_shape %dst [[0, 1]] :
      memref<4x4xf32> into memref<16xf32>
  hivm.hir.load ins(%src : memref<16xf32>) outs(%flat : memref<16xf32>)
  scf.if %condition {
    hivm.hir.load ins(%src : memref<16xf32>) outs(%flat : memref<16xf32>)
  }
  return
}

// -----

// Identical static collapse/expand pairs cancel to the original allocation.
// COMMON-LABEL: func.func @cancel_static_collapse_expand(
// COMMON: %[[SOURCE:.*]] = "test.source"() : () -> memref<2x3xf32>
// COMMON-NOT: memref.collapse_shape
// COMMON-NOT: memref.expand_shape
// COMMON: memref.load %[[SOURCE]][%{{.*}}, %{{.*}}] : memref<2x3xf32>
func.func @cancel_static_collapse_expand() -> f32 {
  %c0 = arith.constant 0 : index
  %source = "test.source"() : () -> memref<2x3xf32>
  %collapsed = memref.collapse_shape %source [[0, 1]] :
      memref<2x3xf32> into memref<6xf32>
  %expanded = memref.expand_shape %collapsed [[0, 1]]
      output_shape [2, 3] : memref<6xf32> into memref<2x3xf32>
  %value = memref.load %expanded[%c0, %c0] : memref<2x3xf32>
  return %value : f32
}

// -----

// A3's memref canonicalization proves the dynamic extent belongs to the
// source dimension. RegBase leaves this pair for A5-compatible pattern order.
// A3-LABEL: func.func @cancel_dynamic_collapse_expand(
// A3: %[[SOURCE:.*]] = "test.dynamic_source"(%arg0) : (index) -> memref<1x?x4xf32, strided<[128, 4, 1]>>
// A3-NOT: memref.collapse_shape
// A3-NOT: memref.expand_shape
// A3: memref.load %[[SOURCE]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<1x?x4xf32, strided<[128, 4, 1]>>
// REGBASE-LABEL: func.func @cancel_dynamic_collapse_expand(
// REGBASE: %[[SOURCE:.*]] = "test.dynamic_source"(%arg0) : (index) -> memref<1x?x4xf32, strided<[128, 4, 1]>>
// REGBASE: %[[COLLAPSED:.*]] = memref.collapse_shape %[[SOURCE]]
// REGBASE: %[[EXPANDED:.*]] = memref.expand_shape %[[COLLAPSED]]
// REGBASE: memref.load %[[EXPANDED]]
func.func @cancel_dynamic_collapse_expand(%size: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source = "test.dynamic_source"(%size) :
      (index) -> memref<1x?x4xf32, strided<[128, 4, 1]>>
  %extent = memref.dim %source, %c1 :
      memref<1x?x4xf32, strided<[128, 4, 1]>>
  %collapsed = memref.collapse_shape %source [[0, 1], [2]] :
      memref<1x?x4xf32, strided<[128, 4, 1]>> into
      memref<?x4xf32, strided<[4, 1]>>
  %expanded = memref.expand_shape %collapsed [[0, 1], [2]]
      output_shape [1, %extent, 4] :
      memref<?x4xf32, strided<[4, 1]>> into
      memref<1x?x4xf32, strided<[?, 4, 1]>>
  %value = memref.load %expanded[%c0, %c0, %c0] :
      memref<1x?x4xf32, strided<[?, 4, 1]>>
  return %value : f32
}

// -----

// An unrelated memref.dim cannot prove cancellation and must leave the pair
// intact.
// COMMON-LABEL: func.func @reject_unrelated_dynamic_extent(
// COMMON: %[[SOURCE:.*]] = "test.dynamic_source"(%arg0) : (index) -> memref<1x?x4xf32, strided<[128, 4, 1]>>
// COMMON: %[[COLLAPSED:.*]] = memref.collapse_shape %[[SOURCE]] {{\[\[}}0, 1], [2]]
// COMMON: %[[EXPANDED:.*]] = memref.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1], [2]]
// COMMON-SAME: output_shape [1, %{{.*}}, 4]
// COMMON: memref.load %[[EXPANDED]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<1x?x4xf32, strided<[?, 4, 1]>>
func.func @reject_unrelated_dynamic_extent(
    %size: index, %other_size: index) -> f32 {
  %c0 = arith.constant 0 : index
  %source = "test.dynamic_source"(%size) :
      (index) -> memref<1x?x4xf32, strided<[128, 4, 1]>>
  %other = memref.alloc(%other_size) : memref<?xi8>
  %unrelated = memref.dim %other, %c0 : memref<?xi8>
  %collapsed = memref.collapse_shape %source [[0, 1], [2]] :
      memref<1x?x4xf32, strided<[128, 4, 1]>> into
      memref<?x4xf32, strided<[4, 1]>>
  %expanded = memref.expand_shape %collapsed [[0, 1], [2]]
      output_shape [1, %unrelated, 4] :
      memref<?x4xf32, strided<[4, 1]>> into
      memref<1x?x4xf32, strided<[?, 4, 1]>>
  %value = memref.load %expanded[%c0, %c0, %c0] :
      memref<1x?x4xf32, strided<[?, 4, 1]>>
  return %value : f32
}

// -----

// The A3-only swap pattern rewrites compatible reassociations. A5 does not
// register that pattern in RegBase.
// A3-LABEL: func.func @swap_compatible_static_shapes(
// A3: %[[SOURCE:.*]] = "test.source"() : () -> memref<2x3x4xf32>
// A3: %[[EXPANDED:.*]] = memref.expand_shape %[[SOURCE]] {{\[\[}}0], [1], [2, 3]]
// A3: %[[COLLAPSED:.*]] = memref.collapse_shape %[[EXPANDED]] {{\[\[}}0, 1], [2], [3]]
// A3: memref.load %[[COLLAPSED]]
// REGBASE-LABEL: func.func @swap_compatible_static_shapes(
// REGBASE: %[[SOURCE:.*]] = "test.source"() : () -> memref<2x3x4xf32>
// REGBASE: %[[COLLAPSED:.*]] = memref.collapse_shape %[[SOURCE]] {{\[\[}}0, 1], [2]]
// REGBASE: %[[EXPANDED:.*]] = memref.expand_shape %[[COLLAPSED]] {{\[\[}}0], [1, 2]]
// REGBASE: memref.load %[[EXPANDED]]
func.func @swap_compatible_static_shapes() -> f32 {
  %c0 = arith.constant 0 : index
  %source = "test.source"() : () -> memref<2x3x4xf32>
  %collapsed = memref.collapse_shape %source [[0, 1], [2]] :
      memref<2x3x4xf32> into memref<6x4xf32>
  %expanded = memref.expand_shape %collapsed [[0], [1, 2]]
      output_shape [6, 2, 2] :
      memref<6x4xf32> into memref<6x2x2xf32>
  %value = memref.load %expanded[%c0, %c0, %c0] : memref<6x2x2xf32>
  return %value : f32
}

// -----

// A result layout that cannot equal the recomputed swapped layout rejects the
// swap, preserving the original collapse-then-expand order.
// COMMON-LABEL: func.func @reject_incompatible_swap_layout(
// COMMON: %[[SOURCE:.*]] = "test.strided_source"() : () -> memref<1x2x3xf32, strided<[100, 3, 1]>>
// COMMON: %[[COLLAPSED:.*]] = memref.collapse_shape %[[SOURCE]] {{\[\[}}0, 1, 2]
// COMMON-SAME: into memref<6xf32, strided<[1]>>
// COMMON: %[[EXPANDED:.*]] = memref.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1]
// COMMON-SAME: into memref<3x2xf32>
// COMMON: memref.load %[[EXPANDED]][%{{.*}}, %{{.*}}] : memref<3x2xf32>
func.func @reject_incompatible_swap_layout() -> f32 {
  %c0 = arith.constant 0 : index
  %source = "test.strided_source"() :
      () -> memref<1x2x3xf32, strided<[100, 3, 1]>>
  %collapsed = memref.collapse_shape %source [[0, 1, 2]] :
      memref<1x2x3xf32, strided<[100, 3, 1]>> into
      memref<6xf32, strided<[1]>>
  %expanded = memref.expand_shape %collapsed [[0, 1]]
      output_shape [3, 2] : memref<6xf32, strided<[1]>> into
      memref<3x2xf32>
  %value = memref.load %expanded[%c0, %c0] : memref<3x2xf32>
  return %value : f32
}
