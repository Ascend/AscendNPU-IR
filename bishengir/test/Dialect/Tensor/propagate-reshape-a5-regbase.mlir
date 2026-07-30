// RUN: bishengir-opt %s -propagate-reshape="for-regbased=false" -allow-unregistered-dialect -verify-each=false -split-input-file | FileCheck %s --check-prefix=A3
// RUN: bishengir-opt %s -propagate-reshape="for-regbased=true" -allow-unregistered-dialect -split-input-file | FileCheck %s --check-prefix=REGBASE

// RegBase moves a legal flip to expanded rank and preserves the shared
// low-rank result with a collapse; A3 leaves the original order.
// A3-LABEL: func.func @flip_expand(
// A3: %[[FLIP:.*]] = hfusion.flip %arg0 : tensor<2x12x5xf32> flip_axis = 2 -> tensor<2x12x5xf32>
// A3: %[[EXP:.*]] = tensor.expand_shape %[[FLIP]] {{\[\[}}0], [1, 2], [3]] output_shape [2, 3, 4, 5]
// REGBASE-LABEL: func.func @flip_expand(
// REGBASE: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1, 2], [3]] output_shape [2, 3, 4, 5]
// REGBASE: %[[FLIP:.*]] = hfusion.flip %[[EXP]] : tensor<2x3x4x5xf32> flip_axis = 3 -> tensor<2x3x4x5xf32>
// REGBASE: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[FLIP]] {{\[\[}}0], [1, 2], [3]]
// REGBASE: return %{{.*}}, %[[COLLAPSE]]
func.func @flip_expand(%arg0: tensor<2x12x5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x12x5xf32>) {
  %flipped = hfusion.flip %arg0 : tensor<2x12x5xf32> flip_axis = 2 -> tensor<2x12x5xf32>
  %expanded = tensor.expand_shape %flipped [[0], [1, 2], [3]]
      output_shape [2, 3, 4, 5] :
      tensor<2x12x5xf32> into tensor<2x3x4x5xf32>
  %used = math.absf %expanded : tensor<2x3x4x5xf32>
  return %used, %flipped : tensor<2x3x4x5xf32>, tensor<2x12x5xf32>
}

// -----

// RegBase moves a legal flip above collapse; A3 leaves it below.
// A3-LABEL: func.func @flip_collapse(
// A3: %[[COLLAPSE:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]]
// A3: %[[FLIP:.*]] = hfusion.flip %[[COLLAPSE]] : tensor<2x12x5xf32> flip_axis = 2
// REGBASE-LABEL: func.func @flip_collapse(
// REGBASE: %[[FLIP:.*]] = hfusion.flip %{{.*}} : tensor<2x3x4x5xf32> flip_axis = 3
// REGBASE: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[FLIP]] {{\[\[}}0], [1, 2], [3]]
func.func @flip_collapse(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x12x5xf32> {
  %source = math.absf %arg0 : tensor<2x3x4x5xf32>
  %collapsed = tensor.collapse_shape %source [[0], [1, 2], [3]] :
      tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
  %flipped = hfusion.flip %collapsed :
      tensor<2x12x5xf32> flip_axis = 2 -> tensor<2x12x5xf32>
  return %flipped : tensor<2x12x5xf32>
}

// -----

// Even RegBase must not move a flip whose axis represents multiple source
// dimensions, because that would change element order inside the group.
// A3-LABEL: func.func @flip_collapse_reject_group_axis(
// A3: %[[COLLAPSE:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]]
// A3-NEXT: %{{.*}} = hfusion.flip %[[COLLAPSE]] : tensor<2x12x5xf32> flip_axis = 1
// REGBASE-LABEL: func.func @flip_collapse_reject_group_axis(
// REGBASE: %[[COLLAPSE:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]]
// REGBASE-NEXT: %{{.*}} = hfusion.flip %[[COLLAPSE]] : tensor<2x12x5xf32> flip_axis = 1
func.func @flip_collapse_reject_group_axis(
    %arg0: tensor<2x3x4x5xf32>) -> tensor<2x12x5xf32> {
  %source = math.absf %arg0 : tensor<2x3x4x5xf32>
  %collapsed = tensor.collapse_shape %source [[0], [1, 2], [3]] :
      tensor<2x3x4x5xf32> into tensor<2x12x5xf32>
  %flipped = hfusion.flip %collapsed :
      tensor<2x12x5xf32> flip_axis = 1 -> tensor<2x12x5xf32>
  return %flipped : tensor<2x12x5xf32>
}

// -----

// RegBase expands extraction coordinates back through the collapse following
// an expand; A3 keeps the low-rank extraction.
// A3-LABEL: func.func @extract_after_expand(
// A3: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1, 2]] output_shape [2, 3, 2]
// A3: %[[SLICE:.*]] = tensor.extract_slice %[[EXP]][0, 0, 0] [2, 3, 1] [1, 1, 2]
// A3: %[[REM:.*]] = arith.remui %arg1, %{{.*}} : index
// A3: %[[DIV:.*]] = arith.divui %arg1, %{{.*}} : index
// A3: tensor.extract %[[SLICE]][%[[DIV]], %[[REM]], %{{.*}}] : tensor<2x3x1xf32>
// REGBASE-LABEL: func.func @extract_after_expand(
// REGBASE: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1, 2]] output_shape [2, 3, 2]
// REGBASE: %[[SLICE:.*]] = tensor.extract_slice %[[EXP]][0, 0, 0] [2, 3, 1] [1, 1, 2]
// REGBASE: tensor.extract %[[SLICE]][%{{.*}}, %{{.*}}, %{{.*}}] : tensor<2x3x1xf32>
func.func @extract_after_expand(
    %arg0: tensor<2x6xf32>, %i: index) -> f32 {
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2]]
      output_shape [2, 3, 2] :
      tensor<2x6xf32> into tensor<2x3x2xf32>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2]] :
      tensor<2x3x2xf32> into tensor<6x2xf32>
  %slice = tensor.extract_slice %collapsed[0, 0] [6, 1] [1, 2] :
      tensor<6x2xf32> to tensor<6x1xf32>
  %flat = tensor.collapse_shape %slice [[0, 1]] :
      tensor<6x1xf32> into tensor<6xf32>
  %value = tensor.extract %flat[%i] : tensor<6xf32>
  return %value : f32
}

// -----

// A3 preserves rank reduction before expand; RegBase expands the source and
// performs the rank-reducing slice at expanded rank.
// A3-LABEL: func.func @rank_reducing_extract_expand(
// A3: %[[SLICE:.*]] = tensor.extract_slice %arg0[0, 0] [1, 6] [1, 1] : tensor<2x6xf32> to tensor<6xf32>
// A3: tensor.expand_shape %[[SLICE]] {{\[\[}}0, 1]] output_shape [2, 3]
// REGBASE-LABEL: func.func @rank_reducing_extract_expand(
// REGBASE: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0], [1, 2]] output_shape [2, 2, 3]
// REGBASE: tensor.extract_slice %[[EXP]][0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<2x2x3xf32> to tensor<2x3xf32>
func.func @rank_reducing_extract_expand(
    %arg0: tensor<2x6xf32>) -> tensor<2x3xf32> {
  %slice = tensor.extract_slice %arg0[0, 0] [1, 6] [1, 1] :
      tensor<2x6xf32> to tensor<6xf32>
  %expanded = tensor.expand_shape %slice [[0, 1]] output_shape [2, 3] :
      tensor<6xf32> into tensor<2x3xf32>
  %used = math.absf %expanded : tensor<2x3xf32>
  return %used : tensor<2x3xf32>
}

// -----

// A3 preserves rank-reducing insertion before expand; RegBase expands both
// operands and inserts the expanded source into the expanded destination.
// A3-LABEL: func.func @rank_reducing_insert_expand(
// A3: %[[INSERT:.*]] = tensor.insert_slice %arg0 into %arg1[0, 0] [1, 6] [1, 1]
// A3: tensor.expand_shape %[[INSERT]] {{\[\[}}0], [1, 2]] output_shape [2, 2, 3]
// REGBASE-LABEL: func.func @rank_reducing_insert_expand(
// REGBASE: %[[SRC:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [2, 3]
// REGBASE: %[[DEST:.*]] = tensor.expand_shape %arg1 {{\[\[}}0], [1, 2]] output_shape [2, 2, 3]
// REGBASE: tensor.insert_slice %[[SRC]] into %[[DEST]][0, 0, 0] [1, 2, 3] [1, 1, 1]
func.func @rank_reducing_insert_expand(
    %src: tensor<6xf32>, %dest: tensor<2x6xf32>) -> tensor<2x2x3xf32> {
  %inserted = tensor.insert_slice %src into %dest[0, 0] [1, 6] [1, 1] :
      tensor<6xf32> into tensor<2x6xf32>
  %expanded = tensor.expand_shape %inserted [[0], [1, 2]]
      output_shape [2, 2, 3] :
      tensor<2x6xf32> into tensor<2x2x3xf32>
  %used = math.absf %expanded : tensor<2x2x3xf32>
  return %used : tensor<2x2x3xf32>
}

// -----

// A3 preserves rank reduction after collapse; RegBase slices the high-rank
// source and restores the requested result rank with a collapse.
// A3-LABEL: func.func @rank_reducing_extract_collapse(
// A3: %[[COLLAPSE:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]]
// A3: tensor.extract_slice %[[COLLAPSE]][0, 0] [1, 4] [1, 1] : tensor<6x4xf32> to tensor<4xf32>
// REGBASE-LABEL: func.func @rank_reducing_extract_collapse(
// REGBASE-NOT: tensor.collapse_shape
// REGBASE: tensor.extract_slice %{{.*}}[0, 0, 0] [1, 1, 4] [1, 1, 1] : tensor<2x3x4xf32> to tensor<4xf32>
func.func @rank_reducing_extract_collapse(
    %arg0: tensor<2x3x4xf32>) -> tensor<4xf32> {
  %source = math.absf %arg0 : tensor<2x3x4xf32>
  %collapsed = tensor.collapse_shape %source [[0, 1], [2]] :
      tensor<2x3x4xf32> into tensor<6x4xf32>
  %slice = tensor.extract_slice %collapsed[0, 0] [1, 4] [1, 1] :
      tensor<6x4xf32> to tensor<4xf32>
  return %slice : tensor<4xf32>
}

// -----

// A3 preserves rank-reducing insertion after collapse; RegBase inserts from
// the high-rank source and then collapses the destination result.
// A3-LABEL: func.func @rank_reducing_insert_collapse(
// A3: %[[COLLAPSE:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]]
// A3: tensor.insert_slice %[[COLLAPSE]] into %arg1[0, 0, 0] [1, 6, 4] [1, 1, 1]
// REGBASE-LABEL: func.func @rank_reducing_insert_collapse(
// REGBASE: %[[DEST:.*]] = tensor.expand_shape %arg1 {{\[\[}}0], [1, 2], [3]] output_shape [2, 2, 3, 4]
// REGBASE: %[[INSERT:.*]] = tensor.insert_slice %{{.*}} into %[[DEST]][0, 0, 0, 0] [1, 2, 3, 4] [1, 1, 1, 1]
// REGBASE: tensor.collapse_shape %[[INSERT]] {{\[\[}}0], [1, 2], [3]]
func.func @rank_reducing_insert_collapse(
    %src: tensor<2x3x4xf32>, %dest: tensor<2x6x4xf32>)
    -> tensor<2x6x4xf32> {
  %source = math.absf %src : tensor<2x3x4xf32>
  %collapsed = tensor.collapse_shape %source [[0, 1], [2]] :
      tensor<2x3x4xf32> into tensor<6x4xf32>
  %inserted = tensor.insert_slice %collapsed into %dest[0, 0, 0]
      [1, 6, 4] [1, 1, 1] :
      tensor<6x4xf32> into tensor<2x6x4xf32>
  return %inserted : tensor<2x6x4xf32>
}

// -----

// A3 crosses to_tensor regardless of layout; RegBase refuses because the
// backing memref's non-identity layout cannot represent the tensor reshape.
// A3-LABEL: sym_name = "to_tensor_non_identity_layout"
// A3: "memref.expand_shape"(%arg0) {{.*}} : (memref<6xf32, strided<[2]>>) -> memref<2x3xf32>
// A3: "bufferization.to_tensor"(%{{.*}}) {{.*}} : (memref<2x3xf32>) -> tensor<2x3xf32>
// REGBASE-LABEL: func.func @to_tensor_non_identity_layout(
// REGBASE: %[[TENSOR:.*]] = bufferization.to_tensor %arg0 restrict writable : memref<6xf32, strided<[2]>>
// REGBASE: tensor.expand_shape %[[TENSOR]] {{\[\[}}0, 1]] output_shape [2, 3]
func.func @to_tensor_non_identity_layout(
    %arg0: memref<6xf32, strided<[2]>>) -> tensor<2x3xf32> {
  %tensor = bufferization.to_tensor %arg0 restrict writable :
      memref<6xf32, strided<[2]>>
  %expanded = tensor.expand_shape %tensor [[0, 1]] output_shape [2, 3] :
      tensor<6xf32> into tensor<2x3xf32>
  %used = math.absf %expanded : tensor<2x3xf32>
  return %used : tensor<2x3xf32>
}

// -----

// A3's unit-only subview path rejects this non-unit split; RegBase moves the
// general rectangular subview above the expand.
// A3-LABEL: func.func @memref_subview_expand(
// A3: %[[SUB:.*]] = memref.subview %arg0[6] [6] [1]
// A3: memref.expand_shape %[[SUB]] {{\[\[}}0, 1]] output_shape [2, 3]
// REGBASE-LABEL: func.func @memref_subview_expand(
// REGBASE: %[[EXP:.*]] = memref.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [8, 3]
// REGBASE: memref.subview %[[EXP]][2, 0] [2, 3] [1, 1] : memref<8x3xf32> to memref<2x3xf32, strided<[3, 1], offset: 6>>
func.func @memref_subview_expand(
    %arg0: memref<24xf32>) -> memref<2x3xf32, strided<[3, 1], offset: ?>> {
  %subview = memref.subview %arg0[6] [6] [1] :
      memref<24xf32> to memref<6xf32, strided<[1], offset: 6>>
  %expanded = memref.expand_shape %subview [[0, 1]] output_shape [2, 3] :
      memref<6xf32, strided<[1], offset: 6>> into
      memref<2x3xf32, strided<[3, 1], offset: 6>>
  %cast = memref.cast %expanded :
      memref<2x3xf32, strided<[3, 1], offset: 6>> to
      memref<2x3xf32, strided<[3, 1], offset: ?>>
  return %cast : memref<2x3xf32, strided<[3, 1], offset: ?>>
}

// -----

// A3's unit-only subview path rejects this non-unit collapse; RegBase computes
// the equivalent rectangular subview at source rank.
// A3-LABEL: func.func @memref_subview_collapse(
// A3: %[[COLLAPSE:.*]] = memref.collapse_shape %{{.*}} {{\[\[}}0, 1]]
// A3: memref.subview %[[COLLAPSE]][6] [12] [1]
// REGBASE-LABEL: func.func @memref_subview_collapse(
// REGBASE: %[[SUB:.*]] = memref.subview %{{.*}}[1, 0] [2, 6] [1, 1] : memref<4x6xf32> to memref<2x6xf32, strided<[6, 1], offset: 6>>
// REGBASE: memref.collapse_shape %[[SUB]] {{\[\[}}0, 1]] : memref<2x6xf32, strided<[6, 1], offset: 6>> into memref<12xf32, strided<[1], offset: 6>>
func.func @memref_subview_collapse()
    -> memref<12xf32, strided<[1], offset: ?>> {
  %source = "test.memref_source"() : () -> memref<4x6xf32>
  %collapsed = memref.collapse_shape %source [[0, 1]] :
      memref<4x6xf32> into memref<24xf32>
  %subview = memref.subview %collapsed[6] [12] [1] :
      memref<24xf32> to memref<12xf32, strided<[1], offset: 6>>
  %cast = memref.cast %subview :
      memref<12xf32, strided<[1], offset: 6>> to
      memref<12xf32, strided<[1], offset: ?>>
  return %cast : memref<12xf32, strided<[1], offset: ?>>
}

// -----

// RegBase computes the collapsed layout from the expanded cast and repairs its
// dynamic layout back to the original static-stride type.
// A3-LABEL: sym_name = "reinterpret_expand_repair"
// A3: "memref.reinterpret_cast"(%arg0) {{.*}} : (memref<?xf32>) -> memref<2x3xf32>
// A3: "memref.collapse_shape"(%{{.*}}) {{.*}} : (memref<2x3xf32>) -> memref<6xf32, strided<[1]>>
// A3-NOT: "memref.reinterpret_cast"(%{{.*}}) {{.*}} : (memref<6xf32>)
// REGBASE-LABEL: func.func @reinterpret_expand_repair(
// REGBASE: %[[VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 3], strides: [3, 1]
// REGBASE: %[[COLLAPSE:.*]] = memref.collapse_shape %[[VIEW]] {{\[\[}}0, 1]] : memref<2x3xf32> into memref<6xf32>
// REGBASE: %[[REPAIR:.*]] = memref.reinterpret_cast %[[COLLAPSE]] to offset: [0], sizes: [6], strides: [1]
// REGBASE: return %[[VIEW]], %[[REPAIR]]
func.func @reinterpret_expand_repair(
    %arg0: memref<?xf32>) -> (memref<2x3xf32>,
                              memref<6xf32, strided<[1]>>) {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [6],
      strides: [1] :
      memref<?xf32> to memref<6xf32, strided<[1]>>
  %expanded = memref.expand_shape %view [[0, 1]] output_shape [2, 3] :
      memref<6xf32, strided<[1]>> into memref<2x3xf32>
  return %expanded, %view :
      memref<2x3xf32>, memref<6xf32, strided<[1]>>
}

// -----

// When the computed collapsed layout is already exact, RegBase needs no
// repair cast.
// A3-LABEL: func.func @reinterpret_expand_no_repair(
// A3: %[[VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 3], strides: [6, 2]
// A3: %[[COLLAPSE:.*]] = memref.collapse_shape %[[VIEW]] {{\[\[}}0, 1]]
// A3: return %[[VIEW]], %[[COLLAPSE]]
// REGBASE-LABEL: func.func @reinterpret_expand_no_repair(
// REGBASE: %[[VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 3], strides: [6, 2]
// REGBASE: %[[COLLAPSE:.*]] = memref.collapse_shape %[[VIEW]] {{\[\[}}0, 1]]
// REGBASE-NOT: memref.reinterpret_cast
// REGBASE: return %[[VIEW]], %[[COLLAPSE]]
func.func @reinterpret_expand_no_repair(
    %arg0: memref<?xf32>) -> (memref<2x3xf32, strided<[6, 2]>>,
                              memref<6xf32, strided<[2]>>) {
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [6],
      strides: [2] :
      memref<?xf32> to memref<6xf32, strided<[2]>>
  %expanded = memref.expand_shape %view [[0, 1]] output_shape [2, 3] :
      memref<6xf32, strided<[2]>> into
      memref<2x3xf32, strided<[6, 2]>>
  return %expanded, %view :
      memref<2x3xf32, strided<[6, 2]>>, memref<6xf32, strided<[2]>>
}

// -----

// Only RegBase redirects fill to the high-rank allocation; the returned
// low-rank view keeps the collapse live.
// A3-LABEL: func.func @memref_collapse_fill(
// A3: %[[SRC:.*]] = "test.memref_source"() : () -> memref<2x3xf32>
// A3: %[[COLLAPSE:.*]] = memref.collapse_shape %[[SRC]] {{\[\[}}0, 1]]
// A3: linalg.fill ins(%{{.*}} : f32) outs(%[[COLLAPSE]] : memref<6xf32>)
// REGBASE-LABEL: func.func @memref_collapse_fill(
// REGBASE: %[[SRC:.*]] = "test.memref_source"() : () -> memref<2x3xf32>
// REGBASE: %[[COLLAPSE:.*]] = memref.collapse_shape %[[SRC]] {{\[\[}}0, 1]]
// REGBASE: linalg.fill ins(%{{.*}} : f32) outs(%[[SRC]] : memref<2x3xf32>)
// REGBASE: return %[[COLLAPSE]]
func.func @memref_collapse_fill() -> memref<6xf32> {
  %cst = arith.constant 0.0 : f32
  %source = "test.memref_source"() : () -> memref<2x3xf32>
  %collapsed = memref.collapse_shape %source [[0, 1]] :
      memref<2x3xf32> into memref<6xf32>
  linalg.fill ins(%cst : f32) outs(%collapsed : memref<6xf32>)
  return %collapsed : memref<6xf32>
}

// -----

// Only RegBase redirects mark to the high-rank allocation; the independent
// return use proves that the shared collapse is retained.
// A3-LABEL: func.func @memref_collapse_mark_shared(
// A3: %[[SRC:.*]] = "test.memref_source"() : () -> memref<2x3xf32>
// A3: %[[COLLAPSE:.*]] = memref.collapse_shape %[[SRC]] {{\[\[}}0, 1]]
// A3: annotation.mark %[[COLLAPSE]] {unit_test} : memref<6xf32>
// REGBASE-LABEL: func.func @memref_collapse_mark_shared(
// REGBASE: %[[SRC:.*]] = "test.memref_source"() : () -> memref<2x3xf32>
// REGBASE: %[[COLLAPSE:.*]] = memref.collapse_shape %[[SRC]] {{\[\[}}0, 1]]
// REGBASE: annotation.mark %[[SRC]] {unit_test} : memref<2x3xf32>
// REGBASE: return %[[COLLAPSE]]
func.func @memref_collapse_mark_shared() -> memref<6xf32> {
  %source = "test.memref_source"() : () -> memref<2x3xf32>
  %collapsed = memref.collapse_shape %source [[0, 1]] :
      memref<2x3xf32> into memref<6xf32>
  annotation.mark %collapsed {unit_test} : memref<6xf32>
  return %collapsed : memref<6xf32>
}

// -----

// A5 permits a single collapse user in a nested region. A3 keeps the original
// region boundary.
// A3-LABEL: func.func @collapse_single_cross_region_user(
// A3: %[[COLLAPSED:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1]
// A3: scf.if
// A3: tensor.extract_slice %[[COLLAPSED]][0, 0] [1, 4] [1, 1]
// REGBASE-LABEL: func.func @collapse_single_cross_region_user(
// REGBASE: scf.if
// REGBASE: %[[HIGH:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0] [1, 1, 4] [1, 1, 1]
// REGBASE: tensor.collapse_shape %[[HIGH]] {{\[\[}}0, 1], [2]
func.func @collapse_single_cross_region_user(
    %arg0: tensor<2x3x4xf32>, %fallback: tensor<1x4xf32>, %condition: i1)
    -> tensor<1x4xf32> {
  %source = math.absf %arg0 : tensor<2x3x4xf32>
  %collapsed = tensor.collapse_shape %source [[0, 1], [2]] :
      tensor<2x3x4xf32> into tensor<6x4xf32>
  %result = scf.if %condition -> tensor<1x4xf32> {
    %used = tensor.extract_slice %collapsed[0, 0] [1, 4] [1, 1] :
        tensor<6x4xf32> to tensor<1x4xf32>
    scf.yield %used : tensor<1x4xf32>
  } else {
    scf.yield %fallback : tensor<1x4xf32>
  }
  return %result : tensor<1x4xf32>
}

// -----

// RegBase propagates both live ReduceWithIndex inputs to expanded rank. The A3
// operation canonicalizer then keeps equivalent low-rank init/results rather
// than A5's explicit expand/collapse pairs.
// A3-LABEL: func.func @reduce_with_index_live_results(
// A3: %[[COLLAPSED:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]
// A3: hfusion.reduce_with_index
// A3-SAME: ins(%[[COLLAPSED]], %arg1 : tensor<1x8x4xf32>, tensor<1x8x4xi32>)
// REGBASE-LABEL: func.func @reduce_with_index_live_results(
// REGBASE: %[[INDEX:.*]] = tensor.expand_shape %arg1 {{\[\[}}0], [1, 2], [3]]
// REGBASE: hfusion.reduce_with_index
// REGBASE-SAME: ins(%{{.*}}, %[[INDEX]] : tensor<1x8x1x4xf32>, tensor<1x8x1x4xi32>)
func.func @reduce_with_index_live_results(
    %arg0: tensor<1x8x1xf32>, %arg1: tensor<1x8x4xi32>)
    -> (tensor<1x4xf32>, tensor<1x4xi32>) {
  %inputInit = tensor.empty() : tensor<1x8x1xf32>
  %input = linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
      ins(%arg0 : tensor<1x8x1xf32>)
      outs(%inputInit : tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
  %broadcastInit = tensor.empty() : tensor<1x8x4xf32>
  %collapsed = tensor.collapse_shape %input [[0], [1, 2]] :
      tensor<1x8x1xf32> into tensor<1x8xf32>
  %broadcasted = linalg.broadcast
      ins(%collapsed : tensor<1x8xf32>)
      outs(%broadcastInit : tensor<1x8x4xf32>) dimensions = [2]
  %valueInit = tensor.empty() : tensor<1x4xf32>
  %indexInit = tensor.empty() : tensor<1x4xi32>
  %value, %index = hfusion.reduce_with_index
      {tie_break_left = true, unsigned_src = false} <max>
      ins(%broadcasted, %arg1 : tensor<1x8x4xf32>, tensor<1x8x4xi32>)
      outs(%valueInit, %indexInit : tensor<1x4xf32>, tensor<1x4xi32>)
      dimensions = [1] -> tensor<1x4xf32>, tensor<1x4xi32>
  return %value, %index : tensor<1x4xf32>, tensor<1x4xi32>
}
