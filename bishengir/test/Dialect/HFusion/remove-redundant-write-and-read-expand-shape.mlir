// RUN: bishengir-opt %s -split-input-file -remove-redundant-write-and-read-pair | FileCheck %s

#flatten = affine_map<(d0) -> (d0 * 8)>
#swap = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// #1071: a masked write to a contiguous row, followed by an identity insert
// and a read of the reshaped row with a unit-axis swap. Preserve read padding
// even when the input vector has non-padding values in its inactive lanes.
func.func @forward_masked_tail(%dst: tensor<1x?xf32>, %v: vector<1x64xf32>,
                              %rows: index) -> vector<8x1x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 7.0 : f32
  %cols = affine.apply #flatten(%rows)
  %wm = vector.create_mask %c1, %cols : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %i = tensor.insert_slice %w into %dst[0, 0] [1, %cols] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
  %e = tensor.expand_shape %i [[0], [1, 2]] output_shape [1, %rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %rm = vector.create_mask %c1, %rows, %c8 : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true], permutation_map = #swap} : tensor<1x?x8xf32>, vector<8x1x8xf32>
  return %r : vector<8x1x8xf32>
}

// CHECK-LABEL: func.func @forward_masked_tail(
// CHECK-SAME: %[[V:[^,:]+]]: vector<1x64xf32>
// CHECK-DAG: %[[PAD:.+]] = arith.constant dense<7.000000e+00> : vector<1x64xf32>
// CHECK-DAG: %[[MASK:.+]] = vector.create_mask {{.*}} : vector<1x64xi1>
// CHECK-NOT: vector.transfer_write
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: vector.transfer_read
// CHECK: %[[SELECT:.+]] = arith.select %[[MASK]], %[[V]], %[[PAD]] : vector<1x64xi1>, vector<1x64xf32>
// CHECK-NEXT: %[[CAST:.+]] = vector.shape_cast %[[SELECT]] : vector<1x64xf32> to vector<8x1x8xf32>
// CHECK-NEXT: return %[[CAST]] : vector<8x1x8xf32>

// -----

func.func @forward_unmasked(%dst: tensor<1x64xf32>,
                            %v: vector<1x64xf32>) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %w = vector.transfer_write %v, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, 8, 8] : tensor<1x64xf32> into tensor<1x8x8xf32>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} : tensor<1x8x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @forward_unmasked(
// CHECK-NEXT: %[[CAST:.+]] = vector.shape_cast %{{.*}} : vector<1x64xf32> to vector<1x8x8xf32>
// CHECK-NEXT: return %[[CAST]] : vector<1x8x8xf32>

// -----

#flatten_rank3 = affine_map<(d0) -> (d0 * 8)>

// Rank-3 writes keep the pair: SelectOpPattern only lowers rank-1 and [1, N]
// padding selects on A5, so the first version stays at write rank <= 2.
// Dynamic masks prevent mask folding from hiding the rank.
func.func @keep_masked_rank_3(%dst: tensor<1x1x?xf32>, %v: vector<1x1x64xf32>,
                              %rows: index) -> vector<1x1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0.0 : f32
  %cols = affine.apply #flatten_rank3(%rows)
  %wm = vector.create_mask %c1, %c1, %cols : vector<1x1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0, %c0], %wm {in_bounds = [true, true, true]} : vector<1x1x64xf32>, tensor<1x1x?xf32>
  %e = tensor.expand_shape %w [[0], [1], [2, 3]] output_shape [1, 1, %rows, 8] : tensor<1x1x?xf32> into tensor<1x1x?x8xf32>
  %rm = vector.create_mask %c1, %c1, %rows, %c8 : vector<1x1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true, true]} : tensor<1x1x?x8xf32>, vector<1x1x8x8xf32>
  return %r : vector<1x1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_masked_rank_3
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: vector.transfer_read %[[E]]

// -----

// Partial constant masks exercise the constant-bound path; canonicalization
// keeps them because they are not all-true.
func.func @forward_constant_mask(%dst: tensor<1x?xf32>,
                                 %v: vector<1x64xf32>,
                                 %rows: index) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %pad = arith.constant 3.0 : f32
  %wm = vector.constant_mask [1, 32] : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %i = tensor.insert_slice %w into %dst[0, 0] [1, %c32] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
  %e = tensor.expand_shape %i [[0], [1, 2]] output_shape [1, %rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %rm = vector.constant_mask [1, 4, 8] : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true]} : tensor<1x?x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @forward_constant_mask(
// CHECK-SAME: %[[V:[^,:]+]]: vector<1x64xf32>
// CHECK: %[[MASK:.+]] = vector.constant_mask [1, 32] : vector<1x64xi1>
// CHECK-NOT: vector.transfer_write
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: vector.transfer_read
// CHECK: %[[SELECT:.+]] = arith.select %[[MASK]], %[[V]], %{{.*}} : vector<1x64xi1>, vector<1x64xf32>
// CHECK-NEXT: %[[CAST:.+]] = vector.shape_cast %[[SELECT]] : vector<1x64xf32> to vector<1x8x8xf32>
// CHECK-NEXT: return %[[CAST]] : vector<1x8x8xf32>

// -----

// A non-constant padding value must be broadcast before the select.
#flatten = affine_map<(d0) -> (d0 * 8)>
#swap = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

func.func @forward_dynamic_padding(%dst: tensor<1x?xf32>, %v: vector<1x64xf32>,
                                   %rows: index, %pad: f32) -> vector<8x1x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cols = affine.apply #flatten(%rows)
  %wm = vector.create_mask %c1, %cols : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %i = tensor.insert_slice %w into %dst[0, 0] [1, %cols] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
  %e = tensor.expand_shape %i [[0], [1, 2]] output_shape [1, %rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %rm = vector.create_mask %c1, %rows, %c8 : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true], permutation_map = #swap} : tensor<1x?x8xf32>, vector<8x1x8xf32>
  return %r : vector<8x1x8xf32>
}

// CHECK-LABEL: func.func @forward_dynamic_padding(
// CHECK-SAME: %[[V:[^,:]+]]: vector<1x64xf32>, %{{.*}}: index, %[[PADARG:[^,:]+]]: f32
// CHECK-DAG: %[[MASK:.+]] = vector.create_mask {{.*}} : vector<1x64xi1>
// CHECK-NOT: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: %[[PAD:.+]] = vector.broadcast %[[PADARG]] : f32 to vector<1x64xf32>
// CHECK: %[[SELECT:.+]] = arith.select %[[MASK]], %[[V]], %[[PAD]] : vector<1x64xi1>, vector<1x64xf32>
// CHECK-NEXT: %[[CAST:.+]] = vector.shape_cast %[[SELECT]] : vector<1x64xf32> to vector<8x1x8xf32>
// CHECK-NEXT: return %[[CAST]] : vector<8x1x8xf32>

// -----

#flatten = affine_map<(d0) -> (d0 * 8)>

// The read can observe old tensor elements if it activates more rows.
func.func @keep_unproven_mask(%dst: tensor<1x?xf32>, %v: vector<1x64xf32>,
                             %write_rows: index, %read_rows: index) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 7.0 : f32
  %cols = affine.apply #flatten(%write_rows)
  %wm = vector.create_mask %c1, %cols : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, %read_rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %rm = vector.create_mask %c1, %read_rows, %c8 : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true]} : tensor<1x?x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_unproven_mask
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: vector.transfer_read %[[E]]

// -----

#flatten = affine_map<(d0) -> (d0 * 8)>

// The caller-provided row count is independent of the reshape. Scaling it can
// overflow the index type even when both transfers stay within the tensor.
func.func @keep_unbounded_mask_rows(%dst: tensor<1x64xf32>, %v: vector<1x64xf32>,
                                    %rows: index) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 7.0 : f32
  %cols = affine.apply #flatten(%rows)
  %wm = vector.create_mask %c1, %cols : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, 8, 8] : tensor<1x64xf32> into tensor<1x8x8xf32>
  %rm = vector.create_mask %c1, %rows, %c8 : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true]} : tensor<1x8x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_unbounded_mask_rows(
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: %[[R:.+]] = vector.transfer_read %[[E]]
// CHECK-NEXT: return %[[R]] : vector<1x8x8xf32>

// -----

#scale4 = affine_map<(d0) -> (d0 * 4)>

// A write bound scaled by a different factor does not define the read prefix.
func.func @keep_wrong_scale(%dst: tensor<1x?xf32>, %v: vector<1x64xf32>,
                            %rows: index) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0.0 : f32
  %cols = affine.apply #scale4(%rows)
  %wm = vector.create_mask %c1, %cols : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %i = tensor.insert_slice %w into %dst[0, 0] [1, %cols] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
  %e = tensor.expand_shape %i [[0], [1, 2]] output_shape [1, %rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %rm = vector.create_mask %c1, %rows, %c8 : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true]} : tensor<1x?x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_wrong_scale
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[I:.+]] = tensor.insert_slice %[[W]]
// CHECK: %[[E:.+]] = tensor.expand_shape %[[I]]
// CHECK: vector.transfer_read %[[E]]

// -----

#flatten = affine_map<(d0) -> (d0 * 8)>

// An unmasked read would consume lanes the write never defined, so the pair
// must stay when only the write is masked.
func.func @keep_single_side_mask(%dst: tensor<1x?xf32>, %v: vector<1x64xf32>,
                                 %rows: index) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.0 : f32
  %cols = affine.apply #flatten(%rows)
  %wm = vector.create_mask %c1, %cols : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, %rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} : tensor<1x?x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_single_side_mask
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: vector.transfer_read %[[E]]

// -----

#transpose = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// Permuting both non-unit axes needs a data permutation, not a shape cast.
func.func @keep_nonunit_transpose(%dst: tensor<1x64xf32>,
                                  %v: vector<1x64xf32>) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %w = vector.transfer_write %v, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, 8, 8] : tensor<1x64xf32> into tensor<1x8x8xf32>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad {in_bounds = [true, true, true], permutation_map = #transpose} : tensor<1x8x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_nonunit_transpose
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: vector.transfer_read %[[E]]

// -----

// The insert updates the second half, while the read observes the first.
func.func @keep_partial_insert(%tile: tensor<1x64xf32>, %dst: tensor<1x128xf32>,
                               %v: vector<1x64xf32>) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %w = vector.transfer_write %v, %tile[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
  %i = tensor.insert_slice %w into %dst[0, 64] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<1x128xf32>
  %e = tensor.expand_shape %i [[0], [1, 2]] output_shape [1, 16, 8] : tensor<1x128xf32> into tensor<1x16x8xf32>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} : tensor<1x16x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_partial_insert
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[I:.+]] = tensor.insert_slice %[[W]]
// CHECK: %[[E:.+]] = tensor.expand_shape %[[I]]
// CHECK: vector.transfer_read %[[E]]

// -----

#flatten = affine_map<(d0) -> (d0 * 8)>

// The insert covers fewer lanes than the write defined, so inactive lanes keep
// old tensor values.
func.func @keep_short_insert(%dst: tensor<1x?xf32>, %v: vector<1x64xf32>,
                             %rows: index) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0.0 : f32
  %cols = affine.apply #flatten(%rows)
  %wm = vector.create_mask %c1, %cols : vector<1x64xi1>
  %w = vector.transfer_write %v, %dst[%c0, %c0], %wm {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %i = tensor.insert_slice %w into %dst[0, 0] [1, %c4] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
  %e = tensor.expand_shape %i [[0], [1, 2]] output_shape [1, %rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %rm = vector.create_mask %c1, %rows, %c8 : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true]} : tensor<1x?x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_short_insert
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[I:.+]] = tensor.insert_slice %[[W]]
// CHECK: %[[E:.+]] = tensor.expand_shape %[[I]]
// CHECK: vector.transfer_read %[[E]]

// -----

// A write mask supplied by the caller can leave lanes untouched inside the
// first four rows that the read observes.
func.func @keep_opaque_mask(%dst: tensor<1x64xf32>, %v: vector<1x64xf32>,
                           %mask: vector<1x64xi1>) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 7.0 : f32
  %w = vector.transfer_write %v, %dst[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, 8, 8] : tensor<1x64xf32> into tensor<1x8x8xf32>
  %rm = vector.constant_mask [1, 4, 8] : vector<1x8x8xi1>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad, %rm {in_bounds = [true, true, true]} : tensor<1x8x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_opaque_mask(
// CHECK-SAME: %[[MASK:[^,:]+]]: vector<1x64xi1>
// CHECK: %[[W:.+]] = vector.transfer_write {{.*}}, %[[MASK]] {
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: %[[R:.+]] = vector.transfer_read %[[E]]
// CHECK-NEXT: return %[[R]] : vector<1x8x8xf32>

// -----

// Equal vector element counts are not sufficient. After expanding to 8x12,
// the read of 8x8 has holes between rows and is not a contiguous prefix.
func.func @keep_partial_inner_dim(%dst: tensor<1x96xf32>,
                                  %v: vector<1x64xf32>) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %w = vector.transfer_write %v, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x96xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, 8, 12] : tensor<1x96xf32> into tensor<1x8x12xf32>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} : tensor<1x8x12xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_partial_inner_dim
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: vector.transfer_read %[[E]]

// -----

// A potentially out-of-bounds read must keep its transfer. The dynamic middle
// dimension prevents the canonicalizer from proving the read in-bounds.
func.func @keep_oob_read(%dst: tensor<1x?xf32>, %v: vector<1x64xf32>,
                         %rows: index) -> vector<1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %w = vector.transfer_write %v, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
  %e = tensor.expand_shape %w [[0], [1, 2]] output_shape [1, %rows, 8] : tensor<1x?xf32> into tensor<1x?x8xf32>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad {in_bounds = [true, false, true]} : tensor<1x?x8xf32>, vector<1x8x8xf32>
  return %r : vector<1x8x8xf32>
}

// CHECK-LABEL: func.func @keep_oob_read
// CHECK: %[[W:.+]] = vector.transfer_write
// CHECK: %[[E:.+]] = tensor.expand_shape %[[W]]
// CHECK: vector.transfer_read %[[E]]
