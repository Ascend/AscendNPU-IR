// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_ReduceWithIndexRAHighPerformance_reduce_dim_0(
// CHECK-SAME: %[[SRC:.*]]: tensor<64x32xf32>, %[[DSTV:.*]]: tensor<32xf32>, %[[DSTI:.*]]: tensor<32xi32>) -> tensor<32xi32> {
// CHECK: %[[SRC_EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
// CHECK: %[[SRC_T:.*]] = hivm.hir.vtranspose ins(%[[SRC]] : tensor<64x32xf32>) outs(%[[SRC_EMPTY]] : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
// CHECK: %[[DSTV_EMPTY:.*]] = tensor.empty() : tensor<32x1xf32>
// CHECK: %[[DSTI_EMPTY:.*]] = tensor.empty() : tensor<32x1xi32>
// CHECK: %[[REDUCE:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%[[SRC_T]] : tensor<32x64xf32>) outs(%[[DSTV_EMPTY]], %[[DSTI_EMPTY]] : tensor<32x1xf32>, tensor<32x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<32x1xf32>, tensor<32x1xi32>
// CHECK: %[[OUTI_EMPTY:.*]] = tensor.empty() : tensor<1x32xi32>
// CHECK: %[[OUTI:.*]] = hivm.hir.vtranspose ins(%[[REDUCE]]#1 : tensor<32x1xi32>) outs(%[[OUTI_EMPTY]] : tensor<1x32xi32>) permutation = [1, 0] -> tensor<1x32xi32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[OUTI]] {{\[\[0, 1\]\]}} : tensor<1x32xi32> into tensor<32xi32>
// CHECK: return %[[COLLAPSED]] : tensor<32xi32>
func.func @test_ReduceWithIndexRAHighPerformance_reduce_dim_0(
    %src : tensor<64x32xf32>, %dstv : tensor<32xf32>,
    %dsti : tensor<32xi32>) -> tensor<32xi32> {
  %dstv_2d = tensor.expand_shape %dstv [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
  %dsti_2d = tensor.expand_shape %dsti [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
  %0:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%src : tensor<64x32xf32>)
      outs(%dstv_2d, %dsti_2d : tensor<1x32xf32>, tensor<1x32xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [0] -> tensor<1x32xf32>, tensor<1x32xi32>
  %1 = tensor.collapse_shape %0#1 [[0, 1]] : tensor<1x32xi32> into tensor<32xi32>
  return %1 : tensor<32xi32>
}

// -----

// CHECK-LABEL: func.func @test_ReduceWithIndexRAHighPerformance_argmax_preserves_nan_mask(
// CHECK-SAME: %[[SRC:.*]]: tensor<64x32xf32>, %[[DSTV:.*]]: tensor<32xf32>, %[[DSTI:.*]]: tensor<32xi32>) -> (tensor<32xf32>, tensor<32xi32>) {
// CHECK: %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[SRC_EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
// CHECK: %[[SRC_T:.*]] = hivm.hir.vtranspose ins(%[[SRC]] : tensor<64x32xf32>) outs(%[[SRC_EMPTY]] : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
// CHECK: %[[EMPTY_MASK:.*]] = tensor.empty() : tensor<32x64xi1>
// CHECK: %[[MASK:.*]] = hivm.hir.vnot
// CHECK: %[[EMPTY_SEL:.*]] = tensor.empty() : tensor<32x64xf32>
// CHECK: %[[SEL:.*]] = hivm.hir.vsel ins(%[[MASK]], %[[NEG_INF]], %[[SRC_T]] : tensor<32x64xi1>, f32, tensor<32x64xf32>) outs(%[[EMPTY_SEL]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK: %[[DSTV_EMPTY:.*]] = tensor.empty() : tensor<32x1xf32>
// CHECK: %[[DSTI_EMPTY:.*]] = tensor.empty() : tensor<32x1xi32>
// CHECK: %[[REDUCE:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%[[SEL]] : tensor<32x64xf32>) outs(%[[DSTV_EMPTY]], %[[DSTI_EMPTY]] : tensor<32x1xf32>, tensor<32x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<32x1xf32>, tensor<32x1xi32>
// CHECK: %[[OUTV_EMPTY:.*]] = tensor.empty() : tensor<1x32xf32>
// CHECK: %[[OUTV:.*]] = hivm.hir.vtranspose ins(%[[REDUCE]]#0 : tensor<32x1xf32>) outs(%[[OUTV_EMPTY]] : tensor<1x32xf32>) permutation = [1, 0] -> tensor<1x32xf32>
// CHECK: %[[OUTI_EMPTY:.*]] = tensor.empty() : tensor<1x32xi32>
// CHECK: %[[OUTI:.*]] = hivm.hir.vtranspose ins(%[[REDUCE]]#1 : tensor<32x1xi32>) outs(%[[OUTI_EMPTY]] : tensor<1x32xi32>) permutation = [1, 0] -> tensor<1x32xi32>
// CHECK: %[[COLLAPSED_V:.*]] = tensor.collapse_shape %[[OUTV]] {{\[\[0, 1\]\]}} : tensor<1x32xf32> into tensor<32xf32>
// CHECK: %[[COLLAPSED_I:.*]] = tensor.collapse_shape %[[OUTI]] {{\[\[0, 1\]\]}} : tensor<1x32xi32> into tensor<32xi32>
// CHECK: return %[[COLLAPSED_V]], %[[COLLAPSED_I]] : tensor<32xf32>, tensor<32xi32>
func.func @test_ReduceWithIndexRAHighPerformance_argmax_preserves_nan_mask(
    %src : tensor<64x32xf32>, %dstv : tensor<32xf32>,
    %dsti : tensor<32xi32>) -> (tensor<32xf32>, tensor<32xi32>) {
  %dstv_2d = tensor.expand_shape %dstv [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
  %dsti_2d = tensor.expand_shape %dsti [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
  %0:2 = hivm.hir.vreduce <max_with_index> ins(%src : tensor<64x32xf32>)
      outs(%dstv_2d, %dsti_2d : tensor<1x32xf32>, tensor<1x32xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [0] -> tensor<1x32xf32>, tensor<1x32xi32>
  %1 = tensor.collapse_shape %0#0 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
  %2 = tensor.collapse_shape %0#1 [[0, 1]] : tensor<1x32xi32> into tensor<32xi32>
  return %1, %2 : tensor<32xf32>, tensor<32xi32>
}

// -----

// CHECK-LABEL: func.func @test_ReduceWithIndexRAHighPerformance_reduce_dim_1_rank3(
// CHECK-SAME: %[[SRC:.*]]: tensor<32x32x128xf32>, %[[DSTV:.*]]: tensor<32x128xf32>, %[[DSTI:.*]]: tensor<32x128xi32>) -> tensor<32x128xi32> {
// CHECK: %[[SRC_EMPTY:.*]] = tensor.empty() : tensor<32x128x32xf32>
// CHECK: %[[SRC_T:.*]] = hivm.hir.vtranspose ins(%[[SRC]] : tensor<32x32x128xf32>) outs(%[[SRC_EMPTY]] : tensor<32x128x32xf32>) permutation = [0, 2, 1] -> tensor<32x128x32xf32>
// CHECK: %[[DSTV_EMPTY:.*]] = tensor.empty() : tensor<32x128x1xf32>
// CHECK: %[[DSTI_EMPTY:.*]] = tensor.empty() : tensor<32x128x1xi32>
// CHECK: %[[REDUCE:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%[[SRC_T]] : tensor<32x128x32xf32>) outs(%[[DSTV_EMPTY]], %[[DSTI_EMPTY]] : tensor<32x128x1xf32>, tensor<32x128x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [2] -> tensor<32x128x1xf32>, tensor<32x128x1xi32>
// CHECK: %[[OUTI_EMPTY:.*]] = tensor.empty() : tensor<32x1x128xi32>
// CHECK: %[[OUTI:.*]] = hivm.hir.vtranspose ins(%[[REDUCE]]#1 : tensor<32x128x1xi32>) outs(%[[OUTI_EMPTY]] : tensor<32x1x128xi32>) permutation = [0, 2, 1] -> tensor<32x1x128xi32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[OUTI]] {{\[\[0\], \[1, 2\]\]}} : tensor<32x1x128xi32> into tensor<32x128xi32>
// CHECK: return %[[COLLAPSED]] : tensor<32x128xi32>
func.func @test_ReduceWithIndexRAHighPerformance_reduce_dim_1_rank3(
    %src : tensor<32x32x128xf32>, %dstv : tensor<32x128xf32>,
    %dsti : tensor<32x128xi32>) -> tensor<32x128xi32> {
  %dstv_3d = tensor.expand_shape %dstv [[0], [1, 2]] output_shape [32, 1, 128] : tensor<32x128xf32> into tensor<32x1x128xf32>
  %dsti_3d = tensor.expand_shape %dsti [[0], [1, 2]] output_shape [32, 1, 128] : tensor<32x128xi32> into tensor<32x1x128xi32>
  %0:2 = hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%src : tensor<32x32x128xf32>)
      outs(%dstv_3d, %dsti_3d : tensor<32x1x128xf32>, tensor<32x1x128xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<32x1x128xf32>, tensor<32x1x128xi32>
  %1 = tensor.collapse_shape %0#1 [[0], [1, 2]] : tensor<32x1x128xi32> into tensor<32x128xi32>
  return %1 : tensor<32x128xi32>
}

// -----

// CHECK-LABEL: func.func @test_ReduceWithIndexRAHighPerformance_last_axis_no_transpose(
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%{{.*}} : tensor<32x128x32xf32>) outs(%{{.*}}, %{{.*}} : tensor<32x128x1xf32>, tensor<32x128x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [2] -> tensor<32x128x1xf32>, tensor<32x128x1xi32>
func.func @test_ReduceWithIndexRAHighPerformance_last_axis_no_transpose(
    %src : tensor<32x128x32xf32>, %dstv : tensor<32x128x1xf32>,
    %dsti : tensor<32x128x1xi32>) -> tensor<32x128x1xi32> {
  %0:2 = hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%src : tensor<32x128x32xf32>)
      outs(%dstv, %dsti : tensor<32x128x1xf32>, tensor<32x128x1xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [2] -> tensor<32x128x1xf32>, tensor<32x128x1xi32>
  return %0#1 : tensor<32x128x1xi32>
}

// -----

// CHECK-LABEL: func.func @test_ReduceWithIndexRAHighPerformance_incompatible_shape_no_transpose(
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%{{.*}} : tensor<7x5x3xf32>) outs(%{{.*}}, %{{.*}} : tensor<7x1x3xf32>, tensor<7x1x3xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<7x1x3xf32>, tensor<7x1x3xi32>
func.func @test_ReduceWithIndexRAHighPerformance_incompatible_shape_no_transpose(
    %src : tensor<7x5x3xf32>, %dstv : tensor<7x1x3xf32>,
    %dsti : tensor<7x1x3xi32>) -> tensor<7x1x3xi32> {
  %0:2 = hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%src : tensor<7x5x3xf32>)
      outs(%dstv, %dsti : tensor<7x1x3xf32>, tensor<7x1x3xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<7x1x3xf32>, tensor<7x1x3xi32>
  return %0#1 : tensor<7x1x3xi32>
}

// -----

// CHECK-LABEL: func.func @test_ReduceWithIndexRAHighPerformance_unsupported_permutation_no_transpose(
// CHECK: %[[SRC_EMPTY1:.*]] = tensor.empty() : tensor<16x32x64x8xf32>
// CHECK: %[[SRC_EMPTY0:.*]] = tensor.empty() : tensor<16x32x8x64xf32>
// CHECK: %[[SRC_T0:.*]] = hivm.hir.vtranspose ins(%[[SRC:.*]] : tensor<16x8x32x64xf32>) outs(%[[SRC_EMPTY0]] : tensor<16x32x8x64xf32>) permutation = [0, 2, 1, 3] -> tensor<16x32x8x64xf32>
// CHECK: %[[SRC_T1:.*]] = hivm.hir.vtranspose ins(%[[SRC_T0]] : tensor<16x32x8x64xf32>) outs(%[[SRC_EMPTY1]] : tensor<16x32x64x8xf32>) permutation = [0, 1, 3, 2] -> tensor<16x32x64x8xf32>
// CHECK: %[[DSTV_EMPTY:.*]] = tensor.empty() : tensor<16x32x64x1xf32>
// CHECK: %[[DSTI_EMPTY:.*]] = tensor.empty() : tensor<16x32x64x1xi32>
// CHECK: %[[REDUCE:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%[[SRC_T1]] : tensor<16x32x64x8xf32>) outs(%[[DSTV_EMPTY]], %[[DSTI_EMPTY]] : tensor<16x32x64x1xf32>, tensor<16x32x64x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [3] -> tensor<16x32x64x1xf32>, tensor<16x32x64x1xi32>
// CHECK: %[[OUTI_EMPTY1:.*]] = tensor.empty() : tensor<16x1x32x64xi32>
// CHECK: %[[OUTI_EMPTY0:.*]] = tensor.empty() : tensor<16x32x1x64xi32>
// CHECK: %[[OUTI_T0:.*]] = hivm.hir.vtranspose ins(%[[REDUCE]]#1 : tensor<16x32x64x1xi32>) outs(%[[OUTI_EMPTY0]] : tensor<16x32x1x64xi32>) permutation = [0, 1, 3, 2] -> tensor<16x32x1x64xi32>
// CHECK: %[[OUTI_T1:.*]] = hivm.hir.vtranspose ins(%[[OUTI_T0]] : tensor<16x32x1x64xi32>) outs(%[[OUTI_EMPTY1]] : tensor<16x1x32x64xi32>) permutation = [0, 2, 1, 3] -> tensor<16x1x32x64xi32>
// CHECK: return %[[OUTI_T1]] : tensor<16x1x32x64xi32>
func.func @test_ReduceWithIndexRAHighPerformance_unsupported_permutation_no_transpose(
    %src : tensor<16x8x32x64xf32>, %dstv : tensor<16x1x32x64xf32>,
    %dsti : tensor<16x1x32x64xi32>) -> tensor<16x1x32x64xi32> {
  %0:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%src : tensor<16x8x32x64xf32>)
      outs(%dstv, %dsti : tensor<16x1x32x64xf32>, tensor<16x1x32x64xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<16x1x32x64xf32>, tensor<16x1x32x64xi32>
  return %0#1 : tensor<16x1x32x64xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeArgMax_hivm_vreduce_nan_mask(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>, %[[INITV:.*]]: tensor<4x1xf32>, %[[INITI:.*]]: tensor<4x1xi32>) -> (tensor<4x1xf32>, tensor<4x1xi32>) {
// CHECK: %[[INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[EMPTY_MASK:.*]] = tensor.empty() : tensor<4x8xi1>
// CHECK: %[[MASK:.*]] = hivm.hir.vnot
// CHECK: %[[EMPTY_SEL:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[SEL:.*]] = hivm.hir.vsel ins(%[[MASK]], %[[INF]], %[[SRC]] : tensor<4x8xi1>, f32, tensor<4x8xf32>) outs(%[[EMPTY_SEL]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[EMPTY_INITV:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK: %[[EMPTY_INITI:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK: %[[REDUCE:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%[[SEL]] : tensor<4x8xf32>) outs(%[[EMPTY_INITV]], %[[EMPTY_INITI]] : tensor<4x1xf32>, tensor<4x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<4x1xf32>, tensor<4x1xi32>
// CHECK: return %[[REDUCE]]#0, %[[REDUCE]]#1 : tensor<4x1xf32>, tensor<4x1xi32>
func.func @test_NormalizeArgMax_hivm_vreduce_nan_mask(
    %src : tensor<4x8xf32>, %initv : tensor<4x1xf32>,
    %initi : tensor<4x1xi32>) -> (tensor<4x1xf32>, tensor<4x1xi32>) {
  %0:2 = hivm.hir.vreduce <max_with_index> ins(%src : tensor<4x8xf32>)
      outs(%initv, %initi : tensor<4x1xf32>, tensor<4x1xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<4x1xf32>, tensor<4x1xi32>
  return %0#0, %0#1 : tensor<4x1xf32>, tensor<4x1xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeArgMin_hivm_vreduce_nan_mask(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>, %[[INITV:.*]]: tensor<4x1xf32>, %[[INITI:.*]]: tensor<4x1xi32>) -> tensor<4x1xf32> {
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[EMPTY_MASK:.*]] = tensor.empty() : tensor<4x8xi1>
// CHECK: %[[MASK:.*]] = hivm.hir.vnot
// CHECK: %[[EMPTY_SEL:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[SEL:.*]] = hivm.hir.vsel ins(%[[MASK]], %[[INF]], %[[SRC]] : tensor<4x8xi1>, f32, tensor<4x8xf32>) outs(%[[EMPTY_SEL]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[EMPTY_INITV:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK: %[[EMPTY_INITI:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK: %[[REDUCE:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%[[SEL]] : tensor<4x8xf32>) outs(%[[EMPTY_INITV]], %[[EMPTY_INITI]] : tensor<4x1xf32>, tensor<4x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<4x1xf32>, tensor<4x1xi32>
// CHECK: return %[[REDUCE]]#0 : tensor<4x1xf32>
func.func @test_NormalizeArgMin_hivm_vreduce_nan_mask(
    %src : tensor<4x8xf32>, %initv : tensor<4x1xf32>,
    %initi : tensor<4x1xi32>) -> tensor<4x1xf32> {
  %0:2 = hivm.hir.vreduce <min_with_index> ins(%src : tensor<4x8xf32>)
      outs(%initv, %initi : tensor<4x1xf32>, tensor<4x1xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<4x1xf32>, tensor<4x1xi32>
  return %0#0 : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeArgMinMax_already_initialized_noop(
// CHECK-NOT: hivm.hir.vcmp
// CHECK-NOT: hivm.hir.vsel
// CHECK: %[[EMPTY_INITV:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK: %[[EMPTY_INITI:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK: hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%{{.*}} : tensor<4x8xf32>) outs(%[[EMPTY_INITV]], %[[EMPTY_INITI]] : tensor<4x1xf32>, tensor<4x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<4x1xf32>, tensor<4x1xi32>
func.func @test_NormalizeArgMinMax_already_initialized_noop(
    %src : tensor<4x8xf32>, %initv : tensor<4x1xf32>,
    %initi : tensor<4x1xi32>) -> tensor<4x1xf32> {
  %0:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index>
      ins(%src : tensor<4x8xf32>)
      outs(%initv, %initi : tensor<4x1xf32>, tensor<4x1xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<4x1xf32>, tensor<4x1xi32>
  return %0#0 : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeArgMinMax_integer_src_noop(
// CHECK-NOT: hivm.hir.vcmp
// CHECK-NOT: hivm.hir.vsel
// CHECK: hivm.hir.vreduce <min_with_index> ins(%{{.*}} : tensor<4x8xi32>) outs(%{{.*}}, %{{.*}} : tensor<4x1xi32>, tensor<4x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<4x1xi32>, tensor<4x1xi32>
func.func @test_NormalizeArgMinMax_integer_src_noop(
    %src : tensor<4x8xi32>, %initv : tensor<4x1xi32>,
    %initi : tensor<4x1xi32>) -> tensor<4x1xi32> {
  %0:2 = hivm.hir.vreduce <min_with_index> ins(%src : tensor<4x8xi32>)
      outs(%initv, %initi : tensor<4x1xi32>, tensor<4x1xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<4x1xi32>, tensor<4x1xi32>
  return %0#0 : tensor<4x1xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeF16ReduceSum_hivm_vreduce(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf16>, %[[INIT:.*]]: tensor<4x1xf16>) -> tensor<4x1xf16> {
// CHECK: %[[EMPTY_IN_F32:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[SRC]] : tensor<4x8xf16>) outs(%[[EMPTY_IN_F32]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[EMPTY_INIT_F32:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK: %[[CAST_INIT:.*]] = hivm.hir.vcast ins(%[[INIT]] : tensor<4x1xf16>) outs(%[[EMPTY_INIT_F32]] : tensor<4x1xf32>) -> tensor<4x1xf32>
// CHECK: %[[REDUCE_F32:.*]] = hivm.hir.vreduce <sum> ins(%[[CAST_IN]] : tensor<4x8xf32>) outs(%[[CAST_INIT]] : tensor<4x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf32>
// CHECK: %[[EMPTY_RES_F16:.*]] = tensor.empty() : tensor<4x1xf16>
// CHECK: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%[[REDUCE_F32]] : tensor<4x1xf32>) outs(%[[EMPTY_RES_F16]] : tensor<4x1xf16>) -> tensor<4x1xf16>
// CHECK: return %[[CAST_OUT]] : tensor<4x1xf16>
func.func @test_NormalizeF16ReduceSum_hivm_vreduce(
    %src : tensor<4x8xf16>, %init : tensor<4x1xf16>) -> tensor<4x1xf16> {
  %0 = hivm.hir.vreduce <sum> ins(%src : tensor<4x8xf16>)
      outs(%init : tensor<4x1xf16>)
      unsigned_src = false
      reduce_dims = [1] -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeF16ReduceSum_hivm_init_f16_only(
// CHECK-SAME: %[[SRC:.*]]: tensor<4x8xf32>, %[[INIT:.*]]: tensor<4x1xf16>) -> tensor<4x1xf16> {
// CHECK: %[[EMPTY_INIT_F32:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK: %[[CAST_INIT:.*]] = hivm.hir.vcast ins(%[[INIT]] : tensor<4x1xf16>) outs(%[[EMPTY_INIT_F32]] : tensor<4x1xf32>) -> tensor<4x1xf32>
// CHECK: %[[REDUCE_F32:.*]] = hivm.hir.vreduce <sum> ins(%[[SRC]] : tensor<4x8xf32>) outs(%[[CAST_INIT]] : tensor<4x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf32>
// CHECK: %[[EMPTY_RES_F16:.*]] = tensor.empty() : tensor<4x1xf16>
// CHECK: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%[[REDUCE_F32]] : tensor<4x1xf32>) outs(%[[EMPTY_RES_F16]] : tensor<4x1xf16>) -> tensor<4x1xf16>
// CHECK: return %[[CAST_OUT]] : tensor<4x1xf16>
func.func @test_NormalizeF16ReduceSum_hivm_init_f16_only(
    %src : tensor<4x8xf32>, %init : tensor<4x1xf16>) -> tensor<4x1xf16> {
  %0 = hivm.hir.vreduce <sum> ins(%src : tensor<4x8xf32>)
      outs(%init : tensor<4x1xf16>)
      unsigned_src = false
      reduce_dims = [1] -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeF16ReduceSum_hivm_non_sum_noop(
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vreduce <max> ins(%{{.*}} : tensor<4x8xf16>) outs(%{{.*}} : tensor<4x1xf16>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf16>
func.func @test_NormalizeF16ReduceSum_hivm_non_sum_noop(
    %src : tensor<4x8xf16>, %init : tensor<4x1xf16>) -> tensor<4x1xf16> {
  %0 = hivm.hir.vreduce <max> ins(%src : tensor<4x8xf16>)
      outs(%init : tensor<4x1xf16>)
      unsigned_src = false
      reduce_dims = [1] -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeF16ReduceSum_hivm_f32_noop(
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vreduce <sum> ins(%{{.*}} : tensor<4x8xf32>) outs(%{{.*}} : tensor<4x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf32>
func.func @test_NormalizeF16ReduceSum_hivm_f32_noop(
    %src : tensor<4x8xf32>, %init : tensor<4x1xf32>) -> tensor<4x1xf32> {
  %0 = hivm.hir.vreduce <sum> ins(%src : tensor<4x8xf32>)
      outs(%init : tensor<4x1xf32>)
      unsigned_src = false
      reduce_dims = [1] -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeReduceWithIndexInitsAndInputs_hivm_to_empty(
// CHECK: %[[OUTV:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK: %[[OUTI:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK: hivm.hir.vreduce <max_with_index> ins(%{{.*}} : tensor<4x8xi32>) outs(%[[OUTV]], %[[OUTI]] : tensor<4x1xi32>, tensor<4x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<4x1xi32>, tensor<4x1xi32>
func.func @test_NormalizeReduceWithIndexInitsAndInputs_hivm_to_empty(
    %src : tensor<4x8xi32>, %initv : tensor<4x1xi32>,
    %initi : tensor<4x1xi32>) -> (tensor<4x1xi32>, tensor<4x1xi32>) {
  %0:2 = hivm.hir.vreduce <max_with_index> ins(%src : tensor<4x8xi32>)
      outs(%initv, %initi : tensor<4x1xi32>, tensor<4x1xi32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1] -> tensor<4x1xi32>, tensor<4x1xi32>
  return %0#0, %0#1 : tensor<4x1xi32>, tensor<4x1xi32>
}
