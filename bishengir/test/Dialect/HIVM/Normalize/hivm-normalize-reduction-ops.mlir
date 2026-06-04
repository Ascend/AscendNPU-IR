// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

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
