// RUN: bishengir-opt -hivm-normalize-matmul %s -split-input-file -verify-diagnostics | FileCheck %s

// Tests that hfusion.disableHfusionVectorize no longer causes NormalizeMatmul
// to skip mmadL1. Mix mode now follows the same counter-based path as SIMD:
// counter creation, may_not_exec, normalized_in_L0C, and tail fallback.

// -----
// Dynamic bounds: loop may be empty (lb >= ub). NormalizeMatmul must create
// counter, set may_not_exec, and build a fallback_not_exec scf.if that
// yields a zero-filled tensor when the loop never ran.

// CHECK-LABEL: func.func @test_mix_dynamic_bounds_tail_fallback
// CHECK: memref.alloca() {normalize_matmul_counter
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {may_not_exec, normalized_in_L0C = [0 : i32]}
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: } {fallback_not_exec}
// CHECK: hivm.hir.vmul
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  func.func @test_mix_dynamic_bounds_tail_fallback(%lb: i32, %ub: i32) -> tensor<16x16xf32> {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %cst = arith.constant 0.0 : f32
    %cst1 = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %a = tensor.empty() : tensor<16x16xf16>
    %b = tensor.empty() : tensor<16x16xf16>
    %0 = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mmad : tensor<16x16xf32>
    }
    %ones = tensor.empty() : tensor<16x16xf32>
    %exp = hivm.hir.vbrc ins(%cst1 : f32) outs(%ones : tensor<16x16xf32>) -> tensor<16x16xf32>
    %1 = hivm.hir.vmul ins(%0, %exp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
}

// -----
// Constant bounds (lb < ub): loop is guaranteed to run. NormalizeMatmul still
// creates counter and sets normalized_in_L0C, but no may_not_exec and no
// fallback scf.if.

// CHECK-LABEL: func.func @test_mix_constant_bounds_no_fallback
// CHECK: memref.alloca() {normalize_matmul_counter
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {normalized_in_L0C = [0 : i32]}
// CHECK-NOT: may_not_exec
// CHECK-NOT: fallback_not_exec
// CHECK: hivm.hir.vmul
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  func.func @test_mix_constant_bounds_no_fallback() -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %cst = arith.constant 0.0 : f32
    %cst1 = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %a = tensor.empty() : tensor<16x16xf16>
    %b = tensor.empty() : tensor<16x16xf16>
    %0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mmad : tensor<16x16xf32>
    }
    %ones = tensor.empty() : tensor<16x16xf32>
    %exp = hivm.hir.vbrc ins(%cst1 : f32) outs(%ones : tensor<16x16xf32>) -> tensor<16x16xf32>
    %1 = hivm.hir.vmul ins(%0, %exp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
}

// -----
// Idempotent: a second NormalizeMatmul run should skip already-processed
// mmadL1 (kNormalizedInL0C attribute prevents double-processing).

// CHECK-LABEL: func.func @test_mix_idempotent
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-NOT: normalize_matmul_counter
// CHECK-NOT: scf.if
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  func.func @test_mix_idempotent(%lb: i32, %ub: i32) -> tensor<16x16xf32> {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<16x16xf32>
    %a = tensor.empty() : tensor<16x16xf16>
    %b = tensor.empty() : tensor<16x16xf16>
    %0 = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %empty) -> (tensor<16x16xf32>) : i32 {
      %mmad = hivm.hir.mmadL1 {normalized_in_L0C} ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mmad : tensor<16x16xf32>
    }
    return %0 : tensor<16x16xf32>
  }
}
