// REQUIRES: regbase
// RUN: bishengir-opt -hivm-insert-fixpipe -hivm-inline-fixpipe %s -split-input-file -verify-diagnostics | FileCheck %s

// On reg-based A5 targets, mmadL1->mmadL1 inserts fixpipe directly to L1
// (fractal NZ shape) instead of the local NZ2ND path.

// -----

// CHECK-LABEL: func.func @dotdot_f32
// CHECK: %[[ARG0:.*]] = hivm.hir.fixpipe {channel_split = true} ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<2x1x16x8xf32>) -> tensor<2x1x16x8xf32>
// CHECK: %[[ARG1:.*]] = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%[[ARG0]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dotdot_f32(%4: tensor<16x16xf32>, %e4: tensor<16x16xf32>, %e5: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %7 = tensor.empty() : tensor<16x16xf32>
    %8 = hivm.hir.mmadL1 ins(%4, %e4, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %9 = tensor.empty() : tensor<16x16xf32>
    %10 = hivm.hir.mmadL1 ins(%8, %e5, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%9 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %10 : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @dotdot_f16f32
// CHECK: %[[ARG0:.*]] = hivm.hir.fixpipe {channel_split = true} ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<2x1x16x8xf32>) -> tensor<2x1x16x8xf32>
// CHECK: %[[ARG1:.*]] = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%[[ARG0]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dotdot_f16f32(%4: tensor<16x16xf16>, %e4: tensor<16x16xf16>, %e5: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %7 = tensor.empty() : tensor<16x16xf32>
    %8 = hivm.hir.mmadL1 ins(%4, %e4, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %9 = tensor.empty() : tensor<16x16xf32>
    %10 = hivm.hir.mmadL1 ins(%8, %e5, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%9 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %10 : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @dotdot_f16f16
// CHECK: %[[ARG0:.*]] = hivm.hir.fixpipe ins(%{{.*}} : tensor<16x16xf16>) outs(%{{.*}} : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK: %[[ARG1:.*]] = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%[[ARG0]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dotdot_f16f16(%4: tensor<16x16xf16>, %e4: tensor<16x16xf16>, %e5: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %7 = tensor.empty() : tensor<16x16xf16>
    %8 = hivm.hir.mmadL1 ins(%4, %e4, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%7 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %9 = tensor.empty() : tensor<16x16xf16>
    %10 = hivm.hir.mmadL1 ins(%8, %e5, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%9 : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %10 : tensor<16x16xf16>
  }
}

// -----

// Vector consumer stays on the local NZ2ND path even on A5.
// CHECK-LABEL: func.func @dot_abs
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: hivm.hir.vabs
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dot_abs(%4: tensor<16x16xf16>, %e4: tensor<16x16xf16>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %7 = tensor.empty() : tensor<16x16xf32>
    %8 = hivm.hir.mmadL1 ins(%4, %e4, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %9 = tensor.empty() : tensor<16x16xf32>
    %10 = hivm.hir.vabs ins(%8 : tensor<16x16xf32>) outs(%9 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %10 : tensor<16x16xf32>
  }
}
