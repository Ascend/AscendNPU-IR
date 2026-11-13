// RUN: bishengir-opt %s -guard-cv-pipeline-ub="simt-vf-dynamic-size=1" -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @overflow_inlines_scopes
// CHECK: hivm.hir.vadd
// CHECK: return
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.preload_num
// CHECK-NOT: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @overflow_inlines_scopes(%in: tensor<64x128xf32>) -> tensor<64x128xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %in) -> (tensor<64x128xf32>) {
      %produced = scope.scope : () -> tensor<64x128xf32> {
        scope.return %acc : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 2 : i32, no_inline}
      %consumed = scope.scope : () -> tensor<64x128xf32> {
        %empty = tensor.empty() : tensor<64x128xf32>
        %added = hivm.hir.vadd ins(%produced, %produced : tensor<64x128xf32>, tensor<64x128xf32>) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
        scope.return %added : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      scf.yield %consumed : tensor<64x128xf32>
    } {hivm.cv_pipelined_loop}
    return %out : tensor<64x128xf32>
  }
}

// -----

// CHECK-LABEL: func.func @fits_keeps_scopes
// CHECK: scope.scope
// CHECK: hivm.preload_num
// CHECK: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @fits_keeps_scopes(%in: tensor<4xf32>) -> tensor<4xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %in) -> (tensor<4xf32>) {
      %produced = scope.scope : () -> tensor<4xf32> {
        scope.return %acc : tensor<4xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 2 : i32, no_inline}
      %consumed = scope.scope : () -> tensor<4xf32> {
        scope.return %produced : tensor<4xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      scf.yield %consumed : tensor<4xf32>
    } {hivm.cv_pipelined_loop}
    return %out : tensor<4xf32>
  }
}

// -----

// CHECK-LABEL: func.func @overflow_restores_snapshot(
// CHECK-NOT: hivm.hir.vadd
// CHECK: arith.addi
// CHECK: return
// CHECK: func.func private @overflow_restores_snapshot.cv_pipeline_snapshot
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @overflow_restores_snapshot(%in: tensor<64x128xf32>) -> tensor<64x128xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %in) -> (tensor<64x128xf32>) {
      %produced = scope.scope : () -> tensor<64x128xf32> {
        scope.return %acc : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 2 : i32, no_inline}
      %consumed = scope.scope : () -> tensor<64x128xf32> {
        %empty = tensor.empty() : tensor<64x128xf32>
        %added = hivm.hir.vadd ins(%produced, %produced : tensor<64x128xf32>, tensor<64x128xf32>) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
        scope.return %added : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      scf.yield %consumed : tensor<64x128xf32>
    } {hivm.cv_pipelined_loop}
    return %out : tensor<64x128xf32>
  }
  func.func private @overflow_restores_snapshot.cv_pipeline_snapshot(%in: tensor<64x128xf32>) -> tensor<64x128xf32>
      attributes {hivm.cv_pipeline_snapshot = @overflow_restores_snapshot, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %in) -> (tensor<64x128xf32>) {
      %c = arith.constant 1 : index
      %added = arith.addi %c, %c : index
      scf.yield %acc : tensor<64x128xf32>
    }
    return %out : tensor<64x128xf32>
  }
}
