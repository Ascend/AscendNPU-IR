// RUN: bishengir-opt %s -hivm-split-mixed-if-conditionals -hivm-mark-tightly-coupled-buffer -hivm-hoist-tightly-coupled-alloc -hivm-split-mix-kernel -split-input-file | FileCheck %s
//
// New SplitMixKernel cases that must run on published builds.
// The original split-mix-kernel.mlir is skipped in published builds.
// Add later cases here as extra split-input sections.

// Preload scopes are kept as skeletons on both split sides. Their scalar
// results are loop counters that the enclosing control flow still reads.
// Stubbing them to 0 made AIC/AIV disagree on trip counts and deadlock.
// CHECK-LABEL: func.func @preload_scope_keeps_scalar_counter_mix_aic(
// CHECK:         %[[FOR:.*]]:2 = scf.for
// CHECK:           %[[CUBE:.*]] = scope.scope : () -> i32
// CHECK:             arith.addi
// CHECK:             scope.return
// CHECK:           %[[VEC:.*]] = scope.scope : () -> i32
// CHECK:           scf.yield %[[CUBE]], %[[VEC]]
// CHECK:         return %[[FOR]]#0
// CHECK-LABEL: func.func @preload_scope_keeps_scalar_counter_mix_aiv(
// CHECK:         %[[FOR:.*]]:2 = scf.for
// CHECK:           %[[CUBE:.*]] = scope.scope : () -> i32
// CHECK:             arith.addi
// CHECK:             scope.return
// CHECK:           %[[VEC:.*]] = scope.scope : () -> i32
// CHECK:           scf.yield %[[CUBE]], %[[VEC]]
// CHECK:         return %[[FOR]]#0
module {
  func.func @preload_scope_keeps_scalar_counter(%init: i32) -> i32
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %step = arith.constant 32 : i32
    %0:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%cube = %init, %vec = %init) -> (i32, i32) : i32 {
      %cube_out = scope.scope : () -> i32 {
        %add = arith.addi %cube, %step : i32
        scope.return %add : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      %vec_out = scope.scope : () -> i32 {
        %add = arith.addi %vec, %step : i32
        scope.return %add : i32
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      scf.yield %cube_out, %vec_out : i32, i32
    }
    return %0#0 : i32
  }
}

// -----

// Preload cube scope returns a tightly coupled UB tensor consumed by vector.
// AIV must keep the scope result; stubbing with tensor.empty is illegal.
// CHECK-LABEL: func.func @scope_tcb_result_mix_aic(
// CHECK: %[[CUBE:.*]] = scope.scope : () -> tensor<64x32xf32>
// CHECK: tightly_coupled_buffer = {{.*}}tightly_coupled_buffer<0>
// CHECK: scope.return
// CHECK-LABEL: func.func @scope_tcb_result_mix_aiv(
// CHECK: %[[CUBE:.*]] = scope.scope : () -> tensor<64x32xf32>
// CHECK: tightly_coupled_buffer = {{.*}}tightly_coupled_buffer<0>
// CHECK: scope.return %[[TCB:.*]] : tensor<64x32xf32>
// CHECK: hivm.hir.vadd ins(%{{.*}}, %[[CUBE]] : tensor<64x32xf32>, tensor<64x32xf32>)
// CHECK-NOT: tensor.empty() : tensor<64x32xf32>
module {
  func.func @scope_tcb_result(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>)
      -> tensor<64x32xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x32xf32>
    %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<64x32xf32>) : i32 {
      %cube = scope.scope : () -> tensor<64x32xf32> {
        %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
        %mm = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<CUBE>} ins(%cst : f32) outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>
        scope.return %t : tensor<64x32xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      %vec = scope.scope : () -> tensor<64x32xf32> {
        %add = hivm.hir.vadd ins(%arg0, %cube : tensor<64x32xf32>, tensor<64x32xf32>) outs(%arg1 : tensor<64x32xf32>) -> tensor<64x32xf32>
        scope.return %add : tensor<64x32xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
      scf.yield %vec : tensor<64x32xf32>
    }
    return %0 : tensor<64x32xf32>
  }
}
