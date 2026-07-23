// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// Test that memref.load on a function argument is lowered to tt.load via
// unrealized_conversion_cast + tt.addptr, and that the cast is eliminated
// after FuncOpPattern + reconcile-unrealized-casts.
// This reproduces the mix-mode SIMT VF scenario where a scalar tensor
// element access (e.g. cu_seqlens_q[pid]) is lowered by triton-ascend's
// TritonToLinalgPass to memref.load on a memref function argument.

module attributes {hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: tt.func @memref_load_on_func_arg
  // CHECK-SAME: %arg0: !tt.ptr<i32>
  // CHECK: %[[ADDR:.*]] = tt.addptr %arg0, %{{.*}} : !tt.ptr<i32>, i64
  // CHECK: tt.load %[[ADDR]] : !tt.ptr<i32>
  // CHECK-NOT: memref.load
  // CHECK-NOT: unrealized_conversion_cast
  func.func @memref_load_on_func_arg(%arg0: memref<?xi32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %c0 = arith.constant 0 : index
    %val = memref.load %arg0[%c0] : memref<?xi32, #hivm.address_space<gm>>
    return
  }
}

// -----

// Test that memref.load with a non-zero index is correctly lowered with
// index * stride offset computation.

module attributes {hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: tt.func @memref_load_with_index
  // CHECK-SAME: %arg0: !tt.ptr<i32>
  // CHECK: arith.muli
  // CHECK: %[[ADDR:.*]] = tt.addptr %arg0, %{{.*}} : !tt.ptr<i32>, i64
  // CHECK: tt.load %[[ADDR]] : !tt.ptr<i32>
  // CHECK-NOT: memref.load
  func.func @memref_load_with_index(%arg0: memref<4xi32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %c2 = arith.constant 2 : index
    %val = memref.load %arg0[%c2] : memref<4xi32, #hivm.address_space<gm>>
    return
  }
}
