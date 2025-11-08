// RUN: bishengir-opt %s -hivm-lift-zero-rank -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_lift_zero_rank(
// CHECK: memref.expand_shape %0 {{.*}} output_shape {{\[}}1] : memref<f32, #hivm.address_space<ub>> into memref<1xf32, #hivm.address_space<ub>>
func.func @test_lift_zero_rank(%arg0: memref<1xf32, #hivm.address_space<gm>>, %arg1: memref<32xf32, #hivm.address_space<gm>>)
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0_i64 = arith.constant 0 : i64
  %collapse_shape = memref.collapse_shape %arg0 [] : memref<1xf32, #hivm.address_space<gm>> into memref<f32, #hivm.address_space<gm>>
  %0 = hivm.hir.pointer_cast(%c0_i64) : memref<f32, #hivm.address_space<ub>>
  %1 = hivm.hir.pointer_cast(%c0_i64) : memref<f32, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%0, %1 : memref<f32, #hivm.address_space<ub>>, memref<f32, #hivm.address_space<ub>>) outs(%1 : memref<f32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%collapse_shape : memref<f32, #hivm.address_space<gm>>) outs(%0 : memref<f32, #hivm.address_space<ub>>)
  return
}

// -----
// CHECK-LABEL: func.func @test_lift_zero_rank_foreach_add(
// CHECK: memref.expand_shape
// CHECK: memref.expand_shape
// CHECK: hivm.hir.load ins({{.*}} : memref<1xf32>) outs({{.*}} : memref<1xf32>)
// CHECK: memref.expand_shape
// CHECK: memref.expand_shape
// CHECK: hivm.hir.load ins({{.*}} : memref<1xf32>) outs({{.*}} : memref<1xf32>)
// CHECK: memref.expand_shape
// CHECK: memref.expand_shape
// CHECK: hivm.hir.store ins({{.*}} : memref<1xf32>) outs({{.*}} : memref<1xf32>)
// CHECK: memref.expand_shape
// CHECK: memref.expand_shape
// CHECK: hivm.hir.store ins({{.*}} : memref<1xf32>) outs({{.*}} : memref<1xf32>)
func.func @test_lift_zero_rank_foreach_add(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>) {
  %cst = arith.constant 1.000000e+00 : f32
  hivm.hir.set_mask_norm
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
  hivm.hir.load ins(%arg0 : memref<f32>) outs(%alloc : memref<f32>)
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<f32>
  hivm.hir.load ins(%arg1 : memref<f32>) outs(%alloc_0 : memref<f32>)
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
  %0 = memref.load %alloc[] : memref<f32>
  %1 = arith.addf %0, %cst : f32
  memref.store %1, %alloc_1[] : memref<f32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<f32>
  %2 = memref.load %alloc_0[] : memref<f32>
  %3 = arith.addf %2, %cst : f32
  memref.store %3, %alloc_2[] : memref<f32>
  hivm.hir.store ins(%alloc_1 : memref<f32>) outs(%arg2 : memref<f32>) atomic = <none>
  hivm.hir.store ins(%alloc_2 : memref<f32>) outs(%arg3 : memref<f32>) atomic = <none>
  return
}