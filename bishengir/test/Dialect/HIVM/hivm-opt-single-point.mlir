// RUN: bishengir-opt %s -hivm-opt-single-point -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_f32_brc_scalar_opt
// CHECK-NOT: hivm.hir.brc
// CHECK: memref.store
func.func @test_f32_brc_scalar_opt(%arg0: memref<1x1xf32>) {
  %cst = arith.constant 0.0 : f32
  hivm.hir.vbrc ins(%cst : f32)
                outs(%arg0 : memref<1x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @test_i64_element_scalar_opt
func.func @test_i64_element_scalar_opt(%arg0: memref<1x1xi64>,
                                       %arg1: memref<1x1xi64>,
                                       %arg2: memref<1x1xi64>) {
  // CHECK-NOT: hivm.hir.vadd
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: memref.store
  hivm.hir.vadd ins(%arg0, %arg1 : memref<1x1xi64>, memref<1x1xi64>)
                outs(%arg2 : memref<1x1xi64>)
  return
}

// -----

// CHECK-LABEL: func @test_f32_element_scalar_opt
func.func @test_f32_element_scalar_opt(%arg0: memref<1x1xf32>,
                                       %arg1: memref<1x1xf32>,
                                       %arg2: memref<1x1xf32>) {
  // CHECK-NOT: hivm.hir.vadd
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: memref.store
  hivm.hir.vadd ins(%arg0, %arg1 : memref<1x1xf32>, memref<1x1xf32>)
                outs(%arg2 : memref<1x1xf32>)

  // CHECK-NOT: hivm.hir.vsub
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.subf
  // CHECK-NEXT: memref.store
  hivm.hir.vsub ins(%arg0, %arg1 : memref<1x1xf32>, memref<1x1xf32>)
                outs(%arg2 : memref<1x1xf32>)

  // CHECK-NOT: hivm.hir.vmul
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.mulf
  // CHECK-NEXT: memref.store
  hivm.hir.vmul ins(%arg0, %arg1 : memref<1x1xf32>, memref<1x1xf32>)
                outs(%arg2 : memref<1x1xf32>)

  // CHECK-NOT: hivm.hir.vdiv
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.divf
  // CHECK-NEXT: memref.store
  hivm.hir.vdiv ins(%arg0, %arg1 : memref<1x1xf32>, memref<1x1xf32>)
                outs(%arg2 : memref<1x1xf32>)

  // CHECK-NOT: hivm.hir.vabs
  // CHECK: memref.load
  // CHECK-NEXT: math.absf
  // CHECK-NEXT: memref.store
  hivm.hir.vabs ins(%arg0 : memref<1x1xf32>)
                outs(%arg1 : memref<1x1xf32>)

  // CHECK-NOT: hivm.hir.vsqrt
  // CHECK: memref.load
  // CHECK-NEXT: math.sqrt
  // CHECK-NEXT: memref.store
  hivm.hir.vsqrt ins(%arg0 : memref<1x1xf32>)
                 outs(%arg1 : memref<1x1xf32>)

  // CHECK-NOT: hivm.hir.vmax
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.maxnumf
  // CHECK-NEXT: memref.store
  hivm.hir.vmax ins(%arg0, %arg1 : memref<1x1xf32>, memref<1x1xf32>)
                outs(%arg2 : memref<1x1xf32>)

  // CHECK-NOT: hivm.hir.vmin
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.minnumf
  // CHECK-NEXT: memref.store
  hivm.hir.vmin ins(%arg0, %arg1 : memref<1x1xf32>, memref<1x1xf32>)
                outs(%arg2 : memref<1x1xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_one_write
// CHECK: hivm.hir.store
func.func @test_one_write(%arg0: memref<24x32x4xf32, #hivm.address_space<gm>>, %arg1: index, %arg2: index) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<24x32x4xf32, #hivm.address_space<gm>> into memref<768x4xf32, #hivm.address_space<gm>>
  %subview = memref.subview %collapse_shape[%arg1, %arg2] [1, 1] [1, 1] : memref<768x4xf32, #hivm.address_space<gm>> to memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1xf32, #hivm.address_space<ub>>
  %collapse_shape_0 = memref.collapse_shape %alloc [[0], [1, 2]] : memref<1x1x1xf32, #hivm.address_space<ub>> into memref<1x1xf32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %collapse_shape_0[0, 0] [1, 1] [1, 1] : memref<1x1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_2 = memref.subview %subview[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_1 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func.func @test_one_read
// CHECK: memref.load
// CHECK: memref.store
func.func @test_one_read(%arg0: memref<24x32x4xf32, #hivm.address_space<gm>>, %arg1: index, %arg2: index) 
attributes {hacc.no_io_alias} {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<24x32x4xf32, #hivm.address_space<gm>> into memref<768x4xf32, #hivm.address_space<gm>>
  %subview = memref.subview %collapse_shape[%arg1, %arg2] [1, 1] [1, 1] : memref<768x4xf32, #hivm.address_space<gm>> to memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1xf32, #hivm.address_space<ub>>
  %collapse_shape_0 = memref.collapse_shape %alloc [[0], [1, 2]] : memref<1x1x1xf32, #hivm.address_space<ub>> into memref<1x1xf32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %collapse_shape_0[0, 0] [1, 1] [1, 1] : memref<1x1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_2 = memref.subview %subview[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.load ins(%subview_2 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func.func @test_many_read
// CHECK: memref.load
// CHECK: memref.store
// CHECK: memref.load
// CHECK: memref.store
func.func @test_many_read(%arg0: memref<24x32x4xf32, #hivm.address_space<gm>>, %arg1: index, %arg2: index) 
attributes {hacc.no_io_alias} {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<24x32x4xf32, #hivm.address_space<gm>> into memref<768x4xf32, #hivm.address_space<gm>>
  %subview = memref.subview %collapse_shape[%arg1, %arg2] [1, 1] [1, 1] : memref<768x4xf32, #hivm.address_space<gm>> to memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_0 = memref.subview %subview[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1xf32, #hivm.address_space<ub>>
  %collapse_shape_1 = memref.collapse_shape %alloc [[0], [1, 2]] : memref<1x1x1xf32, #hivm.address_space<ub>> into memref<1x1xf32, #hivm.address_space<ub>>
  %subview_2 = memref.subview %collapse_shape_1[0, 0] [1, 1] [1, 1] : memref<1x1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%subview_0 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>)
  hivm.hir.load ins(%subview_0 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func.func @test_many_write
// CHECK: hivm.hir.store
// CHECK: hivm.hir.store
func.func @test_many_write(%arg0: memref<24x32x4xf32, #hivm.address_space<gm>>, %arg1: index, %arg2: index) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<24x32x4xf32, #hivm.address_space<gm>> into memref<768x4xf32, #hivm.address_space<gm>>
  %subview = memref.subview %collapse_shape[%arg1, %arg2] [1, 1] [1, 1] : memref<768x4xf32, #hivm.address_space<gm>> to memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1xf32, #hivm.address_space<ub>>
  %collapse_shape_0 = memref.collapse_shape %alloc [[0], [1, 2]] : memref<1x1x1xf32, #hivm.address_space<ub>> into memref<1x1xf32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %collapse_shape_0[0, 0] [1, 1] [1, 1] : memref<1x1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_2 = memref.subview %subview[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_1 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.store ins(%subview_1 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func.func @test_read_and_write
// CHECK: hivm.hir.store
// CHECK: hivm.hir.load
func.func @test_read_and_write(%arg0: memref<24x32x4xf32, #hivm.address_space<gm>>, %arg1: index, %arg2: index) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<24x32x4xf32, #hivm.address_space<gm>> into memref<768x4xf32, #hivm.address_space<gm>>
  %subview = memref.subview %collapse_shape[%arg1, %arg2] [1, 1] [1, 1] : memref<768x4xf32, #hivm.address_space<gm>> to memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1xf32, #hivm.address_space<ub>>
  %collapse_shape_0 = memref.collapse_shape %alloc [[0], [1, 2]] : memref<1x1x1xf32, #hivm.address_space<ub>> into memref<1x1xf32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %collapse_shape_0[0, 0] [1, 1] [1, 1] : memref<1x1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_2 = memref.subview %subview[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_1 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.load ins(%subview_2 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func.func @test_disable_opt_single_point
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
func.func @test_disable_opt_single_point(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  %c1_i64 = arith.constant 1 : i64
  %0 = tensor.empty() : tensor<i64>
  %1 = hivm.hir.load ins(%arg0 : tensor<i64>) outs(%0 : tensor<i64>) -> tensor<i64>
  %2 = hivm.hir.vadd ins(%1, %c1_i64 : tensor<i64>, i64) outs(%0 : tensor<i64>) -> tensor<i64>
  %3 = hivm.hir.store ins(%2 : tensor<i64>) outs(%arg1 : tensor<i64>) atomic = <none> -> tensor<i64>
  return %3 : tensor<i64>
}