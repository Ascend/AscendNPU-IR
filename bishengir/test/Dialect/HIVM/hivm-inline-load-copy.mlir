// RUN: bishengir-opt --hivm-inline-load-copy -split-input-file %s | FileCheck %s


// CHECK-LABEL: func.func @test_load_copy
func.func @test_load_copy(%arg0: memref<5x128xf32>, %arg1: memref<5x128xf32>) {
  // CHECK-NOT: hivm.hir.copy
  // CHECK: hivm.hir.load ins(%arg0 : memref<5x128xf32>) outs(%arg1 : memref<5x128xf32>)
  %empty0 = memref.alloc() : memref<5x128xf32>
  hivm.hir.load ins(%arg0 : memref<5x128xf32>) outs(%empty0 : memref<5x128xf32>)
  hivm.hir.copy ins(%empty0 : memref<5x128xf32>) outs(%arg1 : memref<5x128xf32>)
  return
}

// -----
// CHECK-LABEL: func.func @test_load_copy_subview
func.func @test_load_copy_subview(%arg0: memref<4x128xf32>, %arg1: memref<4x128xf32>, %arg2: memref<5x128xf32>) {
  // CHECK-NOT: hivm.hir.copy
  // CHECK: hivm.hir.load ins(%arg0 : memref<4x128xf32>) outs(%arg1 : memref<4x128xf32>)
  %cst = arith.constant 1.000000e+00 : f32
  %empty0 = memref.alloc() : memref<5x128xf32>
  %subview = memref.subview %empty0[0,0] [4,128] [1,1] : memref<5x128xf32> to memref<4x128xf32>
  hivm.hir.store ins(%empty0 : memref<5x128xf32>) outs(%arg2 : memref<5x128xf32>)
  hivm.hir.load ins(%arg0 : memref<4x128xf32>) outs(%subview : memref<4x128xf32>)
  hivm.hir.copy ins(%subview : memref<4x128xf32>) outs(%arg1 : memref<4x128xf32>)
  hivm.hir.vbrc ins(%cst : f32) outs(%empty0 : memref<5x128xf32>)
  return
}


// -----
// CHECK-LABEL: func.func @test_load_copy_written
func.func @test_load_copy_written(%arg0: memref<4x128xf32>, %arg1: memref<4x128xf32>) {
  // CHECK: hivm.hir.copy
  %cst = arith.constant 1.000000e+00 : f32
  %empty0 = memref.alloc() : memref<5x128xf32>
  %subview = memref.subview %empty0[0,0] [4,128] [1,1] : memref<5x128xf32> to memref<4x128xf32>
  hivm.hir.vbrc ins(%cst : f32) outs(%empty0 : memref<5x128xf32>)
  hivm.hir.load ins(%arg0 : memref<4x128xf32>) outs(%subview : memref<4x128xf32>)
  hivm.hir.copy ins(%subview : memref<4x128xf32>) outs(%arg1 : memref<4x128xf32>)
  return
}

// -----
// CHECK-LABEL: func.func @test_load_copy_read
func.func @test_load_copy_read(%arg0: memref<4x128xf32>, %arg1: memref<4x128xf32>, %arg2: memref<5x128xf32>) {
  // CHECK: hivm.hir.copy
  %empty0 = memref.alloc() : memref<5x128xf32>
  %subview = memref.subview %empty0[0,0] [4,128] [1,1] : memref<5x128xf32> to memref<4x128xf32>
  hivm.hir.load ins(%arg0 : memref<4x128xf32>) outs(%subview : memref<4x128xf32>)
  hivm.hir.copy ins(%subview : memref<4x128xf32>) outs(%arg1 : memref<4x128xf32>)
  hivm.hir.store ins(%empty0 : memref<5x128xf32>) outs(%arg2 : memref<5x128xf32>)
  return
}

// -----
// CHECK-LABEL: func.func @test_copy_before_load
func.func @test_copy_before_load(%arg0: memref<5x128xf32>, %arg1: memref<5x128xf32>) {
  // CHECK: hivm.hir.copy
  %empty0 = memref.alloc() : memref<5x128xf32>
  %empty1 = memref.alloc() : memref<5x128xf32>
  hivm.hir.copy ins(%empty0 : memref<5x128xf32>) outs(%empty1 : memref<5x128xf32>)
  hivm.hir.load ins(%arg0 : memref<5x128xf32>) outs(%empty0 : memref<5x128xf32>)
  return
}

// -----
// CHECK-LABEL: func.func @test_load_copy_write_source
func.func @test_load_copy_write_source(%arg0: memref<4x128xf32>, %arg1: memref<4x128xf32>) {
  // CHECK: hivm.hir.copy
  %cst = arith.constant 1.000000e+00 : f32
  %empty0 = memref.alloc() : memref<5x128xf32>
  %subview = memref.subview %empty0[0,0] [4,128] [1,1] : memref<5x128xf32> to memref<4x128xf32>
  %subview1 = memref.subview %arg0[0,0] [3,128] [1,1] : memref<4x128xf32> to memref<3x128xf32>
  hivm.hir.load ins(%arg0 : memref<4x128xf32>) outs(%subview : memref<4x128xf32>)
  hivm.hir.vbrc ins(%cst : f32) outs(%subview1 : memref<3x128xf32>)
  hivm.hir.copy ins(%subview : memref<4x128xf32>) outs(%arg1 : memref<4x128xf32>)
  return
}