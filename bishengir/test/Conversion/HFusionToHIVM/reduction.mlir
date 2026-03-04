// RUN: bishengir-opt -convert-hfusion-to-hivm %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_reduce_sum_ops
func.func @test_reduce_sum_ops(%src : memref<6x4xf32>, %dst : memref<6xf32>) {
  // CHECK: %[[expanded:.*]] = memref.expand_shape %arg1 {{\[}}[0, 1]] output_shape {{\[}}6, 1] : memref<6xf32> into memref<6x1xf32>
  // CHECK: hivm.hir.vreduce <sum> ins(%arg0 : memref<6x4xf32>) outs(%[[expanded]] : memref<6x1xf32>) unsigned_src = false reduce_dims = [1]
  linalg.reduce { arith.addf }
    ins(%src: memref<6x4xf32>)
    outs(%dst: memref<6xf32>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_prod_ops
func.func @test_reduce_prod_ops(%src : memref<6x4xf32>, %dst : memref<6xf32>) {
  // CHECK: memref.expand_shape {{.*}} {{\[}}[0, 1]]
  // CHECK: hivm.hir.vreduce <prod>
  linalg.reduce { arith.mulf }
    ins(%src: memref<6x4xf32>)
    outs(%dst: memref<6xf32>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_max_ops
func.func @test_reduce_max_ops(%src : memref<6x4xf32>, %dst : memref<6xf32>) {
  // CHECK: memref.expand_shape {{.*}} {{\[}}[0, 1]]
  // CHECK: hivm.hir.vreduce <max>
  linalg.reduce { arith.maximumf }
    ins(%src: memref<6x4xf32>)
    outs(%dst: memref<6xf32>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_ops
func.func @test_reduce_min_ops(%src : memref<6x4xf32>, %dst : memref<6xf32>) {
  // CHECK: memref.expand_shape {{.*}} {{\[}}[0, 1]]
  // CHECK: hivm.hir.vreduce <min>
  linalg.reduce { arith.minimumf }
    ins(%src: memref<6x4xf32>)
    outs(%dst: memref<6xf32>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_ops_3d_outer_axis
func.func @test_reduce_sum_ops_3d_outer_axis(%src : memref<6x4x3xf32>, %dst : memref<4x3xf32>) {
  // CHECK: %[[expanded:.*]] = memref.expand_shape %arg1 {{\[}}[0, 1], [2]] output_shape {{\[}}1, 4, 3] : memref<4x3xf32> into memref<1x4x3xf32>
  // CHECK: hivm.hir.vreduce <sum> ins(%arg0 : memref<6x4x3xf32>) outs(%[[expanded]] : memref<1x4x3xf32>) unsigned_src = false reduce_dims = [0]
  linalg.reduce { arith.addf }
    ins(%src: memref<6x4x3xf32>)
    outs(%dst: memref<4x3xf32>)
    dimensions = [0]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_max_ops_3d_last_axis
func.func @test_reduce_max_ops_3d_last_axis(%src : memref<6x4x3xi32>, %dst : memref<6x4xi32>) {
  // CHECK: %[[expanded:.*]] = memref.expand_shape %arg1 {{\[}}[0], [1, 2]] output_shape {{\[}}6, 4, 1] : memref<6x4xi32> into memref<6x4x1xi32>
  // CHECK: hivm.hir.vreduce <max> ins(%arg0 : memref<6x4x3xi32>) outs(%[[expanded]] : memref<6x4x1xi32>) unsigned_src = false reduce_dims = [2]
  linalg.reduce { arith.maxsi }
    ins(%src: memref<6x4x3xi32>)
    outs(%dst: memref<6x4xi32>)
    dimensions = [2]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_ops_3d_last_axis
func.func @test_reduce_min_ops_3d_last_axis(%src : memref<6x4x3xi32>, %dst : memref<6x4xi32>) {
  // CHECK: %[[expanded:.*]] = memref.expand_shape %arg1 {{\[}}[0], [1, 2]] output_shape {{\[}}6, 4, 1] : memref<6x4xi32> into memref<6x4x1xi32>
  // CHECK: hivm.hir.vreduce <min> ins(%arg0 : memref<6x4x3xi32>) outs(%[[expanded]] : memref<6x4x1xi32>) unsigned_src = false reduce_dims = [2]
  linalg.reduce { arith.minsi }
    ins(%src: memref<6x4x3xi32>)
    outs(%dst: memref<6x4xi32>)
    dimensions = [2]
  return
}

// -----

// CHECK: func.func @test_reduce_outs_stride(%[[arg0:.*]]: memref<6x4x16xf32>, %[[arg1:.*]]: memref<6x16xf32, strided<[16, 1], offset: ?>>) {
func.func @test_reduce_outs_stride(%src : memref<6x4x16xf32>, %dst : memref<6x16xf32, strided<[16,1], offset:?>>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg1]] {{\[}}[0, 1], [2]] output_shape {{\[}}6, 1, 16] : memref<6x16xf32, strided<[16, 1], offset: ?>> into memref<6x1x16xf32, strided<[16, 16, 1], offset: ?>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[arg0]] : memref<6x4x16xf32>) outs(%[[expand_shape]] : memref<6x1x16xf32, strided<[16, 16, 1], offset: ?>>) unsigned_src = false reduce_dims = [1]

  linalg.reduce { arith.addf }
    ins(%src: memref<6x4x16xf32>)
    outs(%dst: memref<6x16xf32, strided<[16, 1], offset:?>>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_sum_tensor_ops
func.func @test_reduce_sum_tensor_ops(%src : tensor<6x4xf32>, %dst : tensor<6xf32>) -> tensor<6xf32> {
  // CHECK: %[[expanded:.*]] = tensor.expand_shape %arg1 {{\[}}[0, 1]] output_shape {{\[}}6, 1] : tensor<6xf32> into tensor<6x1xf32>
  // CHECK: %[[res:.*]] = hivm.hir.vreduce <sum> ins(%arg0 : tensor<6x4xf32>) outs(%[[expanded]] : tensor<6x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<6x1xf32>
  %res = linalg.reduce { arith.addf }
    ins(%src: tensor<6x4xf32>)
    outs(%dst: tensor<6xf32>)
    dimensions = [1]
  return %res : tensor<6xf32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_min_with_index_ops
func.func @test_reduce_min_with_index_ops() {
  // CHECK: %[[SRC:.*]] = memref.alloca() : memref<32x2xf16, #hivm.address_space<ub>>
  // CHECK: %[[DST:.*]] = memref.alloca() : memref<32xf16, #hivm.address_space<ub>>
  // CHECK: %[[DST_INDEX:.*]] = memref.alloca() : memref<32xi32, #hivm.address_space<ub>>
  // CHECK: %[[EXPANDED0:.*]] = memref.expand_shape %[[DST]] {{\[}}[0, 1]] output_shape {{\[}}32, 1] : memref<32xf16, #hivm.address_space<ub>> into memref<32x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[EXPANDED1:.*]] = memref.expand_shape %[[DST_INDEX]] {{\[}}[0, 1]] output_shape {{\[}}32, 1] : memref<32xi32, #hivm.address_space<ub>> into memref<32x1xi32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vreduce <min_with_index> ins(%[[SRC]] : memref<32x2xf16, #hivm.address_space<ub>>) outs(%[[EXPANDED0]], %[[EXPANDED1]] : memref<32x1xf16, #hivm.address_space<ub>>, memref<32x1xi32, #hivm.address_space<ub>>) unsigned_src = false tie_break_left = true reduce_dims = [1]
  %src = memref.alloca() : memref<32x2xf16, #hivm.address_space<ub>>
  %src_index = memref.alloca(): memref<32x2xi32, #hivm.address_space<ub>>
  %dst = memref.alloca() : memref<32xf16, #hivm.address_space<ub>>
  %dst_index = memref.alloca()  : memref<32xi32, #hivm.address_space<ub>>
  hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                 ins(%src, %src_index : memref<32x2xf16, #hivm.address_space<ub>>, memref<32x2xi32, #hivm.address_space<ub>>)
                 outs(%dst, %dst_index : memref<32xf16, #hivm.address_space<ub>>, memref<32xi32, #hivm.address_space<ub>>)
                 dimensions = [1]
  return
}


// -----

// CHECK-LABEL: func.func @test_reduce_min
func.func @test_reduce_min(%src : memref<6x4xf32>, %dst : memref<6xf32>) {
  // CHECK: memref.expand_shape {{.*}} {{\[}}[0, 1]]
  // CHECK: hivm.hir.vreduce <min>
  linalg.reduce { arith.minnumf }
    ins(%src: memref<6x4xf32>)
    outs(%dst: memref<6xf32>)
    dimensions = [1]
  return
}


// -----

// CHECK-LABEL: func.func @test_reduce_max
func.func @test_reduce_max(%src : memref<6x4xf32>, %dst : memref<6xf32>) {
  // CHECK: memref.expand_shape {{.*}} {{\[}}[0, 1]]
  // CHECK: hivm.hir.vreduce <max>
  linalg.reduce { arith.maxnumf }
    ins(%src: memref<6x4xf32>)
    outs(%dst: memref<6xf32>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_scalar
// CHECK-SAME: (%[[IN_TENSOR:.*]]: tensor<32xf32>, %[[OUT_TENSOR:.*]]: tensor<f32>)
func.func @test_reduce_scalar(%arg0: tensor<32xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[OUT_TENSOR]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
  // CHECK: %[[REDUCED:.*]] = hivm.hir.vreduce <sum> ins(%[[IN_TENSOR]] : tensor<32xf32>) outs(%[[EXPANDED]] : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<32xf32>) outs(%arg1 : tensor<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %6 = arith.addf %in, %init : f32
      linalg.yield %6 : f32
    }
  return %reduced : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_any
// CHECK-SAME: (%[[IN_TENSOR:.*]]: tensor<1x1024xi1>, %[[OUT_TENSOR:.*]]: tensor<1xi1>)
func.func @test_reduce_any(%arg0: tensor<1x1024xi1>, %arg1: tensor<1xi1>) -> tensor<1xi1> {
  // CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[OUT_TENSOR]] {{\[}}[0, 1]] output_shape [1, 1] : tensor<1xi1> into tensor<1x1xi1>
  // CHECK: %[[REDUCED:.*]] = hivm.hir.vreduce <any> ins(%[[IN_TENSOR]] : tensor<1x1024xi1>) outs(%[[EXPANDED]] : tensor<1x1xi1>) unsigned_src = false reduce_dims = [1] -> tensor<1x1xi1>
  %reduced = linalg.reduce ins(%arg0 : tensor<1x1024xi1>) outs(%arg1 :  tensor<1xi1>) dimensions = [1]
    (%in: i1, %init: i1) {
      %6 = arith.ori %in, %init : i1
      linalg.yield %6 : i1
    }
  return %reduced : tensor<1xi1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_all
// CHECK-SAME: (%[[IN_TENSOR:.*]]: tensor<1x1024xi1>, %[[OUT_TENSOR:.*]]: tensor<1xi1>)
func.func @test_reduce_all(%arg0: tensor<1x1024xi1>, %arg1: tensor<1xi1>) -> tensor<1xi1> {
  // CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[OUT_TENSOR]] {{\[}}[0, 1]] output_shape [1, 1] : tensor<1xi1> into tensor<1x1xi1>
  // CHECK: %[[REDUCED:.*]] = hivm.hir.vreduce <all> ins(%[[IN_TENSOR]] : tensor<1x1024xi1>) outs(%[[EXPANDED]] : tensor<1x1xi1>) unsigned_src = false reduce_dims = [1] -> tensor<1x1xi1>
  %reduced = linalg.reduce ins(%arg0 : tensor<1x1024xi1>) outs(%arg1 :  tensor<1xi1>) dimensions = [1]
    (%in: i1, %init: i1) {
      %6 = arith.andi %in, %init : i1
      linalg.yield %6 : i1
    }
  return %reduced : tensor<1xi1>
}

// -----

// CHECK-LABEL: func.func @test_reduce_xori_ops
func.func @test_reduce_xori_ops(%src : memref<6x4xi32>, %dst : memref<6xi32>) {
  // CHECK: memref.expand_shape {{.*}} {{\[}}[0, 1]]
  // CHECK: hivm.hir.vreduce <xori>
  linalg.reduce { arith.xori }
    ins(%src: memref<6x4xi32>)
    outs(%dst: memref<6xi32>)
    dimensions = [1]
  return
}

// -----
// CHECK-LABEL: func.func @test_reduce_with_index
module {
  func.func @test_reduce_with_index(%arg0: tensor<256x64xf32>, %arg1: tensor<256x64xi32>) -> tensor<256xf32> {
    %true = arith.constant true
    %0 = tensor.empty() : tensor<256xf32>
    %1 = tensor.empty() : tensor<256xi32>
    // CHECK: %[[REDUCED:.*]]:2 = hivm.hir.vreduce <max_with_index> ins(%[[INPUT0:.*]] : tensor<256x64xf32>) outs(%[[INIT0:.*]], %[[INIT1:.*]] : tensor<256x1xf32>, tensor<256x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<256x1xf32>, tensor<256x1xi32>
    %2:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>) outs(%0, %1 : tensor<256xf32>, tensor<256xi32>) dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
    return %2#0 : tensor<256xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_reduce_with_index
module {
  func.func @test_reduce_with_index(%arg0: memref<256x64xf32>, %arg1: memref<256x64xi32>, %arg2: memref<256xf32>, %arg3: memref<256xi32>) {
    %true = arith.constant true
    // CHECK: hivm.hir.vreduce <max_with_index>
    hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%arg0, %arg1 : memref<256x64xf32>, memref<256x64xi32>) outs(%arg2, %arg3 : memref<256xf32>, memref<256xi32>) dimensions = [1]
    return
  }
}

// -----
// CHECK-LABEL: func.func @test_reduce_with_index
module {
  func.func @test_reduce_with_index(%arg0: tensor<256x64xf32>) -> tensor<256xf32> {
    %true = arith.constant true
    %0 = tensor.empty() : tensor<256xf32>
    %1 = tensor.empty() : tensor<256xi32>
    // CHECK: %[[REDUCED:.*]]:2 = hivm.hir.vreduce <min_with_index> ins(%[[INPUT0:.*]] : tensor<256x64xf32>) outs(%[[INIT0:.*]], %[[INIT1:.*]] : tensor<256x1xf32>, tensor<256x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<256x1xf32>, tensor<256x1xi32>
    %2:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min> ins(%arg0 : tensor<256x64xf32>) outs(%0, %1 : tensor<256xf32>, tensor<256xi32>) dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
    return %2#0 : tensor<256xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_reduce_with_index
module {
  func.func @test_reduce_with_index(%arg0: tensor<256x64xf32>) -> tensor<256xf32> {
    %true = arith.constant true
    %0 = tensor.empty() : tensor<256xf32>
    %1 = tensor.empty() : tensor<256xi32>
    // CHECK: %[[REDUCED:.*]]:2 = hivm.hir.vreduce <min_with_index> ins(%[[INPUT0:.*]] : tensor<256x64xf32>) outs(%[[INIT0:.*]], %[[INIT1:.*]] : tensor<256x1xf32>, tensor<256x1xi32>) unsigned_src = false tie_break_left = false reduce_dims = [1] -> tensor<256x1xf32>, tensor<256x1xi32>
    %2:2 = hfusion.reduce_with_index {tie_break_left = false, unsigned_src = false} <min> ins(%arg0 : tensor<256x64xf32>) outs(%0, %1 : tensor<256xf32>, tensor<256xi32>) dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
    return %2#0 : tensor<256xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_reduce_with_index
module {
  func.func @test_reduce_with_index(%arg0: memref<256x64xf32>, %arg1: memref<256xf32>, %arg2: memref<256xi32>) {
    %true = arith.constant true
    // CHECK: hivm.hir.vreduce <min_with_index>
    hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min> ins(%arg0 : memref<256x64xf32>) outs(%arg1, %arg2 : memref<256xf32>, memref<256xi32>) dimensions = [1]
    return
  }
}
