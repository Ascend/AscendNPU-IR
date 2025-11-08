// RUN: bishengir-opt --hivm-inline-otf-broadcast -split-input-file %s | FileCheck %s
// RUN: bishengir-opt --canonicalize-ext --hivm-inline-otf-broadcast -split-input-file %s | FileCheck %s --check-prefix=CHECK-CANONICALIZE

// -----
// CHECK-LABEL: func.func @test_vmul_ins0
func.func @test_vmul_ins0(%arg0: tensor<1x128xf32>, %arg1: tensor<5x128xf32>) -> tensor<5x128xf32> {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul ins(%[[ins0:.*]], %[[ins1:.*]] : tensor<1x128xf32>, tensor<5x128xf32>) outs(%[[outs:.*]] : tensor<5x128xf32>)
  %empty0 = tensor.empty() : tensor<5x128xf32>
  %empty1 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg0 : tensor<1x128xf32>) outs(%empty0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %ret = hivm.hir.vmul ins(%brc, %arg1 : tensor<5x128xf32>, tensor<5x128xf32>) outs(%empty1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %ret: tensor<5x128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vmul_ins1
func.func @test_vmul_ins1(%arg0: tensor<5x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<5x128xf32> {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul ins(%[[ins0:.*]], %[[ins1:.*]] : tensor<5x128xf32>, tensor<1x128xf32>) outs(%[[outs:.*]] : tensor<5x128xf32>)
  %empty0 = tensor.empty() : tensor<5x128xf32>
  %empty1 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg1 : tensor<1x128xf32>) outs(%empty0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %ret = hivm.hir.vmul ins(%arg0, %brc : tensor<5x128xf32>, tensor<5x128xf32>) outs(%empty1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %ret: tensor<5x128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vmul_outs
func.func @test_vmul_outs(%arg0: tensor<1x128xf32>, %arg1: tensor<5x128xf32>) -> tensor<5x128xf32> {
  // CHECK: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul ins(%[[ins0:.*]], %[[ins1:.*]] : tensor<5x128xf32>, tensor<5x128xf32>) outs(%[[outs:.*]] : tensor<5x128xf32>)
  %empty0 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg0 : tensor<1x128xf32>) outs(%empty0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %ret = hivm.hir.vmul ins(%arg1, %arg1 : tensor<5x128xf32>, tensor<5x128xf32>) outs(%brc : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %ret: tensor<5x128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vsel_non_last_ins0
func.func @test_vsel_non_last_ins0(%arg0: tensor<1x128xi1>, %arg1: tensor<5x128xf32>, %arg2: tensor<5x128xf32>) -> tensor<5x128xf32> {
  // CHECK: hivm.hir.vbrc
  // CHECK: hivm.hir.vsel ins(%[[ins0:.*]], %[[ins1:.*]], %[[ins2:.*]] : tensor<5x128xi1>, tensor<5x128xf32>, tensor<5x128xf32>) outs(%[[outs:.*]] : tensor<5x128xf32>)
  %empty0 = tensor.empty() : tensor<5x128xi1>
  %empty1 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg0 : tensor<1x128xi1>) outs(%empty0 : tensor<5x128xi1>) broadcast_dims = [0] -> tensor<5x128xi1>
  %ret = hivm.hir.vsel ins(%brc, %arg1, %arg2 : tensor<5x128xi1>, tensor<5x128xf32>, tensor<5x128xf32>) outs(%empty1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %ret: tensor<5x128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vsel_non_last_ins1
func.func @test_vsel_non_last_ins1(%arg0: tensor<5x128xi1>, %arg1: tensor<1x128xf32>, %arg2: tensor<5x128xf32>) -> tensor<5x128xf32> {
  // CHECK: hivm.hir.vbrc
  // CHECK: hivm.hir.vsel ins(%[[ins0:.*]], %[[ins1:.*]], %[[ins2:.*]] : tensor<5x128xi1>, tensor<5x128xf32>, tensor<5x128xf32>) outs(%[[outs:.*]] : tensor<5x128xf32>)
  %empty0 = tensor.empty() : tensor<5x128xf32>
  %empty1 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg1 : tensor<1x128xf32>) outs(%empty0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %ret = hivm.hir.vsel ins(%arg0, %brc, %arg2 : tensor<5x128xi1>, tensor<5x128xf32>, tensor<5x128xf32>) outs(%empty1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %ret: tensor<5x128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vsel_non_last_ins2
func.func @test_vsel_non_last_ins2(%arg0: tensor<5x128xi1>, %arg1: tensor<5x128xf32>, %arg2: tensor<1x128xf32>) -> tensor<5x128xf32> {
  // CHECK: hivm.hir.vbrc
  // CHECK: hivm.hir.vsel ins(%[[ins0:.*]], %[[ins1:.*]], %[[ins2:.*]] : tensor<5x128xi1>, tensor<5x128xf32>, tensor<5x128xf32>) outs(%[[outs:.*]] : tensor<5x128xf32>)
  %empty0 = tensor.empty() : tensor<5x128xf32>
  %empty1 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg2 : tensor<1x128xf32>) outs(%empty0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %ret = hivm.hir.vsel ins(%arg0, %arg1, %brc : tensor<5x128xi1>, tensor<5x128xf32>, tensor<5x128xf32>) outs(%empty1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %ret: tensor<5x128xf32>
}

// -----
// CHECK-LABEL: func.func @test_vsel_last_ins2
func.func @test_vsel_last_ins2(%arg0: tensor<128x5xi1>, %arg1: tensor<128x5xf32>, %arg2: tensor<128x1xf32>) -> tensor<128x5xf32> {
  %empty0 = tensor.empty() : tensor<128x5xf32>
  %empty1 = tensor.empty() : tensor<128x5xf32>
  // CHECK: hivm.hir.vbrc
  // CHECK: hivm.hir.vsel ins(%[[ins0:.*]], %[[ins1:.*]], %[[ins2:.*]] : tensor<128x5xi1>, tensor<128x5xf32>, tensor<128x5xf32>) outs(%[[outs:.*]] : tensor<128x5xf32>)
  %brc = hivm.hir.vbrc ins(%arg2 : tensor<128x1xf32>) outs(%empty0 : tensor<128x5xf32>) broadcast_dims = [1] -> tensor<128x5xf32>
  %ret = hivm.hir.vsel ins(%arg0, %arg1, %brc : tensor<128x5xi1>, tensor<128x5xf32>, tensor<128x5xf32>) outs(%empty1 : tensor<128x5xf32>) -> tensor<128x5xf32>
  return %ret: tensor<128x5xf32>
}

// -----
// CHECK-LABEL: func.func @test_vmul_brc_1_3_5
func.func @test_vmul_brc_1_3_5(%arg0: tensor<16x1x16x1x16x1xf32>, %arg1: tensor<16x16x16x16x16x16xf32>) -> tensor<16x16x16x16x16x16xf32> {
  %empty0 = tensor.empty() : tensor<16x16x16x1x16x1xf32>
  %empty1 = tensor.empty() : tensor<16x16x16x16x16x1xf32>
  %empty2 = tensor.empty() : tensor<16x16x16x16x16x16xf32>
  %empty3 = tensor.empty() : tensor<16x16x16x16x16x16xf32>
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul ins(%[[ins0:.*]], %[[ins1:.*]] : tensor<16x1x16x1x16x1xf32>, tensor<16x16x16x16x16x16xf32>) outs(%[[outs:.*]] : tensor<16x16x16x16x16x16xf32>) broadcast = [1, 3, 5] -> tensor<16x16x16x16x16x16xf32>
  %brc0 = hivm.hir.vbrc ins(%arg0 : tensor<16x1x16x1x16x1xf32>) outs(%empty0 : tensor<16x16x16x1x16x1xf32>) broadcast_dims = [1] -> tensor<16x16x16x1x16x1xf32>
  %brc1 = hivm.hir.vbrc ins(%brc0 : tensor<16x16x16x1x16x1xf32>) outs(%empty1 : tensor<16x16x16x16x16x1xf32>) broadcast_dims = [3] -> tensor<16x16x16x16x16x1xf32>
  %brc2 = hivm.hir.vbrc ins(%brc1 : tensor<16x16x16x16x16x1xf32>) outs(%empty2 : tensor<16x16x16x16x16x16xf32>) broadcast_dims = [5] -> tensor<16x16x16x16x16x16xf32>
  %ret = hivm.hir.vmul ins(%brc2, %arg1 : tensor<16x16x16x16x16x16xf32>, tensor<16x16x16x16x16x16xf32>) outs(%empty3 : tensor<16x16x16x16x16x16xf32>) -> tensor<16x16x16x16x16x16xf32>
  return %ret: tensor<16x16x16x16x16x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_vmul_brc_two_op
func.func @test_vmul_brc_two_op(%arg0: tensor<1x16x1x16xf32>, %arg1: tensor<16x16x1x1xf32>) -> tensor<16x16x16x16xf32> {
  %empty0 = tensor.empty() : tensor<16x16x1x16xf32>
  %empty1 = tensor.empty() : tensor<16x16x16x16xf32>
  %empty2 = tensor.empty() : tensor<16x16x16x1xf32>
  %empty3 = tensor.empty() : tensor<16x16x16x16xf32>
  %empty4 = tensor.empty() : tensor<16x16x16x16xf32>
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul ins(%[[ins0:.*]], %[[ins1:.*]] : tensor<1x16x1x16xf32>, tensor<16x16x1x1xf32>) outs(%[[outs:.*]] : tensor<16x16x16x16xf32>) broadcast = [0, 2, 3] -> tensor<16x16x16x16xf32>
  %brc0 = hivm.hir.vbrc ins(%arg0 : tensor<1x16x1x16xf32>) outs(%empty0 : tensor<16x16x1x16xf32>) broadcast_dims = [0] -> tensor<16x16x1x16xf32>
  %brc1 = hivm.hir.vbrc ins(%brc0 : tensor<16x16x1x16xf32>) outs(%empty1 : tensor<16x16x16x16xf32>) broadcast_dims = [2] -> tensor<16x16x16x16xf32>
  %brc2 = hivm.hir.vbrc ins(%arg1 : tensor<16x16x1x1xf32>) outs(%empty2 : tensor<16x16x16x1xf32>) broadcast_dims = [2] -> tensor<16x16x16x1xf32>
  %brc3 = hivm.hir.vbrc ins(%brc2 : tensor<16x16x16x1xf32>) outs(%empty3 : tensor<16x16x16x16xf32>) broadcast_dims = [3] -> tensor<16x16x16x16xf32>
  %ret = hivm.hir.vmul ins(%brc1, %brc3 : tensor<16x16x16x16xf32>, tensor<16x16x16x16xf32>) outs(%empty3 : tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
  return %ret: tensor<16x16x16x16xf32>
}

// -----
// CHECK-CANONICALIZE-LABEL: func.func @test_inline_unit_dim_brc
// CHECK-CANONICALIZE: hivm.hir.vmul ins({{.*}} : tensor<1x?x1xf32>, tensor<1x?x16xf32>) outs({{.*}} : tensor<1x?x16xf32>) broadcast = [2]
func.func @test_inline_unit_dim_brc(%arg0: tensor<1x?x16xf32>, %arg1: tensor<1x?x1xf32>) -> tensor<1x?x16xf32> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<1x?x16xf32>
  %0 = tensor.empty(%dim) : tensor<1x?x16xf32>
  %1 = tensor.empty(%dim) : tensor<1x?x16xf32>
  %2 = tensor.empty(%dim) : tensor<1x?x16xf32>
  %3 = hivm.hir.vbrc ins(%arg0 : tensor<1x?x16xf32>) outs(%0 : tensor<1x?x16xf32>) broadcast_dims = [0] -> tensor<1x?x16xf32>
  %4 = hivm.hir.vbrc ins(%arg1 : tensor<1x?x1xf32>) outs(%1 : tensor<1x?x16xf32>) broadcast_dims = [2] -> tensor<1x?x16xf32>
  %5 = hivm.hir.vmul ins(%4, %3 : tensor<1x?x16xf32>, tensor<1x?x16xf32>) outs(%2 : tensor<1x?x16xf32>) -> tensor<1x?x16xf32>
  return %5 : tensor<1x?x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_vmul_vadds
func.func @test_vmul_vadds(%arg0: tensor<5x128xf32>, %arg1: tensor<1x128xf32>) -> (tensor<5x128xf32>, tensor<1x128xf32>) {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul ins(%[[ins0:.*]], %[[ins1:.*]] : tensor<5x128xf32>, tensor<1x128xf32>) outs(%[[outs:.*]] : tensor<5x128xf32>)
  // CHECK: hivm.hir.vadd ins(%[[ins1:.*]], %[[cst:.*]] : tensor<1x128xf32>, f32) outs(%[[outs1:.*]] : tensor<1x128xf32>)
  %cst = arith.constant 1.000000e+00 : f32
  %empty0 = tensor.empty() : tensor<5x128xf32>
  %empty1 = tensor.empty() : tensor<5x128xf32>
  %empty2 = tensor.empty() : tensor<1x128xf32>
  %brc = hivm.hir.vbrc ins(%arg1 : tensor<1x128xf32>) outs(%empty0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %ret = hivm.hir.vmul ins(%arg0, %brc : tensor<5x128xf32>, tensor<5x128xf32>) outs(%empty1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  %ret1 = hivm.hir.vadd ins(%arg1, %cst : tensor<1x128xf32>, f32) outs(%empty2 : tensor<1x128xf32>) -> tensor<1x128xf32>
  return %ret, %ret1 : tensor<5x128xf32>, tensor<1x128xf32>
}

// -----
// CHECK-CANONICALIZE-LABEL: func.func @simple_erasing_brc
// CHECK-CANONICALIZE-NOT: broadcast
func.func @simple_erasing_brc(%arg0: memref<1x2xf32>, %arg1: memref<1x2xf32>) {
  hivm.hir.vadd ins(%arg0, %arg0 : memref<1x2xf32>, memref<1x2xf32>) outs(%arg1 : memref<1x2xf32>) broadcast = [0]
  return
}

// -----
// CHECK-CANONICALIZE-LABEL: func.func @simple_erasing_brc_tensor
// CHECK-CANONICALIZE-NOT: broadcast
func.func @simple_erasing_brc_tensor(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) {
  %0 = hivm.hir.vadd ins(%arg0, %arg0 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%arg1 : tensor<1x2xf32>) broadcast = [0] -> tensor<1x2xf32>
  return
}

// -----
// CHECK-CANONICALIZE-LABEL: func.func @not_erasing_brc
// CHECK-CANONICALIZE: broadcast = [0]
func.func @not_erasing_brc(%arg0: memref<1x2xf32>, %arg1: memref<2x2xf32>) {
  hivm.hir.vadd ins(%arg0, %arg0 : memref<1x2xf32>, memref<1x2xf32>) outs(%arg1 : memref<2x2xf32>) broadcast = [0]
  return
}
