// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vcmp_i1
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi1>) outs(%{{.*}} : tensor<16xf16>) round_mode = <trunc> -> tensor<16xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi1>) outs(%{{.*}} : tensor<16xf16>) round_mode = <trunc> -> tensor<16xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[LHS]], %[[RHS]] : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<16xi1>) outs(%{{.*}} : tensor<16xi1>) -> tensor<16xi1>
func.func @test_NormalizeToTargetType_hivm_vcmp_i1(%arg0: tensor<16xi1>, %arg1: tensor<16xi1>) -> tensor<16xi1> {
  %dst = tensor.empty() : tensor<16xi1>
  %0 = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<16xi1>, tensor<16xi1>) outs(%dst : tensor<16xi1>) compare_mode = <ne> -> tensor<16xi1>
  return %0 : tensor<16xi1>
}
// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vbrc_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1x16xi1>) outs(%{{.*}} : tensor<1x16xf16>) round_mode = <trunc> -> tensor<1x16xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<5x16xi1>) outs(%{{.*}} : tensor<5x16xf16>) round_mode = <trunc> -> tensor<5x16xf16>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[SRC_F16]] : tensor<1x16xf16>) outs(%[[DST_F16]] : tensor<5x16xf16>) broadcast_dims = [0] -> tensor<5x16xf16>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<5x16xf16>) -> tensor<5x16xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[BRC]], %[[ZERO_BRC]] : tensor<5x16xf16>, tensor<5x16xf16>) outs(%{{.*}} : tensor<5x16xi1>) -> tensor<5x16xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<5x16xi1>) outs(%{{.*}} : tensor<5x16xi1>) -> tensor<5x16xi1>
func.func @test_NormalizeToTargetType_hivm_vbrc_i1(%arg0: tensor<1x16xi1>) -> tensor<5x16xi1> {
  %dst = tensor.empty() : tensor<5x16xi1>
  %0 = hivm.hir.vbrc ins(%arg0 : tensor<1x16xi1>) outs(%dst : tensor<5x16xi1>) broadcast_dims = [0] -> tensor<5x16xi1>
  return %0 : tensor<5x16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vtranspose_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x16xi1>) outs(%{{.*}} : tensor<4x16xf16>) round_mode = <trunc> -> tensor<4x16xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<16x4xi1>) outs(%{{.*}} : tensor<16x4xf16>) round_mode = <trunc> -> tensor<16x4xf16>
// CHECK: %[[TRANS:.*]] = hivm.hir.vtranspose ins(%[[SRC_F16]] : tensor<4x16xf16>) outs(%[[DST_F16]] : tensor<16x4xf16>) permutation = [1, 0] -> tensor<16x4xf16>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<16x4xf16>) -> tensor<16x4xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[TRANS]], %[[ZERO_BRC]] : tensor<16x4xf16>, tensor<16x4xf16>) outs(%{{.*}} : tensor<16x4xi1>) -> tensor<16x4xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<16x4xi1>) outs(%{{.*}} : tensor<16x4xi1>) -> tensor<16x4xi1>
func.func @test_NormalizeToTargetType_hivm_vtranspose_i1(%arg0: tensor<4x16xi1>) -> tensor<16x4xi1> {
  %dst = tensor.empty() : tensor<16x4xi1>
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<4x16xi1>) outs(%dst : tensor<16x4xi1>) permutation = [1, 0] -> tensor<16x4xi1>
  return %0 : tensor<16x4xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vinterleave_i1
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x16x1xi1>) outs(%{{.*}} : tensor<4x16x1xf16>) round_mode = <trunc> -> tensor<4x16x1xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<4x16x1xi1>) outs(%{{.*}} : tensor<4x16x1xf16>) round_mode = <trunc> -> tensor<4x16x1xf16>
// CHECK: %[[INTER:.*]] = hivm.hir.vinterleave ins(%[[LHS]], %[[RHS]] : tensor<4x16x1xf16>, tensor<4x16x1xf16>) outs(%{{.*}} : tensor<4x16x2xf16>) interleave_channel_nums = 2 -> tensor<4x16x2xf16>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<4x16x2xf16>) -> tensor<4x16x2xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[INTER]], %[[ZERO_BRC]] : tensor<4x16x2xf16>, tensor<4x16x2xf16>) outs(%{{.*}} : tensor<4x16x2xi1>) -> tensor<4x16x2xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<4x16x2xi1>) outs(%{{.*}} : tensor<4x16x2xi1>) -> tensor<4x16x2xi1>
func.func @test_NormalizeToTargetType_hivm_vinterleave_i1(%arg0: tensor<4x16x1xi1>, %arg1: tensor<4x16x1xi1>) -> tensor<4x16x2xi1> {
  %dst = tensor.empty() : tensor<4x16x2xi1>
  %0 = hivm.hir.vinterleave ins(%arg0, %arg1 : tensor<4x16x1xi1>, tensor<4x16x1xi1>) outs(%dst : tensor<4x16x2xi1>) interleave_channel_nums = 2 -> tensor<4x16x2xi1>
  return %0 : tensor<4x16x2xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vconcat_i1
// CHECK-DAG: %[[A:.*]] = hivm.hir.vcast ins(%arg0 : tensor<2x16xi1>) outs(%{{.*}} : tensor<2x16xf16>) round_mode = <trunc> -> tensor<2x16xf16>
// CHECK-DAG: %[[B:.*]] = hivm.hir.vcast ins(%arg1 : tensor<2x16xi1>) outs(%{{.*}} : tensor<2x16xf16>) round_mode = <trunc> -> tensor<2x16xf16>
// CHECK: %[[CAT:.*]] = hivm.hir.vconcat dim(1) ins(%[[A]], %[[B]] : tensor<2x16xf16>, tensor<2x16xf16>) outs(%{{.*}} : tensor<2x32xf16>) -> tensor<2x32xf16>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<2x32xf16>) -> tensor<2x32xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[CAT]], %[[ZERO_BRC]] : tensor<2x32xf16>, tensor<2x32xf16>) outs(%{{.*}} : tensor<2x32xi1>) -> tensor<2x32xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<2x32xi1>) outs(%{{.*}} : tensor<2x32xi1>) -> tensor<2x32xi1>
func.func @test_NormalizeToTargetType_hivm_vconcat_i1(%arg0: tensor<2x16xi1>, %arg1: tensor<2x16xi1>) -> tensor<2x32xi1> {
  %dst = tensor.empty() : tensor<2x32xi1>
  %0 = hivm.hir.vconcat dim(1) ins(%arg0, %arg1: tensor<2x16xi1>, tensor<2x16xi1>) outs(%dst : tensor<2x32xi1>) -> tensor<2x32xi1>
  return %0 : tensor<2x32xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vreduce_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x16xi1>) outs(%{{.*}} : tensor<4x16xf16>) round_mode = <trunc> -> tensor<4x16xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<1x16xi1>) outs(%{{.*}} : tensor<1x16xf16>) round_mode = <trunc> -> tensor<1x16xf16>
// CHECK: %[[RED:.*]]:2 = hivm.hir.vreduce <max_with_index> ins(%[[SRC_F16]] : tensor<4x16xf16>) outs(%[[DST_F16]], %{{.*}} : tensor<1x16xf16>, tensor<1x16xi32>) unsigned_src = false tie_break_left = true reduce_dims = [0] -> tensor<1x16xf16>, tensor<1x16xi32>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<1x16xf16>) -> tensor<1x16xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[RED]]#0, %[[ZERO_BRC]] : tensor<1x16xf16>, tensor<1x16xf16>) outs(%{{.*}} : tensor<1x16xi1>) -> tensor<1x16xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<1x16xi1>) outs(%{{.*}} : tensor<1x16xi1>) -> tensor<1x16xi1>
func.func @test_NormalizeToTargetType_hivm_vreduce_i1(%arg0: tensor<4x16xi1>) -> (tensor<1x16xi1>, tensor<1x16xi32>) {
  %dst0 = tensor.empty() : tensor<1x16xi1>
  %dst1 = tensor.empty() : tensor<1x16xi32>
  %0:2 = hivm.hir.vreduce <max_with_index> ins(%arg0 : tensor<4x16xi1>) outs(%dst0, %dst1 : tensor<1x16xi1>, tensor<1x16xi32>) unsigned_src = false tie_break_left = true reduce_dims = [0] -> tensor<1x16xi1>, tensor<1x16xi32>
  return %0#0, %0#1 : tensor<1x16xi1>, tensor<1x16xi32>
}
// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vadd_i8
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[ADD:.*]] = hivm.hir.vadd ins(%[[LHS]], %[[RHS]] : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[ADD]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vadd_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vsub_i8
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[SUB:.*]] = hivm.hir.vsub ins(%[[LHS]], %[[RHS]] : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[SUB]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vsub_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vsub ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vmul_i8
// CHECK-DAG: %[[LHS_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[LHS_F32:.*]] = hivm.hir.vcast ins(%[[LHS_F16]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[RHS_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS_F32:.*]] = hivm.hir.vcast ins(%[[RHS_F16]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[LHS_F32]], %[[RHS_F32]] : tensor<16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MUL_I32:.*]] = hivm.hir.vcast ins(%[[MUL]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xi32>) round_mode = <trunc> -> tensor<16xi32>
// CHECK: annotation.mark %[[MUL_I32]] {overflow_mode = "trunc"} : tensor<16xi32>
// CHECK: hivm.hir.vcast ins(%[[MUL_I32]] : tensor<16xi32>) outs(%{{.*}} : tensor<16xi8>) round_mode = <truncwithoverflow> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vmul_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vmul ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vmax_i8
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[MAX:.*]] = hivm.hir.vmax ins(%[[LHS]], %[[RHS]] : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[MAX]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vmax_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vmax ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vmin_i8
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[MIN:.*]] = hivm.hir.vmin ins(%[[LHS]], %[[RHS]] : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[MIN]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vmin_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vmin ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vabs_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[ABS:.*]] = hivm.hir.vabs ins(%[[SRC_F16]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[ABS]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vabs_i8(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vabs ins(%arg0 : tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vsel_i8
// CHECK-DAG: %[[TRUE_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[FALSE_F16:.*]] = hivm.hir.vcast ins(%arg2 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[SEL:.*]] = hivm.hir.vsel ins(%arg0, %[[TRUE_F16]], %[[FALSE_F16]] : tensor<16xi1>, tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[SEL]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vsel_i8(%arg0: tensor<16xi1>, %arg1: tensor<16xi8>, %arg2: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<16xi1>, tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vbrc_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1x16xi8>) outs(%{{.*}} : tensor<1x16xf16>) round_mode = <trunc> -> tensor<1x16xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<5x16xi8>) outs(%{{.*}} : tensor<5x16xf16>) round_mode = <trunc> -> tensor<5x16xf16>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[SRC_F16]] : tensor<1x16xf16>) outs(%[[DST_F16]] : tensor<5x16xf16>) broadcast_dims = [0] -> tensor<5x16xf16>
// CHECK: hivm.hir.vcast ins(%[[BRC]] : tensor<5x16xf16>) outs(%{{.*}} : tensor<5x16xi8>) round_mode = <trunc> -> tensor<5x16xi8>
func.func @test_NormalizeToTargetType_hivm_vbrc_i8(%arg0: tensor<1x16xi8>) -> tensor<5x16xi8> {
  %dst = tensor.empty() : tensor<5x16xi8>
  %0 = hivm.hir.vbrc ins(%arg0 : tensor<1x16xi8>) outs(%dst : tensor<5x16xi8>) broadcast_dims = [0] -> tensor<5x16xi8>
  return %0 : tensor<5x16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vinterleave_i8
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x16x1xi8>) outs(%{{.*}} : tensor<4x16x1xf16>) -> tensor<4x16x1xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<4x16x1xi8>) outs(%{{.*}} : tensor<4x16x1xf16>) -> tensor<4x16x1xf16>
// CHECK: %[[INTER:.*]] = hivm.hir.vinterleave ins(%[[LHS]], %[[RHS]] : tensor<4x16x1xf16>, tensor<4x16x1xf16>) outs(%{{.*}} : tensor<4x16x2xf16>) interleave_channel_nums = 2 -> tensor<4x16x2xf16>
// CHECK: hivm.hir.vcast ins(%[[INTER]] : tensor<4x16x2xf16>) outs(%{{.*}} : tensor<4x16x2xi8>) round_mode = <trunc> -> tensor<4x16x2xi8>
func.func @test_NormalizeToTargetType_hivm_vinterleave_i8(%arg0: tensor<4x16x1xi8>, %arg1: tensor<4x16x1xi8>) -> tensor<4x16x2xi8> {
  %dst = tensor.empty() : tensor<4x16x2xi8>
  %0 = hivm.hir.vinterleave ins(%arg0, %arg1 : tensor<4x16x1xi8>, tensor<4x16x1xi8>) outs(%dst : tensor<4x16x2xi8>) interleave_channel_nums = 2 -> tensor<4x16x2xi8>
  return %0 : tensor<4x16x2xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vdeinterleave_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<32xi8>) outs(%{{.*}} : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[DEINTER:.*]] = hivm.hir.vdeinterleave ins(%[[SRC_F16]] : tensor<32xf16>) outs(%{{.*}} : tensor<16xf16>) channel_num = 32 index_mode = <CHANNEL_0> -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[DEINTER]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vdeinterleave_i8(%arg0: tensor<32xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vdeinterleave ins(%arg0 : tensor<32xi8>) outs(%dst : tensor<16xi8>) channel_num = 32 index_mode = <CHANNEL_0> -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vreduce_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x16xi8>) outs(%{{.*}} : tensor<4x16xf16>) -> tensor<4x16xf16>
// CHECK: %[[SRC_F32:.*]] = hivm.hir.vcast ins(%[[SRC_F16]] : tensor<4x16xf16>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<1x16xi8>) outs(%{{.*}} : tensor<1x16xf16>) -> tensor<1x16xf16>
// CHECK: %[[DST_F32:.*]] = hivm.hir.vcast ins(%[[DST_F16]] : tensor<1x16xf16>) outs(%{{.*}} : tensor<1x16xf32>) -> tensor<1x16xf32>
// CHECK: %[[RED:.*]] = hivm.hir.vreduce <sum> ins(%[[SRC_F32]] : tensor<4x16xf32>) outs(%[[DST_F32]] : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
// CHECK: %[[RED_I32:.*]] = hivm.hir.vcast ins(%[[RED]] : tensor<1x16xf32>) outs(%{{.*}} : tensor<1x16xi32>) round_mode = <trunc> -> tensor<1x16xi32>
// CHECK: annotation.mark %[[RED_I32]] {overflow_mode = "trunc"} : tensor<1x16xi32>
// CHECK: hivm.hir.vcast ins(%[[RED_I32]] : tensor<1x16xi32>) outs(%{{.*}} : tensor<1x16xi8>) round_mode = <truncwithoverflow> -> tensor<1x16xi8>
func.func @test_NormalizeToTargetType_hivm_vreduce_i8(%arg0: tensor<4x16xi8>) -> tensor<1x16xi8> {
  %dst = tensor.empty() : tensor<1x16xi8>
  %0 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<4x16xi8>) outs(%dst : tensor<1x16xi8>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi8>
  return %0 : tensor<1x16xi8>
}
// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vgather_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<64xi8>) outs(%{{.*}} : tensor<64xf16>) -> tensor<64xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<32xi8>) outs(%{{.*}} : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[GATHER:.*]] = hivm.hir.vgather ins(%[[SRC_F16]] : tensor<64xf16>) indices(%arg1 : tensor<32xi32>) outs(%[[DST_F16]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: hivm.hir.vcast ins(%[[GATHER]] : tensor<32xf16>) outs(%{{.*}} : tensor<32xi8>) round_mode = <trunc> -> tensor<32xi8>
func.func @test_NormalizeToTargetType_hivm_vgather_i8(%arg0: tensor<64xi8>, %arg1: tensor<32xi32>) -> tensor<32xi8> {
  %dst = tensor.empty() : tensor<32xi8>
  %0 = hivm.hir.vgather ins(%arg0 : tensor<64xi8>) indices(%arg1 : tensor<32xi32>) outs(%dst : tensor<32xi8>) -> tensor<32xi8>
  return %0 : tensor<32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vmod_i8
// CHECK-DAG: hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK-DAG: hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK: %[[MOD:.*]] = hivm.hir.vmod ins(%{{.*}}, %{{.*}} : tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi16>) -> tensor<16xi16>
// CHECK: hivm.hir.vcast ins(%[[MOD]] : tensor<16xi16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <truncwithoverflow> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vmod_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vmod ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vmodui_i8
// CHECK-DAG: hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) cast = <cast_unsigned> -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK-DAG: hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) cast = <cast_unsigned> -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK: %[[MOD:.*]] = hivm.hir.vmodui ins(%{{.*}}, %{{.*}} : tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi16>) -> tensor<16xi16>
// CHECK: hivm.hir.vcast ins(%[[MOD]] : tensor<16xi16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <truncwithoverflow> cast = <cast_unsigned> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_hivm_vmodui_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %0 = hivm.hir.vmodui ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// Ascend950 natively supports i8 add/sub/max/min execution.
// Verify these ops are NOT widened, matching HFusion's behavior.

// CHECK-LABEL: func.func @test_DoNotNormalizeToTargetType_hivm_vadd_i8_ascend950
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vadd ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%{{.*}} : tensor<16xi8>) -> tensor<16xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_DoNotNormalizeToTargetType_hivm_vadd_i8_ascend950(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
    %dst = tensor.empty() : tensor<16xi8>
    %0 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
    return %0 : tensor<16xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeToTargetType_hivm_vsub_i8_ascend950
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vsub ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%{{.*}} : tensor<16xi8>) -> tensor<16xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_DoNotNormalizeToTargetType_hivm_vsub_i8_ascend950(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
    %dst = tensor.empty() : tensor<16xi8>
    %0 = hivm.hir.vsub ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
    return %0 : tensor<16xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeToTargetType_hivm_vmax_i8_ascend950
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vmax ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%{{.*}} : tensor<16xi8>) -> tensor<16xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_DoNotNormalizeToTargetType_hivm_vmax_i8_ascend950(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
    %dst = tensor.empty() : tensor<16xi8>
    %0 = hivm.hir.vmax ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
    return %0 : tensor<16xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeToTargetType_hivm_vmin_i8_ascend950
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vmin ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%{{.*}} : tensor<16xi8>) -> tensor<16xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_DoNotNormalizeToTargetType_hivm_vmin_i8_ascend950(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
    %dst = tensor.empty() : tensor<16xi8>
    %0 = hivm.hir.vmin ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>) outs(%dst : tensor<16xi8>) -> tensor<16xi8>
    return %0 : tensor<16xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeF16ToF32Type_hivm_vln_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[EMPTY_IN_F32:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[EMPTY_IN_F32]] : tensor<16xf32>)
// CHECK: %[[EMPTY_OUT_F32:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%[[EMPTY_F16]] : tensor<16xf16>) outs(%[[EMPTY_OUT_F32]] : tensor<16xf32>)
// CHECK: %[[VLN:.*]] = hivm.hir.vln ins(%[[CAST_IN]] : tensor<16xf32>) outs(%[[CAST_OUT]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY_OUT_F16:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[VLN]] : tensor<16xf32>) outs(%[[EMPTY_OUT_F16]] : tensor<16xf16>)
// CHECK: return %[[RES]]
func.func @test_NormalizeF16ToF32Type_hivm_vln_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hivm.hir.vln ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeF16ToF32Type_hivm_vpow_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>, %[[ARG1:.*]]: tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>)
// CHECK: hivm.hir.vcast ins(%[[ARG1]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>)
// CHECK-NOT: hivm.hir.vpow
// CHECK: hivm.hir.vln
// CHECK: hivm.hir.vexp
// CHECK: %[[EMPTY_OUT_F16:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<16xf32>) outs(%[[EMPTY_OUT_F16]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[RES]]
func.func @test_NormalizeF16ToF32Type_hivm_vpow_f16(%arg0: tensor<16xf16>, %arg1: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<16xf16>, tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// PrimaryMath decomposes vrsqrt into vsqrt+vrec before the f16->f32 type
// conversion widens the vsqrt result. This order matches the HIVM pipeline.
// CHECK-LABEL: func.func @test_NormalizeF16ToF32Type_hivm_vrsqrt_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[EMPTY_F16_0:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[EMPTY_F32_0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[EMPTY_F32_0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY_F32_1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST_DST:.*]] = hivm.hir.vcast ins(%[[EMPTY_F16_0]] : tensor<16xf16>) outs(%[[EMPTY_F32_1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY_F32_2:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[SQRT:.*]] = hivm.hir.vsqrt ins(%[[CAST_IN]] : tensor<16xf32>) outs(%[[EMPTY_F32_2]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[REC:.*]] = hivm.hir.vrec ins(%[[SQRT]] : tensor<16xf32>) outs(%[[CAST_DST]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY_F16_1:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[CAST_BACK:.*]] = hivm.hir.vcast ins(%[[REC]] : tensor<16xf32>) outs(%[[EMPTY_F16_1]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[CAST_BACK]]
func.func @test_NormalizeF16ToF32Type_hivm_vrsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hivm.hir.vrsqrt ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeF16ToF32Type_hivm_vln_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[VLN:.*]] = hivm.hir.vln ins(%[[ARG0]] : tensor<16xf32>) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK-NOT: hivm.hir.vcast
// CHECK: return %[[VLN]]
func.func @test_DoNotNormalizeF16ToF32Type_hivm_vln_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeF16ToF32Type_hivm_vln_broadcast_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x16xf16>) -> tensor<5x16xf16>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<5x16xf16>
// CHECK: %[[VLN:.*]] = hivm.hir.vln ins(%[[ARG0]] : tensor<1x16xf16>) outs(%[[EMPTY]] : tensor<5x16xf16>) broadcast = [0] -> tensor<5x16xf16>
// CHECK-NOT: hivm.hir.vcast
// CHECK: return %[[VLN]]
func.func @test_DoNotNormalizeF16ToF32Type_hivm_vln_broadcast_f16(%arg0: tensor<1x16xf16>) -> tensor<5x16xf16> {
  %0 = tensor.empty() : tensor<5x16xf16>
  %1 = hivm.hir.vln ins(%arg0 : tensor<1x16xf16>) outs(%0 : tensor<5x16xf16>) broadcast = [0] -> tensor<5x16xf16>
  return %1 : tensor<5x16xf16>
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeF16ToF32Type_hivm_vrsqrt_transpose_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x4x1x8xf16>) -> tensor<4x1x1x8xf16>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x1x1x8xf16>
// CHECK: %[[VRSQRT:.*]] = hivm.hir.vrsqrt ins(%[[ARG0]] : tensor<1x4x1x8xf16>) outs(%[[EMPTY]] : tensor<4x1x1x8xf16>) transpose = [1, 0, 2, 3] -> tensor<4x1x1x8xf16>
// CHECK-NOT: hivm.hir.vcast
// CHECK: return %[[VRSQRT]]
func.func @test_DoNotNormalizeF16ToF32Type_hivm_vrsqrt_transpose_f16(%arg0: tensor<1x4x1x8xf16>) -> tensor<4x1x1x8xf16> {
  %0 = tensor.empty() : tensor<4x1x1x8xf16>
  %1 = hivm.hir.vrsqrt ins(%arg0 : tensor<1x4x1x8xf16>) outs(%0 : tensor<4x1x1x8xf16>) transpose = [1, 0, 2, 3] -> tensor<4x1x1x8xf16>
  return %1 : tensor<4x1x1x8xf16>
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeF16ToF32Type_hivm_vln_memref_f16
// CHECK-SAME: (%[[SRC:.*]]: memref<?x?xf16>, %[[DST:.*]]: memref<?x?xf16>)
// CHECK: hivm.hir.vln ins(%[[SRC]] : memref<?x?xf16>) outs(%[[DST]] : memref<?x?xf16>)
// CHECK-NOT: hivm.hir.vcast
func.func @test_DoNotNormalizeF16ToF32Type_hivm_vln_memref_f16(%src : memref<?x?xf16>, %dst : memref<?x?xf16>) {
  hivm.hir.vln ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCumOpF16ToF32Type_hivm_vcumsum_f16
// CHECK: %[[SRC_F32:.*]] = hivm.hir.vcast ins(%[[SRC:.*]] : tensor<4x64xf16>) outs(%{{.*}} : tensor<4x64xf32>)
// CHECK: %[[DST_F32:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4x32xf16>) outs(%{{.*}} : tensor<4x32xf32>)
// CHECK: %[[CUM:.*]] = hivm.hir.vcumsum ins(%[[SRC_F32]] : tensor<4x64xf32>) outs(%[[DST_F32]] : tensor<4x32xf32>) cum_dims = [0] reverse = true -> tensor<4x32xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[CUM]] : tensor<4x32xf32>) outs(%{{.*}} : tensor<4x32xf16>) -> tensor<4x32xf16>
func.func @test_NormalizeCumOpF16ToF32Type_hivm_vcumsum_f16(%src : tensor<4x64xf16>) -> tensor<4x32xf16> {
  %dst = tensor.empty() : tensor<4x32xf16>
  %0 = hivm.hir.vcumsum ins(%src : tensor<4x64xf16>) outs(%dst : tensor<4x32xf16>) cum_dims = [0] reverse = true -> tensor<4x32xf16>
  return %0 : tensor<4x32xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCumOpF16ToF32Type_hivm_vcumprod_f16
// CHECK: %[[SRC_F32:.*]] = hivm.hir.vcast ins(%[[SRC:.*]] : tensor<4x64xf16>) outs(%{{.*}} : tensor<4x64xf32>)
// CHECK: %[[DST_F32:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4x32xf16>) outs(%{{.*}} : tensor<4x32xf32>)
// CHECK: %[[CUM:.*]] = hivm.hir.vcumprod ins(%[[SRC_F32]] : tensor<4x64xf32>) outs(%[[DST_F32]] : tensor<4x32xf32>) cum_dims = [1] reverse = true -> tensor<4x32xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[CUM]] : tensor<4x32xf32>) outs(%{{.*}} : tensor<4x32xf16>) -> tensor<4x32xf16>
func.func @test_NormalizeCumOpF16ToF32Type_hivm_vcumprod_f16(%src : tensor<4x64xf16>) -> tensor<4x32xf16> {
  %dst = tensor.empty() : tensor<4x32xf16>
  %0 = hivm.hir.vcumprod ins(%src : tensor<4x64xf16>) outs(%dst : tensor<4x32xf16>) cum_dims = [1] reverse = true -> tensor<4x32xf16>
  return %0 : tensor<4x32xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCumOpI8ToTargetType_hivm_vcumsum_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4x64xi8>) outs(%{{.*}} : tensor<4x64xf16>) -> tensor<4x64xf16>
// CHECK: %[[SRC_F32:.*]] = hivm.hir.vcast ins(%[[SRC_F16]] : tensor<4x64xf16>) outs(%{{.*}} : tensor<4x64xf32>) -> tensor<4x64xf32>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4x32xi8>) outs(%{{.*}} : tensor<4x32xf16>) -> tensor<4x32xf16>
// CHECK: %[[DST_F32:.*]] = hivm.hir.vcast ins(%[[DST_F16]] : tensor<4x32xf16>) outs(%{{.*}} : tensor<4x32xf32>) -> tensor<4x32xf32>
// CHECK: %[[CUM:.*]] = hivm.hir.vcumsum ins(%[[SRC_F32]] : tensor<4x64xf32>) outs(%[[DST_F32]] : tensor<4x32xf32>) cum_dims = [0] reverse = true -> tensor<4x32xf32>
// CHECK: %[[CUM_I32:.*]] = hivm.hir.vcast ins(%[[CUM]] : tensor<4x32xf32>) outs(%{{.*}} : tensor<4x32xi32>) round_mode = <trunc> -> tensor<4x32xi32>
// CHECK: annotation.mark %[[CUM_I32]] {overflow_mode = "trunc"} : tensor<4x32xi32>
// CHECK: hivm.hir.vcast ins(%[[CUM_I32]] : tensor<4x32xi32>) outs(%{{.*}} : tensor<4x32xi8>) round_mode = <truncwithoverflow> -> tensor<4x32xi8>
func.func @test_NormalizeCumOpI8ToTargetType_hivm_vcumsum_i8(%src : tensor<4x64xi8>) -> tensor<4x32xi8> {
  %dst = tensor.empty() : tensor<4x32xi8>
  %0 = hivm.hir.vcumsum ins(%src : tensor<4x64xi8>) outs(%dst : tensor<4x32xi8>) cum_dims = [0] reverse = true -> tensor<4x32xi8>
  return %0 : tensor<4x32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCumOpI8ToTargetType_hivm_vcumprod_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4x64xi8>) outs(%{{.*}} : tensor<4x64xf16>) -> tensor<4x64xf16>
// CHECK: %[[SRC_F32:.*]] = hivm.hir.vcast ins(%[[SRC_F16]] : tensor<4x64xf16>) outs(%{{.*}} : tensor<4x64xf32>) -> tensor<4x64xf32>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4x32xi8>) outs(%{{.*}} : tensor<4x32xf16>) -> tensor<4x32xf16>
// CHECK: %[[DST_F32:.*]] = hivm.hir.vcast ins(%[[DST_F16]] : tensor<4x32xf16>) outs(%{{.*}} : tensor<4x32xf32>) -> tensor<4x32xf32>
// CHECK: %[[CUM:.*]] = hivm.hir.vcumprod ins(%[[SRC_F32]] : tensor<4x64xf32>) outs(%[[DST_F32]] : tensor<4x32xf32>) cum_dims = [1] reverse = true -> tensor<4x32xf32>
// CHECK: %[[CUM_I32:.*]] = hivm.hir.vcast ins(%[[CUM]] : tensor<4x32xf32>) outs(%{{.*}} : tensor<4x32xi32>) round_mode = <trunc> -> tensor<4x32xi32>
// CHECK: annotation.mark %[[CUM_I32]] {overflow_mode = "trunc"} : tensor<4x32xi32>
// CHECK: hivm.hir.vcast ins(%[[CUM_I32]] : tensor<4x32xi32>) outs(%{{.*}} : tensor<4x32xi8>) round_mode = <truncwithoverflow> -> tensor<4x32xi8>
func.func @test_NormalizeCumOpI8ToTargetType_hivm_vcumprod_i8(%src : tensor<4x64xi8>) -> tensor<4x32xi8> {
  %dst = tensor.empty() : tensor<4x32xi8>
  %0 = hivm.hir.vcumprod ins(%src : tensor<4x64xi8>) outs(%dst : tensor<4x32xi8>) cum_dims = [1] reverse = true -> tensor<4x32xi8>
  return %0 : tensor<4x32xi8>
}

// -----

// The I1SelectTemplate is only registered on regbased architectures.
// CHECK-LABEL: func.func @test_NormalizeToTargetType_hivm_vsel_i1
// CHECK: %[[TRUE_I16:.*]] = hivm.hir.vsel ins(%arg1, %{{.*}}, %{{.*}} : tensor<16xi1>, tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi16>) -> tensor<16xi16>
// CHECK: %[[FALSE_I16:.*]] = hivm.hir.vsel ins(%arg2, %{{.*}}, %{{.*}} : tensor<16xi1>, tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi16>) -> tensor<16xi16>
// CHECK: %[[SELECT_I16:.*]] = hivm.hir.vsel ins(%arg0, %[[TRUE_I16]], %[[FALSE_I16]] : tensor<16xi1>, tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi16>) -> tensor<16xi16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[SELECT_I16]], {{.*}} : tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi1>) compare_mode = <ne> -> tensor<16xi1>
// CHECK: return %[[CMP]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeToTargetType_hivm_vsel_i1(%arg0: tensor<16xi1>, %arg1: tensor<16xi1>, %arg2: tensor<16xi1>) -> tensor<16xi1> {
    %dst = tensor.empty() : tensor<16xi1>
    %0 = hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<16xi1>, tensor<16xi1>, tensor<16xi1>) outs(%dst : tensor<16xi1>) -> tensor<16xi1>
    return %0 : tensor<16xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceBoolToLogical_hivm_vreduce_sum_i1
// CHECK: hivm.hir.vreduce <any> ins(%arg0 : tensor<8xi1>) outs(%{{.*}} : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_ReduceBoolToLogical_hivm_vreduce_sum_i1(%arg0: tensor<8xi1>) -> tensor<1xi1> {
    %dst = tensor.empty() : tensor<1xi1>
    %0 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<8xi1>) outs(%dst : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceBoolToLogical_hivm_vreduce_prod_i1
// CHECK: hivm.hir.vreduce <all> ins(%arg0 : tensor<8xi1>) outs(%{{.*}} : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_ReduceBoolToLogical_hivm_vreduce_prod_i1(%arg0: tensor<8xi1>) -> tensor<1xi1> {
    %dst = tensor.empty() : tensor<1xi1>
    %0 = hivm.hir.vreduce <prod> ins(%arg0 : tensor<8xi1>) outs(%dst : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceBoolToLogical_hivm_vreduce_max_i1
// CHECK: hivm.hir.vreduce <any> ins(%arg0 : tensor<8xi1>) outs(%{{.*}} : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_ReduceBoolToLogical_hivm_vreduce_max_i1(%arg0: tensor<8xi1>) -> tensor<1xi1> {
    %dst = tensor.empty() : tensor<1xi1>
    %0 = hivm.hir.vreduce <max> ins(%arg0 : tensor<8xi1>) outs(%dst : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceBoolToLogical_hivm_vreduce_min_i1
// CHECK: hivm.hir.vreduce <all> ins(%arg0 : tensor<8xi1>) outs(%{{.*}} : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_ReduceBoolToLogical_hivm_vreduce_min_i1(%arg0: tensor<8xi1>) -> tensor<1xi1> {
    %dst = tensor.empty() : tensor<1xi1>
    %0 = hivm.hir.vreduce <min> ins(%arg0 : tensor<8xi1>) outs(%dst : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceI1AddToSelectMaxCompare_hivm_vreduce_sum_i1
// CHECK: %[[ONES:.*]] = linalg.fill
// CHECK: %[[ZEROS:.*]] = linalg.fill
// CHECK: %[[SEL:.*]] = hivm.hir.vsel ins(%arg0, %[[ONES]], %[[ZEROS]] : tensor<8xi1>, tensor<8xi16>, tensor<8xi16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[RED:.*]] = hivm.hir.vreduce <max> ins(%[[SEL]] : tensor<8xi16>) outs(%{{.*}} : tensor<1xi16>) unsigned_src = false reduce_dims = [0] -> tensor<1xi16>
// CHECK: hivm.hir.vcmp ins(%[[RED]], {{.*}} : tensor<1xi16>, i16) outs(%{{.*}} : tensor<1xi1>) compare_mode = <ne> -> tensor<1xi1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_ReduceI1AddToSelectMaxCompare_hivm_vreduce_sum_i1(%arg0: tensor<8xi1>) -> tensor<1xi1> {
    %dst = tensor.empty() : tensor<1xi1>
    %0 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<8xi1>) outs(%dst : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceI1AndOrToI16_hivm_vreduce_any_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1x1024xi1>) outs(%{{.*}} : tensor<1x1024xf16>) round_mode = <trunc> -> tensor<1x1024xf16>
// CHECK: %[[SRC_I16:.*]] = hivm.hir.vcast ins(%[[SRC_F16]] : tensor<1x1024xf16>) outs(%{{.*}} : tensor<1x1024xi16>) round_mode = <trunc> -> tensor<1x1024xi16>
// CHECK: %[[INIT_I16:.*]] = linalg.fill
// CHECK: %[[RED:.*]] = hivm.hir.vreduce <max> ins(%[[SRC_I16]] : tensor<1x1024xi16>) outs(%[[INIT_I16]] : tensor<1x1xi16>) unsigned_src = false reduce_dims = [1] -> tensor<1x1xi16>
// CHECK: %[[RED_F16:.*]] = hivm.hir.vcast ins(%[[RED]] : tensor<1x1xi16>) outs(%{{.*}} : tensor<1x1xf16>) -> tensor<1x1xf16>
// CHECK: %[[ZERO_F16:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<1x1xf16>) -> tensor<1x1xf16>
// CHECK: hivm.hir.vcmp ins(%[[RED_F16]], %[[ZERO_F16]] : tensor<1x1xf16>, tensor<1x1xf16>) outs(%{{.*}} : tensor<1x1xi1>) compare_mode = <ne> -> tensor<1x1xi1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_ReduceI1AndOrToI16_hivm_vreduce_any_i1(%arg0: tensor<1x1024xi1>) -> tensor<1x1xi1> {
    %dst = tensor.empty() : tensor<1x1xi1>
    %0 = hivm.hir.vreduce <any> ins(%arg0 : tensor<1x1024xi1>) outs(%dst : tensor<1x1xi1>) unsigned_src = false reduce_dims = [1] -> tensor<1x1xi1>
    return %0 : tensor<1x1xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceI1AndOrToI16_hivm_vreduce_all_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1x1024xi1>) outs(%{{.*}} : tensor<1x1024xf16>) round_mode = <trunc> -> tensor<1x1024xf16>
// CHECK: %[[SRC_I16:.*]] = hivm.hir.vcast ins(%[[SRC_F16]] : tensor<1x1024xf16>) outs(%{{.*}} : tensor<1x1024xi16>) round_mode = <trunc> -> tensor<1x1024xi16>
// CHECK: %[[INIT_I16:.*]] = linalg.fill
// CHECK: %[[RED:.*]] = hivm.hir.vreduce <min> ins(%[[SRC_I16]] : tensor<1x1024xi16>) outs(%[[INIT_I16]] : tensor<1x1xi16>) unsigned_src = false reduce_dims = [1] -> tensor<1x1xi16>
// CHECK: %[[RED_F16:.*]] = hivm.hir.vcast ins(%[[RED]] : tensor<1x1xi16>) outs(%{{.*}} : tensor<1x1xf16>) -> tensor<1x1xf16>
// CHECK: %[[ZERO_F16:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<1x1xf16>) -> tensor<1x1xf16>
// CHECK: hivm.hir.vcmp ins(%[[RED_F16]], %[[ZERO_F16]] : tensor<1x1xf16>, tensor<1x1xf16>) outs(%{{.*}} : tensor<1x1xi1>) compare_mode = <ne> -> tensor<1x1xi1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_ReduceI1AndOrToI16_hivm_vreduce_all_i1(%arg0: tensor<1x1024xi1>) -> tensor<1x1xi1> {
    %dst = tensor.empty() : tensor<1x1xi1>
    %0 = hivm.hir.vreduce <all> ins(%arg0 : tensor<1x1024xi1>) outs(%dst : tensor<1x1xi1>) unsigned_src = false reduce_dims = [1] -> tensor<1x1xi1>
    return %0 : tensor<1x1xi1>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceNormalize910_95_hivm_vreduce_sum_i8
// CHECK: %[[SRC_I16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<5x8x7xi8>) outs(%{{.*}} : tensor<5x8x7xi16>) -> tensor<5x8x7xi16>
// CHECK: %[[DST_I16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<1x8x1xi8>) outs(%{{.*}} : tensor<1x8x1xi16>) -> tensor<1x8x1xi16>
// CHECK: %[[RED:.*]] = hivm.hir.vreduce <sum> ins(%[[SRC_I16]] : tensor<5x8x7xi16>) outs(%[[DST_I16]] : tensor<1x8x1xi16>) unsigned_src = false reduce_dims = [0, 2] -> tensor<1x8x1xi16>
// CHECK: hivm.hir.vcast ins(%[[RED]] : tensor<1x8x1xi16>) outs(%{{.*}} : tensor<1x8x1xi8>) round_mode = <truncwithoverflow> -> tensor<1x8x1xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_ReduceNormalize910_95_hivm_vreduce_sum_i8(%arg0: tensor<5x8x7xi8>) -> tensor<1x8x1xi8> {
    %dst = tensor.empty() : tensor<1x8x1xi8>
    %0 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<5x8x7xi8>) outs(%dst : tensor<1x8x1xi8>) unsigned_src = false reduce_dims = [0, 2] -> tensor<1x8x1xi8>
    return %0 : tensor<1x8x1xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeGatherIndexToI32_hivm_vgather_i16
// CHECK: hivm.hir.vcast ins(%arg1 : tensor<32xi16>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: hivm.hir.vcast ins(%{{.*}} : tensor<32xf32>) outs(%{{.*}} : tensor<32xi32>) round_mode = <trunc>
// CHECK: hivm.hir.vgather ins(%arg0 : tensor<64xf16>) indices(%{{.*}} : tensor<32xi32>)
func.func @test_NormalizeGatherIndexToI32_hivm_vgather_i16(%arg0: tensor<64xf16>, %arg1: tensor<32xi16>) -> tensor<32xf16> {
  %dst = tensor.empty() : tensor<32xf16>
  %0 = hivm.hir.vgather ins(%arg0 : tensor<64xf16>) indices(%arg1 : tensor<32xi16>) outs(%dst : tensor<32xf16>) -> tensor<32xf16>
  return %0 : tensor<32xf16>
}

// -----

// CHECK-LABEL: func.func @test_DoNotNormalizeGatherIndexToI32_hivm_vgather_i32
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vgather ins(%arg0 : tensor<64xf16>) indices(%arg1 : tensor<32xi32>)
func.func @test_DoNotNormalizeGatherIndexToI32_hivm_vgather_i32(%arg0: tensor<64xf16>, %arg1: tensor<32xi32>) -> tensor<32xf16> {
  %dst = tensor.empty() : tensor<32xf16>
  %0 = hivm.hir.vgather ins(%arg0 : tensor<64xf16>) indices(%arg1 : tensor<32xi32>) outs(%dst : tensor<32xf16>) -> tensor<32xf16>
  return %0 : tensor<32xf16>
}
