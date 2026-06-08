// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeToTargetType_compare_i1
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<32xi1>) outs(%{{.*}} : tensor<32xf16>) round_mode = <trunc> -> tensor<32xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<32xi1>) outs(%{{.*}} : tensor<32xf16>) round_mode = <trunc> -> tensor<32xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[LHS]], %[[RHS]] : tensor<32xf16>, tensor<32xf16>) outs(%{{.*}} : tensor<32xi1>) -> tensor<32xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<32xi1>) outs(%{{.*}} : tensor<32xi1>) -> tensor<32xi1>
func.func @test_NormalizeToTargetType_compare_i1(%arg0: tensor<32xi1>, %arg1: tensor<32xi1>) -> tensor<32xi1> {
  %0 = tensor.empty() : tensor<32xi1>
  %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg1 : tensor<32xi1>, tensor<32xi1>)
    outs(%0 : tensor<32xi1>) -> tensor<32xi1>
  return %1 : tensor<32xi1>
}
// CHECK-LABEL: func.func @test_NormalizeToTargetType_select_i8
// CHECK-DAG: %[[TRUE_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[FALSE_F16:.*]] = hivm.hir.vcast ins(%arg2 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[SEL:.*]] = hivm.hir.vsel ins(%arg0, %[[TRUE_F16]], %[[FALSE_F16]] : tensor<16xi1>, tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[SEL]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_select_i8(%arg0: tensor<16xi1>, %arg1: tensor<16xi8>, %arg2: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.select
    ins(%arg0, %arg1, %arg2 : tensor<16xi1>, tensor<16xi8>, tensor<16xi8>)
    outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_broadcast_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<8x1xi1>) outs(%{{.*}} : tensor<8x1xf16>) round_mode = <trunc> -> tensor<8x1xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<8x16xi1>) outs(%{{.*}} : tensor<8x16xf16>) round_mode = <trunc> -> tensor<8x16xf16>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[SRC_F16]] : tensor<8x1xf16>) outs(%[[DST_F16]] : tensor<8x16xf16>) broadcast_dims = [1] -> tensor<8x16xf16>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<8x16xf16>) -> tensor<8x16xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[BRC]], %[[ZERO_BRC]] : tensor<8x16xf16>, tensor<8x16xf16>) outs(%{{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<8x16xi1>) outs(%{{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
func.func @test_NormalizeToTargetType_broadcast_i1(%arg0: tensor<8xi1>) -> tensor<8x16xi1> {
  %0 = tensor.empty() : tensor<8x16xi1>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi1>)
    outs(%0 : tensor<8x16xi1>)
    dimensions = [1]
  return %1 : tensor<8x16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_broadcast_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<8x1xi8>) outs(%{{.*}} : tensor<8x1xf16>) round_mode = <trunc> -> tensor<8x1xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<8x16xi8>) outs(%{{.*}} : tensor<8x16xf16>) round_mode = <trunc> -> tensor<8x16xf16>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[SRC_F16]] : tensor<8x1xf16>) outs(%[[DST_F16]] : tensor<8x16xf16>) broadcast_dims = [1] -> tensor<8x16xf16>
// CHECK: hivm.hir.vcast ins(%[[BRC]] : tensor<8x16xf16>) outs(%{{.*}} : tensor<8x16xi8>) round_mode = <trunc> -> tensor<8x16xi8>
func.func @test_NormalizeToTargetType_broadcast_i8(%arg0: tensor<8xi8>) -> tensor<8x16xi8> {
  %0 = tensor.empty() : tensor<8x16xi8>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi8>)
    outs(%0 : tensor<8x16xi8>)
    dimensions = [1]
  return %1 : tensor<8x16xi8>
}
// CHECK-LABEL: func.func @test_NormalizeToTargetType_transpose_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<32x8xi1>) outs(%{{.*}} : tensor<32x8xf16>) round_mode = <trunc> -> tensor<32x8xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<8x32xi1>) outs(%{{.*}} : tensor<8x32xf16>) round_mode = <trunc> -> tensor<8x32xf16>
// CHECK: %[[TRANS:.*]] = hivm.hir.vtranspose ins(%[[SRC_F16]] : tensor<32x8xf16>) outs(%[[DST_F16]] : tensor<8x32xf16>) permutation = [1, 0] -> tensor<8x32xf16>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<8x32xf16>) -> tensor<8x32xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[TRANS]], %[[ZERO_BRC]] : tensor<8x32xf16>, tensor<8x32xf16>) outs(%{{.*}} : tensor<8x32xi1>) -> tensor<8x32xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<8x32xi1>) outs(%{{.*}} : tensor<8x32xi1>) -> tensor<8x32xi1>
func.func @test_NormalizeToTargetType_transpose_i1(%arg0: tensor<32x8xi1>) -> tensor<8x32xi1> {
  %dst = tensor.empty() : tensor<8x32xi1>
  %transposed = linalg.transpose ins(%arg0 : tensor<32x8xi1>) outs(%dst : tensor<8x32xi1>) permutation = [1, 0]
  return %transposed : tensor<8x32xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_interleave_i1
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x2x32xi1>) outs(%{{.*}} : tensor<4x2x32xf16>) round_mode = <trunc> -> tensor<4x2x32xf16>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<4x2x32xi1>) outs(%{{.*}} : tensor<4x2x32xf16>) round_mode = <trunc> -> tensor<4x2x32xf16>
// CHECK: %[[INTER:.*]] = hivm.hir.vinterleave ins(%[[LHS]], %[[RHS]] : tensor<4x2x32xf16>, tensor<4x2x32xf16>) outs(%{{.*}} : tensor<4x2x64xf16>) interleave_channel_nums = 2 -> tensor<4x2x64xf16>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[INTER]], %[[ZERO_BRC]] : tensor<4x2x64xf16>, tensor<4x2x64xf16>) outs(%{{.*}} : tensor<4x2x64xi1>) -> tensor<4x2x64xi1>
// CHECK: hivm.hir.vnot ins(%[[CMP]] : tensor<4x2x64xi1>) outs(%{{.*}} : tensor<4x2x64xi1>) -> tensor<4x2x64xi1>
func.func @test_NormalizeToTargetType_interleave_i1(%arg0: tensor<4x2x32xi1>, %arg1: tensor<4x2x32xi1>) -> tensor<4x2x64xi1> {
  %1 = tensor.empty() : tensor<4x2x64xi1>
  %2 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi1>, tensor<4x2x32xi1> -> tensor<4x2x64xi1>
  return %2 : tensor<4x2x64xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_interleave_i8
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x2x32xi8>) outs(%{{.*}} : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<4x2x32xi8>) outs(%{{.*}} : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: %[[INTER:.*]] = hivm.hir.vinterleave ins(%[[LHS]], %[[RHS]] : tensor<4x2x32xf16>, tensor<4x2x32xf16>) outs(%{{.*}} : tensor<4x2x64xf16>) interleave_channel_nums = 2 -> tensor<4x2x64xf16>
// CHECK: hivm.hir.vcast ins(%[[INTER]] : tensor<4x2x64xf16>) outs(%{{.*}} : tensor<4x2x64xi8>) round_mode = <trunc> -> tensor<4x2x64xi8>
func.func @test_NormalizeToTargetType_interleave_i8(%arg0: tensor<4x2x32xi8>, %arg1: tensor<4x2x32xi8>) -> tensor<4x2x64xi8> {
  %2 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi8>, tensor<4x2x32xi8> -> tensor<4x2x64xi8>
  return %2 : tensor<4x2x64xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_deinterleave_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<32xi8>) outs(%{{.*}} : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[DEINTER:.*]] = hivm.hir.vdeinterleave ins(%[[SRC_F16]] : tensor<32xf16>) outs(%{{.*}} : tensor<16xf16>) index_mode = <CHANNEL_0> -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[DEINTER]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_deinterleave_i8(%arg0: tensor<32xi8>) -> tensor<16xi8> {
  %1 = hfusion.deinterleave %arg0 channel<0> : tensor<32xi8> -> tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_reduce_with_index_i1
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<32x32xi1>) outs(%{{.*}} : tensor<32x32xf16>) round_mode = <trunc> -> tensor<32x32xf16>
// CHECK: %[[MASKED:.*]] = hivm.hir.vsel ins(%{{.*}}, %{{.*}}, %[[SRC_F16]] : tensor<32x32xi1>, f16, tensor<32x32xf16>) outs(%{{.*}} : tensor<32x32xf16>) -> tensor<32x32xf16>
// CHECK: %[[RED:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <min_with_index> ins(%[[MASKED]] : tensor<32x32xf16>) outs(%{{.*}}, %{{.*}} : tensor<32x1xf16>, tensor<32x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<32x1xf16>, tensor<32x1xi32>
// CHECK: %[[ZERO_BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%{{.*}} : tensor<32x1xf16>) -> tensor<32x1xf16>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[RED]]#0, %[[ZERO_BRC]] : tensor<32x1xf16>, tensor<32x1xf16>) outs(%{{.*}} : tensor<32x1xi1>) -> tensor<32x1xi1>
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<32x1xi1>) outs(%{{.*}} : tensor<32x1xi1>) -> tensor<32x1xi1
// CHECK: %[[BOOL:.*]] = tensor.collapse_shape %[[NOT]]
// CHECK-SAME: tensor<32x1xi1> into tensor<32xi1>
// CHECK: %[[IDX:.*]] = tensor.collapse_shape %{{.*}}#1
// CHECK-SAME: tensor<32x1xi32> into tensor<32xi32>
func.func @test_NormalizeToTargetType_reduce_with_index_i1(%arg0: tensor<32x32xi1>, %arg1: tensor<32x32xi32>) -> (tensor<32xi1>, tensor<32xi32>) {
  %0 = tensor.empty() : tensor<32xi1>
  %1 = tensor.empty() : tensor<32xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                ins(%arg0, %arg1 : tensor<32x32xi1>, tensor<32x32xi32>)
                outs(%0, %1 : tensor<32xi1>, tensor<32xi32>)
                dimensions = [1] -> tensor<32xi1>, tensor<32xi32>
  return %reduced#0, %reduced#1 : tensor<32xi1>, tensor<32xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_reduce_with_index_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x64xi8>) outs(%{{.*}} : tensor<4x64xf16>) -> tensor<4x64xf16>
// CHECK: %[[MASKED:.*]] = hivm.hir.vsel ins(%{{.*}}, %{{.*}}, %[[SRC_F16]] : tensor<4x64xi1>, f16, tensor<4x64xf16>) outs(%{{.*}} : tensor<4x64xf16>) -> tensor<4x64xf16>
// CHECK: %[[RED:.*]]:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%[[MASKED]] : tensor<4x64xf16>) outs(%{{.*}}, %{{.*}} : tensor<4x1xf16>, tensor<4x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<4x1xf16>, tensor<4x1xi32>
// CHECK: hivm.hir.vcast ins(%{{.*}}#0 : tensor<4x1xf16>) outs(%{{.*}} : tensor<4x1xi8>) round_mode = <trunc> -> tensor<4x1xi8>
func.func @test_NormalizeToTargetType_reduce_with_index_i8(%arg0: tensor<4x64xi8>, %arg1: tensor<4x64xi32>) -> (tensor<4xi8>, tensor<4xi32>) {
  %0 = tensor.empty() : tensor<4xi8>
  %1 = tensor.empty() : tensor<4xi32>
  %2:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%arg0, %arg1 : tensor<4x64xi8>, tensor<4x64xi32>) outs(%0, %1 : tensor<4xi8>, tensor<4xi32>) dimensions = [1] -> tensor<4xi8>, tensor<4xi32>
  return %2#0, %2#1 : tensor<4xi8>, tensor<4xi32>
}
// CHECK-LABEL: func.func @test_NormalizeToTargetType_gather_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<4x64xi8>) outs(%{{.*}} : tensor<4x64xf16>) -> tensor<4x64xf16>
// CHECK: %[[DST_F16:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4x32xi8>) outs(%{{.*}} : tensor<4x32xf16>) -> tensor<4x32xf16>
// CHECK: %[[GATHER:.*]] = hivm.hir.vgather ins(%[[SRC_F16]] : tensor<4x64xf16>) indices(%arg1 : tensor<4x32xi32>) outs(%[[DST_F16]] : tensor<4x32xf16>) -> tensor<4x32xf16>
// CHECK: hivm.hir.vcast ins(%[[GATHER]] : tensor<4x32xf16>) outs(%{{.*}} : tensor<4x32xi8>) round_mode = <trunc> -> tensor<4x32xi8>
func.func @test_NormalizeToTargetType_gather_i8(%arg0: tensor<4x64xi8>, %arg1: tensor<4x32xi32>) -> tensor<4x32xi8> {
  %0 = tensor.empty() : tensor<4x32xi8>
  %1 = hfusion.gather ins(%arg0, %arg1 : tensor<4x64xi8>, tensor<4x32xi32>) outs(%0 : tensor<4x32xi8>) axis = 1 -> tensor<4x32xi8>
  return %1 : tensor<4x32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_mod_i8
// CHECK-DAG: hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK-DAG: hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK: %[[MOD:.*]] = hivm.hir.vmod ins(%{{.*}}, %{{.*}} : tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi16>) -> tensor<16xi16>
// CHECK: hivm.hir.vcast ins(%[[MOD]] : tensor<16xi16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <truncwithoverflow> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_mod_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
          ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_modui_i8
// CHECK-DAG: hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) cast = <cast_unsigned> -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK-DAG: hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) cast = <cast_unsigned> -> tensor<16xf16>
// CHECK-DAG: hivm.hir.vcast ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK: %[[MOD:.*]] = hivm.hir.vmodui ins(%{{.*}}, %{{.*}} : tensor<16xi16>, tensor<16xi16>) outs(%{{.*}} : tensor<16xi16>) -> tensor<16xi16>
// CHECK: hivm.hir.vcast ins(%[[MOD]] : tensor<16xi16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <truncwithoverflow> cast = <cast_unsigned> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_modui_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
          ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_add_i8
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[ADD:.*]] = hivm.hir.vadd ins(%[[LHS]], %[[RHS]] : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[ADD]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_add_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
          ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_sub_i8
// CHECK-DAG: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[SUB:.*]] = hivm.hir.vsub ins(%[[LHS]], %[[RHS]] : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[SUB]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_sub_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
          ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_mul_i8
// CHECK-DAG: %[[LHS_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[LHS_F32:.*]] = hivm.hir.vcast ins(%[[LHS_F16]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK-DAG: %[[RHS_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK-DAG: %[[RHS_F32:.*]] = hivm.hir.vcast ins(%[[RHS_F16]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[LHS_F32]], %[[RHS_F32]] : tensor<16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MUL_I32:.*]] = hivm.hir.vcast ins(%[[MUL]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xi32>) round_mode = <trunc> -> tensor<16xi32>
// CHECK: annotation.mark %[[MUL_I32]] {overflow_mode = "trunc"} : tensor<16xi32>
// CHECK: hivm.hir.vcast ins(%[[MUL_I32]] : tensor<16xi32>) outs(%{{.*}} : tensor<16xi8>) round_mode = <truncwithoverflow> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_mul_i8(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
          ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_absi_i8
// CHECK: %[[SRC_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[ABS:.*]] = hivm.hir.vabs ins(%[[SRC_F16]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hivm.hir.vcast ins(%[[ABS]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
func.func @test_NormalizeToTargetType_absi_i8(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>}
          ins(%arg0 : tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}
// Ascend950: linalg.elemwise_binary {add} i8 → hivm.hir.vadd i8.
// Both HFusion and HIVM skip i8 add normalization on Ascend950.

// CHECK-LABEL: func.func @test_DoNotNormalizeToTargetType_add_i8_ascend950
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<16xi8>, tensor<16xi8>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_DoNotNormalizeToTargetType_add_i8_ascend950(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
    %dst = tensor.empty() : tensor<16xi8>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
            ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
            outs(%dst : tensor<16xi8>) -> tensor<16xi8>
    return %res : tensor<16xi8>
  }
}

// -----

// Ascend950: linalg.elemwise_binary {sub} i8 → hivm.hir.vsub i8.
// Both HFusion and HIVM skip i8 sub normalization on Ascend950.

// CHECK-LABEL: func.func @test_DoNotNormalizeToTargetType_sub_i8_ascend950
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vsub ins(%{{.*}}, %{{.*}} : tensor<16xi8>, tensor<16xi8>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_DoNotNormalizeToTargetType_sub_i8_ascend950(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
    %dst = tensor.empty() : tensor<16xi8>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
            ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
            outs(%dst : tensor<16xi8>) -> tensor<16xi8>
    return %res : tensor<16xi8>
  }
}
