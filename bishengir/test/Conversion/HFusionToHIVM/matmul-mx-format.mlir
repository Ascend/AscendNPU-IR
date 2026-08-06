// RUN: bishengir-opt -convert-hfusion-to-hivm %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_formatted_i8_e4m3
  // CHECK: hivm.hir.mmadmxL1 {lhsFormat = 2 : i32, rhsFormat = 2 : i32}
  // CHECK-SAME: tensor<4x8xi8>, tensor<8x16xi8>
  func.func @test_matmulscale_formatted_i8_e4m3(
      %arg0: tensor<4x8xi8>, %arg1: tensor<8x16xi8>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>)
      -> tensor<4x16xf32> {
    %a_empty_fp8 = tensor.empty() : tensor<4x8xf8E4M3FN>
    %b_empty_fp8 = tensor.empty() : tensor<8x16xf8E4M3FN>
    %a_fp8 = hfusion.bitcast ins(%arg0 : tensor<4x8xi8>)
      outs(%a_empty_fp8 : tensor<4x8xf8E4M3FN>) -> tensor<4x8xf8E4M3FN>
    %b_fp8 = hfusion.bitcast ins(%arg1 : tensor<8x16xi8>)
      outs(%b_empty_fp8 : tensor<8x16xf8E4M3FN>) -> tensor<8x16xf8E4M3FN>
    %acc = tensor.empty() : tensor<4x16xf32>
    %res = hfusion.matmul_mx {lhsFormat = 2 : i32, rhsFormat = 2 : i32}
      ins(%a_fp8, %b_fp8, %arg2, %arg3 :
          tensor<4x8xf8E4M3FN>, tensor<8x16xf8E4M3FN>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_do_not_inline_fp4_format
  // CHECK: hivm.hir.bitcast %{{.*}} : tensor<4x8xi8> -> tensor<4x8xf8E5M2>
  // CHECK: hivm.hir.mmadmxL1 {lhsFormat = 3 : i32, rhsFormat = 3 : i32} ins({{.*}} : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>
  func.func @test_matmulscale_do_not_inline_fp4_format(
      %arg0: tensor<4x8xi8>, %arg1: tensor<8x16xi8>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>)
      -> tensor<4x16xf32> {
    %a_empty_fp8 = tensor.empty() : tensor<4x8xf8E5M2>
    %b_empty_fp8 = tensor.empty() : tensor<8x16xf8E5M2>
    %a_fp8 = hfusion.bitcast ins(%arg0 : tensor<4x8xi8>)
      outs(%a_empty_fp8 : tensor<4x8xf8E5M2>) -> tensor<4x8xf8E5M2>
    %b_fp8 = hfusion.bitcast ins(%arg1 : tensor<8x16xi8>)
      outs(%b_empty_fp8 : tensor<8x16xf8E5M2>) -> tensor<8x16xf8E5M2>
    %acc = tensor.empty() : tensor<4x16xf32>
    %res = hfusion.matmul_mx {lhsFormat = 3 : i32, rhsFormat = 3 : i32}
      ins(%a_fp8, %b_fp8, %arg2, %arg3 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_arith_bitcast_i8
  // CHECK: hivm.hir.mmadmxL1 {lhsFormat = 1 : i32, rhsFormat = 1 : i32}
  // CHECK-SAME: tensor<4x8xi8>, tensor<8x16xi8>
  func.func @test_matmulscale_arith_bitcast_i8(
      %arg0: tensor<4x8xi8>, %arg1: tensor<8x16xi8>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>)
      -> tensor<4x16xf32> {
    %a_fp8 = arith.bitcast %arg0 : tensor<4x8xi8> to tensor<4x8xf8E5M2>
    %b_fp8 = arith.bitcast %arg1 : tensor<8x16xi8> to tensor<8x16xf8E5M2>
    %acc = tensor.empty() : tensor<4x16xf32>
    %res = hfusion.matmul_mx {lhsFormat = 1 : i32, rhsFormat = 1 : i32}
      ins(%a_fp8, %b_fp8, %arg2, %arg3 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_accumulate_non_empty_init
  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: hivm.hir.mmadmxL1
  // CHECK-SAME: %[[FALSE]]
  func.func @test_matmulscale_accumulate_non_empty_init(
      %arg0: tensor<4x8xf8E5M2>, %arg1: tensor<8x16xf8E5M2>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>,
      %acc: tensor<4x16xf32>)
      -> tensor<4x16xf32> {
    %res = hfusion.matmul_mx
      ins(%arg0, %arg1, %arg2, %arg3 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_do_not_inline_non_bitcast_def
  // CHECK: hivm.hir.copy
  // CHECK: hivm.hir.mmadmxL1 {lhsFormat = 1 : i32, rhsFormat = 1 : i32} ins({{.*}} : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>
  func.func @test_matmulscale_do_not_inline_non_bitcast_def(
      %arg0: tensor<4x8xf8E5M2>, %arg1: tensor<8x16xf8E5M2>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>)
      -> tensor<4x16xf32> {
    %a_tmp = tensor.empty() : tensor<4x8xf8E5M2>
    %b_tmp = tensor.empty() : tensor<8x16xf8E5M2>
    %a = linalg.copy ins(%arg0 : tensor<4x8xf8E5M2>) outs(%a_tmp : tensor<4x8xf8E5M2>) -> tensor<4x8xf8E5M2>
    %b = linalg.copy ins(%arg1 : tensor<8x16xf8E5M2>) outs(%b_tmp : tensor<8x16xf8E5M2>) -> tensor<8x16xf8E5M2>
    %acc = tensor.empty() : tensor<4x16xf32>
    %res = hfusion.matmul_mx {lhsFormat = 1 : i32, rhsFormat = 1 : i32}
      ins(%a, %b, %arg2, %arg3 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}
