// RUN: bishengir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_elemwise_unary_ops
func.func @test_elemwise_unary_ops(
  %src : memref<6x6xf32>, %dst : memref<6x6xf32>) {
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_elemwise_unary_ops
func.func @test_tensor_elemwise_unary_ops(
  %src : tensor<6x6xf32>, %init : tensor<6x6xf32>) {
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  %res0 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
    ins(%src : tensor<6x6xf32>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
  %res1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
    ins(%src : tensor<6x6xf32>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
  %res2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    ins(%src : tensor<6x6xf32>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
  %res3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
    ins(%src : tensor<6x6xf32>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_elemwise_integer_unary_ops
func.func @test_elemwise_integer_unary_ops(
  %src : memref<6x6xi32>, %dst : memref<6x6xi32>) {
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} 
    ins(%src : memref<6x6xi32>) 
    outs(%dst : memref<6x6xi32>)
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} 
    ins(%src : memref<6x6xi32>) 
    outs(%dst : memref<6x6xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_elemwise_integer_unary_ops
func.func @test_tensor_elemwise_integer_unary_ops(
  %src : tensor<6x6xi32>, %init : tensor<6x6xi32>) {
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
  %res0 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    ins(%src : tensor<6x6xi32>)
    outs(%init : tensor<6x6xi32>) -> tensor<6x6xi32>
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
  %res1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
    ins(%src : tensor<6x6xi32>)
    outs(%init : tensor<6x6xi32>) -> tensor<6x6xi32>
  return
}

// -----

// CHECK-LABEL: func.func @test_not_op
func.func @test_not_op(
  %src : memref<6x6xi1>, %dst : memref<6x6xi1>) {
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} 
    ins(%src : memref<6x6xi1>) 
    outs(%dst : memref<6x6xi1>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_not_op
func.func @test_tensor_not_op(
  %src : tensor<6x6xi1>, %init : tensor<6x6xi1>) {
  //CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
    ins(%src : tensor<6x6xi1>)
    outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  return
}

// -----

// CHECK-LABEL: func.func @test_elemwise_binary_ops
func.func @test_elemwise_binary_ops(
  %src1 : memref<6x6xi32>, %src2 : i1, %dst : memref<6x6xi32>) {
  //CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} 
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} 
    ins(%src1, %src2 : memref<6x6xi32>, i1) 
    outs(%dst : memref<6x6xi32>)
  //CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} 
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} 
    ins(%src1, %src2 : memref<6x6xi32>, i1) 
    outs(%dst : memref<6x6xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_elemwise_binary_ops
func.func @test_tensor_elemwise_binary_ops(
  %src1 : tensor<6x6xi32>, %src2 : i1, %init : tensor<6x6xi32>) {
  //CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
  %res0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
    ins(%src1, %src2 : tensor<6x6xi32>, i1)
    outs(%init : tensor<6x6xi32>) -> tensor<6x6xi32>
  //CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
  %res1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
    ins(%src1, %src2 : tensor<6x6xi32>, i1)
    outs(%init : tensor<6x6xi32>) -> tensor<6x6xi32>
  return
}

// -----

// CHECK-LABEL: func.func @test_elemwise_cmpf_ops
func.func @test_elemwise_cmpf_ops(
  %src1 : memref<6x6xf32>, %src2 : memref<6x6xf32>, %dst : memref<6x6xi1>) {
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<veq>}
    hfusion.compare {fun = #hfusion.compare_fn<veq>}
      ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
      outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vne>}
      hfusion.compare {fun = #hfusion.compare_fn<vne>}
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vle>}
      hfusion.compare {fun = #hfusion.compare_fn<vle>}
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vlt>}
      hfusion.compare {fun = #hfusion.compare_fn<vlt>}
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vge>}
      hfusion.compare {fun = #hfusion.compare_fn<vge>}
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vgt>}
      hfusion.compare {fun = #hfusion.compare_fn<vgt>}
        ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
        outs(%dst : memref<6x6xi1>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_elemwise_cmpf_ops
func.func @test_tensor_elemwise_cmpf_ops(
  %src1 : tensor<6x6xf32>, %src2 : tensor<6x6xf32>, %init : tensor<6x6xi1>) {
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<veq>}
  %res0 = hfusion.compare {fun = #hfusion.compare_fn<veq>}
      ins(%src1, %src2 : tensor<6x6xf32>, tensor<6x6xf32>)
      outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vne>}
  %res1 = hfusion.compare {fun = #hfusion.compare_fn<vne>}
        ins(%src1, %src2 : tensor<6x6xf32>, tensor<6x6xf32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vle>}
  %res2 = hfusion.compare {fun = #hfusion.compare_fn<vle>}
        ins(%src1, %src2 : tensor<6x6xf32>, tensor<6x6xf32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vlt>}
  %res3 = hfusion.compare {fun = #hfusion.compare_fn<vlt>}
        ins(%src1, %src2 : tensor<6x6xf32>, tensor<6x6xf32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vge>}
  %res4 = hfusion.compare {fun = #hfusion.compare_fn<vge>}
        ins(%src1, %src2 : tensor<6x6xf32>, tensor<6x6xf32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vgt>}
  %res5 = hfusion.compare {fun = #hfusion.compare_fn<vgt>}
        ins(%src1, %src2 : tensor<6x6xf32>, tensor<6x6xf32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  return
}

// -----

// CHECK-LABEL: func.func @test_elemwise_cmpi_ops
func.func @test_elemwise_cmpi_ops(
  %src1 : memref<6x6xi32>, %src2 : memref<6x6xi32>, %dst : memref<6x6xi1>) {
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<veq>}
    hfusion.compare {fun = #hfusion.compare_fn<veq>}
      ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
      outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vne>}
      hfusion.compare {fun = #hfusion.compare_fn<vne>}
        ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vlt>}
      hfusion.compare {fun = #hfusion.compare_fn<vlt>}
        ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vgt>}
      hfusion.compare {fun = #hfusion.compare_fn<vgt>}
        ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vle>}
      hfusion.compare {fun = #hfusion.compare_fn<vle>}
        ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
        outs(%dst : memref<6x6xi1>)
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vge>}
      hfusion.compare {fun = #hfusion.compare_fn<vge>}
        ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
        outs(%dst : memref<6x6xi1>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_elemwise_cmpi_ops
func.func @test_tensor_elemwise_cmpi_ops(
  %src1 : tensor<6x6xi32>, %src2 : tensor<6x6xi32>, %init : tensor<6x6xi1>) {
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<veq>}
  %res0 = hfusion.compare {fun = #hfusion.compare_fn<veq>}
      ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
      outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vne>}
  %res1 = hfusion.compare {fun = #hfusion.compare_fn<vne>}
        ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vlt>}
  %res2 = hfusion.compare {fun = #hfusion.compare_fn<vlt>}
        ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vgt>}
  %res3 = hfusion.compare {fun = #hfusion.compare_fn<vgt>}
        ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vle>}
  %res4 = hfusion.compare {fun = #hfusion.compare_fn<vle>}
        ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  //CHECK: hfusion.compare {fun = #hfusion.compare_fn<vge>}
  %res5 = hfusion.compare {fun = #hfusion.compare_fn<vge>}
        ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
        outs(%init : tensor<6x6xi1>) -> tensor<6x6xi1>
  return
}

// -----

// CHECK-LABEL: func.func @test_selectfi_ops
func.func @test_selectfi_ops(
  %src1 : memref<6x6xi1>, %src2 : memref<6x6xi32>, %src3 : memref<6x6xi32>, %dst : memref<6x6xi32>) {
  //CHECK: hfusion.select
    hfusion.select
      ins(%src1, %src2, %src3 : memref<6x6xi1>, memref<6x6xi32>, memref<6x6xi32>)
      outs(%dst : memref<6x6xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_selectfi_ops
func.func @test_tensor_selectfi_ops(
  %src1 : tensor<6x6xi1>, %src2 : tensor<6x6xi32>, %src3 : tensor<6x6xi32>, %init : tensor<6x6xi32>) {
  //CHECK: hfusion.select
  %res0 = hfusion.select
      ins(%src1, %src2, %src3 : tensor<6x6xi1>, tensor<6x6xi32>, tensor<6x6xi32>)
      outs(%init : tensor<6x6xi32>) -> tensor<6x6xi32>
  return
}

// -----

// CHECK-LABEL: func.func @test_selectf_ops
func.func @test_selectf_ops(
  %src1 : memref<6x6xi1>, %src2 : memref<6x6xf32>, %src3 : memref<6x6xf32>, %dst : memref<6x6xf32>) {
  //CHECK: hfusion.select
    hfusion.select 
    ins(%src1, %src2, %src3 : memref<6x6xi1>, memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_selectf_ops
func.func @test_tensor_selectf_ops(
  %src1 : tensor<6x6xi1>, %src2 : tensor<6x6xf32>, %src3 : tensor<6x6xf32>, %init : tensor<6x6xf32>) {
  //CHECK: hfusion.select
  %res0 = hfusion.select
    ins(%src1, %src2, %src3 : tensor<6x6xi1>, tensor<6x6xf32>, tensor<6x6xf32>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_cast_f32_f16
func.func @test_cast_f32_f16(
  %src : memref<6x6xf32>, %dst : memref<6x6xf16>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  hfusion.cast {mode = #hfusion.round_mode<rint>} 
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_cast_f32_f16
func.func @test_tensor_cast_f32_f16(
  %src : tensor<6x6xf32>, %init : tensor<6x6xf16>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  %res0 = hfusion.cast {mode = #hfusion.round_mode<rint>}
    ins(%src : tensor<6x6xf32>)
    outs(%init : tensor<6x6xf16>) -> tensor<6x6xf16>
  return
}

// -----

// CHECK-LABEL: func.func @test_cast_f32_bf16
func.func @test_cast_f32_bf16(
  %src : memref<6x6xf32>, %dst : memref<6x6xbf16>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  hfusion.cast {mode = #hfusion.round_mode<rint>} 
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xbf16>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_cast_f32_bf16
func.func @test_tensor_cast_f32_bf16(
  %src : tensor<6x6xf32>, %init : tensor<6x6xbf16>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  %res0 = hfusion.cast {mode = #hfusion.round_mode<rint>}
    ins(%src : tensor<6x6xf32>)
    outs(%init : tensor<6x6xbf16>) -> tensor<6x6xbf16>
  return
}

// -----

// CHECK-LABEL: func.func @test_cast_f16_f32
func.func @test_cast_f16_f32(
  %src : memref<6x6xf16>, %dst : memref<6x6xf32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  hfusion.cast {mode = #hfusion.round_mode<rint>} 
    ins(%src : memref<6x6xf16>)
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_cast_f16_f32
func.func @test_tensor_cast_f16_f32(
  %src : tensor<6x6xf16>, %init : tensor<6x6xf32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  %res0 = hfusion.cast {mode = #hfusion.round_mode<rint>}
    ins(%src : tensor<6x6xf16>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_cast_bf16_f32
func.func @test_cast_bf16_f32(
  %src : memref<6x6xbf16>, %dst : memref<6x6xf32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  hfusion.cast {mode = #hfusion.round_mode<rint>} 
    ins(%src : memref<6x6xbf16>)
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_cast_bf16_f32
func.func @test_tensor_cast_bf16_f32(
  %src : tensor<6x6xbf16>, %init : tensor<6x6xf32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<rint>}
  %res0 = hfusion.cast {mode = #hfusion.round_mode<rint>}
    ins(%src : tensor<6x6xbf16>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_cast_f32_i32
func.func @test_cast_f32_i32(
  %src : memref<6x6xf32>, %dst : memref<6x6xi32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<trunc>}
  hfusion.cast {mode = #hfusion.round_mode<trunc>} 
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_cast_f32_i32
func.func @test_tensor_cast_f32_i32(
  %src : tensor<6x6xf32>, %init : tensor<6x6xi32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<trunc>}
  %res0 = hfusion.cast {mode = #hfusion.round_mode<trunc>}
    ins(%src : tensor<6x6xf32>)
    outs(%init : tensor<6x6xi32>) -> tensor<6x6xi32>
  return
}

// -----

// CHECK-LABEL: func.func @test_cast_i32_f32
func.func @test_cast_i32_f32(
  %src : memref<6x6xi32>, %dst : memref<6x6xf32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<trunc>}
  hfusion.cast {mode = #hfusion.round_mode<trunc>} 
    ins(%src : memref<6x6xi32>)
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_cast_i32_f32
func.func @test_tensor_cast_i32_f32(
  %src : tensor<6x6xi32>, %init : tensor<6x6xf32>) {
  //CHECK: hfusion.cast {mode = #hfusion.round_mode<trunc>}
  %res0 = hfusion.cast {mode = #hfusion.round_mode<trunc>}
    ins(%src : tensor<6x6xi32>)
    outs(%init : tensor<6x6xf32>) -> tensor<6x6xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index_without_index_input
func.func @test_reduce_with_index_without_index_input(
  %input : memref<6x7xf32>,
  %output : memref<6xf32>, %output_index : memref<6xi32>) {
  // CHECK: hfusion.reduce_with_index {tie_break_left = true} <min>
  hfusion.reduce_with_index {tie_break_left = true} <min> 
    ins(%input : memref<6x7xf32>)
    outs(%output, %output_index : memref<6xf32>, memref<6xi32>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_reduce_with_index_without_index_input
func.func @test_tensor_reduce_with_index_without_index_input(
  %input : tensor<6x7xf32>,
  %output : tensor<6xf32>, %output_index : tensor<6xi32>) {
  // CHECK: hfusion.reduce_with_index {tie_break_left = true} <min>
  %data, %index = hfusion.reduce_with_index {tie_break_left = true} <min>
    ins(%input : tensor<6x7xf32>)
    outs(%output, %output_index : tensor<6xf32>, tensor<6xi32>)
    dimensions = [1]
    -> tensor<6xf32>, tensor<6xi32>
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index_with_index_input
func.func @test_reduce_with_index_with_index_input(
  %input : memref<6x7xf32>, %input_index : memref<6x7xi32>,
  %output : memref<7xf32>, %output_index : memref<7xi32>) {
  // CHECK: hfusion.reduce_with_index {tie_break_left = true} <max>
  hfusion.reduce_with_index {tie_break_left = true} <max> 
    ins(%input, %input_index : memref<6x7xf32>, memref<6x7xi32>)
    outs(%output, %output_index : memref<7xf32>, memref<7xi32>)
    dimensions = [0]
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_reduce_with_index_with_index_input
func.func @test_tensor_reduce_with_index_with_index_input(
  %input : tensor<6x7xf32>, %input_index : tensor<6x7xi32>,
  %output : tensor<7xf32>, %output_index : tensor<7xi32>) {
  // CHECK: hfusion.reduce_with_index {tie_break_left = true} <max>
  %data, %index = hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%input, %input_index : tensor<6x7xf32>, tensor<6x7xi32>)
    outs(%output, %output_index : tensor<7xf32>, tensor<7xi32>)
    dimensions = [0]
    -> tensor<7xf32>, tensor<7xi32>
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index_without_index_input_int
func.func @test_reduce_with_index_without_index_input_int(
  %input : memref<6x7xi32>,
  %output : memref<6xi32>, %output_index : memref<6xi32>) {
  // CHECK: hfusion.reduce_with_index {tie_break_left = true} <max>
  hfusion.reduce_with_index {tie_break_left = true} <max> 
    ins(%input : memref<6x7xi32>)
    outs(%output, %output_index : memref<6xi32>, memref<6xi32>)
    dimensions = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index_with_index_input_int
func.func @test_reduce_with_index_with_index_input_int(
  %input : memref<6x7xi32>, %input_index : memref<6x7xi32>,
  %output : memref<7xi32>, %output_index : memref<7xi32>) {
  // CHECK: hfusion.reduce_with_index {tie_break_left = true} <min>
  hfusion.reduce_with_index {tie_break_left = true} <min> 
    ins(%input, %input_index : memref<6x7xi32>, memref<6x7xi32>)
    outs(%output, %output_index : memref<7xi32>, memref<7xi32>)
    dimensions = [0]
  return
}

// -----

// CHECK-LABEL: func.func @test_cumsum_tensor
func.func @test_cumsum_tensor(%input : tensor<6x?x?xi32>) {
  %s16 = tensor.empty() : tensor<6x7xi16>
  %s32 = tensor.empty() : tensor<6x7xi32>
  %s64 = tensor.empty() : tensor<6x7xi64>
  %f16 = tensor.empty() : tensor<6x7xf16>
  %f32 = tensor.empty() : tensor<6x7xf32>
  hfusion.cumsum %f16 : tensor<6x7xf16> cum_dims = [0] reverse = false -> tensor<6x7xf16>
  hfusion.cumsum %f32 : tensor<6x7xf32> cum_dims = [1] reverse = false -> tensor<6x7xf32>
  hfusion.cumsum %s16 : tensor<6x7xi16> cum_dims = [0] reverse = false -> tensor<6x7xi16>
  hfusion.cumsum %s32 : tensor<6x7xi32> cum_dims = [1] reverse = false -> tensor<6x7xi32>
  hfusion.cumsum %s64 : tensor<6x7xi64> cum_dims = [0] reverse = false -> tensor<6x7xi64>
  hfusion.cumsum %input : tensor<6x?x?xi32> cum_dims = [2] reverse = false -> tensor<6x?x?xi32>
  return
}

// -----

// CHECK-LABEL: func.func @test_cumprod_tensor
func.func @test_cumprod_tensor(%input : tensor<6x?x?xi32>) {
  %s16 = tensor.empty() : tensor<6x7xi16>
  %s32 = tensor.empty() : tensor<6x7xi32>
  %s64 = tensor.empty() : tensor<6x7xi64>
  %f16 = tensor.empty() : tensor<6x7xf16>
  %f32 = tensor.empty() : tensor<6x7xf32>
  hfusion.cumprod %f16 : tensor<6x7xf16> cum_dims = [0] reverse = false -> tensor<6x7xf16>
  hfusion.cumprod %f32 : tensor<6x7xf32> cum_dims = [1] reverse = false -> tensor<6x7xf32>
  hfusion.cumprod %s16 : tensor<6x7xi16> cum_dims = [0] reverse = false -> tensor<6x7xi16>
  hfusion.cumprod %s32 : tensor<6x7xi32> cum_dims = [1] reverse = false -> tensor<6x7xi32>
  hfusion.cumprod %s64 : tensor<6x7xi64> cum_dims = [0] reverse = false -> tensor<6x7xi64>
  hfusion.cumprod %input : tensor<6x?x?xi32> cum_dims = [2] reverse = false -> tensor<6x?x?xi32>
  return
}

// -----

func.func @test_reduce_with_index_negative_dimension(
  %input : memref<6x7xi32>,
  %output : memref<6xi32>, %output_index : memref<6xi32>) {
  // expected-error @+1 {{dimensions for reduction should be in the range [0, 1]}}
  hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%input : memref<6x7xi32>)
    outs(%output, %output_index : memref<6xi32>, memref<6xi32>)
    dimensions = [-1]
  return
}

// -----

func.func @test_reduce_with_index_out_of_bound_dimension(
  %input : memref<6x7xi32>,
  %output : memref<6xi32>, %output_index : memref<6xi32>) {
  // expected-error @+1 {{dimensions for reduction should be in the range [0, 1]}}
  hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%input : memref<6x7xi32>)
    outs(%output, %output_index : memref<6xi32>, memref<6xi32>)
    dimensions = [2]
  return
}

// -----

func.func @test_reduce_with_index_decreasing(
  %input : memref<6x7xi32>,
  %output : memref<6xi32>, %output_index : memref<6xi32>) {
  // expected-error @+1 {{dense array attribute should be in increasing order}}
  hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%input : memref<6x7xi32>)
    outs(%output, %output_index : memref<6xi32>, memref<6xi32>)
    dimensions = [1, 0]
  return
}

// -----

func.func @test_reduce_with_index_non_increasing(
  %input : memref<6x7xi32>,
  %output : memref<6xi32>, %output_index : memref<6xi32>) {
  // expected-error @+1 {{dense array attribute should be in increasing order}}
  hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%input : memref<6x7xi32>)
    outs(%output, %output_index : memref<6xi32>, memref<6xi32>)
    dimensions = [0, 0]
  return
}
