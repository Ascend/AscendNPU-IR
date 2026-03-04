// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @test_element_type_only_f16_i32(%a : memref<16xf16>, %b : memref<16xi32>) {
  bishengir_test.element_type_only_f16_i32 ins(%a, %b : memref<16xf16>, memref<16xi32>)
  return
}

// -----

func.func @test_element_type_only_f16_i32_invalid(%a : memref<16xi32>, %b : memref<16xi32>) {
  // expected-error@+1 {{failed to verify that operand at idx 0 should have element type 16-bit float}}
  bishengir_test.element_type_only_f16_i32 ins(%a, %b : memref<16xi32>, memref<16xi32>)
  return
}

// -----

func.func @element_type_only_f16f32_i8i32() {
  %f16_tensor = tensor.empty() : tensor<16xf16>
  %i8_tensor = tensor.empty() : tensor<16xi8>
  %f32_tensor = tensor.empty() : tensor<16xf32>
  %i32_tensor = tensor.empty() : tensor<16xi32>
  %f16 = arith.constant 1.0 : f16
  %i32 = arith.constant 1 : i32
  bishengir_test.element_type_only_f16f32_i8i32 ins(%f16_tensor, %i8_tensor : tensor<16xf16>, tensor<16xi8>)
  bishengir_test.element_type_only_f16f32_i8i32 ins(%f16, %i32_tensor : f16, tensor<16xi32>)
  bishengir_test.element_type_only_f16f32_i8i32 ins(%f32_tensor, %i32 : tensor<16xf32>, i32)
  return
}

// -----

func.func @element_type_only_int_float() {
  %bf16_tensor = tensor.empty() : tensor<16xbf16>
  %i1_tensor = tensor.empty() : tensor<16xi1>
  bishengir_test.element_type_only_int_float ins(%i1_tensor, %bf16_tensor : tensor<16xi1>, tensor<16xbf16>)

  %f64_tensor = tensor.empty() : tensor<16xf64>
  %ui8_tensor = tensor.empty() : tensor<16xui8>
  bishengir_test.element_type_only_int_float ins(%ui8_tensor, %f64_tensor : tensor<16xui8>, tensor<16xf64>)
  return
}

// -----

func.func @element_type_only_int_float_invalid() {
  %complex_tensor = tensor.empty() : tensor<16xcomplex<f64>>
  %i1_tensor = tensor.empty() : tensor<16xi1>
  // expected-error@+1 {{failed to verify that operand at idx 1 should have element type floating-point}}
  bishengir_test.element_type_only_int_float ins(%i1_tensor, %complex_tensor : tensor<16xi1>, tensor<16xcomplex<f64>>)
  return
}