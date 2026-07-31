// RUN: bishengir-opt %s --hivm-normalize-ops -split-input-file | FileCheck %s

// CHECK-LABEL: @vmax_to_vrelu_f32
// CHECK-NOT: hivm.hir.vmax
// CHECK: hivm.hir.vrelu ins(%{{.*}} : tensor<64x64xf32>) outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
func.func @vmax_to_vrelu_f32(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<64x64xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mmad_out = tensor.empty() : tensor<64x64xf32>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x64xf32>) -> tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %result = hivm.hir.vmax ins(%mmad, %zero : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: @vmax_to_vrelu_f16
// CHECK-NOT: hivm.hir.vmax
// CHECK: hivm.hir.vrelu ins(%{{.*}} : tensor<4x64x32xf16>) outs(%{{.*}} : tensor<4x64x32xf16>) -> tensor<4x64x32xf16>
func.func @vmax_to_vrelu_f16(%arg0: tensor<4x64x32xf16>, %arg1: tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<4x64x32xf16>
  %zero = linalg.fill ins(%cst : f16) outs(%empty : tensor<4x64x32xf16>) -> tensor<4x64x32xf16>
  %mmad_out = tensor.empty() : tensor<4x64x32xf16>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<4x64x32xf16>, tensor<4x64x32xf16>, i1, index, index, index) outs(%mmad_out : tensor<4x64x32xf16>) -> tensor<4x64x32xf16>
  %out = tensor.empty() : tensor<4x64x32xf16>
  %result = hivm.hir.vmax ins(%mmad, %zero : tensor<4x64x32xf16>, tensor<4x64x32xf16>)
      outs(%out : tensor<4x64x32xf16>) -> tensor<4x64x32xf16>
  return %result : tensor<4x64x32xf16>
}

// -----

// Test commuted operand order: vmax(zero, mmad_result) -> vrelu(mmad_result)
// CHECK-LABEL: @vmax_to_vrelu_commuted
// CHECK-NOT: hivm.hir.vmax
// CHECK: hivm.hir.vrelu ins(%{{.*}} : tensor<64x64xf32>) outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
func.func @vmax_to_vrelu_commuted(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<64x64xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mmad_out = tensor.empty() : tensor<64x64xf32>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x64xf32>) -> tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %result = hivm.hir.vmax ins(%zero, %mmad : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test scalar zero: vmax(mmad_result, scalar_zero) -> vrelu(mmad_result)
// CHECK-LABEL: @vmax_to_vrelu_scalar_zero
// CHECK-NOT: hivm.hir.vmax
// CHECK: hivm.hir.vrelu
func.func @vmax_to_vrelu_scalar_zero(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %mmad_out = tensor.empty() : tensor<64x64xf32>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x64xf32>) -> tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %result = hivm.hir.vmax ins(%mmad, %cst : tensor<64x64xf32>, f32)
      outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test signed i32: vmax(mmad_result, zero) with is_signed=true -> vrelu
// CHECK-LABEL: @vmax_to_vrelu_i32_signed
// CHECK-NOT: hivm.hir.vmax
// CHECK: hivm.hir.vrelu
func.func @vmax_to_vrelu_i32_signed(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xi32> {
  %cst = arith.constant 0 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %mmad_out = tensor.empty() : tensor<64x64xi32>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x64xi32>) -> tensor<64x64xi32>
  %empty = tensor.empty() : tensor<64x64xi32>
  %zero = linalg.fill ins(%cst : i32) outs(%empty : tensor<64x64xi32>) -> tensor<64x64xi32>
  %out = tensor.empty() : tensor<64x64xi32>
  %result = hivm.hir.vmax ins(%mmad, %zero : tensor<64x64xi32>, tensor<64x64xi32>)
      outs(%out : tensor<64x64xi32>) is_signed = true -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// Negative test: vmax with non-zero operand should NOT be converted
// CHECK-LABEL: @vmax_no_convert_nonzero
// CHECK: hivm.hir.vmax
// CHECK-NOT: hivm.hir.vrelu
func.func @vmax_no_convert_nonzero(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %mmad_out = tensor.empty() : tensor<64x64xf32>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x64xf32>) -> tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %result = hivm.hir.vmax ins(%mmad, %arg2 : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Negative test: unsupported type (bf16) should NOT be converted
// CHECK-LABEL: @vmax_no_convert_unsupported_type
// CHECK: hivm.hir.vmax
// CHECK-NOT: hivm.hir.vrelu
func.func @vmax_no_convert_unsupported_type(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %mmad_out = tensor.empty() : tensor<64x64xbf16>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %empty = tensor.empty() : tensor<64x64xbf16>
  %zero = linalg.fill ins(%cst : bf16) outs(%empty : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %out = tensor.empty() : tensor<64x64xbf16>
  %result = hivm.hir.vmax ins(%mmad, %zero : tensor<64x64xbf16>, tensor<64x64xbf16>)
      outs(%out : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %result : tensor<64x64xbf16>
}

// -----

// Negative test: unsigned i32 (is_signed=false) should NOT be converted
// CHECK-LABEL: @vmax_no_convert_unsigned_i32
// CHECK: hivm.hir.vmax
// CHECK-NOT: hivm.hir.vrelu
func.func @vmax_no_convert_unsigned_i32(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xi32> {
  %cst = arith.constant 0 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %mmad_out = tensor.empty() : tensor<64x64xi32>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x64xi32>) -> tensor<64x64xi32>
  %empty = tensor.empty() : tensor<64x64xi32>
  %zero = linalg.fill ins(%cst : i32) outs(%empty : tensor<64x64xi32>) -> tensor<64x64xi32>
  %out = tensor.empty() : tensor<64x64xi32>
  %result = hivm.hir.vmax ins(%mmad, %zero : tensor<64x64xi32>, tensor<64x64xi32>)
      outs(%out : tensor<64x64xi32>) is_signed = false -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// Negative test: no mmad producer - should NOT be converted
// CHECK-LABEL: @vmax_no_convert_no_mmad
// CHECK: hivm.hir.vmax
// CHECK-NOT: hivm.hir.vrelu
func.func @vmax_no_convert_no_mmad(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %result = hivm.hir.vmax ins(%arg0, %zero : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test with broadcast attribute preserved
// CHECK-LABEL: @vmax_to_vrelu_with_broadcast
// CHECK-NOT: hivm.hir.vmax
// CHECK: hivm.hir.vrelu ins(%{{.*}} : tensor<64x1xf32>) outs(%{{.*}} : tensor<64x64xf32>) broadcast = [0, 1] -> tensor<64x64xf32>
func.func @vmax_to_vrelu_with_broadcast(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %mmad_out = tensor.empty() : tensor<64x1xf32>
  %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%mmad_out : tensor<64x1xf32>) -> tensor<64x1xf32>
  %empty = tensor.empty() : tensor<64x64xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %result = hivm.hir.vmax ins(%mmad, %zero : tensor<64x1xf32>, tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>) broadcast = [0, 1] -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}
