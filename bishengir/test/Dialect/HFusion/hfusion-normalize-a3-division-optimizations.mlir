// RUN: bishengir-opt %s --hfusion-normalize-ops -split-input-file | FileCheck %s --check-prefix=HF --enable-var-scope
// RUN: bishengir-opt %s --hfusion-normalize-ops --convert-hfusion-to-hivm -split-input-file | FileCheck %s --check-prefix=HIVM --enable-var-scope

// Scalar guards, safe denominator and sign remain scalar. Both floating
// divisions consume the same scalar f32 denominator.
// HF-LABEL: func.func @scalar_i32_divisor(
// HF: %[[ONE:.*]] = arith.cmpi eq, %arg1,
// HF: %[[NEGONE:.*]] = arith.cmpi eq, %arg1,
// HF: %[[SPECIAL:.*]] = arith.ori %[[ONE]], %[[NEGONE]]
// HF: %[[SAFE:.*]] = arith.select %[[SPECIAL]],
// HF: %[[BF:.*]] = arith.sitofp %[[SAFE]] : i32 to f32
// HF: arith.cmpf olt, %[[BF]],
// HF: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}}, %[[BF]] : tensor<16xf32>, f32)
// HF: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}}, %[[BF]] : tensor<16xf32>, f32)
// HF: hfusion.select ins(%[[SPECIAL]],
// HF: return
// HIVM-LABEL: func.func @scalar_i32_divisor(
// HIVM: arith.select
// HIVM-COUNT-2: hivm.hir.vdiv
// HIVM: return
func.func @scalar_i32_divisor(%a: tensor<16xi32>, %b: i32) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %q : tensor<16xi32>
}
