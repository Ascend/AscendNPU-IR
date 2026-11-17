// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// Negative test: f64->f32 cast must survive PreVectorizationFusion unchanged.
// Without this guard, PreVectorizationFusion would outline the cast into scalar
// loops (linalg.generic + scf.for), which CCEC cannot lower (no fp64 vector ALU).
// CHECK-LABEL: func.func @test_preserve_f64_to_f32_cast
// CHECK-NOT: linalg.generic
// CHECK-NOT: scf.for
// CHECK: hfusion.cast {{.*}}ins(%{{.*}} : tensor<512xf64>) outs(%{{.*}} : tensor<512xf32>) -> tensor<512xf32>
func.func @test_preserve_f64_to_f32_cast(%arg0: tensor<512xf64>) -> tensor<512xf32> {
  %empty = tensor.empty() : tensor<512xf32>
  %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<512xf64>) outs(%empty : tensor<512xf32>) -> tensor<512xf32>
  return %cast : tensor<512xf32>
}
