// REQUIRES: regbase
// TODO: enable after migrating regbase pipeline dependencies
// RUN: bishengir-opt -convert-math-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

// === math.exp ===

// CHECK-LABEL: func.func @test_scalar_exp_f32
func.func @test_scalar_exp_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  // CHECK:       tensor.extract
  %ret = math.exp %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_exp_bf16
func.func @test_scalar_exp_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.exp %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.exp2 ===

// CHECK-LABEL: func.func @test_scalar_exp2_f32
func.func @test_scalar_exp2_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>}
  // CHECK:       tensor.extract
  %ret = math.exp2 %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_exp2_bf16
func.func @test_scalar_exp2_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.exp2 %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.log ===

// CHECK-LABEL: func.func @test_scalar_log_f32
func.func @test_scalar_log_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
  // CHECK:       tensor.extract
  %ret = math.log %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_log_bf16
func.func @test_scalar_log_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.log %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.log2 ===

// CHECK-LABEL: func.func @test_scalar_log2_f32
func.func @test_scalar_log2_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>}
  // CHECK:       tensor.extract
  %ret = math.log2 %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_log2_bf16
func.func @test_scalar_log2_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.log2 %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.ceil ===

// CHECK-LABEL: func.func @test_scalar_ceil_f32
func.func @test_scalar_ceil_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // CHECK:       tensor.extract
  %ret = math.ceil %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_ceil_bf16
func.func @test_scalar_ceil_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.ceil %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.floor ===

// CHECK-LABEL: func.func @test_scalar_floor_f32
func.func @test_scalar_floor_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
  // CHECK:       tensor.extract
  %ret = math.floor %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_floor_bf16
func.func @test_scalar_floor_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.floor %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.sin ===

// CHECK-LABEL: func.func @test_scalar_sin_f32
func.func @test_scalar_sin_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
  // CHECK:       tensor.extract
  %ret = math.sin %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_sin_bf16
func.func @test_scalar_sin_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.sin %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.cos ===

// CHECK-LABEL: func.func @test_scalar_cos_f32
func.func @test_scalar_cos_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
  // CHECK:       tensor.extract
  %ret = math.cos %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_cos_bf16
func.func @test_scalar_cos_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.cos %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.erf ===

// CHECK-LABEL: func.func @test_scalar_erf_f32
func.func @test_scalar_erf_f32(%arg0 : f32) -> f32 {
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
  // CHECK:       tensor.extract
  %ret = math.erf %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_erf_bf16
func.func @test_scalar_erf_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
  // CHECK:       tensor.extract
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.erf %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.absf ===
// f32 and f16 are natively supported by hardware — scalar op stays unchanged.
// bf16 goes through scalar f32 fallback (not tensor promotion).

// CHECK-LABEL: func.func @test_scalar_absf_f32
func.func @test_scalar_absf_f32(%arg0 : f32) -> f32 {
  // CHECK:       math.absf {{.*}} : f32
  %ret = math.absf %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_absf_f16
func.func @test_scalar_absf_f16(%arg0 : f16) -> f16 {
  // CHECK:       math.absf {{.*}} : f16
  %ret = math.absf %arg0 : f16
  return %ret : f16
}

// -----

// CHECK-LABEL: func.func @test_scalar_absf_bf16
func.func @test_scalar_absf_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       math.absf {{.*}} : f32
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.absf %arg0 : bf16
  return %ret : bf16
}

// -----

// === math.rsqrt ===
// f32 is natively supported — scalar op stays unchanged.
// bf16 goes through scalar f32 fallback (not tensor promotion).
// f16 is NOT natively supported — goes through tensor promotion.

// CHECK-LABEL: func.func @test_scalar_rsqrt_f32
func.func @test_scalar_rsqrt_f32(%arg0 : f32) -> f32 {
  // CHECK:       math.rsqrt {{.*}} : f32
  %ret = math.rsqrt %arg0 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_rsqrt_bf16
func.func @test_scalar_rsqrt_bf16(%arg0 : bf16) -> bf16 {
  // CHECK:       arith.extf {{.*}} : bf16 to f32
  // CHECK:       math.rsqrt {{.*}} : f32
  // CHECK:       arith.truncf {{.*}} : f32 to bf16
  %ret = math.rsqrt %arg0 : bf16
  return %ret : bf16
}

// -----

// CHECK-LABEL: func.func @test_scalar_rsqrt_f16
func.func @test_scalar_rsqrt_f16(%arg0 : f16) -> f16 {
  // CHECK:       tensor.from_elements
  // CHECK:       hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
  // CHECK:       tensor.extract
  %ret = math.rsqrt %arg0 : f16
  return %ret : f16
}

// -----

// === math.fma ===
// f32 is natively supported — scalar op stays unchanged.
// f16 and bf16 go through tensor promotion and are decomposed into
// linalg.elemwise_binary mul + add.

// CHECK-LABEL: func.func @test_scalar_fma_f32
func.func @test_scalar_fma_f32(%arg0 : f32, %arg1 : f32, %arg2 : f32) -> f32 {
  // CHECK:       math.fma {{.*}} : f32
  %ret = math.fma %arg0, %arg1, %arg2 : f32
  return %ret : f32
}

// -----

// CHECK-LABEL: func.func @test_scalar_fma_f16
func.func @test_scalar_fma_f16(%arg0 : f16, %arg1 : f16, %arg2 : f16) -> f16 {
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
  // CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
  // CHECK:       tensor.extract
  %ret = math.fma %arg0, %arg1, %arg2 : f16
  return %ret : f16
}

// -----

// CHECK-LABEL: func.func @test_scalar_fma_bf16
func.func @test_scalar_fma_bf16(%arg0 : bf16, %arg1 : bf16, %arg2 : bf16) -> bf16 {
  // CHECK:       tensor.from_elements
  // CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
  // CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
  // CHECK:       tensor.extract
  %ret = math.fma %arg0, %arg1, %arg2 : bf16
  return %ret : bf16
}
