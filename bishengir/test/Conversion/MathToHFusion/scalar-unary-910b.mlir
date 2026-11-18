// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910_9382 --convert-math-to-hfusion | FileCheck %s
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B4-1 --convert-math-to-hfusion | FileCheck %s

// Scalar unary promotion on 910B/910_93. Ops with a working native scalar
// path for a dtype must stay scalar; all other dtype/op combinations must be
// promoted to a one-element tensor.

module {
  // absf f32 is native (HIVMOptSinglePoint -> hivm.vabs).
  // CHECK-LABEL: func.func @scalar_absf_f32
  // CHECK-NOT: tensor.from_elements
  // CHECK: math.absf
  func.func @scalar_absf_f32(%x: f32) -> f32 {
    %y = math.absf %x : f32
    return %y : f32
  }

  // absf f16 has no native scalar path; promote it through a tensor.
  // CHECK-LABEL: func.func @scalar_absf_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
  // CHECK: tensor.extract
  func.func @scalar_absf_f16(%x: f16) -> f16 {
    %y = math.absf %x : f16
    return %y : f16
  }

  // absf bf16 uses the scalar bf16 fallback: extf -> absf f32 -> truncf.
  // CHECK-LABEL: func.func @scalar_absf_bf16
  // CHECK-NOT: tensor.from_elements
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: math.absf {{.*}} : f32
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_absf_bf16(%x: bf16) -> bf16 {
    %y = math.absf %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_ceil_f32
  // CHECK: tensor.from_elements
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // CHECK: tensor.extract
  func.func @scalar_ceil_f32(%x: f32) -> f32 {
    %y = math.ceil %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_floor_f32
  // CHECK: tensor.from_elements
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
  // CHECK: tensor.extract
  func.func @scalar_floor_f32(%x: f32) -> f32 {
    %y = math.floor %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_floor_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
  // CHECK: tensor.extract
  func.func @scalar_floor_f16(%x: f16) -> f16 {
    %y = math.floor %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_floor_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_floor_bf16(%x: bf16) -> bf16 {
    %y = math.floor %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_exp_f32
  // CHECK: tensor.from_elements
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  // CHECK: tensor.extract
  func.func @scalar_exp_f32(%x: f32) -> f32 {
    %y = math.exp %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_exp2_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>}
  // CHECK: tensor.extract
  func.func @scalar_exp2_f32(%x: f32) -> f32 {
    %y = math.exp2 %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_expm1_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>}
  // CHECK: tensor.extract
  func.func @scalar_expm1_f32(%x: f32) -> f32 {
    %y = math.expm1 %x : f32
    return %y : f32
  }

  // log f32/f16 are native (HIVMDecomposeOp -> hivm.vln).
  // CHECK-LABEL: func.func @scalar_log_f32
  // CHECK-NOT: tensor.from_elements
  // CHECK: math.log
  func.func @scalar_log_f32(%x: f32) -> f32 {
    %y = math.log %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_log_f16
  // CHECK-NOT: tensor.from_elements
  // CHECK: math.log
  func.func @scalar_log_f16(%x: f16) -> f16 {
    %y = math.log %x : f16
    return %y : f16
  }

  // bf16 log is promoted through an f32 tensor.
  // CHECK-LABEL: func.func @scalar_log_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_log_bf16(%x: bf16) -> bf16 {
    %y = math.log %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_log2_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>}
  // CHECK: tensor.extract
  func.func @scalar_log2_f32(%x: f32) -> f32 {
    %y = math.log2 %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_log10_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log10>}
  // CHECK: tensor.extract
  func.func @scalar_log10_f32(%x: f32) -> f32 {
    %y = math.log10 %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_log1p_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>}
  // CHECK: tensor.extract
  func.func @scalar_log1p_f32(%x: f32) -> f32 {
    %y = math.log1p %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_sin_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
  // CHECK: tensor.extract
  func.func @scalar_sin_f32(%x: f32) -> f32 {
    %y = math.sin %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_cos_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
  // CHECK: tensor.extract
  func.func @scalar_cos_f32(%x: f32) -> f32 {
    %y = math.cos %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_tan_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>}
  // CHECK: tensor.extract
  func.func @scalar_tan_f32(%x: f32) -> f32 {
    %y = math.tan %x : f32
    return %y : f32
  }

  // sqrt f32 is native (HIVMOptSinglePoint -> hivm.vsqrt).
  // CHECK-LABEL: func.func @scalar_sqrt_f32
  // CHECK-NOT: tensor.from_elements
  // CHECK: math.sqrt
  func.func @scalar_sqrt_f32(%x: f32) -> f32 {
    %y = math.sqrt %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_sqrt_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  // CHECK: tensor.extract
  func.func @scalar_sqrt_f16(%x: f16) -> f16 {
    %y = math.sqrt %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_sqrt_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_sqrt_bf16(%x: bf16) -> bf16 {
    %y = math.sqrt %x : bf16
    return %y : bf16
  }

  // rsqrt f32 is native (1/sqrt).
  // CHECK-LABEL: func.func @scalar_rsqrt_f32
  // CHECK-NOT: tensor.from_elements
  // CHECK: math.rsqrt
  func.func @scalar_rsqrt_f32(%x: f32) -> f32 {
    %y = math.rsqrt %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_rsqrt_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
  // CHECK: tensor.extract
  func.func @scalar_rsqrt_f16(%x: f16) -> f16 {
    %y = math.rsqrt %x : f16
    return %y : f16
  }

  // rsqrt bf16 uses the scalar bf16 fallback: extf -> rsqrt f32 -> truncf.
  // CHECK-LABEL: func.func @scalar_rsqrt_bf16
  // CHECK-NOT: tensor.from_elements
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: math.rsqrt {{.*}} : f32
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_rsqrt_bf16(%x: bf16) -> bf16 {
    %y = math.rsqrt %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_tanh_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
  // CHECK: tensor.extract
  func.func @scalar_tanh_f32(%x: f32) -> f32 {
    %y = math.tanh %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_erf_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
  // CHECK: tensor.extract
  func.func @scalar_erf_f32(%x: f32) -> f32 {
    %y = math.erf %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_roundeven_f32
  // CHECK: tensor.from_elements
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<nearbyint>}
  // CHECK: tensor.extract
  func.func @scalar_roundeven_f32(%x: f32) -> f32 {
    %y = math.roundeven %x : f32
    return %y : f32
  }

  // CHECK-LABEL: func.func @scalar_ceil_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // CHECK: tensor.extract
  func.func @scalar_ceil_f16(%x: f16) -> f16 {
    %y = math.ceil %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_ceil_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_ceil_bf16(%x: bf16) -> bf16 {
    %y = math.ceil %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_exp_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  // CHECK: tensor.extract
  func.func @scalar_exp_f16(%x: f16) -> f16 {
    %y = math.exp %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_exp_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_exp_bf16(%x: bf16) -> bf16 {
    %y = math.exp %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_exp2_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>}
  // CHECK: tensor.extract
  func.func @scalar_exp2_f16(%x: f16) -> f16 {
    %y = math.exp2 %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_exp2_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_exp2_bf16(%x: bf16) -> bf16 {
    %y = math.exp2 %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_expm1_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>}
  // CHECK: tensor.extract
  func.func @scalar_expm1_f16(%x: f16) -> f16 {
    %y = math.expm1 %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_expm1_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_expm1_bf16(%x: bf16) -> bf16 {
    %y = math.expm1 %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_log2_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>}
  // CHECK: tensor.extract
  func.func @scalar_log2_f16(%x: f16) -> f16 {
    %y = math.log2 %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_log2_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_log2_bf16(%x: bf16) -> bf16 {
    %y = math.log2 %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_log10_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log10>}
  // CHECK: tensor.extract
  func.func @scalar_log10_f16(%x: f16) -> f16 {
    %y = math.log10 %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_log10_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log10>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_log10_bf16(%x: bf16) -> bf16 {
    %y = math.log10 %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_log1p_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>}
  // CHECK: tensor.extract
  func.func @scalar_log1p_f16(%x: f16) -> f16 {
    %y = math.log1p %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_log1p_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_log1p_bf16(%x: bf16) -> bf16 {
    %y = math.log1p %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_sin_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
  // CHECK: tensor.extract
  func.func @scalar_sin_f16(%x: f16) -> f16 {
    %y = math.sin %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_sin_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_sin_bf16(%x: bf16) -> bf16 {
    %y = math.sin %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_cos_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
  // CHECK: tensor.extract
  func.func @scalar_cos_f16(%x: f16) -> f16 {
    %y = math.cos %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_cos_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_cos_bf16(%x: bf16) -> bf16 {
    %y = math.cos %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_tan_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>}
  // CHECK: tensor.extract
  func.func @scalar_tan_f16(%x: f16) -> f16 {
    %y = math.tan %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_tan_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_tan_bf16(%x: bf16) -> bf16 {
    %y = math.tan %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_tanh_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
  // CHECK: tensor.extract
  func.func @scalar_tanh_f16(%x: f16) -> f16 {
    %y = math.tanh %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_tanh_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_tanh_bf16(%x: bf16) -> bf16 {
    %y = math.tanh %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_erf_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
  // CHECK: tensor.extract
  func.func @scalar_erf_f16(%x: f16) -> f16 {
    %y = math.erf %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_erf_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_erf_bf16(%x: bf16) -> bf16 {
    %y = math.erf %x : bf16
    return %y : bf16
  }

  // CHECK-LABEL: func.func @scalar_roundeven_f16
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<nearbyint>}
  // CHECK: tensor.extract
  func.func @scalar_roundeven_f16(%x: f16) -> f16 {
    %y = math.roundeven %x : f16
    return %y : f16
  }

  // CHECK-LABEL: func.func @scalar_roundeven_bf16
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<nearbyint>}
  // CHECK: tensor.extract
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_roundeven_bf16(%x: bf16) -> bf16 {
    %y = math.roundeven %x : bf16
    return %y : bf16
  }
}
