// RUN: bishengir-opt %s -debug-only=triton-remap -triton-remap | FileCheck %s

// CHECK-DAG: llvm.func @_mlir_ciface_simt_tanh_float(f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_tanh_half(f16, f16) -> f16
// CHECK-DAG: llvm.func @_mlir_ciface_simt_rsqrt_float(f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_sin_float(f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_pow_float(f32, f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_umulhi_uint32_t(i32, i32) -> i32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_log1p_float(f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_log1p_half(f16) -> f16
// CHECK-DAG: llvm.func @_mlir_ciface_simt_atan_float(f32, f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_atan_half(f16, f16) -> f16
// CHECK-DAG: llvm.func @_mlir_ciface_simt_ilogb_float(f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_ilogb_half(f16) -> f16
// CHECK-DAG: llvm.func @_mlir_ciface_simt_relu_float(f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_relu_half(f16) -> f16
// CHECK-DAG: llvm.func @_mlir_ciface_simt_round_float(f32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_ldexp_float(f32, i32) -> f32
// CHECK-DAG: llvm.func @_mlir_ciface_simt_ldexp_half(f16, i32) -> f16
// CHECK-DAG: llvm.func @_mlir_ciface_simt_log2_float(f32) -> f32

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @remap_rsqrt
  llvm.func @remap_rsqrt(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__nv_rsqrtf
    // CHECK: llvm.call @_mlir_ciface_simt_rsqrt_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__nv_rsqrtf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__nv_rsqrtf(f32) -> f32

  // CHECK-LABEL: @remap_tanhf
  llvm.func @remap_tanhf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_tanhf
    // CHECK: llvm.call @_mlir_ciface_simt_tanh_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_tanhf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_tanhf(f32) -> f32

  // CHECK-LABEL: @remap_tanhDh
  llvm.func @remap_tanhDh(%arg11: f16, %arg12: f16)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_tanhDh
    // CHECK: llvm.call @_mlir_ciface_simt_tanh_half
    %0 = llvm.fadd %arg11, %arg12 : f16
    %1 = llvm.call @__hmf_tanhDh(%0, %0) : (f16, f16) -> f16
    llvm.return
  }
  llvm.func @__hmf_tanhDh(f16, f16) -> f16

  // CHECK-LABEL: @remap_sinf
  llvm.func @remap_sinf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__nv_sinf
    // CHECK: llvm.call @_mlir_ciface_simt_sin_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__nv_sinf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__nv_sinf(f32) -> f32
  
  // CHECK-LABEL: @remap_cosf
  llvm.func @remap_cosf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__nv_cosf
    // CHECK: llvm.call @_mlir_ciface_simt_cos_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__nv_cosf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__nv_cosf(f32) -> f32

  // CHECK-LABEL: @remap_powf
  llvm.func @remap_powf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_powf
    // CHECK: llvm.call @_mlir_ciface_simt_pow_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_powf(%0, %0) : (f32, f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_powf(f32, f32) -> f32

  // CHECK-LABEL: @remap_tanf
  llvm.func @remap_tanf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_tanf
    // CHECK: llvm.call @_mlir_ciface_simt_tan_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_tanf(%0, %0) : (f32, f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_tanf(f32, f32) -> f32

  // CHECK-LABEL: @remap_tanDh
  llvm.func @remap_tanDh(%arg11: f16, %arg12: f16)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_tanDh
    // CHECK: llvm.call @_mlir_ciface_simt_tan_half
    %0 = llvm.fadd %arg11, %arg12 : f16
    %1 = llvm.call @__hmf_tanDh(%0, %0) : (f16, f16) -> f16
    llvm.return
  }
  llvm.func @__hmf_tanDh(f16, f16) -> f16

    // CHECK-LABEL: @remap_powi
  llvm.func @remap_powi(%arg11: i32, %arg12: i32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_powi
    // CHECK: llvm.call @_mlir_ciface_simt_pow_int32_t
    %0 = llvm.add %arg11, %arg12 : i32
    %1 = llvm.call @__hmf_powi(%0, %0) : (i32, i32) -> i32
    llvm.return
  }
  llvm.func @__hmf_powi(i32, i32) -> i32

   // CHECK-LABEL: @remap_umulhi
  llvm.func @remap_umulhi(%arg11: i32, %arg12: i32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__nv_umulhi
    // CHECK: llvm.call @_mlir_ciface_simt_umulhi_uint32_t
    %0 = llvm.add %arg11, %arg12 : i32
    %1 = llvm.call @__nv_umulhi(%0, %0) : (i32, i32) -> i32
    llvm.return
  }
  llvm.func @__nv_umulhi(i32,i32) -> i32

  // CHECK-LABEL: @remap_recipf
  llvm.func @remap_recipf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_recipf
    // CHECK: llvm.call @_mlir_ciface_simt_recip_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_recipf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_recipf(f32) -> f32

  // CHECK-LABEL: @remap_recipDh
  llvm.func @remap_recipDh(%arg11: f16, %arg12: f16)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_recipDh
    // CHECK: llvm.call @_mlir_ciface_simt_recip_half
    %0 = llvm.fadd %arg11, %arg12 : f16
    %1 = llvm.call @__hmf_recipDh(%0) : (f16) -> f16
    llvm.return
  }
  llvm.func @__hmf_recipDh(f16) -> f16

  // CHECK-LABEL: @remap_ldexpf
  llvm.func @remap_ldexpf(%arg11: f32, %arg12: i32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_ldexpf
    // CHECK: llvm.call @_mlir_ciface_simt_ldexp_float
    %1 = llvm.call @__hmf_ldexpf(%arg11, %arg12) : (f32, i32) -> f32
    llvm.return
  }
  llvm.func @__hmf_ldexpf(f32, i32) -> f32

  // CHECK-LABEL: @remap_ldexpDh
  llvm.func @remap_ldexpDh(%arg11: f16, %arg12: i32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_ldexpDh
    // CHECK: llvm.call @_mlir_ciface_simt_ldexp_half
    %1 = llvm.call @__hmf_ldexpDh(%arg11, %arg12) : (f16, i32) -> f16
    llvm.return
  }
  llvm.func @__hmf_ldexpDh(f16, i32) -> f16

  // CHECK-LABEL: @remap_log1p
  llvm.func @remap_log1p(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_log1pf
    // CHECK: llvm.call @_mlir_ciface_simt_log1p_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_log1pf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_log1pf(f32) -> f32

  // CHECK-LABEL: @remap_log1p_half
  llvm.func @remap_log1p_half(%arg11: f16, %arg12: f16)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_log1pDh
    // CHECK: llvm.call @_mlir_ciface_simt_log1p_half
    %0 = llvm.fadd %arg11, %arg12 : f16
    %1 = llvm.call @__hmf_log1pDh(%0) : (f16) -> f16
    llvm.return
  }
  llvm.func @__hmf_log1pDh(f16) -> f16

  // CHECK-LABEL: @remap_log2
  llvm.func @remap_log2(%arg11: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__nv_log2f
    // CHECK: llvm.call @_mlir_ciface_simt_log2_float
    %0 = llvm.call @__nv_log2f(%arg11) : (f32) -> f32
    llvm.return
  }
  llvm.func @__nv_log2f(f32) -> f32

  // CHECK-LABEL: @remap_atanf
  llvm.func @remap_atanf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_atanf
    // CHECK: llvm.call @_mlir_ciface_simt_atan_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_atanf(%0, %0) : (f32, f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_atanf(f32, f32) -> f32

  // CHECK-LABEL: @remap_atanDh
  llvm.func @remap_atanDh(%arg11: f16, %arg12: f16)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_atanDh
    // CHECK: llvm.call @_mlir_ciface_simt_atan_half
    %0 = llvm.fadd %arg11, %arg12 : f16
    %1 = llvm.call @__hmf_atanDh(%0, %0) : (f16, f16) -> f16
    llvm.return
  }
  llvm.func @__hmf_atanDh(f16, f16) -> f16

  // CHECK-LABEL: @remap_ilogb
  llvm.func @remap_ilogb(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_ilogbf
    // CHECK: llvm.call @_mlir_ciface_simt_ilogb_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_ilogbf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_ilogbf(f32) -> f32

  // CHECK-LABEL: @remap_ilogb_half
  llvm.func @remap_ilogb_half(%arg11: f16, %arg12: f16)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_ilogbDh
    // CHECK: llvm.call @_mlir_ciface_simt_ilogb_half
    %0 = llvm.fadd %arg11, %arg12 : f16
    %1 = llvm.call @__hmf_ilogbDh(%0) : (f16) -> f16
    llvm.return
  }
  llvm.func @__hmf_ilogbDh(f16) -> f16

  // CHECK-LABEL: @remap_reluf
  llvm.func @remap_reluf(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_reluf
    // CHECK: llvm.call @_mlir_ciface_simt_relu_float
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_reluf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_reluf(f32) -> f32

  // CHECK-LABEL: @remap_reludh
  llvm.func @remap_reludh(%arg11: f16, %arg12: f16)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_reluDh
    // CHECK: llvm.call @_mlir_ciface_simt_relu_half
    %0 = llvm.fadd %arg11, %arg12 : f16
    %1 = llvm.call @__hmf_reluDh(%0) : (f16) -> f16
    llvm.return
  }
  llvm.func @__hmf_reluDh(f16) -> f16

  // CHECK-LABEL: @remap_round
  llvm.func @remap_round(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_roundf
    // CHECK: llvm.call @_mlir_ciface_simt_round
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__hmf_roundf(%0) : (f32) -> f32
    llvm.return
  }
  llvm.func @__hmf_roundf(f32) -> f32

  // CHECK-LABEL: @remap_divrn
  llvm.func @remap_divrn(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__nv_fdiv_rn
    // CHECK: llvm.call @_mlir_ciface_simt_divrn
    %0 = llvm.fadd %arg11, %arg12 : f32
    %1 = llvm.call @__nv_fdiv_rn(%0, %0) : (f32, f32) -> f32
    llvm.return
  }
  llvm.func @__nv_fdiv_rn(f32, f32) -> f32

  // CHECK-LABEL: @remap_float_as_int
  llvm.func @remap_float_as_int(%arg11: f32)
    attributes {nvvm.kernel = 1 : ui1} {
    // CHECK-NOT: llvm.call @__hmf_float_as_int_fp32
    // CHECK: llvm.call @_mlir_ciface_simt_float_as_int_float
    %0 = llvm.call @__hmf_float_as_int_fp32(%arg11) : (f32) -> i32
    llvm.return
  }
  llvm.func @__hmf_float_as_int_fp32(f32) -> i32
}
