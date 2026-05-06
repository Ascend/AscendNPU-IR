// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins --split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck --enable-var-scope %s

// The organization of this test file is designed specifically to fully test every
// code path in AscendDPXMathOpsLowering.cpp. If the way math ops lowering changes,
// The organization of this file should change accordingly.

// CHECK-LABEL: @ascend_dpx_unary_direct_lowering_f32
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @ascend_dpx_unary_direct_lowering_f32(%arg1 : f32) {
    // CHECK: hivm_regbaseintrins.ceilf 
    // CHECK-SAME: %[[ARG]]
    %0 = ascend_dpx.ceil %arg1 : (f32) -> f32
    // CHECK: llvm.intr.exp
    // CHECK-SAME: %[[ARG]]
    %1 = ascend_dpx.exp %arg1 : (f32) -> f32
    // CHECK: hivm_regbaseintrins.floorf 
    // CHECK-SAME: %[[ARG]]
    %2 = ascend_dpx.floor %arg1 : (f32) -> f32
    // CHECK: llvm.intr.log
    // CHECK-SAME: %[[ARG]]
    %3 = ascend_dpx.log %arg1 : (f32) -> f32
    // CHECK: hivm_regbaseintrins.rintf 
    // CHECK-SAME: %[[ARG]]
    %4 = ascend_dpx.rint %arg1 : (f32) -> f32
    // CHECK: llvm.intr.sqrt
    // CHECK-SAME: %[[ARG]]
    %5 = ascend_dpx.sqrt %arg1 : (f32) -> f32
    return
}

// CHECK-LABEL: @ascend_dpx_unary_direct_lowering_f16
// CHECK-SAME: %[[ARG:.*]]: f16
func.func @ascend_dpx_unary_direct_lowering_f16(%arg1 : f16) {
    // CHECK: hivm_regbaseintrins.ceilh 
    // CHECK-SAME: %[[ARG]]
    %0 = ascend_dpx.ceil %arg1 : (f16) -> f16
    // CHECK: hivm_regbaseintrins.floorh 
    // CHECK-SAME: %[[ARG]]
    %1 = ascend_dpx.floor %arg1 : (f16) -> f16
    // CHECK: llvm.intr.log
    // CHECK-SAME: %[[ARG]]
    %2 = ascend_dpx.log %arg1 : (f16) -> f16
    // CHECK: hivm_regbaseintrins.rinth 
    // CHECK-SAME: %[[ARG]]
    %3 = ascend_dpx.rint %arg1 : (f16) -> f16
    // CHECK: llvm.intr.sqrt
    // CHECK-SAME: %[[ARG]]
    %4 = ascend_dpx.sqrt %arg1 : (f16) -> f16
    return
}

// CHECK-LABEL: @ascend_dpx_unary_direct_lowering_bf16
// CHECK-SAME: %[[ARG:.*]]: bf16
func.func @ascend_dpx_unary_direct_lowering_bf16(%arg1 : bf16) {
    // CHECK: hivm_regbaseintrins.ceily
    // CHECK-SAME: %[[ARG]]
    %0 = ascend_dpx.ceil %arg1 : (bf16) -> bf16
    // CHECK: hivm_regbaseintrins.floory
    // CHECK-SAME: %[[ARG]]
    %1 = ascend_dpx.floor %arg1 : (bf16) -> bf16
    // CHECK: llvm.intr.log
    // CHECK-SAME: %[[ARG]]
    %2 = ascend_dpx.log %arg1 : (bf16) -> bf16
    // CHECK: hivm_regbaseintrins.rinty
    // CHECK-SAME: %[[ARG]]
    %3 = ascend_dpx.rint %arg1 : (bf16) -> bf16
    // CHECK: llvm.intr.sqrt
    // CHECK-SAME: %[[ARG]]
    %4 = ascend_dpx.sqrt %arg1 : (bf16) -> bf16
    return
}

// CHECK-LABEL: @ascend_dpx_binary_direct_lowering_f32
// CHECK-SAME: %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32
func.func @ascend_dpx_binary_direct_lowering_f32(%arg1 : f32, %arg2 : f32) {
    // CHECK: llvm.fdiv %[[ARG1]], %[[ARG2]]
    %0 = ascend_dpx.div %arg1, %arg2 : (f32, f32) -> f32
    return
}

// CHECK-LABEL: @ascend_dpx_binary_direct_lowering_i32
// CHECK-SAME: %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32
func.func @ascend_dpx_binary_direct_lowering_i32(%arg1 : i32, %arg2 : i32) {
    // CHECK: llvm.sdiv %[[ARG1]], %[[ARG2]]
    %0 = ascend_dpx.div %arg1, %arg2 : (i32, i32) -> i32
    // CHECK: llvm.udiv %[[ARG1]], %[[ARG2]]
    %1 = ascend_dpx.udiv %arg1, %arg2 : (i32, i32) -> i32
    return
}

// CHECK-LABEL: @ascend_dpx_unary_libcall_lowering_f32
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @ascend_dpx_unary_libcall_lowering_f32(%arg1 : f32) {
    // CHECK: _mlir_ciface_simt_atan_float(%[[ARG]])
    %0 = ascend_dpx.atan %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_cos_float(%[[ARG]])
    %1 = ascend_dpx.cos %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_erf_float(%[[ARG]])
    %2 = ascend_dpx.erf %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_ilogb_float(%[[ARG]])
    %3 = ascend_dpx.ilogb %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_isfinite_float(%[[ARG]])
    %4 = ascend_dpx.isfinite %arg1 : (f32) -> i1
    // CHECK: _mlir_ciface_simt_isinf_float(%[[ARG]])
    %5 = ascend_dpx.isinf %arg1 : (f32) -> i1
    // CHECK: _mlir_ciface_simt_isnan_float(%[[ARG]])
    %6 = ascend_dpx.isnan %arg1 : (f32) -> i1
    // CHECK: _mlir_ciface_simt_log1p_float(%[[ARG]])
    %7 = ascend_dpx.log1p %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_log2_float(%[[ARG]])
    %8 = ascend_dpx.log2 %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_recip_float(%[[ARG]])
    %9 = ascend_dpx.recip %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_relu_float(%[[ARG]])
    %10 = ascend_dpx.relu %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_round_float(%[[ARG]])
    %11 = ascend_dpx.round %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_rsqrt_float(%[[ARG]])
    %12 = ascend_dpx.rsqrt %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_sin_float(%[[ARG]])
    %13 = ascend_dpx.sin %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_tan_float(%[[ARG]])
    %14 = ascend_dpx.tan %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_tanh_float(%[[ARG]])
    %15 = ascend_dpx.tanh %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_float_as_int_float(%[[ARG]])
    %16 = ascend_dpx.float_as_int %arg1 : (f32) -> i32
    // CHECK: _mlir_ciface_simt_trunc_float(%[[ARG]])
    %17 = ascend_dpx.trunc %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_nearbyint_float(%[[ARG]])
    %18 = ascend_dpx.nearbyint %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_log10_float(%[[ARG]])
    %19 = ascend_dpx.log10 %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_asin_float(%[[ARG]])
    %20 = ascend_dpx.asin %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_acos_float(%[[ARG]])
    %21 = ascend_dpx.acos %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_sinh_float(%[[ARG]])
    %22 = ascend_dpx.sinh %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_cosh_float(%[[ARG]])
    %23 = ascend_dpx.cosh %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_asinh_float(%[[ARG]])
    %24 = ascend_dpx.asinh %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_acosh_float(%[[ARG]])
    %25 = ascend_dpx.acosh %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_atanh_float(%[[ARG]])
    %26 = ascend_dpx.atanh %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_expm1_float(%[[ARG]])
    %27 = ascend_dpx.expm1 %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_cyl_bessel_i0_float(%[[ARG]])
    %28 = ascend_dpx.cyl_bessel_i0 %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_erfinv_float(%[[ARG]])
    %29 = ascend_dpx.erfinv %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_lgamma_float(%[[ARG]])
    %30 = ascend_dpx.lgamma %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_signbit_float(%[[ARG]])
    %31 = ascend_dpx.signbit %arg1 : (f32) -> i1
    // CHECK: _mlir_ciface_simt_abs_float(%[[ARG]])
    %32 = ascend_dpx.abs %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_saturatef_float(%[[ARG]])
    %35 = ascend_dpx.saturatef %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_exp10_float(%[[ARG]])
    %37 = ascend_dpx.exp10 %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_rcp_rn_float(%[[ARG]])
    %39 = ascend_dpx.rcp_rn %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_rcp_rz_float(%[[ARG]])
    %40 = ascend_dpx.rcp_rz %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_rcp_rd_float(%[[ARG]])
    %41 = ascend_dpx.rcp_rd %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_rcp_ru_float(%[[ARG]])
    %42 = ascend_dpx.rcp_ru %arg1 : (f32) -> f32
    // CHECK: _mlir_ciface_simt_rsqrt_rn_float(%[[ARG]])
    %43 = ascend_dpx.rsqrt_rn %arg1 : (f32) -> f32
    return
}

// CHECK-LABEL: @ascend_dpx_unary_libcall_lowering_f16
// CHECK-SAME: %[[ARG:.*]]: f16
func.func @ascend_dpx_unary_libcall_lowering_f16(%arg1 : f16) {
    // CHECK: @_mlir_ciface_simt_atan_half(%[[ARG]])
    %0 = ascend_dpx.atan %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_cos_half(%[[ARG]])
    %1 = ascend_dpx.cos %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_erf_half(%[[ARG]])
    %2 = ascend_dpx.erf %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_ilogb_half(%[[ARG]])
    %3 = ascend_dpx.ilogb %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_log1p_half(%[[ARG]])
    %4 = ascend_dpx.log1p %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_recip_half(%[[ARG]])
    %5 = ascend_dpx.recip %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_relu_half(%[[ARG]])
    %6 = ascend_dpx.relu %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_rsqrt_half(%[[ARG]])
    %7 = ascend_dpx.rsqrt %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_sin_half(%[[ARG]])
    %8 = ascend_dpx.sin %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_tan_half(%[[ARG]])
    %9 = ascend_dpx.tan %arg1 : (f16) -> f16
    // CHECK: @_mlir_ciface_simt_tanh_half(%[[ARG]])
    %10 = ascend_dpx.tanh %arg1 : (f16) -> f16
    return
}

// CHECK-LABEL: @ascend_dpx_unary_libcall_lowering_bf16
// CHECK-SAME: %[[ARG:.*]]: bf16
func.func @ascend_dpx_unary_libcall_lowering_bf16(%arg1 : bf16) {
    // CHECK: @_mlir_ciface_simt_cos_bfloat16_t(%[[ARG]])
    %0 = ascend_dpx.cos %arg1 : (bf16) -> bf16
    // CHECK: @_mlir_ciface_simt_erf_bfloat16_t(%[[ARG]])
    %1 = ascend_dpx.erf %arg1 : (bf16) -> bf16
    // CHECK: @_mlir_ciface_simt_rsqrt_bfloat16_t(%[[ARG]])
    %2 = ascend_dpx.rsqrt %arg1 : (bf16) -> bf16
    // CHECK: @_mlir_ciface_simt_sin_bfloat16_t(%[[ARG]])
    %3 = ascend_dpx.sin %arg1 : (bf16) -> bf16
    // CHECK: @_mlir_ciface_simt_tanh_bfloat16_t(%[[ARG]])
    %4 = ascend_dpx.tanh %arg1 : (bf16) -> bf16
    return
}

// CHECK-LABEL: @ascend_dpx_unary_libcall_lowering_int
func.func @ascend_dpx_unary_libcall_lowering_int(%arg : i8) {
    %arg_i16 = arith.extsi %arg : i8 to i16
    %arg_i32 = arith.extsi %arg : i8 to i32
    %arg_i64 = arith.extsi %arg : i8 to i64
    // CHECK: llvm.call @_mlir_ciface_simt_abs_int32_t
    %0 = ascend_dpx.abs %arg_i32 : (i32) -> i32
    // CHECK: llvm.call @_mlir_ciface_simt_clz_int32_t
    %1 = ascend_dpx.clz %arg_i32 : (i32) -> i32
    // CHECK: llvm.call @_mlir_ciface_simt_brev_int32_t
    %2 = ascend_dpx.brev %arg_i32 : (i32) -> i32
    // CHECK: llvm.call @_mlir_ciface_simt_ffs_int32_t
    %3 = ascend_dpx.ffs %arg_i32 : (i32) -> i32
    // CHECK: llvm.call @_mlir_ciface_simt_popc_int32_t
    %4 = ascend_dpx.popc %arg_i32 : (i32) -> i32
    return
}
 
// CHECK-LABEL: @ascend_dpx_binary_libcall_lowering_int
func.func @ascend_dpx_binary_libcall_lowering_int(%arg1 : i8, %arg2 : i8) {
    %arg1_i16 = arith.extsi %arg1 : i8 to i16
    %arg2_i16 = arith.extsi %arg2 : i8 to i16
    %arg1_i32 = arith.extsi %arg1 : i8 to i32
    %arg2_i32 = arith.extsi %arg2 : i8 to i32
    %arg1_i64 = arith.extsi %arg1 : i8 to i64
    %arg2_i64 = arith.extsi %arg2 : i8 to i64
    // CHECK: llvm.call @_mlir_ciface_simt_pow_int8_t
    %0 = ascend_dpx.pow %arg1, %arg2 : (i8, i8) -> i8
    // CHECK: llvm.call @_mlir_ciface_simt_pow_int16_t
    %1 = ascend_dpx.pow %arg1_i16, %arg2_i16 : (i16, i16) -> i16
    // CHECK: llvm.call @_mlir_ciface_simt_pow_int32_t
    %2 = ascend_dpx.pow %arg1_i32, %arg2_i32 : (i32, i32) -> i32
    // CHECK: llvm.call @_mlir_ciface_simt_pow_int64_t
    %3 = ascend_dpx.pow %arg1_i64, %arg2_i64 : (i64, i64) -> i64
    // CHECK: _mlir_ciface_simt_umulhi_uint32_t
    %4 = ascend_dpx.umulhi %arg1_i32, %arg2_i32 : (i32, i32) -> i32
    // CHECK: _mlir_ciface_simt_mulhi_int32_t
    %5 = ascend_dpx.mulhi %arg1_i32, %arg2_i32 : (i32, i32) -> i32
    // CHECK: _mlir_ciface_simt_mul24_int32_t
    %6 = ascend_dpx.mul24 %arg1_i32, %arg2_i32 : (i32, i32) -> i32
    // CHECK: _mlir_ciface_simt_hadd_int32_t
    %7 = ascend_dpx.hadd %arg1_i32, %arg2_i32 : (i32, i32) -> i32
    // CHECK: _mlir_ciface_simt_rhadd_int32_t
    %8 = ascend_dpx.rhadd %arg1_i32, %arg2_i32 : (i32, i32) -> i32
    return
}

// CHECK-LABEL: @ascend_dpx_binary_libcall_lowering_f32
// CHECK-SAME: %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32
func.func @ascend_dpx_binary_libcall_lowering_f32(%arg1 : f32, %arg2 : f32) {
    // CHECK: @_mlir_ciface_simt_pow_float(%[[ARG1]], %[[ARG2]])
    %0 = ascend_dpx.pow %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_ldexp_float(%[[ARG1]], %[[ARG2]])
    %1 = ascend_dpx.ldexp %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_copysign_float(%[[ARG1]], %[[ARG2]])
    %2 = ascend_dpx.copysign %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_atan2_float(%[[ARG1]], %[[ARG2]])
    %3 = ascend_dpx.atan2 %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_nextafter_float(%[[ARG1]], %[[ARG2]])
    %4 = ascend_dpx.nextafter %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_hypot_float(%[[ARG1]], %[[ARG2]])
    %5 = ascend_dpx.hypot %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_fdim_float(%[[ARG1]], %[[ARG2]])
    %6 = ascend_dpx.fdim %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_fast_dividef_float(%[[ARG1]], %[[ARG2]])
    %7 = ascend_dpx.fast_dividef %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_div_rn_float(%[[ARG1]], %[[ARG2]])
    %8 = ascend_dpx.div_rn %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_div_rz_float(%[[ARG1]], %[[ARG2]])
    %9 = ascend_dpx.div_rz %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_div_rd_float(%[[ARG1]], %[[ARG2]])
    %10 = ascend_dpx.div_rd %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_div_ru_float(%[[ARG1]], %[[ARG2]])
    %11 = ascend_dpx.div_ru %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_fmod_float(%[[ARG1]], %[[ARG2]])
    %12 = ascend_dpx.fmod %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_remainder_float(%[[ARG1]], %[[ARG2]])
    %13 = ascend_dpx.remainder %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_add_rn_float(%[[ARG1]], %[[ARG2]])
    %14 = ascend_dpx.add_rn %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_add_rz_float(%[[ARG1]], %[[ARG2]])
    %15 = ascend_dpx.add_rz %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_add_rd_float(%[[ARG1]], %[[ARG2]])
    %16 = ascend_dpx.add_rd %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_add_ru_float(%[[ARG1]], %[[ARG2]])
    %17 = ascend_dpx.add_ru %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_mul_rn_float(%[[ARG1]], %[[ARG2]])
    %18 = ascend_dpx.mul_rn %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_mul_rz_float(%[[ARG1]], %[[ARG2]])
    %19 = ascend_dpx.mul_rz %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_mul_rd_float(%[[ARG1]], %[[ARG2]])
    %20 = ascend_dpx.mul_rd %arg1, %arg2 : (f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_mul_ru_float(%[[ARG1]], %[[ARG2]])
    %21 = ascend_dpx.mul_ru %arg1, %arg2 : (f32, f32) -> f32
    return
}

// CHECK-LABEL: @ascend_dpx_binary_libcall_lowering_f16
// CHECK-SAME: %[[ARG1:.*]]: f16, %[[ARG2:.*]]: f16
func.func @ascend_dpx_binary_libcall_lowering_f16(%arg1 : f16, %arg2 : f16) {
    // CHECK: @_mlir_ciface_simt_pow_half(%[[ARG1]], %[[ARG2]])
    %0 = ascend_dpx.pow %arg1, %arg2 : (f16, f16) -> f16
    // CHECK: @_mlir_ciface_simt_ldexp_half(%[[ARG1]], %[[ARG2]])
    %1 = ascend_dpx.ldexp %arg1, %arg2 : (f16, f16) -> f16
    return
}

// CHECK-LABEL: @ascend_dpx_binary_libcall_lowering_bf16
// CHECK-SAME: %[[ARG1:.*]]: bf16, %[[ARG2:.*]]: bf16
func.func @ascend_dpx_binary_libcall_lowering_bf16(%arg1 : bf16, %arg2 : bf16) {
    // CHECK: @_mlir_ciface_simt_pow_bfloat16_t(%[[ARG1]], %[[ARG2]])
    %0 = ascend_dpx.pow %arg1, %arg2 : (bf16, bf16) -> bf16
    return
}

// CHECK-LABEL: @ascend_dpx_ternary_libcall_lowering_int
func.func @ascend_dpx_ternary_libcall_lowering_int(%arg1 : i8, %arg2 : i8, %arg3 : i8) {
    %arg1_i32 = arith.extsi %arg1 : i8 to i32
    %arg2_i32 = arith.extsi %arg2 : i8 to i32
    %arg3_i32 = arith.extsi %arg3 : i8 to i32
    
    // CHECK: @_mlir_ciface_simt_byte_perm_int32_t
    %0 = ascend_dpx.byte_perm %arg1_i32, %arg2_i32, %arg3_i32 : (i32, i32, i32) -> i32
    // CHECK: @_mlir_ciface_simt_sad_int32_t
    %1 = ascend_dpx.sad %arg1_i32, %arg2_i32, %arg3_i32 : (i32, i32, i32) -> i32
    return
}
 
// CHECK-LABEL: @ascend_dpx_ternary_libcall_lowering_f32
// CHECK-SAME: %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32
func.func @ascend_dpx_ternary_libcall_lowering_f32(%arg1 : f32, %arg2 : f32, %arg3 : f32) {
    // CHECK: @_mlir_ciface_simt_fma_rn_float(%[[ARG1]], %[[ARG2]], %[[ARG3]])
    %0 = ascend_dpx.fma_rn %arg1, %arg2, %arg3 : (f32, f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_fma_rz_float(%[[ARG1]], %[[ARG2]], %[[ARG3]])
    %1 = ascend_dpx.fma_rz %arg1, %arg2, %arg3 : (f32, f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_fma_rd_float(%[[ARG1]], %[[ARG2]], %[[ARG3]])
    %2 = ascend_dpx.fma_rd %arg1, %arg2, %arg3 : (f32, f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_fma_ru_float(%[[ARG1]], %[[ARG2]], %[[ARG3]])
    %3 = ascend_dpx.fma_ru %arg1, %arg2, %arg3 : (f32, f32, f32) -> f32
    // CHECK: @_mlir_ciface_simt_fma_float(%[[ARG1]], %[[ARG2]], %[[ARG3]])
    %4 = ascend_dpx.fma %arg1, %arg2, %arg3 : (f32, f32, f32) -> f32
    return
}

// CHECK-LABEL: @ascend_dpx_exp2_lowering_f32
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @ascend_dpx_exp2_lowering_f32(%arg1 : f32) {
    // CHECK: %[[CONST:.*]] =
    // CHECK-SAME: constant
    // CHECK-SAME: 0.69
    // CHECK-NEXT: %[[MUL:.*]] = llvm.fmul %[[ARG]]
    // CHECK-NEXT: llvm.intr.exp(%[[MUL]])
    %0 = ascend_dpx.exp2 %arg1 : (f32) -> f32
    return
}
