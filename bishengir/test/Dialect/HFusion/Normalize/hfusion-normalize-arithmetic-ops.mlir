// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hfusion_rsqrt_to_hfusion_sqrt
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf32>)
// CHECK: %[[ONE:.*]]: tensor<5x1xf32>
// CHECK: %[[TWO:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<5x1xf32>) outs(%[[ONE:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[THREE:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[TWO:.*]]: tensor<5x1xf32>) outs(%[[ZERO:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[THREE:.*]]
func.func @test_NormalizeRSqrt_hfusion_rsqrt_to_hfusion_sqrt(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hfusion_rsqrt_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf16>)
// CHECK: %[[EMPTY0:.*]]: tensor<16xf16>
// CHECK: %[[EMPTY1:.*]]: tensor<16xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins(%[[arg0:.*]] : tensor<16xf16>) outs(%[[EMPTY1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]]: tensor<16xf32>
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins(%[[EMPTY0:.*]] : tensor<16xf16>) outs(%[[EMPTY2:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY3:.*]]: tensor<16xf32>
// CHECK: %[[SQRT0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[CAST0:.*]]: tensor<16xf32>) outs(%[[EMPTY3:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[REC0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[SQRT0:.*]]: tensor<16xf32>) outs(%[[CAST1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY4:.*]]: tensor<16xf16>
// CHECK: %[[CAST2:.*]] = hfusion.cast {{.*}} ins(%[[REC0:.*]] : tensor<16xf32>) outs(%[[EMPTY4:.*]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[REC0:.*]] : tensor<16xf16>
func.func @test_NormalizeRSqrt_hfusion_rsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hfusion_rsqrt_to_hfusion_sqrt_dynshape
// CHECK: %[[ARG0:.*]]: tensor<5x?xf32>, %[[ARG1:.*]]: index
// CHECK: %[[ONE:.*]]: tensor<5x?xf32>
// CHECK: %[[TWO:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<5x?xf32>) outs(%[[ONE:.*]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: %[[THREE:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[TWO:.*]]: tensor<5x?xf32>) outs(%[[ZERO:.*]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: return %[[THREE:.*]]
func.func @test_NormalizeRSqrt_hfusion_rsqrt_to_hfusion_sqrt_dynshape(%s: tensor<5x?xf32>, %d : index) -> tensor<5x?xf32> {
  %0 = tensor.empty(%d) : tensor<5x?xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%s : tensor<5x?xf32>) outs(%0 : tensor<5x?xf32>) -> tensor<5x?xf32>
  return %1 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivVSToRec_linalg_div_to_hfusion_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>)
// CHECK: %[[cast0:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<5x1xf16>)
// CHECK: %[[rec:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[cast0:.*]]: tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf32>)
// CHECK: %[[cast1:.*]] = hfusion.cast {{.*}} ins(%[[rec]] : tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf16>)
// CHECK: return %[[cast1]]
func.func @test_NormalizeDivVSToRec_linalg_div_to_hfusion_rec(%src: tensor<5x1xf16>) -> tensor<5x1xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty() : tensor<5x1xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %src : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %1 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivVSToRec_linalg_div_f32_no_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf32>)
// CHECK: %[[cst:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK-NOT: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
// CHECK: %[[div:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[cst]], %[[arg0]] : f32, tensor<5x1xf32>) outs(%[[empty]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[div]]
func.func @test_NormalizeDivVSToRec_linalg_div_f32_no_rec(%src: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %src : f32, tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivVSToRec_dyn_rec_mul
// CHECK: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[dim:.*]] = tensor.dim {{.*}}, %[[c0]] : tensor<?x14336xf16>
// CHECK: %[[empty:.*]] = tensor.empty(%[[dim]]) : tensor<?x14336xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}}, {{.*}} : tensor<?x14336xf16>, tensor<?x14336xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<?x14336xf16>) -> tensor<?x14336xf16>
func.func @test_NormalizeDivVSToRec_dyn_rec_mul(%arg0: tensor<?x4096xf16>, %arg1: tensor<14336x4096xf16>, %arg2: tensor<?x14336xf16>) -> tensor<?x14336xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
  %0 = tensor.empty(%dim) : tensor<?x14336xf16>
  %1 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x4096xf16>, tensor<14336x4096xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
  %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg2 : tensor<?x14336xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%1, %2 : tensor<?x14336xf16>, tensor<?x14336xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
  return %3 : tensor<?x14336xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_linalg_divi_to_divf
// CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHSFP:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHSFP:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSFP]] : tensor<48xf32>, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RET:.*]] = hfusion.cast {{.*}} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
func.func @test_NormalizeDivSI_linalg_divi_to_divf(%arg0: tensor<48xi32>, %arg1: tensor<48xi32>) -> tensor<48xi32> {
  %res = tensor.empty() : tensor<48xi32>
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
  return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_linalg_divi_to_divf_vs
// CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHSFP:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RHSCASTED:.*]] = arith.sitofp %arg1 : i32 to f32
// CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSCASTED]] : tensor<48xf32>, f32) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RET:.*]] = hfusion.cast {{.*}} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
func.func @test_NormalizeDivSI_linalg_divi_to_divf_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
  %res = tensor.empty() : tensor<48xi32>
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%res : tensor<48xi32>) -> tensor<48xi32>
  return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_linalg_divi_to_divf_sv
// CHECK: %[[LHSCASTED:.*]] = arith.sitofp %arg0 : i32 to f32
// CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHSFP:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSCASTED]], %[[RHSFP]] : f32, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RET:.*]] = hfusion.cast {{.*}} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
func.func @test_NormalizeDivSI_linalg_divi_to_divf_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
  %res = tensor.empty() : tensor<48xi32>
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
  return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_divui
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<6x6xi32>
// CHECK: %[[VAL_7:.*]] = hfusion.cast {{.*}}
func.func @test_NormalizeDivUI_divui(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_divui_vs
// CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHSFP:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RHSCASTED:.*]] = arith.sitofp %arg1 : i32 to f32
// CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSCASTED]] : tensor<48xf32>, f32) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RET:.*]] = hfusion.cast {{.*}} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
func.func @test_NormalizeDivUI_divui_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
  %res = tensor.empty() : tensor<48xi32>
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%res : tensor<48xi32>) -> tensor<48xi32>
  return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_divui_sv
// CHECK: %[[LHSCASTED:.*]] = arith.sitofp %arg0 : i32 to f32
// CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHSFP:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSCASTED]], %[[RHSFP]] : f32, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RET:.*]] = hfusion.cast {{.*}} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
func.func @test_NormalizeDivUI_divui_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
  %res = tensor.empty() : tensor<48xi32>
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
  return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_normalize_div_int
// CHECK: %[[cstMin:.*]] = arith.constant -128 : i16
// CHECK: %[[cstMax:.*]] = arith.constant 128 : i16
// CHECK: %[[lhsEmpty:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[lhs:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg0 : tensor<1024xi8>) outs(%[[lhsEmpty]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[rhsEmpty:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[rhs:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg1 : tensor<1024xi8>) outs(%[[rhsEmpty]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[divEmpty:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[div:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[lhs]], %[[rhs]] : tensor<1024xi16>, tensor<1024xi16>) outs(%[[divEmpty]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[maskEmpty:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[mask:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[div]], %[[cstMax]] : tensor<1024xi16>, i16) outs(%[[maskEmpty]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[selected:.*]] = hfusion.select ins(%[[mask]], %[[cstMin]], %[[div]] : tensor<1024xi1>, i16, tensor<1024xi16>) outs(%[[div]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[f16Empty:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[f16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, enable_saturate = true, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[selected]] : tensor<1024xi16>) outs(%[[f16Empty]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[i8Empty:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[result:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, enable_saturate = true, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[f16]] : tensor<1024xf16>) outs(%[[i8Empty]] : tensor<1024xi8>) -> tensor<1024xi8>
// CHECK: return %[[result]] : tensor<1024xi8>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeDivSI_normalize_div_int(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_normalize_div_uint_regbase
// CHECK: %[[lhsEmpty:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[lhs:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg0 : tensor<1024xi8>) outs(%[[lhsEmpty]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[rhsEmpty:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[rhs:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg1 : tensor<1024xi8>) outs(%[[rhsEmpty]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[divEmpty:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[div:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%[[lhs]], %[[rhs]] : tensor<1024xi16>, tensor<1024xi16>) outs(%[[divEmpty]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK-NOT: hfusion.compare
// CHECK-NOT: hfusion.select
// CHECK: %[[resEmpty:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[result:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[div]] : tensor<1024xi16>) outs(%[[resEmpty]] : tensor<1024xi8>) -> tensor<1024xi8>
// CHECK: return %[[result]] : tensor<1024xi8>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeDivUI_normalize_div_uint_regbase(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_uint
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins({{.*}}, {{.*}} : tensor<1024xi16>, tensor<1024xi16>) outs({{.*}} : tensor<1024xi16>) -> tensor<1024xi16>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeDivUI_uint(%arg0: tensor<1024xi16>, %arg1: tensor<1024xi16>) -> tensor<1024xi16> {
    %0 = tensor.empty() : tensor<1024xi16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<1024xi16>, tensor<1024xi16>) outs(%0 : tensor<1024xi16>) -> tensor<1024xi16>
    return %1 : tensor<1024xi16>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCDivandFloorDivInt_floordivsi
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_7:.*]] = hfusion.cast {{.*}}
func.func @test_NormalizeCDivandFloorDivInt_floordivsi(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCDivandFloorDivInt_ceildivsi
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_7:.*]] = hfusion.cast {{.*}}
func.func @test_NormalizeCDivandFloorDivInt_ceildivsi(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCDivandFloorDivInt_ceildivui
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {{.*}}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_7:.*]] = hfusion.cast {{.*}}
func.func @test_NormalizeCDivandFloorDivInt_ceildivui(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivui>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeMulExt_mulext_i8_high_bits
// CHECK: %[[cst_8:.*]] = arith.constant 8 : i16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg0_f16:.*]] = hfusion.cast {{.*}} ins(%[[arg0:.*]] : tensor<4x2xi8>) outs(%[[empty0:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg0_i16:.*]] = hfusion.cast {{.*}} ins(%[[arg0_f16:.*]] : tensor<4x2xf16>) outs(%[[empty1:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg1_f16:.*]] = hfusion.cast {{.*}} ins(%[[arg1:.*]] : tensor<4x2xi8>) outs(%[[empty2:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg1_i16:.*]] = hfusion.cast {{.*}} ins(%[[arg1_f16:.*]] : tensor<4x2xf16>) outs(%[[empty3:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0_i16:.*]], %[[arg1_i16:.*]] : tensor<4x2xi16>, tensor<4x2xi16>) outs(%[[empty4:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[res_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[mul:.*]], %[[cst_8:.*]] : tensor<4x2xi16>, i16) outs(%[[empty5:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<4x2xi8>
// CHECK: %[[res_i8:.*]] = hfusion.cast {{.*}} ins(%[[res_i16]] : tensor<4x2xi16>) outs(%[[empty6]] : tensor<4x2xi8>) -> tensor<4x2xi8>
// CHECK: return %[[res_i8]] : tensor<4x2xi8>
func.func @test_NormalizeMulExt_mulext_i8_high_bits(%arg0: tensor<4x2xi8>, %arg1: tensor<4x2xi8>) -> tensor<4x2xi8> {
  %low, %high = hfusion.mulext %arg0, %arg1 : tensor<4x2xi8>
  return %high : tensor<4x2xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeMulExt_mulext_i8_low_bits
// CHECK: %[[cst_8:.*]] = arith.constant 8 : i16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg0_f16:.*]] = hfusion.cast {{.*}} ins(%[[arg0:.*]] : tensor<4x2xi8>) outs(%[[empty0:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg0_i16:.*]] = hfusion.cast {{.*}} ins(%[[arg0_f16:.*]] : tensor<4x2xf16>) outs(%[[empty1:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg1_f16:.*]] = hfusion.cast {{.*}} ins(%[[arg1:.*]] : tensor<4x2xi8>) outs(%[[empty2:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg1_i16:.*]] = hfusion.cast {{.*}} ins(%[[arg1_f16:.*]] : tensor<4x2xf16>) outs(%[[empty3:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0_i16:.*]], %[[arg1_i16:.*]] : tensor<4x2xi16>, tensor<4x2xi16>) outs(%[[empty4:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[shl:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[mul:.*]], %[[cst_8:.*]] : tensor<4x2xi16>, i16) outs(%[[empty5:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[res_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[shl:.*]], %c8_i16 : tensor<4x2xi16>, i16) outs(%[[empty6:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<4x2xi8>
// CHECK: %[[res_i8:.*]] = hfusion.cast {{.*}} ins(%[[res_i16]] : tensor<4x2xi16>) outs(%[[empty7]] : tensor<4x2xi8>) -> tensor<4x2xi8>
// CHECK: return %[[res_i8]] : tensor<4x2xi8>
func.func @test_NormalizeMulExt_mulext_i8_low_bits(%arg0: tensor<4x2xi8>, %arg1: tensor<4x2xi8>) -> tensor<4x2xi8> {
  %low, %high = hfusion.mulext %arg0, %arg1 : tensor<4x2xi8>
  return  %low : tensor<4x2xi8>
}
