// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s
// This test verifies the correctness of HIVM normalization transformations
// applied to HFusion operators after conversion to HIVM dialect.

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf32>)
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<5x1xf32>) outs(%{{.*}} : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<5x1xf32>) outs(%{{.*}} : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> {
    %0 = tensor.empty() : tensor<5x1xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
    return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>)
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[VSQRT]] : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[REC:.*]] = hivm.hir.vrec ins(%[[CAST0]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[REC]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
    %0 = tensor.empty() : tensor<16xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
    return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt_dynshape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x?xf32>, %[[ARG1:.*]]: index)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<5x?xf32>
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<5x?xf32>) outs(%{{.*}} : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<5x?xf32>) outs(%{{.*}} : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt_dynshape(%s: tensor<5x?xf32>, %d : index) -> tensor<5x?xf32> {
    %0 = tensor.empty(%d) : tensor<5x?xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%s : tensor<5x?xf32>) outs(%0 : tensor<5x?xf32>) -> tensor<5x?xf32>
    return %1 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_hivm_mul_div_by_one
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f16) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[VDIV:.*]] = hivm.hir.vdiv ins(%[[VBRC]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[VMUL0:.*]] = hivm.hir.vmul ins(%[[VDIV]], %[[ARG1]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[VMUL1:.*]] = hivm.hir.vmul ins(%[[VMUL0]], %[[VDIV]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[VMUL1]]
func.func @test_NormalizeMulRec_hivm_mul_div_by_one(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %arg0 : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_hivm_mul_div_by_one_rec
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV0:.*]] = hivm.hir.vdiv ins(%[[ARG1]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV1:.*]] = hivm.hir.vdiv ins(%[[DIV0]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[DIV1]]
func.func @test_NormalizeMulRec_hivm_mul_div_by_one_rec(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSubVSToVMulAndVAdd_hivm
// CHECK: %[[NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY_ADD:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[EMPTY_MUL:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%arg0, %[[NEG1]] : tensor<16xf32>, f32) outs(%[[EMPTY_MUL]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ADD:.*]] = hivm.hir.vadd ins(%[[MUL]], %arg1 : tensor<16xf32>, f32) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[ADD]]
func.func @test_NormalizeSubVSToVMulAndVAdd_hivm(%arg0: tensor<16xf32>, %arg1: f32) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg1, %arg0 : f32, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSubVSToVMulAndVAdd_hivm_zero_scalar_ascend950
// CHECK: %[[NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%arg0, %[[NEG1]] : tensor<16xf32>, f32) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK-NOT: hivm.hir.vadd
// CHECK: return %[[MUL]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeSubVSToVMulAndVAdd_hivm_zero_scalar_ascend950(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%c0, %arg0 : f32, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_linalg_divi_to_divf
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<48xi32>) outs(%[[LHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<48xi32>) outs(%[[RHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<48xf32>, tensor<48xf32>) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_linalg_divi_to_divf(%arg0: tensor<48xi32>, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%0 : tensor<48xi32>) -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_linalg_divi_to_divf_vs
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<48xi32>) outs(%[[LHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[RHS:.*]] = arith.sitofp %arg1 : i32 to f32
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<48xf32>, f32) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_linalg_divi_to_divf_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%0 : tensor<48xi32>) -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_linalg_divi_to_divf_sv
// CHECK: %[[FROM:.*]] = tensor.from_elements %arg0 : tensor<1xi32>
// CHECK: %[[CAST0_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[FROM]] : tensor<1xi32>) outs(%[[CAST0_EMPTY]] : tensor<1xf32>) round_mode = <trunc> -> tensor<1xf32>
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[CAST0]][%{{.*}}] : tensor<1xf32>
// CHECK: %[[BRC_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACT]] : f32) outs(%[[BRC_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<48xi32>) outs(%[[RHS_EMPTY:.*]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[BRC]], %[[RHS]] : tensor<48xf32>, tensor<48xf32>) outs(%[[DIV_EMPTY:.*]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY:.*]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_linalg_divi_to_divf_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%0 : tensor<48xi32>) -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_divui
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<6x6xi32>) outs(%[[LHS_EMPTY:.*]] : tensor<6x6xf32>) round_mode = <trunc> -> tensor<6x6xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<6x6xi32>) outs(%[[RHS_EMPTY:.*]] : tensor<6x6xf32>) round_mode = <trunc> -> tensor<6x6xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[DIV_EMPTY:.*]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<6x6xf32>) outs(%[[RES_EMPTY:.*]] : tensor<6x6xi32>) round_mode = <trunc> -> tensor<6x6xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_divui(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty() : tensor<6x6xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_divui_vs
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<48xi32>) outs(%[[LHS_EMPTY:.*]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[RHS:.*]] = arith.sitofp %arg1 : i32 to f32
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<48xf32>, f32) outs(%[[DIV_EMPTY:.*]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY:.*]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_divui_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%0 : tensor<48xi32>) -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_divui_sv
// CHECK: %[[FROM:.*]] = tensor.from_elements %arg0 : tensor<1xi32>
// CHECK: %[[CAST0_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[FROM]] : tensor<1xi32>) outs(%[[CAST0_EMPTY]] : tensor<1xf32>) round_mode = <trunc> -> tensor<1xf32>
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[CAST0]][%{{.*}}] : tensor<1xf32>
// CHECK: %[[BRC_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACT]] : f32) outs(%[[BRC_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<48xi32>) outs(%[[RHS_EMPTY:.*]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[BRC]], %[[RHS]] : tensor<48xf32>, tensor<48xf32>) outs(%[[DIV_EMPTY:.*]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY:.*]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_divui_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%0 : tensor<48xi32>) -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_linalg_divi_to_i8_divi16
// CHECK: %[[CSTMIN:.*]] = arith.constant -128 : i16
// CHECK: %[[CSTMAX:.*]] = arith.constant 128 : i16
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[MASK:.*]] = hivm.hir.vcmp ins(%[[DIV]], %[[CSTMAX]] : tensor<1024xi16>, i16) outs(%[[MASK_EMPTY:.*]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[MIN:.*]] = hivm.hir.vbrc ins(%[[CSTMIN]] : i16) outs(%[[MIN_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[SELECT:.*]] = hivm.hir.vsel ins(%[[MASK]], %[[MIN]], %[[DIV]] : tensor<1024xi1>, tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[F16:.*]] = hivm.hir.vcast ins(%[[SELECT]] : tensor<1024xi16>) outs(%[[F16_EMPTY]] : tensor<1024xf16>) round_mode = <trunc> -> tensor<1024xf16>
// CHECK: %[[I8_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[F16]] : tensor<1024xf16>) outs(%[[I8_EMPTY]] : tensor<1024xi8>) round_mode = <trunc> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeDivSI_linalg_divi_to_i8_divi16(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_membase_i8
// CHECK: %[[LHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[LHS_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[LHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LHS_F32:.*]] = hivm.hir.vcast ins(%[[LHS_F16]] : tensor<1024xf16>) outs(%[[LHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RHS_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[RHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RHS_F32:.*]] = hivm.hir.vcast ins(%[[RHS_F16]] : tensor<1024xf16>) outs(%[[RHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS_F32]], %[[RHS_F32]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DIV_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[I32_EMPTY:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK: %[[I32:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xf32>) outs(%[[I32_EMPTY]] : tensor<1024xi32>) round_mode = <trunc> -> tensor<1024xi32>
// CHECK: %[[I8_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[I32]] : tensor<1024xi32>) outs(%[[I8_EMPTY]] : tensor<1024xi8>) round_mode = <truncwithoverflow> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_membase_i8(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_normalize_div_int
// CHECK: %[[CSTMIN:.*]] = arith.constant -128 : i16
// CHECK: %[[CSTMAX:.*]] = arith.constant 128 : i16
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[MASK:.*]] = hivm.hir.vcmp ins(%[[DIV]], %[[CSTMAX]] : tensor<1024xi16>, i16) outs(%[[MASK_EMPTY:.*]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[MIN:.*]] = hivm.hir.vbrc ins(%[[CSTMIN]] : i16) outs(%[[MIN_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[SELECT:.*]] = hivm.hir.vsel ins(%[[MASK]], %[[MIN]], %[[DIV]] : tensor<1024xi1>, tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[F16:.*]] = hivm.hir.vcast ins(%[[SELECT]] : tensor<1024xi16>) outs(%[[F16_EMPTY]] : tensor<1024xf16>) round_mode = <trunc> -> tensor<1024xf16>
// CHECK: %[[I8_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[F16]] : tensor<1024xf16>) outs(%[[I8_EMPTY]] : tensor<1024xi8>) round_mode = <trunc> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeDivSI_normalize_div_int(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_regbase_i8_no_ascend950
// CHECK: %[[LHS_EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[LHS0:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_EMPTY0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[LHS_EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%[[LHS0]] : tensor<1024xf16>) outs(%[[LHS_EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RHS_EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RHS0:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_EMPTY0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[RHS_EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%[[RHS0]] : tensor<1024xf16>) outs(%[[RHS_EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DIV_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES_EMPTY0:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK: %[[RES0:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xf32>) outs(%[[RES_EMPTY0]] : tensor<1024xi32>) round_mode = <trunc> -> tensor<1024xi32>
// CHECK: %[[RES_EMPTY1:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[RES0]] : tensor<1024xi32>) outs(%[[RES_EMPTY1]] : tensor<1024xi8>) round_mode = <truncwithoverflow> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_regbase_i8_no_ascend950(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_normalize_div_uint_regbase
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_EMPTY]] : tensor<1024xi16>) cast = <cast_unsigned> -> tensor<1024xi16>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_EMPTY]] : tensor<1024xi16>) cast = <cast_unsigned> -> tensor<1024xi16>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV_EMPTY]] : tensor<1024xi16>) isSigned = false -> tensor<1024xi16>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xi16>) outs(%[[RES_EMPTY]] : tensor<1024xi8>) round_mode = <trunc> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeDivUI_normalize_div_uint_regbase(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_membase_i8
// CHECK: %[[LHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[LHS_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[LHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LHS_F32:.*]] = hivm.hir.vcast ins(%[[LHS_F16]] : tensor<1024xf16>) outs(%[[LHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RHS_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[RHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RHS_F32:.*]] = hivm.hir.vcast ins(%[[RHS_F16]] : tensor<1024xf16>) outs(%[[RHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS_F32]], %[[RHS_F32]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DIV_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[I32_EMPTY:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK: %[[I32:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xf32>) outs(%[[I32_EMPTY]] : tensor<1024xi32>) round_mode = <trunc> -> tensor<1024xi32>
// CHECK: %[[I8_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[I32]] : tensor<1024xi32>) outs(%[[I8_EMPTY]] : tensor<1024xi8>) round_mode = <truncwithoverflow> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_membase_i8(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}
