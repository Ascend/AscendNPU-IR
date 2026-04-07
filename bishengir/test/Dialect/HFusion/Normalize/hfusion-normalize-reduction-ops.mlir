// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_NormalizeArgMinMax_normalize_reduce_with_index_ra_to_ar
// CHECK: %[[res0:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK: %[[res1:.*]] = tensor.empty() : tensor<32x128xi32>
// CHECK: %[[tmp_buf0:.*]] = tensor.empty() : tensor<32x128x32xf32>
// CHECK: %[[transposed0:.*]] = linalg.transpose ins(%[[arg0:.*]] : tensor<32x32x128xf32>) outs(%[[tmp_buf0]] : tensor<32x128x32xf32>) permutation = [0, 2, 1]
// CHECK: %[[tmp_buf1:.*]] = tensor.empty() : tensor<32x128x32xi32>
// CHECK: %[[transposed1:.*]] = linalg.transpose ins(%[[arg1:.*]] : tensor<32x32x128xi32>) outs(%[[tmp_buf1]] : tensor<32x128x32xi32>) permutation = [0, 2, 1]
// CHECK: hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min> ins(%[[transposed0]], %[[transposed1]] : tensor<32x128x32xf32>, tensor<32x128x32xi32>) outs(%[[res0]], %[[res1]] : tensor<32x128xf32>, tensor<32x128xi32>) dimensions = [2]
func.func @test_NormalizeArgMinMax_normalize_reduce_with_index_ra_to_ar(%arg0: tensor<32x32x128xf32>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi32> {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = tensor.empty() : tensor<32x128xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xf32>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xf32>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xf32>, tensor<32x128xi32>

  return %reduced#1 : tensor<32x128xi32>
}

// -----

// CHECK-LABEL: @test_ReduceWithIndexRAHighPerformance_reduce_dim_0
// CHECK: %[[res0:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[res1:.*]] = tensor.empty() : tensor<32xi32>
// CHECK: %[[tmp_buf0:.*]] = tensor.empty() : tensor<32x64xf32>
// CHECK: %[[transposed0:.*]] = linalg.transpose ins(%[[arg0:.*]] : tensor<64x32xf32>) outs(%[[tmp_buf0]] : tensor<32x64xf32>) permutation = [1, 0]
// CHECK: %[[tmp_buf1:.*]] = tensor.empty() : tensor<32x64xi32>
// CHECK: %[[transposed1:.*]] = linalg.transpose ins(%[[arg1:.*]] : tensor<64x32xi32>) outs(%[[tmp_buf1]] : tensor<32x64xi32>) permutation = [1, 0]
// CHECK: hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%[[transposed0]], %[[transposed1]] : tensor<32x64xf32>, tensor<32x64xi32>) outs(%[[res0]], %[[res1]] : tensor<32xf32>, tensor<32xi32>) dimensions = [1]
func.func @test_ReduceWithIndexRAHighPerformance_reduce_dim_0(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xi32>) -> tensor<32xi32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = tensor.empty() : tensor<32xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
                ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<64x32xi32>)
                outs(%0, %1 : tensor<32xf32>, tensor<32xi32>)
                dimensions = [0] -> tensor<32xf32>, tensor<32xi32>

  return %reduced#1 : tensor<32xi32>
}

// -----

// CHECK-LABEL: @test_ReduceWithIndexRAHighPerformance_last_axis_no_transpose
// CHECK-NOT: linalg.transpose
// CHECK: hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min> ins(%arg0, %arg1 : tensor<32x128x32xf32>, tensor<32x128x32xi32>) outs(%[[res0:.*]], %[[res1:.*]] : tensor<32x128xf32>, tensor<32x128xi32>) dimensions = [2]
func.func @test_ReduceWithIndexRAHighPerformance_last_axis_no_transpose(%arg0: tensor<32x128x32xf32>, %arg1: tensor<32x128x32xi32>) -> tensor<32x128xi32> {
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = tensor.empty() : tensor<32x128xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                ins(%arg0, %arg1 : tensor<32x128x32xf32>, tensor<32x128x32xi32>)
                outs(%0, %1 : tensor<32x128xf32>, tensor<32x128xi32>)
                dimensions = [2] -> tensor<32x128xf32>, tensor<32x128xi32>

  return %reduced#1 : tensor<32x128xi32>
}

// -----

// CHECK-LABEL: @test_ReduceWithIndexRAHighPerformance_incompatible_shape_no_transpose
// CHECK-NOT: linalg.transpose
// CHECK: hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min> ins(%arg0, %arg1 : tensor<7x5x3xf32>, tensor<7x5x3xi32>) outs(%[[res0:.*]], %[[res1:.*]] : tensor<7x3xf32>, tensor<7x3xi32>) dimensions = [1]
func.func @test_ReduceWithIndexRAHighPerformance_incompatible_shape_no_transpose(%arg0: tensor<7x5x3xf32>, %arg1: tensor<7x5x3xi32>) -> tensor<7x3xi32> {
  %0 = tensor.empty() : tensor<7x3xf32>
  %1 = tensor.empty() : tensor<7x3xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                ins(%arg0, %arg1 : tensor<7x5x3xf32>, tensor<7x5x3xi32>)
                outs(%0, %1 : tensor<7x3xf32>, tensor<7x3xi32>)
                dimensions = [1] -> tensor<7x3xf32>, tensor<7x3xi32>

  return %reduced#1 : tensor<7x3xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeReduceIndexToI32_reduce_max_with_index
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<1x1xf32>
// CHECK: %[[EMPTY_I32_1:.*]] = tensor.empty() : tensor<1x1xi32>
// CHECK: %[[REDUCE:.*]] = hfusion.reduce_with_index {already_initialize_init, tie_break_left = true, unsigned_src = false} <max> ins({{.*}}, {{.*}} : tensor<1x6x1xf32>, tensor<1x6x1xi32>) outs({{.*}}, {{.*}} : tensor<1x1xf32>, tensor<1x1xi32>) dimensions = [1]  -> tensor<1x1xf32>, tensor<1x1xi32>
// CHECK: %[[EMPTY_I64:.*]] = tensor.empty() : tensor<1x1xi64>
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins(%[[REDUCE:.*]]#1 : tensor<1x1xi32>) outs(%[[EMPTY_I64]] : tensor<1x1xi64>) -> tensor<1x1xi64>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeReduceIndexToI32_reduce_max_with_index(%arg0: tensor<1x6x1xf32>, %arg1: tensor<1x6x1xi64>) -> (tensor<1x1xf32>, tensor<1x1xi64>) {
    %cst = arith.constant 1.0 : f32
    %0 = tensor.empty() : tensor<1x1xf32>
    %init_val = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %c1_i64 = arith.constant 1: i64
    %a = tensor.empty() : tensor<1x1xi64>
    %b = tensor.empty() : tensor<1x1xf32>
    %c:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%arg0, %arg1 : tensor<1x6x1xf32>, tensor<1x6x1xi64>) outs(%b, %a : tensor<1x1xf32>, tensor<1x1xi64>) dimensions = [1] -> tensor<1x1xf32>, tensor<1x1xi64>
    return %c#0, %c#1 : tensor<1x1xf32>, tensor<1x1xi64>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeArgMinMax_reduce_max_with_index
// CHECK: %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast ins(%arg0 : tensor<1x6x1xf32>) outs(%{{.*}} : tensor<1x6x1xi32>) -> tensor<1x6x1xi32>
// CHECK: %[[MASK_EMPTY:.*]] = tensor.empty() : tensor<1x6x1xi1>
// CHECK: %[[MASK:.*]] = hfusion.compare {{.*}} ins({{.*}}, {{.*}} : tensor<1x6x1xf32>, f32) outs(%[[MASK_EMPTY]] : tensor<1x6x1xi1>) -> tensor<1x6x1xi1>
// CHECK: %[[SELECT_EMPTY:.*]] = tensor.empty() : tensor<1x6x1xf32>
// CHECK: %[[SELECT:.*]] = hfusion.select ins(%[[MASK]], %[[NEG_INF]], %arg0 : tensor<1x6x1xi1>, f32, tensor<1x6x1xf32>) outs(%[[SELECT_EMPTY]] : tensor<1x6x1xf32>) -> tensor<1x6x1xf32>
// CHECK: %[[OUT0:.*]] = tensor.empty() : tensor<1x1xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<1x1xi32>
// CHECK: hfusion.reduce_with_index {already_initialize_init, tie_break_left = true, unsigned_src = false} <max> ins(%[[SELECT]], %arg1 : tensor<1x6x1xf32>, tensor<1x6x1xi32>) outs(%[[OUT0]], %[[OUT1]] : tensor<1x1xf32>, tensor<1x1xi32>) dimensions = [1]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {  
  func.func @test_NormalizeArgMinMax_reduce_max_with_index(%arg0: tensor<1x6x1xf32>, %arg1: tensor<1x6x1xi32>) -> tensor<1x1xf32> {
    %0 = tensor.empty() : tensor<1x1xf32>
    %1 = tensor.empty() : tensor<1x1xi32>
    %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
                  ins(%arg0, %arg1 : tensor<1x6x1xf32>, tensor<1x6x1xi32>)
                  outs(%0, %1 : tensor<1x1xf32>, tensor<1x1xi32>)
                  dimensions = [1] -> tensor<1x1xf32>, tensor<1x1xi32>
    return %reduced#0 : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeArgMinMax_reduce_min_with_index
// CHECK: %[[POS_INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast ins(%arg0 : tensor<1x6x1xf32>) outs(%{{.*}} : tensor<1x6x1xi32>) -> tensor<1x6x1xi32>
// CHECK: %[[MASK_EMPTY:.*]] = tensor.empty() : tensor<1x6x1xi1>
// CHECK: %[[MASK:.*]] = hfusion.compare {{.*}} ins({{.*}}, {{.*}} : tensor<1x6x1xf32>, f32) outs(%[[MASK_EMPTY]] : tensor<1x6x1xi1>) -> tensor<1x6x1xi1>
// CHECK: %[[SELECT_EMPTY:.*]] = tensor.empty() : tensor<1x6x1xf32>
// CHECK: %[[SELECT:.*]] = hfusion.select ins(%[[MASK]], %[[POS_INF]], %arg0 : tensor<1x6x1xi1>, f32, tensor<1x6x1xf32>) outs(%[[SELECT_EMPTY]] : tensor<1x6x1xf32>) -> tensor<1x6x1xf32>
// CHECK: %[[OUT0:.*]] = tensor.empty() : tensor<1x1xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<1x1xi32>
// CHECK: hfusion.reduce_with_index {already_initialize_init, tie_break_left = true, unsigned_src = false} <min> ins(%[[SELECT]], %arg1 : tensor<1x6x1xf32>, tensor<1x6x1xi32>) outs(%[[OUT0]], %[[OUT1]] : tensor<1x1xf32>, tensor<1x1xi32>) dimensions = [1]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeArgMinMax_reduce_min_with_index(%arg0: tensor<1x6x1xf32>, %arg1: tensor<1x6x1xi32>) -> tensor<1x1xf32> {
    %0 = tensor.empty() : tensor<1x1xf32>
    %1 = tensor.empty() : tensor<1x1xi32>
    %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                  ins(%arg0, %arg1 : tensor<1x6x1xf32>, tensor<1x6x1xi32>)
                  outs(%0, %1 : tensor<1x1xf32>, tensor<1x1xi32>)
                  dimensions = [1] -> tensor<1x1xf32>, tensor<1x1xi32>
    return %reduced#0 : tensor<1x1xf32>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeReduceWithIndexInitsAndInputs_fill_to_empty
// CHECK: %[[OUT0:.*]] = tensor.empty() : tensor<1x1xi32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<1x1xi32>
// CHECK: %[[IDX:.*]] = tensor.empty() : tensor<1x6x1xi32>
// CHECK: hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%arg0, %[[IDX]] : tensor<1x6x1xi32>, tensor<1x6x1xi32>) outs(%[[OUT0]], %[[OUT1]] : tensor<1x1xi32>, tensor<1x1xi32>) dimensions = [1]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeReduceWithIndexInitsAndInputs_fill_to_empty(%arg0: tensor<1x6x1xi32>) -> (tensor<1x1xi32>, tensor<1x1xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<1x1xi32>
    %1 = tensor.empty() : tensor<1x6x1xi32>
    %init_val = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1x1xi32>) -> tensor<1x1xi32>
    %init_idx = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1x1xi32>) -> tensor<1x1xi32>
    %idx = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<1x6x1xi32>) -> tensor<1x6x1xi32>
    %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
                  ins(%arg0, %idx : tensor<1x6x1xi32>, tensor<1x6x1xi32>)
                  outs(%init_val, %init_idx : tensor<1x1xi32>, tensor<1x1xi32>)
                  dimensions = [1] -> tensor<1x1xi32>, tensor<1x1xi32>
    return %reduced#0, %reduced#1 : tensor<1x1xi32>, tensor<1x1xi32>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeF16ReduceSum_addf
// CHECK: %[[EMPTY_IN_F32:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[CAST_IN:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<4x8xf16>) outs(%[[EMPTY_IN_F32]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[EMPTY_INIT_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_INIT:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<4xf16>) outs(%[[EMPTY_INIT_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[REDUCE:.*]] = linalg.reduce { arith.addf } ins(%[[CAST_IN]] : tensor<4x8xf32>) outs(%[[CAST_INIT]] : tensor<4xf32>) dimensions = [1]
// CHECK: %[[EMPTY_RES_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK: %[[CAST_RES:.*]] = hfusion.cast {{.*}} ins(%[[REDUCE]] : tensor<4xf32>) outs(%[[EMPTY_RES_F16]] : tensor<4xf16>) -> tensor<4xf16>
func.func @test_NormalizeF16ReduceSum_addf(%arg0: tensor<4x8xf16>, %arg1: tensor<4xf16>) -> tensor<4xf16> {
  %reduce = linalg.reduce { arith.addf } ins(%arg0 : tensor<4x8xf16>) outs(%arg1 : tensor<4xf16>) dimensions = [1]
  return %reduce : tensor<4xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeF16ReduceSum_non_addf
// CHECK-NOT: hfusion.cast
// CHECK: linalg.reduce { arith.maximumf } ins(%arg0 : tensor<4x8xf16>) outs(%arg1 : tensor<4xf16>) dimensions = [1]
func.func @test_NormalizeF16ReduceSum_non_addf(%arg0: tensor<4x8xf16>, %arg1: tensor<4xf16>) -> tensor<4xf16> {
  %reduce = linalg.reduce { arith.maximumf } ins(%arg0 : tensor<4x8xf16>) outs(%arg1 : tensor<4xf16>) dimensions = [1]
  return %reduce : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeReduceMinMaxNumF_softmax_f32_8_8192
// CHECK: %[[INFINITY:.*]] : f32
// CHECK: %[[SRC_0_NAN_MASK:.*]] -> tensor<8x8192xi1>
// CHECK: %[[EMPTY_0:.*]] = tensor.empty() : tensor<8x8192xf32>
// CHECK: %[[NEW_SRC_0:.*]] = hfusion.select ins(%[[SRC_0_NAN_MASK:.*]], %[[INFINITY:.*]], %arg0 : tensor<8x8192xi1>, f32, tensor<8x8192xf32>) outs(%[[EMPTY_0:.*]] : tensor<8x8192xf32>) -> tensor<8x8192xf32>
// CHECK: %[[REDUCE:.*]] = linalg.reduce { arith.maximumf } ins(%[[NEW_SRC_0:.*]] : tensor<8x8192xf32>) outs(%[[DST:.*]] : tensor<8xf32>) dimensions = [1]
func.func @test_NormalizeReduceMinMaxNumF_softmax_f32_8_8192(%arg0: tensor<8x8192xf32>) -> tensor<8x8192xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %0 = tensor.empty() : tensor<8xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
  %reduced = linalg.reduce { arith.maxnumf } ins(%arg0 : tensor<8x8192xf32>) outs(%1 : tensor<8xf32>) dimensions = [1]
  %2 = tensor.empty() : tensor<8x8192xf32>
  %broadcasted = linalg.broadcast ins(%reduced : tensor<8xf32>) outs(%2 : tensor<8x8192xf32>) dimensions = [1]
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %broadcasted : tensor<8x8192xf32>, tensor<8x8192xf32>) outs(%2 : tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<8x8192xf32>) outs(%2 : tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
  %reduced_1 = linalg.reduce { arith.addf } ins(%4 : tensor<8x8192xf32>) outs(%5 : tensor<8xf32>) dimensions = [1]
  %broadcasted_2 = linalg.broadcast ins(%reduced_1 : tensor<8xf32>) outs(%2 : tensor<8x8192xf32>) dimensions = [1]
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%4, %broadcasted_2 : tensor<8x8192xf32>, tensor<8x8192xf32>) outs(%2 : tensor<8x8192xf32>) -> tensor<8x8192xf32>
  return %6 : tensor<8x8192xf32>
}
