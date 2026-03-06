// RUN: bishengir-opt -canonicalize -cse -split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0, 0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @_attn_fwd_scope_0_outlined_manual_vf_0(%arg0: tensor<64x64xf32>, %arg1: tensor<1x64xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1x64xf16>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<64x64xf16>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>) -> (tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.constant_mask [1] : vector<64xi1>
    %1:4 = scf.for %arg11 = %arg4 to %arg5 step %arg6 iter_args(%arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9, %arg15 = %arg10) -> (tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
      %2 = arith.index_cast %arg11 : i32 to index
      %extracted_slice = tensor.extract_slice %arg0[%2, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg13[%2] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_1 = tensor.extract_slice %arg14[%2] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %3 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %4 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = arith.mulf %3, %4 : vector<1x64xf32>
      %6 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %7 = vector.multi_reduction <maximumf>, %5, %6 [1] : vector<1x64xf32> to vector<1xf32>
      %8 = vector.transfer_write %7, %arg2[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %9 = vector.transfer_read %extracted_slice_0[%c0], %cst, %0 {in_bounds = [true]} : tensor<1xf32>, vector<64xf32>
      // CHECK: %[[ARG0:.*]] = vector.multi_reduction <maximumf>
      // Check: %[[ARG1:.*]] = vector.broadcast %[[ARG0:.*]] : vector<1xf32> to vector<64xf32>
      // Check: arith.maximumf %{{.*}}, %[[ARG1:.*]]
      %10 = vector.transfer_read %8[%c0], %cst, %0 {in_bounds = [true]} : tensor<1xf32>, vector<64xf32>
      %11 = arith.maximumf %9, %10 : vector<64xf32>
      %12 = vector.transfer_write %11, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<1xf32>
      // CHECK: %[[ARG2:.*]] = arith.maximumf
      // CHECK: %[[ARG3:.*]] = vector.extract %[[ARG2]][0] : f32 from vector<64xf32>
      // CHECK: %[[ARG4:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<1x64xf32>
      // CHECK: arith.subf %{{.*}}, %[[ARG4:.*]]
      %13 = vector.transfer_read %12[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<1xf32>, vector<1x64xf32>
      %14 = arith.subf %5, %13 : vector<1x64xf32>
      %15 = math.exp %14 : vector<1x64xf32>
      %16 = arith.truncf %15 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %17 = vector.transfer_write %16, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x64xf16>
      %18 = vector.multi_reduction <add>, %15, %6 [1] : vector<1x64xf32> to vector<1xf32>
      %19 = vector.transfer_write %18, %arg2[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %17 into %arg12[%2, 0] [1, 64] [1, 1] : tensor<1x64xf16> into tensor<64x64xf16>
      %20 = arith.subf %9, %11 : vector<64xf32>
      %21 = math.exp %20 : vector<64xf32>
      %22 = vector.transfer_write %21, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<1xf32>
      %23 = vector.transfer_read %extracted_slice_1[%c0], %cst, %0 {in_bounds = [true]} : tensor<1xf32>, vector<64xf32>
      // CHECK: %[[ARG4:.*]] = vector.multi_reduction <add>
      // CHECK: %[[ARG5:.*]] = vector.broadcast %[[ARG4:.*]] : vector<1xf32> to vector<64xf32>
      // CHECK: arith.addf %{{.*}}, %[[ARG5:.*]] : vector<64xf32>
      %24 = vector.transfer_read %19[%c0], %cst, %0 {in_bounds = [true]} : tensor<1xf32>, vector<64xf32>
      %25 = arith.mulf %23, %21 : vector<64xf32>
      %26 = arith.addf %25, %24 : vector<64xf32>
      %27 = vector.transfer_write %26, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<1xf32>
      %inserted_slice_2 = tensor.insert_slice %12 into %arg13[%2] [1] [1] : tensor<1xf32> into tensor<64xf32>
      %inserted_slice_3 = tensor.insert_slice %27 into %arg14[%2] [1] [1] : tensor<1xf32> into tensor<64xf32>
      %inserted_slice_4 = tensor.insert_slice %22 into %arg15[%2] [1] [1] : tensor<1xf32> into tensor<64xf32>
      scf.yield %inserted_slice, %inserted_slice_2, %inserted_slice_3, %inserted_slice_4 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
    }
    return %1#0, %1#1, %1#2, %1#3 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
  }
}

// -----

#map = affine_map<(d0) -> (d0, 0)>
func.func @_attn_fwd_scope_3_outlined_manual_vf_0(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: index, %arg3: f32, %arg4: tensor<1x64xf32>, %arg5: tensor<1xf32>, %arg6: vector<64xi1>, %arg7: tensor<1x64xf32>, %arg8: tensor<4x1x16xf16>, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: tensor<64x64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<4x64x16xf16>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<4x64x16xf16>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %0:3 = scf.for %arg15 = %arg9 to %arg10 step %arg11 iter_args(%arg16 = %arg12, %arg17 = %arg13, %arg18 = %arg14) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<4x64x16xf16>)  : i32 {
    %1 = arith.index_cast %arg15 : i32 to index
    %extracted_slice = tensor.extract_slice %arg0[%1, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%1] [1] [1] : tensor<64xf32> to tensor<1xf32>
    %2 = vector.transfer_read %extracted_slice[%c0, %c0], %arg3 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %3 = vector.transfer_read %arg4[%c0, %c0], %arg3 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %4 = arith.mulf %2, %3 : vector<1x64xf32>
    %5 = vector.transfer_read %arg5[%c0], %arg3 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
    %6 = vector.multi_reduction <maximumf>, %4, %5 [1] : vector<1x64xf32> to vector<1xf32>
    %7 = vector.transfer_write %6, %arg5[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
    %8 = vector.mask %arg6 { vector.transfer_read %extracted_slice_0[%c0], %arg3 {in_bounds = [true]} : tensor<1xf32>, vector<64xf32> } : vector<64xi1> -> vector<64xf32>
    %9 = vector.mask %arg6 { vector.transfer_read %7[%c0], %arg3 {in_bounds = [true]} : tensor<1xf32>, vector<64xf32> } : vector<64xi1> -> vector<64xf32>
    %10 = arith.maximumf %8, %9 : vector<64xf32>
    %11 = vector.mask %arg6 { vector.transfer_write %10, %arg5[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<1xf32> } : vector<64xi1> -> tensor<1xf32>
    %12 = vector.transfer_read %11[%c0], %arg3 {in_bounds = [true, true], permutation_map = #map} : tensor<1xf32>, vector<1x64xf32>
    %13 = arith.subf %4, %12 : vector<1x64xf32>
    %14 = math.exp %13 : vector<1x64xf32>
    %15 = vector.transfer_write %14, %arg7[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    %expanded = tensor.expand_shape %15 [[0], [1, 2]] output_shape [1, 4, 16] : tensor<1x64xf32> into tensor<1x4x16xf32>
    %16 = vector.transfer_read %expanded[%c0, %c0, %c0], %arg3 {in_bounds = [true, true, true]} : tensor<1x4x16xf32>, vector<1x4x16xf32>
    %17 = arith.truncf %16 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x4x16xf32> to vector<1x4x16xf16>
    // CHECK: %[[EXP:.*]] = math.exp %{{.*}} : vector<1x64xf32>
    // CHECK-NEXT: %[[WRITE:.*]] = vector.transfer_write %[[EXP:.*]], {{.*}} {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    // CHECK-NEXT: %[[CAST:.*]] = vector.shape_cast %[[EXP:.*]] : vector<1x64xf32> to vector<1x4x16xf32>
    // CHECK-NEXT: %[[TRUNC:.*]] = arith.truncf %[[CAST:.*]] {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x4x16xf32> to vector<1x4x16xf16>
    %18 = vector.transpose %17, [1, 0, 2] : vector<1x4x16xf16> to vector<4x1x16xf16>
    %19 = vector.transfer_write %18, %arg8[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, tensor<4x1x16xf16>
    %inserted_slice = tensor.insert_slice %15 into %arg16[%1, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
    %inserted_slice_1 = tensor.insert_slice %11 into %arg17[%1] [1] [1] : tensor<1xf32> into tensor<64xf32>
    %inserted_slice_2 = tensor.insert_slice %19 into %arg18[0, %1, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<4x64x16xf16>
    scf.yield %inserted_slice, %inserted_slice_1, %inserted_slice_2 : tensor<64x64xf32>, tensor<64xf32>, tensor<4x64x16xf16>
  }
  return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64xf32>, tensor<4x64x16xf16>
}

// -----

// CHECK-LABLE: func.func @_attn_fwd_mix_aiv_outlined_vf_10
#map6 = affine_map<(d0) -> (-d0 + 80, 64)>
  func.func @_attn_fwd_mix_aiv_outlined_vf_10(%arg0: tensor<80xf32>, %arg1: tensor<80xi1>, %arg2: f32, %arg3: tensor<80xi32>, %arg4: tensor<80xi1>) -> tensor<80xi1> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<64xi32>
    %cst_0 = arith.constant dense<-1> : vector<64xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<6.400000e+02> : vector<64xf32>
    %false = arith.constant false
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c80 = arith.constant 80 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg5 = %c0 to %c80 step %c64 iter_args(%arg6 = %arg4) -> (tensor<80xi1>) {
      %1 = affine.min #map6(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%arg5] [%1] [1] : tensor<80xf32> to tensor<?xf32>
      %extracted_slice_3 = tensor.extract_slice %arg1[%arg5] [%1] [1] : tensor<80xi1> to tensor<?xi1>
      %2 = vector.create_mask %1 : vector<64xi1>
      %3 = vector.transfer_read %extracted_slice[%c0], %cst_2, %2 {in_bounds = [true]} : tensor<?xf32>, vector<64xf32>
      %4 = vector.broadcast %arg2 : f32 to vector<64xf32>
      %5 = arith.addf %3, %4 : vector<64xf32>
      %6 = arith.cmpf olt, %5, %cst_1 : vector<64xf32>
      // CHECK: vector.transfer_write %6, %extracted_slice_3[%c0], %2
      %7 = vector.transfer_write %6, %extracted_slice_3[%c0], %2 {in_bounds = [true]} : vector<64xi1>, tensor<?xi1>
      %extracted_slice_4 = tensor.extract_slice %arg3[%arg5] [%1] [1] : tensor<80xi32> to tensor<?xi32>
      %dim = tensor.dim %7, %c0 : tensor<?xi1>
      %8 = vector.create_mask %dim : vector<64xi1>
      %9 = vector.transfer_read %7[%c0], %false, %8 {in_bounds = [true]} : tensor<?xi1>, vector<64xi1>
      %10 = vector.transfer_read %extracted_slice_4[%c0], %c0_i32, %8 {in_bounds = [true]} : tensor<?xi32>, vector<64xi32>
      %11 = arith.select %9, %10, %cst_0 : vector<64xi1>, vector<64xi32>
      %12 = arith.cmpi sge, %11, %cst : vector<64xi32>
      %13 = vector.transfer_write %12, %extracted_slice_3[%c0], %8 {in_bounds = [true]} : vector<64xi1>, tensor<?xi1>
      %extracted_slice_5 = tensor.extract_slice %arg6[%arg5] [%1] [1] : tensor<80xi1> to tensor<?xi1>
      %dim_6 = tensor.dim %13, %c0 : tensor<?xi1>
      %14 = vector.create_mask %dim_6 : vector<64xi1>
      %15 = vector.transfer_read %13[%c0], %false, %14 {in_bounds = [true]} : tensor<?xi1>, vector<64xi1>
      %16 = vector.transfer_read %7[%c0], %false, %14 {in_bounds = [true]} : tensor<?xi1>, vector<64xi1>
      %17 = arith.andi %15, %16 : vector<64xi1>
      %18 = vector.transfer_write %17, %extracted_slice_5[%c0], %14 {in_bounds = [true]} : vector<64xi1>, tensor<?xi1>
      %inserted_slice = tensor.insert_slice %18 into %arg6[%arg5] [%1] [1] : tensor<?xi1> into tensor<80xi1>
      scf.yield %inserted_slice : tensor<80xi1>
    }
    return %0 : tensor<80xi1>
  }
