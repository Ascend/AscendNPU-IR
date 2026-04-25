// RUN: bishengir-opt --convert-arith-to-hivmave %s | FileCheck %s 

func.func @triton_unk_fused__to_copy_add_arange_lt_repeat_4_outlined_vf_1() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst4 = arith.constant dense<[-2.0, -4.0, -6.0, -6.0, -6.0]> : vector<5xf32>
    // CHECK: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[num0:.+]] = ave.hir.vci %[[cst]], <INCREASE> : f32, vector<5xf32>
    // CHECK: %[[num1:.+]] = ave.hir.pge <ALL> : vector<5xi1>
    // CHECK: %[[cst_0:.+]] = arith.constant -2.000000e+00 : f32
    // CHECK: %[[num2:.+]] = ave.hir.vmuls %[[num0]], %[[cst_0]], %[[num1]] : vector<5xf32>, f32, vector<5xi1>
    // CHECK: %[[cst_1:.+]] = arith.constant -2.000000e+00 : f32
    // CHECK: %[[num3:.+]] = ave.hir.vadds %[[num2]], %[[cst_1]], %[[num1]] : vector<5xf32>, f32, vector<5xi1>
    // CHECK: %[[cst_2:.+]] = arith.constant -6.000000e+00 : f32
    // CHECK: %[[num4:.+]] = ave.hir.pge <ALL> : vector<5xi1>
    // CHECK: %[[num5:.+]] = ave.hir.broadcast %[[cst_2]], %[[num4]] : f32, vector<5xi1> -> vector<5xf32>
    // CHECK: %[[num6:.+]] = ave.hir.pge <VL2> : vector<5xi1>
    // CHECK: %[[num7:.+]] = ave.hir.vsel %[[num6]], %[[num3]], %[[num5]] : vector<5xi1>, vector<5xf32>
    
    %cst3 = arith.constant dense<[2.0, 4.0, 6.0, 6.0, 6.0]> : vector<5xf16>
    // CHECK: %[[cst_3:.+]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[num8:.+]] = ave.hir.vci %[[cst_3]], <INCREASE> : f16, vector<5xf16>
    // CHECK: %[[num9:.+]] = ave.hir.pge <ALL> : vector<5xi1>
    // CHECK: %[[cst_4:.+]] = arith.constant 2.000000e+00 : f16
    // CHECK: %[[num10:.+]] = ave.hir.vmuls %[[num8]], %[[cst_4]], %[[num9]] : vector<5xf16>, f16, vector<5xi1>
    // CHECK: %[[cst_5:.+]] = arith.constant 2.000000e+00 : f16
    // CHECK: %[[num11:.+]] = ave.hir.vadds %[[num10]], %[[cst_5]], %[[num9]] : vector<5xf16>, f16, vector<5xi1>
    // CHECK: %[[cst_6:.+]] = arith.constant 6.000000e+00 : f16
    // CHECK: %[[num12:.+]] = ave.hir.pge <ALL> : vector<5xi1>
    // CHECK: %[[num13:.+]] = ave.hir.broadcast %[[cst_6]], %[[num12]] : f16, vector<5xi1> -> vector<5xf16>
    // CHECK: %[[num14:.+]] = ave.hir.pge <VL2> : vector<5xi1>
    // CHECK: %[[num15:.+]] = ave.hir.vsel %[[num14]], %[[num11]], %[[num13]] : vector<5xi1>, vector<5xf16>

    %cst2 = arith.constant dense<[2, 3, 4, 5, 6]> : vector<5xi32>
    // CHECK: %[[c2_i32:.+]] = arith.constant 2 : i32
    // CHECK: %[[num16:.+]] = ave.hir.vci %[[c2_i32]], <INCREASE> : i32, vector<5xi32>
    return
}

func.func @triton_unk_fused__softmax_mul_where_3_outlined_vf_4(%arg0: memref<8xi32, #hivm.address_space<ub>>, %arg1: memref<8x50xf32, strided<[56, 1]>, #hivm.address_space<ub>>, %arg2: f32, %arg3: memref<8x50xf32, strided<[56, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c50 = arith.constant 50 : index
  %cst = arith.constant dense<-2.14748365E+9> : vector<64xf32>
  %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xi64>
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  scf.for %arg4 = %c0 to %c8 step %c1 {
    %subview = memref.subview %arg0[%arg4] [1] [1] : memref<8xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg4, 0] [1, 50] [1, 1] : memref<8x50xf32, strided<[56, 1]>, #hivm.address_space<ub>> to memref<1x50xf32, strided<[56, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg3[%arg4, 0] [1, 50] [1, 1] : memref<8x50xf32, strided<[56, 1]>, #hivm.address_space<ub>> to memref<1x50xf32, strided<[56, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = ave.hir.vload <BRC_B32> %subview[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
    %res, %new_true_shape = ave.hir.plt %c50 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
    %subview_3 = memref.subview %subview_1[0, 0] [1, 50] [1, 1] : memref<1x50xf32, strided<[56, 1], offset: ?>, #hivm.address_space<ub>> to memref<50xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %1 = ave.hir.vload <NORM> %subview_3[%c0] : memref<50xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
    %2 = ave.hir.scalar_broadcast %arg2 : f32 -> vector<64xf32>
    annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
    %3 = arith.addf %1, %2 : vector<64xf32>
    annotation.mark %3 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
    annotation.mark %0 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
    %4 = arith.extsi %0 {round_mode = #hfusion.round_mode<rint>} : vector<64xi32> to vector<64xi64>
    annotation.mark %4 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
    %5 = arith.cmpi sgt, %4, %cst_0 : vector<64xi64>
    annotation.mark %5 {reached_mask_ops_idx = 0 : i32} : vector<64xi1>
    %6 = arith.select %5, %3, %cst : vector<64xi1>, vector<64xf32>
    annotation.mark %6 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
    %subview_4 = memref.subview %subview_2[0, 0] [1, 50] [1, 1] : memref<1x50xf32, strided<[56, 1], offset: ?>, #hivm.address_space<ub>> to memref<50xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_4[%c0], %res, %6 : memref<50xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}
