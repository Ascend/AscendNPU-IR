// RUN: bishengir-opt -optimize-reduction-loop -split-input-file %s | FileCheck %s

// CHECK-LABEL: @triton_unk_fused_mean_0_outlined_vf_2
func.func @triton_unk_fused_mean_0_outlined_vf_2(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<16x784xf16, #hivm.address_space<ub>>, %arg2: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  // CHECK: %c384 = arith.constant 384 : index
  // CHECK: %c768 = arith.constant 768 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c784 = arith.constant 784 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 7.840000e+02 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst_0, %0 : f32, vector<64xi1> -> vector<64xf32>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %7:2 = scf.for %arg4 = %c0 to %c64 step %c64 iter_args(%arg5 = %1, %arg6 = %1) -> (vector<64xf32>, vector<64xf32>) {
    // CHECK:   %21 = arith.addi %arg4, %c384 : index
    // CHECK:   %22 = affine.min #map(%arg4)
    // CHECK:   %subview_5 = memref.subview %arg1[%arg3, %arg4] [1, %22] [1, 1] : memref<16x784xf16, #hivm.address_space<ub>> to memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK:   %subview_6 = memref.subview %subview_5[0, 0] [1, %22] [1, 1] : memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf16, #map1, #hivm.address_space<ub>>
    // CHECK:   %res_7 = ave.hir.vload <NORM> %subview_6[%c0] : memref<?xf16, #map1, #hivm.address_space<ub>> into vector<64xf16>
    // CHECK:   %23 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK:   %24 = ave.hir.vextf %res_7, <part_even>, %23 : vector<64xf16>, vector<64xf32>, vector<64xi1>
    // CHECK-NOT: ave.hir.vadd
    // CHECK:   %25 = affine.min #map(%21)
    // CHECK:   %subview_8 = memref.subview %arg1[%arg3, %21] [1, %25] [1, 1] : memref<16x784xf16, #hivm.address_space<ub>> to memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK:   %subview_9 = memref.subview %subview_8[0, 0] [1, %25] [1, 1] : memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf16, #map1, #hivm.address_space<ub>>
    // CHECK:   %res_10 = ave.hir.vload <NORM> %subview_9[%c0] : memref<?xf16, #map1, #hivm.address_space<ub>> into vector<64xf16>
    // CHECK:   %26 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK:   %27 = ave.hir.vextf %res_10, <part_even>, %26 : vector<64xf16>, vector<64xf32>, vector<64xi1>
    // CHECK-NOT: ave.hir.vadd
    // CHECK:   scf.yield %24, %27 : vector<64xf32>, vector<64xf32>
    // CHECK: } {splitDepth = 1 : i64}
    // CHECK: %8:2 = scf.for %arg4 = %c64 to %c384 step %c64 iter_args(%arg5 = %7#0, %arg6 = %7#1) -> (vector<64xf32>, vector<64xf32>) {
    %7 = scf.for %arg4 = %c0 to %c784 step %c64 iter_args(%arg5 = %1) -> (vector<64xf32>) {
      // CHECK: %21 = arith.addi %arg4, %c384 : index
      %12 = affine.min affine_map<(d0) -> (-d0 + 784, 64)>(%arg4)
      %subview_2 = memref.subview %arg1[%arg3, %arg4] [1, %12] [1, 1] : memref<16x784xf16, #hivm.address_space<ub>> to memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>
      %13 = ave.hir.pltm %arg4, %c784 : vector<64xi1>
      %subview_3 = memref.subview %subview_2[0, 0] [1, %12] [1, 1] : memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<?xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf16>
      %14 = ave.hir.pge <ALL> : vector<64xi1>
      %15 = ave.hir.vextf %res_4, <part_even>, %14 : vector<64xf16>, vector<64xf32>, vector<64xi1>
      %16 = ave.hir.vsel %13, %15, %1 : vector<64xi1>, vector<64xf32>
      %17 = ave.hir.pge <ALL> : vector<64xi1>
      %18 = ave.hir.vadd %16, %arg5, %17 {reductionOp} : vector<64xf32>, vector<64xi1>
      scf.yield %18 : vector<64xf32>
      // CHECK: %27 = affine.min #map(%21)
      // CHECK: %subview_8 = memref.subview %arg1[%arg3, %21] [1, %27] [1, 1] : memref<16x784xf16, #hivm.address_space<ub>> to memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: %subview_9 = memref.subview %subview_8[0, 0] [1, %27] [1, 1] : memref<1x?xf16, strided<[784, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf16, #map1, #hivm.address_space<ub>>
      // CHECK: %res_10 = ave.hir.vload <NORM> %subview_9[%c0] : memref<?xf16, #map1, #hivm.address_space<ub>> into vector<64xf16>
      // CHECK: %28 = ave.hir.pge <ALL> : vector<64xi1>
      // CHECK: %29 = ave.hir.vextf %res_10, <part_even>, %28 : vector<64xf16>, vector<64xf32>, vector<64xi1>
      // CHECK: %30 = ave.hir.pge <ALL> : vector<64xi1>
      // CHECK: %31 = ave.hir.vadd %29, %arg6, %30 {reductionOp} : vector<64xf32>, vector<64xi1>
      // CHECK: scf.yield %26, %31 : vector<64xf32>, vector<64xf32>
    } {reductionLoop}
    // CHECK: %10 = ave.hir.vadd %8#0, %8#1, %9 : vector<64xf32>, vector<64xi1>
    %subview_1 = memref.subview %subview[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %8 = ave.hir.pge <ALL> : vector<64xi1>
    %9 = ave.hir.reduction <add>, %7, %8 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %10 = builtin.unrealized_conversion_cast %9 : vector<64xf32> to vector<1xf32>
    %11 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %subview_1[], %11, %10 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  }
  %4 = ave.hir.pge <VL16> {mask_op_idx = 1 : i32} : vector<64xi1>
  annotation.mark %4 {mask_op_idx = 1 : i32} : vector<64xi1>
  %res = ave.hir.vload <NORM> %arg0[%c0] : memref<16xf32, #hivm.address_space<ub>> into vector<64xf32>
  annotation.mark %res {reached_mask_ops_idx = 1 : i32} : vector<64xf32>
  %5 = ave.hir.vdiv %res, %3, %4 {cast = #hivm.cast<cast_unsigned>} : vector<64xf32>, vector<64xi1>
  annotation.mark %5 {reached_mask_ops_idx = 1 : i32} : vector<64xf32>
  %6 = ave.hir.vtruncf %5, <rint>, false, <part_even>, %4 : vector<64xf32>, vector<64xf16>, vector<64xi1>
  annotation.mark %6 {reached_mask_ops_idx = 1 : i32} : vector<64xf16>
  ave.hir.masked_store <NORM_B16> %arg2[%c0], %4, %6 : memref<16xf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  return
}


// CHECK-LABEL: @no_reuse_mask
// CHECK: ave.hir.plt %{{.*}} {mask_op_idx = 0 : i32}
// CHECK-NOT: ave.hir.plt %{{.*}} {mask_op_idx = 0 : i32}
// CHECK: ave.hir.plt %{{.*}}
func.func @no_reuse_mask(%arg0: memref<1x64xf32, #hivm.address_space<ub>>, %arg1: memref<8x100xf32, #hivm.address_space<ub>>, %arg2: memref<8x100xf32, #hivm.address_space<ub>>, %arg3: memref<8x100xf32, #hivm.address_space<ub>>, %arg4: memref<8xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst_0, %0 : f32, vector<64xi1> -> vector<64xf32>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg5 = %c0 to %c8 step %c1 {
    %subview = memref.subview %arg4[%arg5] [1] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_1[%c0], %4, %3 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    scf.for %arg6 = %c0 to %c100 step %c64 {
      %9 = affine.min affine_map<(d0) -> (-d0 + 100, 64)>(%arg6)
      %subview_3 = memref.subview %arg1[%arg5, %arg6] [1, %9] [1, 1] : memref<8x100xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[100, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %arg3[%arg5, %arg6] [1, %9] [1, 1] : memref<8x100xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[100, 1], offset: ?>, #hivm.address_space<ub>>
      %res_5, %new_true_shape = ave.hir.plt %9 {mask_op_idx = 0 : i32} : vector<64xi1>, index
      annotation.mark %res_5 {mask_op_idx = 0 : i32} : vector<64xi1>
      %subview_6 = memref.subview %subview_3[0, 0] [1, %9] [1, 1] : memref<1x?xf32, strided<[100, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_7 = ave.hir.vload <NORM> %subview_6[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      annotation.mark %res_7 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      %10 = ave.hir.vmul %res_7, %1, %res_5 : vector<64xf32>, vector<64xi1>
      annotation.mark %10 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      %subview_8 = memref.subview %subview_4[0, 0] [1, %9] [1, 1] : memref<1x?xf32, strided<[100, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B32> %subview_8[%c0], %res_5, %10 {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous} : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %subview_9 = memref.subview %arg2[%arg5, %arg6] [1, %9] [1, 1] : memref<8x100xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[100, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_10 = memref.subview %arg0[0, 0] [1, %9] [1, 1] : memref<1x64xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
      %subview_11 = memref.subview %subview_9[0, 0] [1, %9] [1, 1] : memref<1x?xf32, strided<[100, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_12 = ave.hir.vload <NORM> %subview_11[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      annotation.mark %res_12 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      %subview_13 = memref.subview %subview_10[0, 0] [1, %9] [1, 1] : memref<1x?xf32, strided<[64, 1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %res_14 = ave.hir.vload <NORM> %subview_13[%c0] : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
      annotation.mark %res_14 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      %11 = ave.hir.vadd %res_12, %res_14, %res_5 {reductionOp} : vector<64xf32>, vector<64xi1>
      annotation.mark %11 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      ave.hir.masked_store <NORM_B32> %subview_13[%c0], %res_5, %11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    } {reductionLoop}
    %res = ave.hir.vload <NORM> %subview_1[%c0] : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
    %subview_2 = memref.subview %subview[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.reduction <add>, %res, %5 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %7 = builtin.unrealized_conversion_cast %6 : vector<64xf32> to vector<1xf32>
    %8 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %subview_2[], %8, %7 {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous} : memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  }
  return
}