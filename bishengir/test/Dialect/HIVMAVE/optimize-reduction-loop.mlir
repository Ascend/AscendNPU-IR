// RUN: bishengir-opt -optimize-reduction-loop -split-input-file %s | FileCheck %s

// OptimizeReductionLoopHIVMAVE splits reduction loops and peels the tail.
// The split loops' cloned `affine.min affine_map<(-d0 + tailUb, step)>(iv)`
// evaluate to `step` on every iteration (the real tail is peeled out), so
// FoldAffineMinToStepPattern folds them to the constant `step`. This makes
// the memref.subview sizes static and avoids a later scalar `min` chain.

// CHECK-LABEL: @triton_unk_fused_mean_0_outlined_vf_2(
// CHECK-SAME:  %[[ARG0:.*]]: memref{{.*}}, %[[ARG1:.*]]: memref{{.*}}, %[[ARG2:.*]]: memref{{.*}})
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
  // CHECK: scf.for %[[OUTER_IV:.*]] = %c0 to %c16 step %c1 {
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  // The 0..784 reduction loop has a genuine tail (len 16), so it is split into
  // two halves and the tail peeled out; the split loops fold affine.min to %c64
  // (static subviews) while the second half keeps the addi-form IV.

  // CHECK:   %[[HALF1:.*]]:2 = scf.for %[[SPLIT_IV:.*]] = %c0 to %c64 step %c64 iter_args(%[[ACC0:.*]] = %{{.*}}, %[[ACC1:.*]] = %{{.*}}){{.*}} {
  // CHECK:     %[[ADD:.*]] = arith.addi %[[SPLIT_IV]], %c384
  // CHECK-NOT: affine.min
  // CHECK:     memref.subview %[[ARG1]][%[[OUTER_IV]], %[[SPLIT_IV]]] [1, %c64] [1, 1]
  // CHECK-NOT: ave.hir.vadd
  // CHECK:     memref.subview %[[ARG1]][%[[OUTER_IV]], %[[ADD]]] [1, %c64] [1, 1]
  // CHECK-NOT: ave.hir.vadd
  // CHECK:     scf.yield %{{.*}}, %{{.*}}
  // CHECK:   } {splitDepth = 1 : i64}
  // CHECK:   %[[HALF2:.*]]:2 = scf.for %[[IV:.*]] = %c64 to %c384 step %c64 iter_args(%[[ACC0]] = %[[HALF1]]#0, %[[ACC1]] = %[[HALF1]]#1){{.*}} {
    %7 = scf.for %arg4 = %c0 to %c784 step %c64 iter_args(%arg5 = %1) -> (vector<64xf32>) {
  // CHECK:     %[[SECONDIV:.*]] = arith.addi %[[IV]], %c384
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
  // CHECK:     %[[FIRST_SV0:.*]] = memref.subview %[[ARG1]][%[[OUTER_IV]], %[[IV]]] [1, %c64] [1, 1]
  // CHECK:     %[[FIRST_SV1:.*]] = memref.subview %[[FIRST_SV0]][0, 0] [1, %c64] [1, 1]
  // CHECK:     %[[FIRST_VLOAD:.*]] = ave.hir.vload <NORM> %[[FIRST_SV1]][%c0]
  // CHECK:     %[[FIRST_PGE0:.*]] = ave.hir.pge <ALL>
  // CHECK:     %[[FIRST_EXTF:.*]] = ave.hir.vextf %[[FIRST_VLOAD]], <part_even>, %[[FIRST_PGE0]]
  // CHECK:     %[[FIRST_PGE1:.*]] = ave.hir.pge <ALL>
  // CHECK:     %[[FIRST_RES:.*]] = ave.hir.vadd %[[FIRST_EXTF]], %[[ACC0]], %[[FIRST_PGE1]] {reductionOp}
  // CHECK:     %[[SECOND_SV0:.*]] = memref.subview %[[ARG1]][%[[OUTER_IV]], %[[SECONDIV]]] [1, %c64] [1, 1]
  // CHECK:     %[[SECOND_SV1:.*]] = memref.subview %[[SECOND_SV0]][0, 0] [1, %c64] [1, 1]
  // CHECK:     %[[SECOND_VLOAD:.*]] = ave.hir.vload <NORM> %[[SECOND_SV1]][%c0]
  // CHECK:     %[[SECOND_PGE0:.*]] = ave.hir.pge <ALL>
  // CHECK:     %[[SECOND_EXTF:.*]] = ave.hir.vextf %[[SECOND_VLOAD]], <part_even>, %[[SECOND_PGE0]]
  // CHECK:     %[[SECOND_PGE1:.*]] = ave.hir.pge <ALL>
  // CHECK:     %[[SECOND_RES:.*]] = ave.hir.vadd %[[SECOND_EXTF]], %[[ACC1]], %[[SECOND_PGE1]] {reductionOp}
  // CHECK:     scf.yield %[[FIRST_RES]], %[[SECOND_RES]]
  // CHECK:   } {splitDepth = 1 : i64}
    } {reductionLoop}
  // CHECK:   ave.hir.vadd %[[HALF2]]#0, %[[HALF2]]#1
  // CHECK:   memref.subview %[[ARG1]][%[[OUTER_IV]], %c768] [1, %c16] [1, 1]
  // CHECK:   ave.hir.pltm %c768, %c784
  // CHECK:   ave.hir.vsel %{{.*}}, %{{.*}}, %{{.*}}
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

// -----

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

// -----

// CHECK-LABEL: func.func @mixed_vumin_vumax_reduction
// CHECK: %[[SPLIT:.*]]:4 = scf.for %[[IV:.*]] = %c64 to %c256 step %c64
// CHECK: } {splitDepth = 1 : i64}
// CHECK: %[[MIN_COMBINE:.*]] = ave.hir.vumin %[[SPLIT]]#0, %[[SPLIT]]#1
// CHECK: %[[MAX_COMBINE:.*]] = ave.hir.vumax %[[SPLIT]]#2, %[[SPLIT]]#3
func.func @mixed_vumin_vumax_reduction(
    %arg0: memref<2x512xi1, #hivm.address_space<ub>>,
    %arg1: memref<2x512xi32, #hivm.address_space<ub>>,
    %arg2: memref<2xi32, #hivm.address_space<ub>>,
    %arg3: memref<2xi32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c-1_i32 = arith.constant -1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c65535_i32 = arith.constant 65535 : i32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c65535_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %c0_i32, %2 : i32, vector<64xi1> -> vector<64xi32>
  %4 = ave.hir.pge <ALL> : vector<64xi1>
  %5 = ave.hir.broadcast %c-1_i32, %4 : i32, vector<64xi1> -> vector<64xi32>
  scf.for %arg4 = %c0 to %c2 step %c1 {
    %subview = memref.subview %arg2[%arg4] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg3[%arg4] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %6:2 = scf.for %arg5 = %c0 to %c512 step %c64 iter_args(%arg6 = %5, %arg7 = %3) -> (vector<64xi32>, vector<64xi32>) {
      %subview_3 = memref.subview %arg0[%arg4, %arg5] [1, 64] [1, 1] : memref<2x512xi1, #hivm.address_space<ub>> to memref<1x64xi1, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %arg1[%arg4, %arg5] [1, 64] [1, 1] : memref<2x512xi32, #hivm.address_space<ub>> to memref<1x64xi32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_5 = memref.subview %subview_3[0, 0] [1, 64] [1, 1] : memref<1x64xi1, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_5[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi1>
      %subview_6 = memref.subview %subview_4[0, 0] [1, 64] [1, 1] : memref<1x64xi32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_7 = ave.hir.vload <NORM> %subview_6[%c0] : memref<64xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi32>
      %15 = ave.hir.vsel %res, %res_7, %1 : vector<64xi1>, vector<64xi32>
      %16 = ave.hir.pge <ALL> : vector<64xi1>
      %17 = ave.hir.vumin %15, %arg6, %16 {reductionOp} : vector<64xi32>, vector<64xi1>
      %18 = ave.hir.vsel %res, %res_7, %3 : vector<64xi1>, vector<64xi32>
      %19 = ave.hir.pge <ALL> : vector<64xi1>
      %20 = ave.hir.vumax %18, %arg7, %19 {reductionOp} : vector<64xi32>, vector<64xi1>
      scf.yield %17, %20 : vector<64xi32>, vector<64xi32>
    } {reductionLoop}
    %subview_1 = memref.subview %subview[0] [1] [1] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.reduction <umin>, %6#0, %7 : vector<64xi32>, vector<64xi1> -> vector<64xi32>
    %9 = builtin.unrealized_conversion_cast %8 : vector<64xi32> to vector<1xi32>
    %10 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %subview_1[], %10, %9 {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous} : memref<i32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi32>
    %subview_2 = memref.subview %subview_0[0] [1] [1] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.reduction <umax>, %6#1, %11 : vector<64xi32>, vector<64xi1> -> vector<64xi32>
    %13 = builtin.unrealized_conversion_cast %12 : vector<64xi32> to vector<1xi32>
    %14 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %subview_2[], %14, %13 {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous} : memref<i32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi32>
  }
  return
}

// -----

// The original (0..8777 step 64) reduction loop has a genuine tail (len 9),
// so it is split+peeled; the surviving split loops must have their affine.min
// folded to %c64.
//
// CHECK-LABEL: func.func @rmsnorm_vf2
// In the split loops, no affine.min remains and subviews are static [1, %c64].
// CHECK-NOT: affine.min
// CHECK: memref.subview{{.*}} [1, %c64]
// The peeled real tail stays as plt %c9 (dynamic, not folded).
// CHECK: ave.hir.plt %c9
func.func @rmsnorm_vf2(%arg0: memref<1x8777xf16, #hivm.address_space<ub>>, %arg1: memref<1xf32, #hivm.address_space<ub>>, %arg2: memref<1x8777xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c64 = arith.constant 64 : index
  %c8777 = arith.constant 8777 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
  %2 = scf.for %arg3 = %c0 to %c8777 step %c64 iter_args(%arg4 = %1) -> (vector<64xf32>) {
    %7 = affine.min affine_map<(d0) -> (-d0 + 8777, 64)>(%arg3)
    %subview_0 = memref.subview %arg2[0, %arg3] [1, %7] [1, 1] : memref<1x8777xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[8777, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg0[0, %arg3] [1, %7] [1, 1] : memref<1x8777xf16, #hivm.address_space<ub>> to memref<1x?xf16, strided<[8777, 1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %7 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
    %subview_2 = memref.subview %subview_1[0, 0] [1, %7] [1, 1] : memref<1x?xf16, strided<[8777, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_3 = ave.hir.vload <NORM> %subview_2[%c0] : memref<?xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf16>
    annotation.mark %res_3 {reached_mask_ops_idx = 0 : i32} : vector<64xf16>
    %8 = ave.hir.pge <ALL> : vector<64xi1>
    %9 = ave.hir.vextf %res_3, <part_even>, %8 : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %subview_4 = memref.subview %subview_0[0, 0] [1, %7] [1, 1] : memref<1x?xf32, strided<[8777, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_4[%c0], %res, %9 {hivm.is_continuous} : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %10 = ave.hir.pge <ALL> : vector<64xi1>
    %11 = ave.hir.vmul %9, %9, %10 : vector<64xf32>, vector<64xi1>
    %12 = ave.hir.vsel %res, %11, %1 : vector<64xi1>, vector<64xf32>
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    %14 = ave.hir.vadd %12, %arg4, %13 {reductionOp} : vector<64xf32>, vector<64xi1>
    scf.yield %14 : vector<64xf32>
  } {reductionLoop}
  %subview = memref.subview %arg1[0] [1] [1] : memref<1xf32, #hivm.address_space<ub>> to memref<f32, #hivm.address_space<ub>>
  %3 = ave.hir.pge <ALL> : vector<64xi1>
  %4 = ave.hir.reduction <add>, %2, %3 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  %5 = builtin.unrealized_conversion_cast %4 : vector<64xf32> to vector<1xf32>
  %6 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B32> %subview[], %6, %5 : memref<f32, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  return
}

// -----

// A reduction loop whose range is exactly divisible by step (no tail). After
// splitting the full-vector loops still fold affine.min to step; the (absent)
// tail means no dynamic affine.min anywhere.
// CHECK-LABEL: func.func @divisible_range
// CHECK-NOT: affine.min
// CHECK: memref.subview{{.*}} [1, %c64]
func.func @divisible_range(%arg0: memref<1x768xf16, #hivm.address_space<ub>>, %arg1: memref<1xf32, #hivm.address_space<ub>>, %arg2: memref<1x768xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c64 = arith.constant 64 : index
  %c768 = arith.constant 768 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
  %2 = scf.for %arg3 = %c0 to %c768 step %c64 iter_args(%arg4 = %1) -> (vector<64xf32>) {
    %7 = affine.min affine_map<(d0) -> (-d0 + 768, 64)>(%arg3)
    %subview_0 = memref.subview %arg2[0, %arg3] [1, %7] [1, 1] : memref<1x768xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[768, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg0[0, %arg3] [1, %7] [1, 1] : memref<1x768xf16, #hivm.address_space<ub>> to memref<1x?xf16, strided<[768, 1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %7 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %subview_2 = memref.subview %subview_1[0, 0] [1, %7] [1, 1] : memref<1x?xf16, strided<[768, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_3 = ave.hir.vload <NORM> %subview_2[%c0] : memref<?xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf16>
    %9 = ave.hir.vextf %res_3, <part_even>, %res : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %subview_4 = memref.subview %subview_0[0, 0] [1, %7] [1, 1] : memref<1x?xf32, strided<[768, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_4[%c0], %res, %9 {hivm.is_continuous} : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %11 = ave.hir.vmul %9, %9, %res : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    %14 = ave.hir.vadd %11, %arg4, %13 {reductionOp} : vector<64xf32>, vector<64xi1>
    scf.yield %14 : vector<64xf32>
  } {reductionLoop}
  %subview = memref.subview %arg1[0] [1] [1] : memref<1xf32, #hivm.address_space<ub>> to memref<f32, #hivm.address_space<ub>>
  %3 = ave.hir.pge <ALL> : vector<64xi1>
  %4 = ave.hir.reduction <add>, %2, %3 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  %5 = builtin.unrealized_conversion_cast %4 : vector<64xf32> to vector<1xf32>
  %6 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B32> %subview[], %6, %5 : memref<f32, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  return
}
