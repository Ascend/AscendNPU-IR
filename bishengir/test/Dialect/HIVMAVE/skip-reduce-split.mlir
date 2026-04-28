// RUN: bishengir-opt -optimize-reduction-loop -split-input-file %s | FileCheck %s

func.func @_attn_bwd_dq_dsink_mix_aiv_outlined_vf_7(%arg0: memref<1x1x64xf32, #hivm.address_space<ub>>, %arg1: memref<16x1x512xf32, #hivm.address_space<ub>>, %arg2: memref<16x1xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg2[%arg3, 0] [1, 1] [1, 1] : memref<16x1xf32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[0, 0, 0] [1, 1, 64] [1, 1, 1] : memref<1x1x64xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_0[%c0], %2, %1 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    // CHECK: scf.for %arg4 = %c0 to %c64 step %c64 {
    // CHECK:   %subview_2 = memref.subview %arg1[%arg3, 0, %arg4] [1, 1, 64] [1, 1, 1] : memref<16x1x512xf32, #hivm.address_space<ub>> to memref<1x1x64xf32, strided<[512, 512, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK:   %subview_3 = memref.subview %subview_2[0, 0, 0] [1, 1, 64] [1, 1, 1] : memref<1x1x64xf32, strided<[512, 512, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    // CHECK:   %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
    // CHECK:   %res_5 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
    // CHECK:   %10 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK:   %11 = ave.hir.vadd %res_4, %res_5, %10 {reductionOp} : vector<64xf32>, vector<64xi1>
    // CHECK:   %12 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK:   ave.hir.masked_store <NORM_B32> %subview_0[%c0], %12, %11 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    // CHECK: }
    // CHECK: scf.for %arg4 = %c64 to %c512 step %c64 {
    scf.for %arg4 = %c0 to %c512 step %c64 {
      %subview_2 = memref.subview %arg1[%arg3, 0, %arg4] [1, 1, 64] [1, 1, 1] : memref<16x1x512xf32, #hivm.address_space<ub>> to memref<1x1x64xf32, strided<[512, 512, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview_2[0, 0, 0] [1, 1, 64] [1, 1, 1] : memref<1x1x64xf32, strided<[512, 512, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      %res_5 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
      %10 = ave.hir.pge <ALL> : vector<64xi1>
      %11 = ave.hir.vadd %res_4, %res_5, %10 {reductionOp} : vector<64xf32>, vector<64xi1>
      %12 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview_0[%c0], %12, %11 : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    } {reductionLoop}
    %res = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
    %subview_1 = memref.subview %subview[0, 0] [1, 1] [1, 1] : memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>> to memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %3 = ave.hir.pge <ALL> : vector<64xi1>
    %4 = ave.hir.reduction <add>, %res, %3 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %5 = builtin.unrealized_conversion_cast %4 : vector<64xf32> to vector<1xf32>
    %6 = builtin.unrealized_conversion_cast %5 : vector<1xf32> to f32
    %7 = builtin.unrealized_conversion_cast %6 : f32 to vector<1x1xf32>
    %8 = builtin.unrealized_conversion_cast %7 : vector<1x1xf32> to vector<1xf32>
    %9 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %subview_1[], %9, %8 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  }
  return
}