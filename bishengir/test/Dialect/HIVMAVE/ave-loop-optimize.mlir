// RUN: bishengir-opt -ave-loop-optimize %s -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @causal_conv1d_update_kernel_bdt_fwd_outlined_vf_3

// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: scf.for {{.*}} = %c0 to %c256 step %[[C128]]


func.func @causal_conv1d_update_kernel_bdt_fwd_outlined_vf_3(
    %arg0: memref<32x4xf16, #hivm.address_space<ub>>,
    %arg1: index,
    %arg2: memref<32x256xf32, #hivm.address_space<ub>>,
    %arg3: memref<32x259xf16, #hivm.address_space<ub>>,
    %arg4: memref<32x256xf32, #hivm.address_space<ub>>) 
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  scf.for %arg5 = %c0 to %c32 step %c1 {
    %subview = memref.subview %arg0[%arg5, %arg1] [1, 1] [1, 1] : memref<32x4xf16, #hivm.address_space<ub>> to memref<1xf16, strided<[4], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg6 = %c0 to %c256 step %c64 {
      %subview_0 = memref.subview %arg2[%arg5, %arg6] [1, 64] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>      
      %0 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg6, %arg1]
      %subview_1 = memref.subview %arg3[%arg5, %0] [1, 64] [1, 1] : memref<32x259xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[259, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg4[%arg5, %arg6] [1, 64] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      %subview_4 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[259, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_5 = ave.hir.vload <NORM> %subview_4[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf16>
      %res_6 = ave.hir.vload <BRC_B16> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xf16, strided<[4], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
      %1 = ave.hir.pge <ALL> : vector<64xi1>
      %2 = ave.hir.vextf %res_6, <part_even>, %1 : vector<64xf16>, vector<64xf32>, vector<64xi1>
      %3 = ave.hir.pge <ALL> : vector<64xi1>
      %4 = ave.hir.vextf %res_5, <part_even>, %3 : vector<64xf16>, vector<64xf32>, vector<64xi1> 
      %5 = ave.hir.pge <ALL> : vector<64xi1>
      %6 = ave.hir.vmul %4, %2, %5 : vector<64xf32>, vector<64xi1>
      %7 = ave.hir.pge <ALL> : vector<64xi1>
      %8 = ave.hir.vadd %res, %6, %7 : vector<64xf32>, vector<64xi1>
      %subview_7 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>       
      %9 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview_7[%c0], %9, %8 {hivm.is_continuous} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @peel_epilogue_canon_test
// split bound folded to constant by AffineApplyOp canonicalization
// CHECK-DAG: %[[C192:.*]] = arith.constant 192 : index
// affine.apply in promoted epilogue body folded to constant
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// main loop uses folded split bound as upper bound
// CHECK: scf.for {{.*}} = %c0 to %[[C192]] step %c64
// no second scf.for — epilogue promoted by scf::ForOp canonicalization
// CHECK-NOT: scf.for
// epilogue body inlined: src subview uses folded %c256, dst subview uses %c192
// CHECK: memref.subview %{{.*}}[0, %[[C256]]]
// CHECK: memref.subview %{{.*}}[0, %[[C192]]]

func.func @peel_epilogue_canon_test(
    %arg0: memref<32x320xf16, #hivm.address_space<ub>>,
    %arg1: memref<32x320xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c200 = arith.constant 200 : index
  scf.for %iv = %c0 to %c200 step %c64 {
    %offset = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%iv]
    %sub = memref.subview %arg0[0, %offset] [1, 64] [1, 1] : memref<32x320xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
    %sub_flat = memref.subview %sub[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %vload = ave.hir.vload <NORM> %sub_flat[%c0] : memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf16>
    %pge = ave.hir.pge <ALL> : vector<64xi1>
    %vextf = ave.hir.vextf %vload, <part_even>, %pge : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %sub_dst = memref.subview %arg1[0, %iv] [1, 64] [1, 1] : memref<32x320xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
    %sub_dst_flat = memref.subview %sub_dst[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[320, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %sub_dst_flat[%c0], %pge, %vextf {hivm.is_continuous} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}
