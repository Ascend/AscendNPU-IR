// RUN: bishengir-opt -ave-process-vsstb %s -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @auto_unroll
// CHECK-NOT: scf.for %arg4
// CHECK: ave.hir.vload <DINTLV_B32>
// CHECK: ave.hir.pge <ALL> : vector<128xi1>
// CHECK: ave.hir.vtruncf %{{.*}}, <round>, false, <part_even>, %{{.*}} : vector<64xf32>, vector<128xf16>, vector<64xi1>
// CHECK: ave.hir.vtruncf %{{.*}}, <round>, false, <part_odd>, %{{.*}} : vector<64xf32>, vector<128xf16>, vector<64xi1>
// CHECK: ave.hir.vor %{{.*}}, %{{.*}}, %{{.*}} : vector<128xf16>, vector<128xi1>
func.func @auto_unroll(%arg0: memref<64x8x16xf32, #hivm.address_space<ub>>, %arg1: memref<64x8x16xf16, #hivm.address_space<ub>>, %arg2: memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c1040 = arith.constant 1040 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c4 {
      %subview = memref.subview %arg0[%arg3, %arg4, 0] [1, 4, 16] [1, 1, 1] : memref<64x8x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %subview[0, 0, 0] [1, 4, 16] [1, 1, 1] : memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_0[%c0, %c0] : memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      %0 = ave.hir.pge <ALL> : vector<64xi1>
      %1 = ave.hir.vtruncf %res, <round>, false, <part_even>, %0 : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %subview_1 = memref.subview %arg2[%arg4, %arg3, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_1[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.store_with_stride %subview_2[%c0, %c0], %c1040, %2, %1 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    } {unroll_for_vsstb}
  }
  return
}

// CHECK-LABEL: func.func @merge_load
// CHECK: ave.hir.vload <DINTLV_B32>
// CHECK: ave.hir.pge <VL16> : vector<128xi1>
// CHECK: ave.hir.vtruncf %{{.*}}, <rint>, false, <part_even>, %{{.*}} : vector<64xf32>, vector<128xf16>, vector<64xi1>
// CHECK: ave.hir.vtruncf %{{.*}}, <rint>, false, <part_odd>, %{{.*}} : vector<64xf32>, vector<128xf16>, vector<64xi1>
// CHECK: ave.hir.vor %{{.*}}, %{{.*}}, %{{.*}} : vector<128xf16>, vector<128xi1>
func.func @merge_load(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64x128xf32, #hivm.address_space<ub>>, %arg7: index, %arg8: f32, %arg9: memref<1xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline, noinline, outline = true, vector_mode = "simd"} {
  %c1040 = arith.constant 1040 : index
  %c0 = arith.constant 0 : index
  scf.for %arg10 = %arg0 to %arg1 step %arg2  : i32 {
    %2 = arith.index_cast %arg10 : i32 to index
    %subview = memref.subview %arg5[%2] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg6[%2, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg6[%2, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %3 = memref.load %subview[%arg7] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_2[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %4 = ave.hir.scalar_broadcast %3 : f32 -> vector<64xf32>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vsub %res, %4, %5 : vector<64xf32>, vector<64xi1>
    %subview_3 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vsub %res_4, %4, %7 : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vexp %6, %9 : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.vexp %8, %11 : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    %14 = ave.hir.vtruncf %10, <rint>, false, <part_even>, %13 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %subview_5 = memref.subview %arg3[0, %2, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %subview_5[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
    %15 = ave.hir.pge <VL8> : vector<64xi1>
    ave.hir.store_with_stride %subview_6[%c0, %c0], %c1040, %15, %14 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    %16 = ave.hir.pge <ALL> : vector<64xi1>
    %17 = ave.hir.vtruncf %12, <rint>, false, <part_even>, %16 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %subview_7 = memref.subview %arg3[4, %2, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %subview_7[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
    %18 = ave.hir.pge <VL8> : vector<64xi1>
    ave.hir.store_with_stride %subview_8[%c0, %c0], %c1040, %18, %17 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  }
  return
}

// CHECK-LABEL: func.func @no_merge_load
// CHECK-NOT: ave.hir.vload <DINTLV_B32>
// CHECK: ave.hir.vtruncf %{{.*}}, <rint>, false, <part_even>, %{{.*}} : vector<64xf32>, vector<64xf16>, vector<64xi1>
// CHECK: ave.hir.vtruncf %{{.*}}, <rint>, false, <part_even>, %{{.*}} : vector<64xf32>, vector<64xf16>, vector<64xi1>
// CHECK: ave.hir.pge <ALL> : vector<128xi1>
// CHECK: ave.hir.vdintlv %{{.*}}, %{{.*}} : vector<64xf16>, vector<128xf16>
func.func @no_merge_load(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64x128xf32, #hivm.address_space<ub>>, %arg7: index, %arg8: f32, %arg9: memref<1xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline, noinline, outline = true, vector_mode = "simd"} {
  %c1040 = arith.constant 1040 : index
  %c0 = arith.constant 0 : index
  scf.for %arg10 = %arg0 to %arg1 step %arg2  : i32 {
    %2 = arith.index_cast %arg10 : i32 to index
    %subview = memref.subview %arg5[%2] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg6[%2, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg6[%2, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %3 = memref.load %subview[%arg7] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_2[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %4 = ave.hir.scalar_broadcast %3 : f32 -> vector<64xf32>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vsub %res, %4, %5 : vector<64xf32>, vector<64xi1>
    %subview_3 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vsub %res_4, %4, %7 : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vexp %6, %9 : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.vexp %8, %11 : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_0[%c0], %13, %10 : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    ave.hir.masked_store <NORM_B32> %subview_1[%c0], %13, %12 : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %14 = ave.hir.vtruncf %10, <rint>, false, <part_even>, %13 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %subview_5 = memref.subview %arg3[0, %2, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %subview_5[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
    %15 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.store_with_stride %subview_6[%c0, %c0], %c1040, %15, %14 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    %16 = ave.hir.pge <ALL> : vector<64xi1>
    %17 = ave.hir.vtruncf %12, <rint>, false, <part_even>, %16 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %subview_7 = memref.subview %arg3[4, %2, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %subview_7[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
    %18 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.store_with_stride %subview_8[%c0, %c0], %c1040, %18, %17 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  }
  return
}
