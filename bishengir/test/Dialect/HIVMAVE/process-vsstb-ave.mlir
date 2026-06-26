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

// CHECK-LABEL: func.func @no_merge_load_due_to_non_elementwise
// CHECK-NOT: ave.hir.vload <DINTLV_B32>
// CHECK: ave.hir.vtruncf %{{.*}}, <rint>, false, <part_even>, %{{.*}} : vector<64xf32>, vector<64xbf16>, vector<64xi1>
// CHECK: ave.hir.vtruncf %{{.*}}, <rint>, false, <part_even>, %{{.*}} : vector<64xf32>, vector<64xbf16>, vector<64xi1>
// CHECK: ave.hir.pge <ALL> : vector<128xi1>
// CHECK: ave.hir.vdintlv %{{.*}}, %{{.*}} : vector<64xbf16>, vector<128xbf16>
// Test case: multiple loads exist, but non-elementwise ops (vcmp, vsel) on the
// data path prevent load merging. Should fall back to vdintlv path.
func.func @no_merge_load_due_to_non_elementwise(%arg0: memref<32xf32, #hivm.address_space<ub>>, %arg1: memref<32xf32, #hivm.address_space<ub>>, %arg2: memref<32x128xi8, #hivm.address_space<ub>>, %arg3: memref<32x128xi8, #hivm.address_space<ub>>, %arg4: memref<32x128xf32, #hivm.address_space<ub>>, %arg5: memref<32x128xf32, #hivm.address_space<ub>>, %arg6: f32, %arg7: memref<8x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c528 = arith.constant 528 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0_i8 = arith.constant 0 : i8
  %cst = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %c0_i8, %2 : i8, vector<64xi1> -> vector<64xi8>
  scf.for %arg8 = %c0 to %c32 step %c1 {
    %subview = memref.subview %arg0[%arg8] [1] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg8] [1] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg9 = %c0 to %c8 step %c4 {
      %4 = affine.apply affine_map<(d0) -> (d0 * 16)>(%arg9)
      %subview_1 = memref.subview %arg2[%arg8, %4] [1, 64] [1, 1] : memref<32x128xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_2[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi8>
      %subview_3 = memref.subview %arg3[%arg8, %4] [1, 64] [1, 1] : memref<32x128xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %subview_3[0, 0] [1, 64] [1, 1] : memref<1x64xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_5 = ave.hir.vload <NORM> %subview_4[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi8>
      %5 = ave.hir.pge <ALL> : vector<64xi1>
      %6 = ave.hir.vor %res, %res_5, %5 : vector<64xi8>, vector<64xi1>
      %subview_6 = memref.subview %arg4[%arg8, %4] [1, 64] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %arg5[%arg8, %4] [1, 64] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_8 = memref.subview %subview_6[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_9 = ave.hir.vload <NORM> %subview_8[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      %res_10 = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %subview_11 = memref.subview %subview_7[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %res_12 = ave.hir.vload <NORM> %subview_11[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      %res_13 = ave.hir.vload <BRC_B32> %subview_0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %7 = ave.hir.pge <ALL> : vector<64xi1>
      %8 = ave.hir.vsub %res_12, %res_13, %7 : vector<64xf32>, vector<64xi1>
      %9 = ave.hir.scalar_broadcast %arg6 : f32 -> vector<64xf32>
      %10 = ave.hir.pge <ALL> : vector<64xi1>
      %11 = ave.hir.vmul %res_9, %9, %10 : vector<64xf32>, vector<64xi1>
      %12 = ave.hir.pge <ALL> : vector<64xi1>
      %13 = ave.hir.vsub %11, %res_10, %12 : vector<64xf32>, vector<64xi1>
      %14 = ave.hir.pge <ALL> : vector<64xi1>
      %15 = ave.hir.vexp %13, %14 : vector<64xf32>, vector<64xi1>
      // Non-elementwise op: vcmp breaks the elementwise chain
      %16 = ave.hir.pge <ALL> : vector<64xi1>
      %17 = ave.hir.vcmp <NE> %6, %3, %16 : vector<64xi8>, vector<64xi1> -> vector<64xi1>
      // Non-elementwise op: vsel breaks the elementwise chain
      %18 = ave.hir.vsel %17, %15, %1 : vector<64xi1>, vector<64xf32>
      %19 = ave.hir.pge <ALL> : vector<64xi1>
      %20 = ave.hir.vmul %18, %8, %19 : vector<64xf32>, vector<64xi1>
      %21 = ave.hir.pge <ALL> : vector<64xi1>
      %22 = ave.hir.vmul %20, %9, %21 : vector<64xf32>, vector<64xi1>
      %23 = ave.hir.pge <ALL> : vector<64xi1>
      %24 = ave.hir.vtruncf %22, <rint>, false, <part_even>, %23 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
      %subview_14 = memref.subview %arg7[%arg9, %arg8, 0] [4, 1, 16] [1, 1, 1] : memref<8x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xbf16, strided<[528, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_15 = memref.subview %subview_14[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xbf16, strided<[528, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 528 + d1 + s0)>, #hivm.address_space<ub>>
      %25 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.store_with_stride %subview_15[%c0, %c0], %c528, %25, %24 : memref<4x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 528 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    } {unroll_for_vsstb}
  }
  return
}
