// RUN: bishengir-opt -convert-hivmave-to-std -expand-strided-metadata \
// RUN: -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s
// CHECK-LABEL: @test_template_mask_store_op_mask
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_template_mask_store_op_mask(%arg0: memref<256x4xi64, #hivm.address_space<ub>>, %arg1: memref<256x4xi64, #hivm.address_space<ub>>, %arg2: memref<256x8xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = ave.hir.pge <VL8> {mask_op_idx = 1 : i32} : vector<64xi1>
    // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
    scf.for %arg3 = %c0 to %c256 step %c1 {
      %subview = memref.subview %arg0[%arg3, 0] [1, 4] [1, 1] : memref<256x4xi64, #hivm.address_space<ub>> to memref<1x4xi64, strided<[4, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3, 0] [1, 4] [1, 1] : memref<256x4xi64, #hivm.address_space<ub>> to memref<1x4xi64, strided<[4, 1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview[%c0, %c0] : memref<1x4xi64, strided<[4, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
      %res_1 = ave.hir.vload <NORM> %subview_0[%c0, %c0]  : memref<1x4xi64, strided<[4, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
      %res1, %res2 = ave.hir.vintlv %res, %res_1  : vector<64xi64>, vector<64xi64>
      ave.hir.masked_store <NORM_B64> %arg2[%arg3, %c0], %0, %res1  : memref<256x8xi64, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    }
    return
  }
  // CHECK-LABEL: @test_pld_pand_pst
  // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"
  // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"
  // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"
  // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b8"
  // CHECK: "hivm_regbaseintrins.intr.hivm.pand.z"
  // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b8"
  // CHECK: "hivm_regbaseintrins.intr.hivm.init.vector.align.data"
  // CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
  func.func @test_pld_pand_pst(%arg0: memref<512xf32, #hivm.address_space<ub>>, %arg1: memref<512xi1, #hivm.address_space<ub>>, %arg2: memref<512xf32, #hivm.address_space<ub>>, %arg3: f32, %arg4: i32 {hivm.constant_value = 31 : i64}, %arg5: i32 {hivm.constant_value = -1 : i64}, %arg6: memref<512xi1, #hivm.address_space<ub>>, %arg7: memref<512xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 2.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>> -> vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
    scf.for %arg8 = %c0 to %c512 step %c64 {
      %subview = memref.subview %arg0[%arg8] [64] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg8] [64] [1] : memref<512xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %6 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %7 = ave.hir.vtrc %res, <floor>, %6 : vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %8 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %9 = ave.hir.vcmp <EQ> %7, %res, %8 : vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>> -> vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %10 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      ave.hir.masked_store <NORM_B8> %subview_0[%c0], %10, %9 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>, hivm.is_continuous} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %subview_1 = memref.subview %arg2[%arg8] [64] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %11 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %12 = ave.hir.vabs %res, %11 : vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %13 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      ave.hir.masked_store <NORM_B32> %subview_1[%c0], %13, %12 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
    }
    %2 = arith.bitcast %arg3 : f32 to i32
    %3 = arith.shrsi %2, %arg4 : i32
    %4 = arith.cmpi eq, %3, %arg5 : i32
    %5 = ave.hir.scalar_broadcast %4 : i1 -> vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
    scf.for %arg8 = %c0 to %c512 step %c64 {
      %subview = memref.subview %arg1[%arg8] [64] [1] : memref<512xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg6[%arg8] [64] [1] : memref<512xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
      %6 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
      %7 = ave.hir.preg.and <b8> %5, %res, %6 : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
      %8 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
      ave.hir.masked_store <NORM_B8> %subview_0[%c0], %8, %7 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>, hivm.is_continuous} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
      %subview_1 = memref.subview %arg2[%arg8] [64] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg7[%arg8] [64] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res_3 = ave.hir.vload <NORM> %subview_1[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %9 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %10 = ave.hir.vdivfhp %res_3, %1, %9 : vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %11 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %12 = ave.hir.vtrc %10, <trunc>, %11 : vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %13 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %14 = ave.hir.vmuls %12, %cst, %13 : vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, f32, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %15 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %16 = ave.hir.vsub %res_3, %14, %15 : vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      %17 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
      ave.hir.masked_store <NORM_B32> %subview_2[%c0], %17, %16 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>, vector<64xf32, #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>>
    }
    return
  }
}
