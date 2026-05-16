// RUN: bishengir-opt -analyze-vector-layout -analyze-alignment-bitwidth \
// RUN: -remove-vector-layout-attr -convert-hivmave-to-std -expand-strided-metadata \
// RUN: -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s
// CHECK-LABEL: @test_template_mask_store_op_mask
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_template_mask_store_op_mask(%arg0: memref<256x4xi64, #hivm.address_space<ub>>, %arg1: memref<256x4xi64, #hivm.address_space<ub>>, %arg2: memref<256x8xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 8 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
    } {element_alignment_bit_width = -1 : i32}
    return
  }
 }
