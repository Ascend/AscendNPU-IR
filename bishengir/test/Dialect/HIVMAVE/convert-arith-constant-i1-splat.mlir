// RUN: bishengir-opt -convert-arith-to-hivmave %s | FileCheck %s

#map = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<()[s0] -> (s0 * 89)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @calc_cube_vector_mix_aiv_outlined_vf_2(%arg0: memref<1x1xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<true> : vector<1x256xi1>
    // CHECK-NOT: vector.shape_cast
    // CHECK: ave.hir.pge <ALL> : vector<256xi1>
    // CHECK: builtin.unrealized_conversion_cast
    %c0 = arith.constant 0 : index
    %0 = ave.hir.pge <VL1> : vector<256xi1>
    %1 = builtin.unrealized_conversion_cast %0 : vector<256xi1> to vector<1x256xi1>
    annotation.mark %1 {mask_op_idx = -1 : i32} : vector<1x256xi1>
    %2 = builtin.unrealized_conversion_cast %cst : vector<1x256xi1> to vector<256xi1>
    ave.hir.masked_store <NORM_B8> %arg0[%c0, %c0], %0, %2 : memref<1x1xi1, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
    return
  }
}
