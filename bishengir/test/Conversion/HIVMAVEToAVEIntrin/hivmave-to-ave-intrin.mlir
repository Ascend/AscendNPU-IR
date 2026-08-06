// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 -analyze-vector-layout -remove-vector-layout-attr -ave-normalize-ops -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @was_bool_to_int8_plds_pk4
// CHECK: "hivm_regbaseintrins.intr.hivm.plds.b8"
// CHECK: "hivm_regbaseintrins.intr.hivm.pand.z"
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @was_bool_to_int8_plds_pk4(%mask: memref<16xi8, #hivm.address_space<ub>>, %data: memref<16xf32, #hivm.address_space<ub>>, %dst: memref<16xf32, #hivm.address_space<ub>>, %scale: f32, %fallback: f32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %full = ave.hir.pge <ALL> : vector<64xi1>
    %zero = ave.hir.broadcast %c0_i8, %full : i8, vector<64xi1> -> vector<64xi8>
    %fallback_vec = ave.hir.broadcast %fallback, %full : f32, vector<64xi1> -> vector<64xf32>
    %valid = ave.hir.pge <VL16> {mask_op_idx = 0 : i32} : vector<64xi1>
    %mask_i8 = ave.hir.vload <NORM> %mask[%c0] {was_bool_to_int8 = true} : memref<16xi8, #hivm.address_space<ub>> into vector<64xi8>
    %data_vec = ave.hir.vload <NORM> %data[%c0] : memref<16xf32, #hivm.address_space<ub>> into vector<64xf32>
    %scaled = ave.hir.vmuls %data_vec, %scale, %valid : vector<64xf32>, f32, vector<64xi1>
    %pred = ave.hir.vcmp <NE> %mask_i8, %zero, %valid : vector<64xi8>, vector<64xi1> -> vector<64xi1>
    %selected = ave.hir.vsel %pred, %scaled, %fallback_vec : vector<64xi1>, vector<64xf32>
    ave.hir.masked_store <NORM_B32> %dst[%c0], %valid, %selected : memref<16xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @was_bool_to_int8_plds_all_mask_no_pand
// CHECK: "hivm_regbaseintrins.intr.hivm.plds.b8"
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @was_bool_to_int8_plds_all_mask_no_pand(%mask: memref<16xi8, #hivm.address_space<ub>>, %data: memref<16xf32, #hivm.address_space<ub>>, %dst: memref<16xf32, #hivm.address_space<ub>>, %fallback: f32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %full = ave.hir.pge <ALL> : vector<64xi1>
    %zero = ave.hir.broadcast %c0_i8, %full : i8, vector<64xi1> -> vector<64xi8>
    %fallback_vec = ave.hir.broadcast %fallback, %full : f32, vector<64xi1> -> vector<64xf32>
    %mask_i8 = ave.hir.vload <NORM> %mask[%c0] {was_bool_to_int8 = true} : memref<16xi8, #hivm.address_space<ub>> into vector<64xi8>
    %data_vec = ave.hir.vload <NORM> %data[%c0] : memref<16xf32, #hivm.address_space<ub>> into vector<64xf32>
    %pred = ave.hir.vcmp <NE> %mask_i8, %zero, %full : vector<64xi8>, vector<64xi1> -> vector<64xi1>
    %selected = ave.hir.vsel %pred, %data_vec, %fallback_vec : vector<64xi1>, vector<64xf32>
    ave.hir.masked_store <NORM_B32> %dst[%c0], %full, %selected : memref<16xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @plain_i8_vcmp_ne_zero_keeps_vcmp
// CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"
// CHECK: "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @plain_i8_vcmp_ne_zero_keeps_vcmp(%mask: memref<16xi8, #hivm.address_space<ub>>, %data: memref<16xf32, #hivm.address_space<ub>>, %dst: memref<16xf32, #hivm.address_space<ub>>, %fallback: f32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %full = ave.hir.pge <ALL> : vector<64xi1>
    %zero = ave.hir.broadcast %c0_i8, %full : i8, vector<64xi1> -> vector<64xi8>
    %fallback_vec = ave.hir.broadcast %fallback, %full : f32, vector<64xi1> -> vector<64xf32>
    %valid = ave.hir.pge <VL16> {mask_op_idx = 0 : i32} : vector<64xi1>
    %mask_i8 = ave.hir.vload <NORM> %mask[%c0] : memref<16xi8, #hivm.address_space<ub>> into vector<64xi8>
    %data_vec = ave.hir.vload <NORM> %data[%c0] : memref<16xf32, #hivm.address_space<ub>> into vector<64xf32>
    %pred = ave.hir.vcmp <NE> %mask_i8, %zero, %valid : vector<64xi8>, vector<64xi1> -> vector<64xi1>
    %selected = ave.hir.vsel %pred, %data_vec, %fallback_vec : vector<64xi1>, vector<64xf32>
    ave.hir.masked_store <NORM_B32> %dst[%c0], %valid, %selected : memref<16xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}
