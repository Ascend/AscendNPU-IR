// RUN: bishengir-opt  -convert-hivmave-to-ave-intrin %s  | FileCheck %s

// CHECK-LABEL: func.func @test_vpack(
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_vpack(%arg0: vector<64xi32>) 
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    // CHECK: "hivm_regbaseintrins.intr.hivm.vpack.x"(%{{.*}}, %{{.*}}) : (vector<64xi32>, i32) -> vector<64xi32>
    %res = "ave.hir.vpack"(%arg0) : 
      (vector<64xi32>) -> (vector<64xi32>)
    "test.result_check"(%res) : (vector<64xi32>) -> ()
    return
  }
}

// -----// CHECK-LABEL: func.func @test_vunpack(
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_vunpack(%arg0: vector<128xi16>) 
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    // CHECK: "hivm_regbaseintrins.intr.hivm.vzunpack"(%{{.*}}, %{{.*}}) : (vector<128xi16>, i32) -> vector<128xi16>
    %res = "ave.hir.vunpack"(%arg0) : 
      (vector<128xi16>) -> (vector<128xi16>)
    "test.result_check"(%res) : (vector<128xi16>) -> ()
    return
  }
}