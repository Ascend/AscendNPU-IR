// RUN: bishengir-opt --hoist-vstas %s | FileCheck %s
 
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: func @test_hoist_loop
  func.func @test_hoist_loop(%arg0: memref<64xf32, #hivm.address_space<ub>>, %step: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, noinline, outline = true, vector_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
 
    %stride_4 = llvm.mlir.constant(4 : i32) : i32
 
    %vec_payload = llvm.mlir.undef : vector<64xf32>
 
    %base, %offset, %size, %stride = memref.extract_strided_metadata %arg0 : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
 
    // CHECK: %[[INIT_VEC:.*]] = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"()
    // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[CARRIED_VEC:.*]] = %[[INIT_VEC]], %[[CARRIED_PTR:.*]] = %{{.*}})
    scf.for %iv = %c0 to %c64 step %step {
       %iv_i64 = arith.index_cast %iv : index to i64
       %iv_idx = builtin.unrealized_conversion_cast %iv_i64 : i64 to index
       
       %view = memref.reinterpret_cast %base to offset: [%iv_idx], sizes: [1], strides: [1] 
               : memref<f32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
       
       %struct = builtin.unrealized_conversion_cast %view : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
       %ptr_base = llvm.extractvalue %struct[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
       %ptr_offset = llvm.extractvalue %struct[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
       %ptr_real = llvm.getelementptr %ptr_base[%ptr_offset] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
 
       %init_align = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"() : () -> vector<32xi8>
 
       // CHECK: %[[RES_STRUCT:.*]] = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%{{.*}}, %{{.*}}, %{{.*}}, %[[CARRIED_VEC]])
       %vstus_res = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%vec_payload, %ptr_real, %stride_4, %init_align) {hivm.is_continuous}
                    : (vector<64xf32>, !llvm.ptr<6>, i32, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
       
       %res_vec = llvm.extractvalue %vstus_res[0] : !llvm.struct<(vector<32xi8>, ptr<6>)>
       %res_ptr = llvm.extractvalue %vstus_res[1] : !llvm.struct<(vector<32xi8>, ptr<6>)>
 
       // CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"
       "hivm_regbaseintrins.intr.hivm.vstas"(%res_vec, %res_ptr, %c0_i32, %c0_i32) {hivm.is_continuous}
           : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()
       
       // CHECK: scf.yield %{{.*}}, %{{.*}}
    }
    // CHECK: }
    // CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
    return
  }
 
  // CHECK-LABEL: func @test_no_hoist_exceeds_threshold
  func.func @test_no_hoist_exceeds_threshold(%arg0: memref<64xf32, #hivm.address_space<ub>>, %step: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, noinline, outline = true, vector_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %stride_4 = llvm.mlir.constant(4 : i32) : i32
    %vec_payload = llvm.mlir.undef : vector<64xf32>
    %base, %offset, %size, %stride = memref.extract_strided_metadata %arg0 : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index

    scf.for %iv = %c0 to %c64 step %step {
      %iv_i64 = arith.index_cast %iv : index to i64
      %iv_idx = builtin.unrealized_conversion_cast %iv_i64 : i64 to index
      %view = memref.reinterpret_cast %base to offset: [%iv_idx], sizes: [1], strides: [1]
              : memref<f32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %struct = builtin.unrealized_conversion_cast %view : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
      %ptr_base = llvm.extractvalue %struct[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
      %ptr_offset = llvm.extractvalue %struct[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
      %ptr_real = llvm.getelementptr %ptr_base[%ptr_offset] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32

      %init_align_0 = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"() : () -> vector<32xi8>
      %vstus_res_0 = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%vec_payload, %ptr_real, %stride_4, %init_align_0) {hivm.is_continuous}
                    : (vector<64xf32>, !llvm.ptr<6>, i32, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_vec_0 = llvm.extractvalue %vstus_res_0[0] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_ptr_0 = llvm.extractvalue %vstus_res_0[1] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      "hivm_regbaseintrins.intr.hivm.vstas"(%res_vec_0, %res_ptr_0, %c0_i32, %c0_i32) {hivm.is_continuous}
          : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()

      %init_align_1 = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"() : () -> vector<32xi8>
      %vstus_res_1 = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%vec_payload, %ptr_real, %stride_4, %init_align_1) {hivm.is_continuous}
                    : (vector<64xf32>, !llvm.ptr<6>, i32, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_vec_1 = llvm.extractvalue %vstus_res_1[0] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_ptr_1 = llvm.extractvalue %vstus_res_1[1] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      "hivm_regbaseintrins.intr.hivm.vstas"(%res_vec_1, %res_ptr_1, %c0_i32, %c0_i32) {hivm.is_continuous}
          : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()

      %init_align_2 = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"() : () -> vector<32xi8>
      %vstus_res_2 = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%vec_payload, %ptr_real, %stride_4, %init_align_2) {hivm.is_continuous}
                    : (vector<64xf32>, !llvm.ptr<6>, i32, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_vec_2 = llvm.extractvalue %vstus_res_2[0] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_ptr_2 = llvm.extractvalue %vstus_res_2[1] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      "hivm_regbaseintrins.intr.hivm.vstas"(%res_vec_2, %res_ptr_2, %c0_i32, %c0_i32) {hivm.is_continuous}
          : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()

      %init_align_3 = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"() : () -> vector<32xi8>
      %vstus_res_3 = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%vec_payload, %ptr_real, %stride_4, %init_align_3) {hivm.is_continuous}
                    : (vector<64xf32>, !llvm.ptr<6>, i32, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_vec_3 = llvm.extractvalue %vstus_res_3[0] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      %res_ptr_3 = llvm.extractvalue %vstus_res_3[1] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      "hivm_regbaseintrins.intr.hivm.vstas"(%res_vec_3, %res_ptr_3, %c0_i32, %c0_i32) {hivm.is_continuous}
          : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()
    }
    return
  }
  
  func.func @main_caller(%arg0: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, noinline, outline = true, vector_mode = "simd"} {
    %c1_step = arith.constant 1 : index
    func.call @test_hoist_loop(%arg0, %c1_step) : (memref<64xf32, #hivm.address_space<ub>>, index) -> ()
    return
  }
}
