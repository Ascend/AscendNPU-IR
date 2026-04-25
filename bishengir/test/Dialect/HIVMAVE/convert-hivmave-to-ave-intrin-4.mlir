// RUN: bishengir-opt -append-vector-layout -annotate-dist-op-layout \
// RUN: -eliminate-vector-layout -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s
// CHECK-LABEL: @test_ext_user_is_call
#map = affine_map<()[s0, s1, s2] -> (s0 * 256 + s1 * 64 + s2 * 8)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_ext_user_is_call(%arg0: memref<8x4x8x8xi8, #hivm.address_space<ub>>, %arg1: memref<8x4x8x8xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c8 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        scf.for %arg4 = %c0 to %c8 step %c1 {
          %1 = ave.hir.pge <VL8> {mask_op_idx = 0 : i32} : vector<64xi1>
          %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %arg0 : memref<8x4x8x8xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
          %2 = affine.apply #map()[%arg2, %arg3, %arg4]
          %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%2], sizes: [8], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<8xi8, #map1, #hivm.address_space<ub>>
          %res = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 32 : i32} : memref<8xi8, #map1, #hivm.address_space<ub>> into vector<64xi8>
          %3 = ave.hir.vextui %res, %1 {element_alignment_bit_width = 32 : i32, pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<64xi1>
          // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"
          // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s8"
          // CHECK: "hivm_regbaseintrins.intr.hivm.vintlv"
          // CHECK: "hivm_regbaseintrins.intr.hivm.vintlv"
          // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u82u32.x"
          %4 = ave.hir.pge <ALL> : vector<64xi1>
          %5 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
          %6 = builtin.unrealized_conversion_cast %4 : vector<64xi1> to vector<256xi1>
          func.call @_mlir_ciface_cast_uint32_t_to_int64_t(%5, %3, %6) : (!llvm.ptr, vector<64xi32>, vector<256xi1>) -> ()
          %base_buffer_0, %offset_1, %sizes_2:4, %strides_3:4 = memref.extract_strided_metadata %arg1 : memref<8x4x8x8xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index, index
          %7 = affine.apply #map()[%arg2, %arg3, %arg4]
          %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%7], sizes: [8], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<8xi64, #map1, #hivm.address_space<ub>>
          %cast = memref.cast %reinterpret_cast_4 : memref<8xi64, #map1, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %8 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
          func.call @masked_store_NORM_B64_int64_t_rank1(%cast, %c0_i64, %8, %5) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64, vector<256xi1>, !llvm.ptr) -> ()
        } {element_alignment_bit_width = 32 : i32}
      } {element_alignment_bit_width = 32 : i32}
    } {element_alignment_bit_width = 32 : i32}
    return
  }
  func.func private @_mlir_ciface_cast_uint32_t_to_int64_t(!llvm.ptr, vector<64xi32>, vector<256xi1>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @masked_store_NORM_B64_int64_t_rank1(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64, vector<256xi1>, !llvm.ptr) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface} 
}