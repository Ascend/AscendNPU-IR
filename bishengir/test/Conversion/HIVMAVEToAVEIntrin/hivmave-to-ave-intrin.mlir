// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 -analyze-vector-layout -analyze-alignment-bitwidth -remove-vector-layout-attr -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 -convert-hivmave-to-ave-intrin -cse -canonicalize %s -split-input-file | FileCheck %s --check-prefix=CSE

// CHECK-LABEL: func.func @cast_to_nd_with_overflow_outlined_vf_0
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @cast_to_nd_with_overflow_outlined_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %res = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.vfptosi %res, <truncwithoverflow>, %0 {sat = true} : vector<64xf32>, vector<64xi1>, vector<64xi32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    %3 = ave.hir.vsitofp %1, %2 {rnd = #hivm.round_mode<trunc>} : vector<64xi32>, vector<64xi1>, vector<64xf32>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    %5 = ave.hir.vtruncf %3, <trunc>, true, <part_even>, %4 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x"(%46, %48, %49, %50) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<64xi32>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s322u16.x"(%51, %48, %52, %53) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
    // CHECK: builtin.unrealized_conversion_cast %55 : vector<64xi16> to vector<128xi16>
    %7 = ave.hir.vfptoui %5, <trunc>, true, <part_even>, %6 : vector<64xf16>, vector<64xi1>, vector<64xi16>
    %8 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B16> %arg1[%c0], %8, %7 : memref<64xi16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
    return
  }
}

// -----

// CHECK-LABEL: @test_trunci_i642u32_sat
#map = affine_map<(d0) -> (-d0 + 125, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_trunci_i642u32_sat(%arg0: memref<125xi64, #hivm.address_space<ub>>, %arg1: memref<125xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c125 = arith.constant 125 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c125 step %c64 {
      %1 = affine.min #map(%arg2)
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<125xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg2], sizes: [%1], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<125xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%arg2], sizes: [%1], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = ave.hir.pltm %arg2, %c125 : vector<64xi1>
      %3 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %cast = memref.cast %reinterpret_cast : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vload_NORM_int64_t_rank1(%3, %cast, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
      %4 = builtin.unrealized_conversion_cast %2 : vector<64xi1> to vector<256xi1>
      // CHECK: func.call @_mlir_ciface_cast_int64_t_to_uint32_t_sat(%17, %18) : (!llvm.ptr, vector<256xi1>) -> vector<64xi32>
      %5 = func.call @_mlir_ciface_cast_int64_t_to_uint32_t_sat(%3, %4) : (!llvm.ptr, vector<256xi1>) -> vector<64xi32>
      ave.hir.masked_store <NORM_B32> %reinterpret_cast_4[%c0], %2, %5 {hivm.is_continuous} : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
  func.func private @load_gm_to_ubuf_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @_mlir_ciface_cast_int64_t_to_uint32_t_sat(!llvm.ptr, vector<256xi1>) -> vector<64xi32> attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @vload_NORM_int64_t_rank1(!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----

// CHECK-LABEL: @test_trunci_u642u16_sat
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_trunci_u642u16_sat(%arg0: memref<13xi64, #hivm.address_space<ub>>, %arg1: memref<13xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c13 = arith.constant 13 : index
    %res, %new_true_shape = ave.hir.plt %c13 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %1 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
    %cast = memref.cast %arg0 : memref<13xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vload_NORM_int64_t_rank1(%1, %cast, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
    %2 = builtin.unrealized_conversion_cast %res : vector<64xi1> to vector<256xi1>
    // CHECK: call @_mlir_ciface_cast_uint64_t_to_uint32_t_sat(%10, %11) : (!llvm.ptr, vector<256xi1>) -> vector<64xi32>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u322u16.x"(%13, %7, %15, %16) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
    %3 = call @_mlir_ciface_cast_uint64_t_to_uint32_t_sat(%1, %2) : (!llvm.ptr, vector<256xi1>) -> vector<64xi32>
    %4 = ave.hir.vtrunci %3, true, %res {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi32>, vector<64xi16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %arg1[%c0], %res, %4 : memref<13xi16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
    return
  }
  func.func private @load_gm_to_ubuf_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int16_t(memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @_mlir_ciface_cast_uint64_t_to_uint32_t_sat(!llvm.ptr, vector<256xi1>) -> vector<64xi32> attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @vload_NORM_int64_t_rank1(!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----

// CHECK-LABEL: @test_trunci_u642i8_sat
#map = affine_map<(d0) -> (-d0 + 300, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_trunci_u642i8_sat(%arg0: memref<300xi64, #hivm.address_space<ub>>, %arg1: memref<300xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c300 = arith.constant 300 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c300 step %c64 {
      %1 = affine.min #map(%arg2)
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<300xi64, #hivm.address_space<ub>> -> memref<i64, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg2], sizes: [%1], strides: [1] : memref<i64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<300xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%arg2], sizes: [%1], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = ave.hir.pltm %arg2, %c300 : vector<64xi1>
      %3 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %cast = memref.cast %reinterpret_cast : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vload_NORM_int64_t_rank1(%3, %cast, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
      %4 = builtin.unrealized_conversion_cast %2 : vector<64xi1> to vector<256xi1>
      // CHECK: func.call @_mlir_ciface_cast_uint64_t_to_uint32_t_sat({{.*}}, {{.*}}) : (!llvm.ptr, vector<256xi1>) -> vector<64xi32>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u322u8.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<256xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtif.u82f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xf16>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s8.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
      %5 = func.call @_mlir_ciface_cast_uint64_t_to_uint32_t_sat(%3, %4) : (!llvm.ptr, vector<256xi1>) -> vector<64xi32>
      %6 = ave.hir.vtrunci %5, true, %2 {pp = #ave.vcvt_pp_type<pp0>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
      %7 = ave.hir.vuitofp %6, <part_even>, %2 : vector<64xi8>, vector<64xi1>, vector<64xf16>
      %8 = ave.hir.vfptosi %7, <trunc>, %2 {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xf16>, vector<64xi1>, vector<64xi8>
      ave.hir.masked_store <NORM_B8> %reinterpret_cast_4[%c0], %2, %8 {hivm.is_continuous} : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
  func.func private @load_gm_to_ubuf_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int8_t(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @_mlir_ciface_cast_uint64_t_to_uint32_t_sat(!llvm.ptr, vector<256xi1>) -> vector<64xi32> attributes {hacc.always_inline, llvm.emit_c_interface}
  func.func private @vload_NORM_int64_t_rank1(!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) attributes {hacc.always_inline, llvm.emit_c_interface}
}

// -----

// CHECK-LABEL: @test_trunci_u162i8_sat
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_trunci_u162i8_sat(%arg0: memref<9xi16, #hivm.address_space<ub>>, %arg1: memref<9xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 16 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    %res, %new_true_shape = ave.hir.plt %c9 {mask_op_idx = 0 : i32} : vector<128xi1>, index
    %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<9xi16, #hivm.address_space<ub>> into vector<128xi16>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u162u8.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xi16>, vector<256xi1>, i32, i32) -> vector<256xi8>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtif.u82f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xf16>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s8.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
    %0 = ave.hir.vtrunci %res_0, true, %res {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2ui>} : vector<128xi16>, vector<128xi8>, vector<128xi1>
    %1 = ave.hir.vuitofp %0, <part_even>, %res : vector<128xi8>, vector<128xi1>, vector<128xf16>
    %2 = ave.hir.vfptosi %1, <trunc>, %res {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<128xf16>, vector<128xi1>, vector<128xi8>
    ave.hir.masked_store <NORM_B8> %arg1[%c0], %res, %2 : memref<9xi8, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi8>
    return
  }
}

// -----

// CHECK-LABEL: @test_trunci_u322i8_sat
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_trunci_u322i8_sat(%arg0: memref<9xi32, #hivm.address_space<ub>>, %arg1: memref<9xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    %res, %new_true_shape = ave.hir.plt %c9 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<9xi32, #hivm.address_space<ub>> into vector<64xi32>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xi32>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u322s16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtif.s162f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xi16>, vector<256xi1>, i32) -> vector<128xf16>
    %0 = ave.hir.vtrunci %res_0, true, %res {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2si>} : vector<64xi32>, vector<64xi16>, vector<64xi1>
    %1 = ave.hir.vsitofp %0, %res {rnd = #hivm.round_mode<trunc>} : vector<64xi16>, vector<64xi1>, vector<64xf16>
    %2 = ave.hir.vfptosi %1, <trunc>, %res {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xf16>, vector<64xi1>, vector<64xi8>
    ave.hir.masked_store <NORM_B8> %arg1[%c0], %res, %2 : memref<9xi8, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
    return
  }
}

// -----

// CHECK-LABEL: @test_trunci_u322i16_sat
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_trunci_u322i16_sat(%arg0: memref<9xi32, #hivm.address_space<ub>>, %arg1: memref<9xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    %res, %new_true_shape = ave.hir.plt %c9 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<9xi32, #hivm.address_space<ub>> into vector<64xi32>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u322s16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
    %0 = ave.hir.vtrunci %res_0, true, %res {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2si>} : vector<64xi32>, vector<64xi16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %arg1[%c0], %res, %0 : memref<9xi16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_lower_reduction_for_vector_length(
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_lower_reduction_for_vector_length(%arg0: memref<64xf16, #hivm.address_space<ub>>, %arg1: memref<1xi16, #hivm.address_space<ub>>)
  attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %res = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf16, #hivm.address_space<ub>> into vector<64xi16>
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK: %[[src:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi16> to vector<128xi16>
    // CHECK: %[[mask:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
    // CHECK: %[[res:.*]] = "hivm_regbaseintrins.intr.hivm.vcmin.s.x"(%[[src]], %[[mask]]) : (vector<128xi16>, vector<256xi1>) -> vector<128xi16>
    // CHECK: builtin.unrealized_conversion_cast %[[res]] : vector<128xi16> to vector<64xi16>
    %1 = ave.hir.reduction <min>, %res, %0 : vector<64xi16>, vector<64xi1> -> vector<64xi16>
    %2 = builtin.unrealized_conversion_cast %1 : vector<64xi16> to vector<1xi16>
    %3 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <NORM_B16> %arg1[%c0], %3, %2 : memref<1xi16, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi16>
    return
  }
}

// -----

// CSE-LABEL: @plds_i1_as_msk_of_vsel_f32
// CSE: %[[PLDS_B8:.*]] = "hivm_regbaseintrins.intr.hivm.plds.b8"
// CSE: %[[PGE_B8:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b8"
// CSE: %[[PINTLV0:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b8"(%[[PLDS_B8]], %[[PGE_B8]]) : (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CSE: %[[PINTLV00:.*]] = llvm.extractvalue %[[PINTLV0]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CSE: %[[PINTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b8"(%[[PINTLV00]], %[[PGE_B8]]) : (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CSE: %[[PINTLV10:.*]] = llvm.extractvalue %[[PINTLV1]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CSE: "hivm_regbaseintrins.intr.hivm.vsel"(%[[LHS:.*]], %[[RHS:.*]], %[[PINTLV10]]) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
#map4 = affine_map<()[s0] -> (s0 * 16)>
#map15 = affine_map<(d0)[s0] -> (d0 + s0)>
#map16 = affine_map<()[s0] -> (s0 * 256)>
#map18 = affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>
func.func @plds_i1_as_msk_of_vsel_f32(%arg0: memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %arg5: memref<4x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c1040 = arith.constant 1040 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %T = ave.hir.pge <ALL> : vector<64xi1>
  %0 = ave.hir.broadcast %cst_0, %T {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
  %1 = ave.hir.broadcast %cst_1, %T {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg6 = %c0 to %c64 step %c1 {
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg0 : memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index
    %2 = affine.apply #map16()[%arg6]
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%2], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map15, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %reinterpret_cast_5[%c0] {element_alignment_bit_width = 8 : i32} : memref<64xi1, #map15, #hivm.address_space<ub>> into vector<64xi1>
    %9 = ave.hir.vsel %res, %0, %1 {element_alignment_bit_width = 32 : i32} : vector<64xi1>, vector<64xf32>
    %11 = ave.hir.vtruncf %9, <rint>, false, <part_even>, %T {element_alignment_bit_width = 32 : i32} : vector<64xf32>, vector<128xbf16>, vector<64xi1>
    %base_buffer_19, %offset_20, %sizes_21:3, %strides_22:3 = memref.extract_strided_metadata %arg5 : memref<4x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index
    %13 = affine.apply #map4()[%arg6]
    %14 = ave.hir.pge <VL64> : vector<128xi1>
    %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [%13], sizes: [4, 16], strides: [1040, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x16xbf16, #map18, #hivm.address_space<ub>>
    ave.hir.store_with_stride %reinterpret_cast_23[%c0, %c0], %c1040, %14, %11 {element_alignment_bit_width = 16 : i32} : memref<4x16xbf16, #map18, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
  } {element_alignment_bit_width = -1 : i32}
  return
}

// -----

// CHECK-LABEL: @test_bf16_gather_with_misaligned_element_alignment
// When vgather has element_alignment_bit_width = 16 but user has 32,
// pintlv SHOULD be applied after vgather2_v300.v128bf16 for alignment
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_bf16_gather_with_misaligned_element_alignment(%arg0: memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1: index) -> () attributes {element_alignment_bit_width = -1 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %mask = ave.hir.pge <ALL> : vector<128xi1>
    %v_index = ave.hir.vci %c0_i16, <INCREASE> : i16, vector<128xi16>
    %v_mask = ave.hir.pge <ALL> : vector<128xi1>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128bf16"
    // CHECK: "hivm_regbaseintrins.intr.hivm.vintlv"
    %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%v_index], %v_mask {element_alignment_bit_width = 16 : i32} : memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xbf16>
    ave.hir.masked_store <NORM_B16> %arg0[%c0, %c0], %mask, %0 {element_alignment_bit_width = 32 : i32} : memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
    return
  }
}

// -----

// CHECK-LABEL: @test_bf16_gather_with_aligned_element_alignment
// When vgather has element_alignment_bit_width = 32 and user also has 32,
// pintlv should NOT be applied after vgather2_v300.v128bf16
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_bf16_gather_with_aligned_element_alignment(%arg0: memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1: index) -> () attributes {element_alignment_bit_width = -1 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %mask = ave.hir.pge <ALL> : vector<128xi1>
    %v_index = ave.hir.vci %c0_i16, <INCREASE> : i16, vector<128xi16>
    %v_mask = ave.hir.pge <ALL> : vector<128xi1>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128bf16"
    // CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vintlv"
    %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%v_index], %v_mask {element_alignment_bit_width = 32 : i32} : memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xbf16>
    ave.hir.masked_store <NORM_B16> %arg0[%c0, %c0], %mask, %0 {element_alignment_bit_width = 32 : i32} : memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_vadds(
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_vadds(%arg0: memref<33xi8, #hivm.address_space<ub>>, %arg1: i8, %arg2: memref<33xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 8 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c33 = arith.constant 33 : index
    %c0 = arith.constant 0 : index
    %res, %new_true_shape = ave.hir.plt %c33 {mask_op_idx = 0 : i32} : vector<256xi1>, index
    %res_0 = ave.hir.vload <NORM> %arg0[%c0] {element_alignment_bit_width = 8 : i32} : memref<33xi8, #hivm.address_space<ub>> into vector<256xi8>
    // CHECK-NOT: llvm.sext %{{.*}} : i8 to i16
    // CHECK: "hivm_regbaseintrins.intr.hivm.vadds.s.x"(%{{.*}}, %{{.*}}1, %{{.*}}) : (vector<256xi8>, i8, vector<256xi1>) -> vector<256xi8>
    %1 = ave.hir.vadds %res_0, %arg1, %res {element_alignment_bit_width = 8 : i32} : vector<256xi8>, i8, vector<256xi1>
    ave.hir.masked_store <NORM_B8> %arg2[%c0], %res, %1 {element_alignment_bit_width = 8 : i32} : memref<33xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
    return
  }
}