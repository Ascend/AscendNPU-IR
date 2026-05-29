// RUN: bishengir-opt %s -convert-hivmave-to-ave-intrin --split-input-file | FileCheck %s

// CHECK-LABEL: @test_plt
func.func @test_plt(%arg0: index) {
  // CHECK: %{{.*}} = llvm.trunc %{{.*}} : i64 to i32
  // CHECK-NEXT: %{{.*}} = "hivm_regbaseintrins.intr.hivm.plt.b8.v300"(%{{.*}}) : (i32) -> !llvm.struct<(vector<256xi1>, i32)>
  // CHECK-NEXT: %{{.*}} = llvm.extractvalue %{{.*}}[0] {mask_bit_width = 8 : i32} : !llvm.struct<(vector<256xi1>, i32)>
  // CHECK-NEXT: %{{.*}} = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(vector<256xi1>, i32)>
  %res, %new_true_shape = ave.hir.plt %arg0 {functionType = #ave.func_dist_type<pb8>}: vector<256xi1>, index
  "test.test"(%res, %new_true_shape) : (vector<256xi1>, index) -> ()
  return
}

// -----

// CHECK-LABEL: @test_vsel
func.func @test_vsel(%arg0 : vector<256xi1>, %arg1 : vector<64xf32>, %arg2 : vector<64xf32>) -> vector<64xf32> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vsel"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vsel %arg0, %arg1, %arg2: vector<256xi1>, vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_vsel_bf16
func.func @test_vsel_bf16(%arg0 : vector<256xi1>, %arg1 : vector<128xbf16>, %arg2 : vector<128xbf16>) -> vector<128xbf16> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vsel"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vsel %arg0, %arg1, %arg2: vector<256xi1>, vector<128xbf16>
  return %0 : vector<128xbf16>
}

// -----

// CHECK-LABEL: @test_reduction_add_v64f32
func.func @test_reduction_add_v64f32(%arg0 : vector<64xi1>, %arg1 : vector<64xf32>) -> vector<64xf32> {
  // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : vector<64xi1> to vector<256xi1>
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%{{.*}}, %{{.*}}) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.reduction <add>, %arg1, %arg0 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_reduction_max_v64f32
func.func @test_reduction_max_v64f32(%arg0 : vector<64xi1>, %arg1 : vector<64xf32>) -> vector<64xf32> {
  // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : vector<64xi1> to vector<256xi1>
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%{{.*}}, %{{.*}}) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.reduction <max>, %arg1, %arg0 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_broadcast_scalar
func.func @test_broadcast_scalar(%arg0 : f32) -> vector<64xf32> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vbr"(%{{.*}}) : (f32) -> vector<64xf32>
  %0 = ave.hir.scalar_broadcast %arg0 : f32 -> vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_broadcast_scalar_i1
func.func @test_broadcast_scalar_i1(%arg0 : i1) -> vector<64xi1> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vbr"(%{{.*}}) : (i8) -> vector<256xi8>
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vbr"(%{{.*}}) : (i8) -> vector<256xi8>
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi1>
  %0 = ave.hir.scalar_broadcast %arg0 : i1 -> vector<64xi1>
  return %0 : vector<64xi1>
}

// -----

// CHECK-LABEL: @test_broadcast_scalar_f8e4m3
func.func @test_broadcast_scalar_f8e4m3(%arg0 : f8E4M3FN) -> vector<256xf8E4M3FN> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vbr"(%{{.*}}) : (f8E4M3FN) -> vector<256xf8E4M3FN>
  %0 = ave.hir.scalar_broadcast %arg0 : f8E4M3FN -> vector<256xf8E4M3FN>
  return %0 : vector<256xf8E4M3FN>
}

// -----

// CHECK-LABEL: @test_broadcast_scalar_f8e5m2
func.func @test_broadcast_scalar_f8e5m2(%arg0 : f8E5M2) -> vector<256xf8E5M2> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vbr"(%{{.*}}) : (f8E5M2) -> vector<256xf8E5M2>
  %0 = ave.hir.scalar_broadcast %arg0 : f8E5M2 -> vector<256xf8E5M2>
  return %0 : vector<256xf8E5M2>
}

// -----

// CHECK-LABEL: @test_vtruncf_f322f16
func.func @test_vtruncf_f322f16(%arg0 : vector<64xf32>, %mask: vector<256xi1>) -> vector<128xf16> attributes {hivm.vector_function} {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xf16>
  %0 = ave.hir.vtruncf %arg0, <rint>, true, <part_even>, %mask : vector<64xf32>, vector<128xf16>, vector<256xi1>
  return %0 : vector<128xf16>
}

// -----

// CHECK-LABEL: @test_vtruncf_f322bf16
func.func @test_vtruncf_f322bf16(%arg0 : vector<64xf32>, %mask: vector<256xi1>) -> vector<128xbf16> attributes {hivm.vector_function} {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
  %0 = ave.hir.vtruncf %arg0, <rint>, true, <part_even>, %mask  : vector<64xf32>, vector<128xbf16>, vector<256xi1>
  return %0 : vector<128xbf16>
}

// -----

// CHECK-LABEL: @test_vextf_f162f32
func.func @test_vextf_f162f32(%arg0 : vector<128xf16>, %mask: vector<256xi1>) -> vector<64xf32> attributes {hivm.vector_function} {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
  %0 = ave.hir.vextf %arg0, <part_even>, %mask : vector<128xf16>, vector<64xf32>, vector<256xi1>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_vintlv_f8e5m2
func.func @test_vintlv_f8e5m2(
    %arg0 : vector<256xf8E5M2>,
    %arg1 : vector<256xf8E5M2>)
    -> vector<256xf8E5M2> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vintlv"(%{{.*}}, %{{.*}}) : (vector<256xf8E5M2>, vector<256xf8E5M2>) -> !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  %r0, %r1 = ave.hir.vintlv %arg0, %arg1 : vector<256xf8E5M2>, vector<256xf8E5M2>
  return %r0 : vector<256xf8E5M2>
}

// -----

// CHECK-LABEL: @test_vextf_bf162f32
func.func @test_vextf_bf162f32(%arg0 : vector<128xbf16>, %mask: vector<256xi1>) -> vector<64xf32> attributes {hivm.vector_function} {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
  %0 = ave.hir.vextf %arg0, <part_even>, %mask : vector<128xbf16>, vector<64xf32>, vector<256xi1>
  return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_vcmp_ule_uint8
#map = affine_map<(d0) -> (-d0 + 3978, 256)>
func.func @test_vcmp_ule_uint8(%arg0: memref<3978xi8, #hivm.address_space<ub>>, %arg1: memref<3978xi8, #hivm.address_space<ub>>, %arg2: memref<3978xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c3978 = arith.constant 3978 : index
  %c256 = arith.constant 256 : index
  %c0_i16 = arith.constant 0 : i16
  %c1_i16 = arith.constant 1 : i16
  %0 = ave.hir.pge <ALL> : vector<256xi1>
  %1 = ave.hir.broadcast %c1_i16, %0 : i16, vector<256xi1> -> vector<256xi8>
  %2 = ave.hir.pge <ALL> : vector<256xi1>
  %3 = ave.hir.broadcast %c0_i16, %2 : i16, vector<256xi1> -> vector<256xi8>
  scf.for %arg3 = %c0 to %c3978 step %c256 {
    %4 = affine.min #map(%arg3)
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<3978xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg3], sizes: [%4], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<3978xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%arg3], sizes: [%4], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg2 : memref<3978xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%arg3], sizes: [%4], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %5 = ave.hir.pltm %arg3, %c3978 : vector<256xi1>
    %6 = ave.hir.vload <NORM> %reinterpret_cast[%c0] : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
    %7 = ave.hir.vload <NORM> %reinterpret_cast_4[%c0] : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>

    // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vcmp.le.u.z"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi1>
    %8 = ave.hir.vcmp <ULE> %6, %7, %5 : vector<256xi8>, vector<256xi1> -> vector<256xi1>
    %9 = ave.hir.vsel %8, %1, %3 : vector<256xi1>, vector<256xi8>
    ave.hir.masked_store <NORM_B8> %reinterpret_cast_9[%c0], %5, %9 : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  }
  return
}

// -----

// CHECK-LABEL: @triton_xor_sum_dim_outlined_vf_0
#map1 = affine_map<(d0) -> (-d0 + 40000, 256)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_xor_sum_dim_outlined_vf_0(%arg0: memref<1x40000xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c40000 = arith.constant 40000 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %0 = ave.hir.pge <ALL> : vector<256xi1>
    %1 = ave.hir.broadcast %c0_i16, %0 : i16, vector<256xi1> -> vector<256xi8>
    scf.for %arg1 = %c0 to %c40000 step %c256 {
      %2 = affine.min #map1(%arg1)
      // CHECK: llvm.sdiv %{{.*}}, %{{.*}}  : i64
      // CHECK: llvm.trunc %{{.*}} : i64 to i16
      %3 = ave.hir.pltm %arg1, %c40000 : vector<256xi1>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<1x40000xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg1], sizes: [%2], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<?xi8, #map2, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B8> %reinterpret_cast[%c0], %3, %1 : memref<?xi8, #map2, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
    }
    return
  }
}

// -----

// CHECK-LABEL: @test_plt_b8_in_top_region_used_by_b32_op
func.func @test_plt_b8_in_top_region_used_by_b32_op(%arg0: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %res, %new_true_shape = ave.hir.plt %arg0 {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>, index
  "test.test"(%res, %new_true_shape) {element_alignment_bit_width = 32 : i32} : (vector<64xi1>, index) -> ()
  return
}
// CHECK: "hivm_regbaseintrins.intr.hivm.plt.b32.v300"
// CHECK: llvm.extractvalue {{.*}}[0] {mask_bit_width = 32 : i32

// -----

// CHECK-LABEL: @test_plt_b8_in_top_region_used_by_b16_op
func.func @test_plt_b8_in_top_region_used_by_b16_op(%arg0: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %res, %new_true_shape = ave.hir.plt %arg0 {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<pb16>} : vector<64xi1>, index
  "test.test"(%res, %new_true_shape) : (vector<64xi1>, index) -> ()
  return
}
// CHECK: "hivm_regbaseintrins.intr.hivm.plt.b16.v300"
// CHECK: llvm.extractvalue {{.*}}[0] {mask_bit_width = 16 : i32

// -----

// CHECK-LABEL: @test_plt_b8_in_top_region_user_has_no_elem_align
func.func @test_plt_b8_in_top_region_user_has_no_elem_align(%arg0: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %res, %new_true_shape = ave.hir.plt %arg0 {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>} : vector<256xi1>, index
  "test.test"(%res) : (vector<256xi1>) -> ()
  return
}
// CHECK: "hivm_regbaseintrins.intr.hivm.plt.b8.v300"
// CHECK: llvm.extractvalue {{.*}}[0] {mask_bit_width = 8 : i32

// -----

// CHECK-LABEL: @cast_to_nd_with_overflow_outlined_vf_0
#map3 = affine_map<()[s0, s1] -> (s0 * 2048 + s1 * 256)>
#map4 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @cast_to_nd_with_overflow_outlined_vf_0(%arg0: memref<9x8x32xi1, strided<[2048, 256, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2304_i32 = arith.constant 2304 : i32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %c2304_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
    %2 = ave.hir.vci %c0_i32, <INCREASE> : i32, vector<64xi32>
    scf.for %arg1 = %c0 to %c9 step %c1 {
      %3 = arith.index_cast %arg1 : index to i32
      %4 = ave.hir.scalar_broadcast %3 : i32 -> vector<64xi32>
      %5 = ave.hir.pge <ALL> : vector<64xi1>
      %6 = ave.hir.vmuls %4, %c256_i32, %5 : vector<64xi32>, i32, vector<64xi1>
      scf.for %arg2 = %c0 to %c8 step %c1 {
        %7 = arith.index_cast %arg2 : index to i32
        %8 = ave.hir.scalar_broadcast %7 : i32 -> vector<64xi32>
        %9 = ave.hir.pge <ALL> : vector<64xi1>
        %10 = ave.hir.vmuls %8, %c32_i32, %9 : vector<64xi32>, i32, vector<64xi1>
        %11 = ave.hir.pge <ALL> : vector<64xi1>
        %12 = ave.hir.vadd %6, %10, %11 : vector<64xi32>, vector<64xi1>
        %13 = ave.hir.pge <ALL> : vector<64xi1>
        %14 = ave.hir.vadd %12, %2, %13 : vector<64xi32>, vector<64xi1>
        %15 = ave.hir.pge <ALL> : vector<64xi1>
        %16 = ave.hir.vcmp <LT> %14, %1, %15 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
        %17 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<64xi1>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<9x8x32xi1, strided<[2048, 256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index, index, index
        %18 = affine.apply #map3()[%arg1, %arg2]
        // CHECK: %[[DESC:.*]] = builtin.unrealized_conversion_cast %reinterpret_cast
        // CHECK-SAME: to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: %[[BASE_PTR:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: %[[BIT_OFFSET:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 : index) : i64
        // CHECK: %[[BYTE_OFFSET:.*]] = llvm.udiv %[[BIT_OFFSET]], %[[C8]] : i64
        %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%18], sizes: [32], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<32xi1, #map4, #hivm.address_space<ub>>
        ave.hir.masked_store <NORM_B8> %reinterpret_cast[%c0], %17, %16 : memref<32xi1, #map4, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
      }
    }
    return
  }
}
