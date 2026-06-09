// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin %s | FileCheck %s
 func.func @triton_unk_fused__npu_dtype_cast_eq_ge_masked__0_outlined_vf_3(%arg0: memref<128xi1, #hivm.address_space<ub>>, %arg1: memref<128xi64, #hivm.address_space<ub>>, %arg2: memref<128xi64, #hivm.address_space<ub>>, %arg3: memref<128xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %c0_i64 = arith.constant 0 : i64
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
      // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
      %1 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
      %2 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
      %5 = builtin.unrealized_conversion_cast %1 : vector<64xi1> to vector<256xi1>
      func.call @_mlir_ciface_vsel_int64_t(%2, %5, %2, %2) : (!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) -> ()
    }
    return
  }
 func.func private @_mlir_ciface_vsel_int64_t(!llvm.ptr, vector<256xi1>, !llvm.ptr, !llvm.ptr) attributes {hacc.always_inline, llvm.emit_c_interface}

func.func @test_load_b8(%arg0: memref<128xi32, #hivm.address_space<ub>>, %arg1: memref<128xi1, #hivm.address_space<ub>>, %arg2: memref<128xi32, #hivm.address_space<ub>>, %arg3: memref<128xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c-1_i32 = arith.constant -1 : i32
  %0 = ave.hir.pge <ALL>  : vector<64xi1>
  %1 = ave.hir.broadcast %c-1_i32, %0  : i32, vector<64xi1> -> vector<64xi32>
  %2 = ave.hir.pge <ALL>  : vector<64xi1>
  %3 = ave.hir.broadcast %c0_i32, %2 {element_alignment_bit_width = 32 : i32} : i32, vector<64xi1> -> vector<64xi32>
  scf.for %arg4 = %c0 to %c128 step %c64 {
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg4], sizes: [64], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<128xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%arg4], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg2 : memref<128xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%arg4], sizes: [64], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %arg3 : memref<128xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer_10 to offset: [%arg4], sizes: [64], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %reinterpret_cast[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
    // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
    %res_15 = ave.hir.vload <NORM> %reinterpret_cast_4[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
    %res_16 = ave.hir.vload <NORM> %reinterpret_cast_9[%c0] : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    %5 = ave.hir.preg.not <b8> %res_15, %4 : vector<64xi1>
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    %7 = ave.hir.vcmp <LT> %res, %3, %6 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
    %8 = ave.hir.pge <ALL> : vector<64xi1>
    %9 = ave.hir.preg.or <b8> %7, %5, %8 : vector<64xi1>
    %10 = ave.hir.vsel %9, %1, %res_16 : vector<64xi1>, vector<64xi32>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %reinterpret_cast_14[%c0], %11, %10 {hivm.is_continuous} : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  }
  return
}

#map = affine_map<(d0) -> (-d0 + 104, 64)>
#map1 = affine_map<()[s0, s1] -> (s0 * 104 + s1)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map4 = affine_map<()[s0, s1] -> (s0 * 256 + s1)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1)>
#map6 = affine_map<()[s0] -> (s0 * 4)>
  func.func @triton_poi_fused_140_outlined_vf_11(%arg0: memref<8xi8, #hivm.address_space<ub>>, %arg1: memref<8x104xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %arg2: memref<8x104xf32, #hivm.address_space<ub>>, %arg3: memref<8x104xf32, #hivm.address_space<ub>>, %arg4: memref<8x104xf32, #hivm.address_space<ub>>, %arg5: memref<8x104xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c104 = arith.constant 104 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0_i16 = arith.constant 0 : i16
    %cst = arith.constant 0.000000e+00 : f32
    %2 = ave.hir.pge <ALLF> : vector<64xi1>
    %3 = ave.hir.pge <ALL>: vector<64xi1>
    %4 = ave.hir.broadcast %c0_i16, %3  : i16, vector<64xi1> -> vector<64xi8>
    scf.for %arg6 = %c0 to %c8 step %c1 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<8xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg6], sizes: [1], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<1xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
      scf.for %arg7 = %c0 to %c104 step %c64 {
        %5 = affine.min #map(%arg7)
        scf.for %arg8 = %c0 to %5 step %c64 {
          %14 = affine.min #map3(%arg8)[%5]
          %res_27 = ave.hir.vload <BRC_B8> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
          %15 = ave.hir.pge <ALL>  : vector<64xi1>
          %16 = ave.hir.vcmp <NE> %res_27, %4, %15  : vector<64xi8>, vector<64xi1> -> vector<64xi1>
          %17 = ave.hir.pge <ALL>  : vector<64xi1>
          %18 = ave.hir.preg.xor <b8> %16, %2, %17  : vector<64xi1>
          %res_28, %new_true_shape = ave.hir.plt %14 {mask_op_idx = 0 : i32} : vector<64xi1>, index
          %base_buffer_29, %offset_30, %sizes_31:2, %strides_32:2 = memref.extract_strided_metadata %arg1 : memref<8x104xi1, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index
          %19 = affine.apply #map4()[%arg6, %arg7]
          %20 = affine.apply #map5()[%19, %arg8]
          %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [%20], sizes: [%14], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<?xi1, #map2, #hivm.address_space<ub>>
          // CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi1>, !llvm.ptr<6>, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
          ave.hir.masked_store <NORM_B8> %reinterpret_cast_33[%c0], %res_28, %18 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>, hivm.is_continuous} : memref<?xi1, #map2, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
        }
      }
    }
    return
  }
 
func.func @test_load_normb8_i1_unalign(%arg0: memref<80xi1, #hivm.address_space<ub>>, %arg1: memref<16x80xf32, #hivm.address_space<ub>>, %arg2: memref<1xf32, #hivm.address_space<ub>>, %arg3: memref<16x80xf32, #hivm.address_space<ub>>, %arg4: memref<16x80xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
%c0 = arith.constant 0 : index
%c80 = arith.constant 80 : index
%c64 = arith.constant 64 : index
%c16 = arith.constant 16 : index
%c1 = arith.constant 1 : index
%cst = arith.constant -9.99999979E+10 : f32
%c2147483647_i32 = arith.constant 2147483647 : i32
%0 = ave.hir.pge <ALL> {element_alignment_bit_width = -1 : i32} : vector<64xi1>
%1 = ave.hir.broadcast %c2147483647_i32, %0 {element_alignment_bit_width = 32 : i32} : i32, vector<64xi1> -> vector<64xi32>
%2 = ave.hir.pge <ALL> {element_alignment_bit_width = -1 : i32} : vector<64xi1>
%3 = ave.hir.broadcast %cst, %2 {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
%4 = ave.hir.pge <ALLF> {element_alignment_bit_width = -1 : i32} : vector<64xi1>
scf.for %arg5 = %c0 to %c16 step %c1 {
  scf.for %arg6 = %c0 to %c80 step %c64 {
    %5 = affine.min affine_map<(d0) -> (-d0 + 80, 64)>(%arg6)
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<80xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg6], sizes: [%5], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<?xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %6 = ave.hir.pltm %arg6, %c80 {element_alignment_bit_width = -1 : i32} : vector<64xi1>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"({{.*}}) : (!llvm.ptr<6>) -> vector<32xi8>
    // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<64xi32>, vector<32xi8>, ptr<6>)>
    // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
    %res = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb32>} : memref<?xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
    %res_6 = ave.hir.vload <BRC_B32> %arg2[%c0] {element_alignment_bit_width = 32 : i32} : memref<1xf32, #hivm.address_space<ub>> into vector<64xf32>
    %8 = ave.hir.vmul %res_6, %res_6, %6 {element_alignment_bit_width = 32 : i32} : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.preg.xor <b8> %res, %4, %6 {element_alignment_bit_width = 32 : i32} : vector<64xi1>
    %10 = ave.hir.vsel %9, %8, %3 {element_alignment_bit_width = 32 : i32} : vector<64xi1>, vector<64xf32>
    %base_buffer_7, %offset_8, %sizes_9:2, %strides_10:2 = memref.extract_strided_metadata %arg3 : memref<16x80xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %11 = affine.apply affine_map<()[s0, s1] -> (s0 * 80 + s1)>()[%arg5, %arg6]
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [%11], sizes: [%5], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %reinterpret_cast_11[%c0], %6, %10 {element_alignment_bit_width = 32 : i32, hivm.is_continuous} : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
}
return
}

// -----

// Test for getMaxDataTypeWidths function with element_alignment_bit_width attribute
func.func @test_element_alignment_bit_width(%arg0: memref<64xi1, #hivm.address_space<ub>>, %arg1: memref<64xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  
  // Test case where element_alignment_bit_width is explicitly set to 32
  // This should trigger the code path at line 1055-1058
  %0 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32} : vector<64xi1>
  
  // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}, {{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
  %res = ave.hir.vload <NORM> %arg0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #hivm.address_space<ub>> into vector<64xi1>
  
  // Store result to verify the element_alignment_bit_width is correctly propagated
  %1 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32} : vector<64xi1>
  ave.hir.masked_store <NORM_B32> %arg1[%c0], %1, %res {element_alignment_bit_width = 32 : i32} : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  
  return
}
