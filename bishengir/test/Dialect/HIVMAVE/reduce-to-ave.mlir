// RUN: bishengir-opt -convert-hivmave-to-ave-intrin %s | FileCheck %s
#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func.func @reduction_max
func.func @reduction_max(%arg0: memref<16xi16, #hivm.address_space<ub>>, %arg1: memref<16x16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 16 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c-32768_i16 = arith.constant -32768 : i16
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  %1 = ave.hir.broadcast %c-32768_i16, %0 : i16, vector<128xi1> -> vector<128xi16>
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %4 = ave.hir.pge <VL16> : vector<128xi1>
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<16x16xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index, index, index
    %5 = affine.apply #map()[%arg3]
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%5], sizes: [16], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<16xi8, #map1, #hivm.address_space<ub>>
    %res_0 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<16xi8, #map1, #hivm.address_space<ub>> into vector<128xi8>
    %6 = ave.hir.pge <ALL> : vector<128xi1>
    %7 = ave.hir.vextsi %res_0, %6 {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi16>, vector<128xi1>
    %8 = ave.hir.vsel %4, %7, %1 : vector<128xi1>, vector<128xi16>
    %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %arg0 : memref<16xi16, #hivm.address_space<ub>> -> memref<i16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%arg3], sizes: [], strides: [] : memref<i16, #hivm.address_space<ub>> to memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
    %9 = ave.hir.pge <ALL> : vector<128xi1>
    %10 = ave.hir.reduction <max>, %8, %9 : vector<128xi16>, vector<128xi1> -> vector<128xi16>
    %11 = builtin.unrealized_conversion_cast %10 : vector<128xi16> to vector<1xi16>
    %12 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B16> %reinterpret_cast_5[], %12, %11 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi16>
  } {element_alignment_bit_width = 16 : i32}
  %2 = ave.hir.pge <VL16> {mask_op_idx = 1 : i32} : vector<128xi1>
  %res = ave.hir.vload <NORM> %arg0[%c0] : memref<16xi16, #hivm.address_space<ub>> into vector<128xi16>
  %3 = ave.hir.vtrunci %res, false, %2 {part = #ave.vcvt_part_type<part_even>} : vector<128xi16>, vector<128xi8>, vector<128xi1>
  ave.hir.masked_store <NORM_B8> %arg2[%c0], %2, %3 : memref<16xi8, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi8>
  return
}

// CHECK-NOT: ave.hir.reduction <max>
// CHECK: "hivm_regbaseintrins.intr.hivm.vcmax.s.x"


// CHECK-LABEL: func.func @reduction_umax
func.func @reduction_umax(%arg0: memref<16xi16, #hivm.address_space<ub>>, %arg1: memref<16x16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 16 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i16 = arith.constant 0 : i16
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  %1 = ave.hir.broadcast %c0_i16, %0 : i16, vector<128xi1> -> vector<128xi16>
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %4 = ave.hir.pge <VL16> : vector<128xi1>
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<16x16xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index, index, index
    %5 = affine.apply #map()[%arg3]
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%5], sizes: [16], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<16xi8, #map1, #hivm.address_space<ub>>
    %res_0 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<16xi8, #map1, #hivm.address_space<ub>> into vector<128xi8>
    %6 = ave.hir.pge <ALL> : vector<128xi1>
    %7 = ave.hir.vextui %res_0, %6 {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi16>, vector<128xi1>
    %8 = ave.hir.vsel %4, %7, %1 : vector<128xi1>, vector<128xi16>
    %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %arg0 : memref<16xi16, #hivm.address_space<ub>> -> memref<i16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%arg3], sizes: [], strides: [] : memref<i16, #hivm.address_space<ub>> to memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
    %9 = ave.hir.pge <ALL> : vector<128xi1>
    %10 = ave.hir.reduction <umax>, %8, %9 : vector<128xi16>, vector<128xi1> -> vector<128xi16>
    %11 = builtin.unrealized_conversion_cast %10 : vector<128xi16> to vector<1xi16>
    %12 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B16> %reinterpret_cast_5[], %12, %11 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi16>
  } {element_alignment_bit_width = 16 : i32}
  %2 = ave.hir.pge <VL16> {mask_op_idx = 1 : i32} : vector<128xi1>
  %res = ave.hir.vload <NORM> %arg0[%c0] : memref<16xi16, #hivm.address_space<ub>> into vector<128xi16>
  %3 = ave.hir.vtrunci %res, false, %2 {part = #ave.vcvt_part_type<part_even>} : vector<128xi16>, vector<128xi8>, vector<128xi1>
  ave.hir.masked_store <NORM_B8> %arg2[%c0], %2, %3 : memref<16xi8, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi8>
  return
}

// CHECK-NOT: ave.hir.reduction <umax>
// CHECK: "hivm_regbaseintrins.intr.hivm.vcmax.u.x"
