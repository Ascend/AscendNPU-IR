// RUN: bishengir-opt -hivm-recognize-deinterleave-op -split-input-file %s | FileCheck %s

// CHECK-LABEL: recognize_deinterleave_for_load_0
#map = affine_map<()[s0] -> (s0 * 8320 + 129)>
#map1 = affine_map<()[s0] -> (s0 + 4)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
module {
  func.func @recognize_deinterleave_for_load_0(%arg0: i64, %arg1: memref<?xf16, #hivm.address_space<gm>>, %arg2: memref<?xf16, #hivm.address_space<gm>>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) 
  attributes {func_dyn_memref_args = dense<[false, true, true, true, false, false, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg8, %arg7 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg6 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = affine.apply #map()[%6]
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [4, 64], strides: [8320, 130] : memref<?xf16, #hivm.address_space<gm>> to memref<4x64xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg5 : i32 to index
    %9 = arith.maxsi %8, %c0 : index
    %10 = arith.minsi %9, %c64 : index
    %11 = affine.apply #map1()[%6]
    %12 = arith.index_cast %arg4 : i32 to index
    %13 = arith.maxsi %6, %12 : index
    %14 = arith.minsi %11, %13 : index
    %15 = affine.apply #map2()[%14, %6]
    %16 = arith.minsi %15, %c4 : index
    %17 = arith.minsi %10, %c64 : index
    // CHECK: %[[VAL_33:.*]] = memref.alloc() : memref<4x64xf16, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[VAL_33]] : memref<4x64xf16, #hivm.address_space<ub>>
    // CHECK: %[[VAL_34:.*]] = memref.subview %[[VAL_22:.*]][0, 0] [%[[VAL_31:.*]], %[[VAL_32:.*]]] [1, 1] : memref<4x64xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>
    // CHECK: %[[VAL_35:.*]] = memref.alloc() : memref<4x64xf16, #hivm.address_space<ub>>
    // CHECK: %[[VAL_36:.*]] = memref.subview %[[VAL_33]][0, 0] [%[[VAL_31]], %[[VAL_32]]] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    // CHECK: %[[VAL_37:.*]] = memref.subview %[[VAL_35]][0, 0] [%[[VAL_31]], %[[VAL_32]]] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[VAL_37]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load ins(%[[VAL_34]] : memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>) outs(%[[VAL_37]] : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %[[VAL_13:.*]] : f16 left_padding_num = %[[VAL_9:.*]] : index
    // CHECK: hivm.hir.vdeinterleave ins(%[[VAL_37]] : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%[[VAL_36]] : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) channel_num = 16 index_mode = <CHANNEL_0>
    %alloc = memref.alloc() : memref<4x64xf16, #hivm.address_space<ub>>
    annotation.mark %alloc : memref<4x64xf16, #hivm.address_space<ub>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%16, %17] [1, 1] : memref<4x64xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %alloc[0, 0] [%16, %17] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%subview : memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
    return
  }
}

// -----

// CHECK-LABEL: recognize_deinterleave_for_load_to_alloc
// CHECK: hivm.hir.vdeinterleave
// CHECK: hivm.hir.vdeinterleave
func.func @recognize_deinterleave_for_load_to_alloc(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xf16, #hivm.address_space<gm>>, %arg2: memref<?xf16, #hivm.address_space<gm>>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: i32, %arg5: i32, %arg6: i32) attributes {func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c86 = arith.constant 86 : index
  %c4_i32 = arith.constant 4 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %arg6, %arg5 : i32
  %3 = arith.divsi %1, %2 : i32
  %4 = arith.remsi %3, %arg4 : i32
  hivm.hir.set_mask_norm
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32, #hivm.address_space<ub>>
  annotation.mark %alloc : memref<4x32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<4x32xf32, #hivm.address_space<ub>>)
  %5 = arith.muli %4, %c4_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %7 = affine.apply affine_map<()[s0] -> (s0 * 4160 + 129)>()[%6]
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [4, 32], strides: [4160, 130] : memref<?xf16, #hivm.address_space<gm>> to memref<4x32xf16, strided<[4160, 130], offset: ?>, #hivm.address_space<gm>>
  %alloc_1 = memref.alloc() : memref<4x32xf16, #hivm.address_space<ub>>
  %8 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%6]
  %9 = arith.maxsi %6, %c86 : index
  %10 = arith.minsi %8, %9 : index
  %11 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%10, %6]
  %12 = arith.minsi %11, %c4 : index
  %13 = arith.cmpi slt, %12, %c4 : index
  %subview = memref.subview %reinterpret_cast[0, 0] [%12, 32] [1, 1] : memref<4x32xf16, strided<[4160, 130], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf16, strided<[4160, 130], offset: ?>, #hivm.address_space<gm>>
  %subview_2 = memref.subview %alloc_1[0, 0] [%12, 32] [1, 1] : memref<4x32xf16, #hivm.address_space<ub>> to memref<?x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%subview : memref<?x32xf16, strided<[4160, 130], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x32xf16, strided<[32, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f16 left_padding_num = %c0 : index
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%alloc_1 : memref<4x32xf16, #hivm.address_space<ub>>) outs(%alloc_3 : memref<4x32xf32, #hivm.address_space<ub>>)
  %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [32], sizes: [1, 32], strides: [32, 33] : memref<?xf16, #hivm.address_space<gm>> to memref<1x32xf16, strided<[32, 33], offset: 32>, #hivm.address_space<gm>>
  %alloc_5 = memref.alloc() : memref<1x32xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_4 : memref<1x32xf16, strided<[32, 33], offset: 32>, #hivm.address_space<gm>>) outs(%alloc_5 : memref<1x32xf16, #hivm.address_space<ub>>)
  return
}

// -----
// CHECK-LABEL: func.func @recognize_deinterleave_for_i8_tensor
func.func @recognize_deinterleave_for_i8_tensor(%arg0: index, %arg1: memref<128xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>) {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  // CHECK: %[[VAL_4:.*]] = memref.alloc() : memref<128xi8, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[VAL_4]] : memref<128xi8, #hivm.address_space<ub>>
  // CHECK: %[[VAL_5:.*]] = memref.subview %[[VAL_1:.*]][0] {{\[}}%[[VAL_0:.*]]] [1] : memref<128xi8, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>
  // CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<128xi8, #hivm.address_space<ub>>
  // CHECK: %[[VAL_7:.*]] = memref.subview %[[VAL_4]][0] {{\[}}%[[VAL_0]]] [1] : memref<128xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  // CHECK: %[[VAL_8:.*]] = memref.subview %[[VAL_6]][0] {{\[}}%[[VAL_0]]] [1] : memref<128xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[VAL_8]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  // CHECK:           hivm.hir.load ins(%[[VAL_5]] : memref<?xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>) outs(%[[VAL_8]] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %[[VAL_2:.*]] : i8 left_padding_num = %[[VAL_3:.*]] : index init_out_buffer = false
  // CHECK:           hivm.hir.vdeinterleave ins(%[[VAL_8]] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) outs(%[[VAL_7]] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) channel_num = 32 index_mode = <CHANNEL_0>
  %alloc = memref.alloc() : memref<128xi8, #hivm.address_space<ub>>
  annotation.mark %alloc : memref<128xi8, #hivm.address_space<ub>>
  %subview = memref.subview %arg1[0] [%arg0] [1] : memref<128xi8, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>
  %subview_0 = memref.subview %alloc[0] [%arg0] [1] : memref<128xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%subview : memref<?xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i8 : i8 left_padding_num = %c0 : index init_out_buffer = false
  return
}

// -----
// CHECK-LABEL: func.func @recognize_deinterleave_with_reshape_op
func.func @recognize_deinterleave_with_reshape_op(%arg0: index, %arg1: memref<16x8xi8, strided<[16,2], offset: ?>, #hivm.address_space<gm>>, %arg2: index) {
  // CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<32x16xi8, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[VAL_3]] : memref<32x16xi8, #hivm.address_space<ub>>
  // CHECK: %[[VAL_4:.*]] = memref.alloc() : memref<32x16xi8, #hivm.address_space<ub>>
  // CHECK: %[[VAL_5:.*]] = memref.subview %[[VAL_3]][0, 0] [%[[VAL_0:.*]], 16] [1, 1] : memref<32x16xi8, #hivm.address_space<ub>> to memref<?x16xi8, strided<[16, 1]>, #hivm.address_space<ub>>
  // CHECK: %[[VAL_6:.*]] = memref.collapse_shape %[[VAL_5]] {{\[}}[0, 1]] : memref<?x16xi8, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  // CHECK: %[[VAL_7:.*]] = memref.expand_shape %[[VAL_6]] {{\[}}[0, 1]] output_shape [%[[VAL_2:.*]], 8] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>> into memref<?x8xi8, #hivm.address_space<ub>>
  // CHECK: %[[VAL_8:.*]] = memref.subview %[[VAL_4]][0, 0] [%[[VAL_0]], 16] [1, 1] : memref<32x16xi8, #hivm.address_space<ub>> to memref<?x16xi8, strided<[16, 1]>, #hivm.address_space<ub>>
  // CHECK: %[[VAL_9:.*]] = memref.collapse_shape %[[VAL_8]] {{\[}}[0, 1]] : memref<?x16xi8, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  // CHECK: %[[VAL_10:.*]] = memref.expand_shape %[[VAL_9]] {{\[}}[0, 1]] output_shape [%[[VAL_2]], 8] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>> into memref<?x8xi8, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[VAL_10]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x8xi8, #hivm.address_space<ub>>
  // CHECK: hivm.hir.load ins(%[[VAL_1:.*]] : memref<16x8xi8, strided<[16, 2], offset: ?>, #hivm.address_space<gm>>) outs(%[[VAL_10]] : memref<?x8xi8, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.vdeinterleave ins(%[[VAL_10]] : memref<?x8xi8, #hivm.address_space<ub>>) outs(%[[VAL_7]] : memref<?x8xi8, #hivm.address_space<ub>>) channel_num = 32 index_mode = <CHANNEL_0>
  %alloc = memref.alloc() : memref<32x16xi8, #hivm.address_space<ub>>
  annotation.mark %alloc : memref<32x16xi8, #hivm.address_space<ub>>
  %subview = memref.subview %alloc[0,0] [%arg0,16] [1,1] : memref<32x16xi8, #hivm.address_space<ub>> to memref<?x16xi8, strided<[16, 1]>, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<?x16xi8, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [%arg2, 8] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>> into memref<?x8xi8, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg1 : memref<16x8xi8, strided<[16,2], offset: ?>, #hivm.address_space<gm>>) outs(%expand_shape : memref<?x8xi8, #hivm.address_space<ub>>)
  return
}