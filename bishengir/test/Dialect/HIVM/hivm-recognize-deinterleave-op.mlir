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
    // CHECK: %[[alloc:.*]] = memref.alloc() : memref<4x64xf16
    // CHECK: %[[subview_0:.*]] = memref.subview %[[alloc]]{{\[}}0, 0] {{\[}}{{.*}}, {{.*}}] {{\[}}1, 1] : memref<4x64xf16,{{.*}}> to memref<?x?xf16, strided<[64, 1]>,{{.*}}>
    // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<4x64xf16
    // CHECK: %[[subview_1:.*]] = memref.subview %[[alloc_1]]{{\[}}0, 0] {{\[}}{{.*}}, {{.*}}] {{\[}}1, 1] : memref<4x64xf16,{{.*}}> to memref<?x?xf16, strided<[64, 1]>,{{.*}}>
    // CHECK: annotation.mark %[[subview_1]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
    // CHECK: hivm.hir.load ins({{.*}} : memref<?x?xf16, strided<[8320, 130], offset: ?>,{{.*}}>) outs(%[[subview_1]] : memref<?x?xf16, strided<[64, 1]>,{{.*}}>)
    // CHECK: hivm.hir.vdeinterleave ins(%[[subview_1]] : memref<?x?xf16, strided<[64, 1]>,{{.*}}>) outs(%[[subview_0]] : memref<?x?xf16, strided<[64, 1]>,{{.*}}>) channel_num = 16 index_mode = <CHANNEL_0>
    %alloc = memref.alloc() : memref<4x64xf16, #hivm.address_space<ub>>
    annotation.mark %alloc : memref<4x64xf16, #hivm.address_space<ub>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%16, %17] [1, 1] : memref<4x64xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %alloc[0, 0] [%16, %17] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%subview : memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf16, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
    return
  }
}

// -----

// CHECK-LABEL: recognize_deinterleave_for_copy_0
#map = affine_map<()[s0] -> (s0 * 8320 + 129)>
#map1 = affine_map<()[s0] -> (s0 + 4)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
module {
  func.func @recognize_deinterleave_for_copy_0(%arg0: i64, %arg1: memref<?xf16, #hivm.address_space<gm>>, %arg2: memref<?xf16, #hivm.address_space<gm>>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) 
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
    // CHECK: %[[alloc:.*]] = memref.alloc() : memref<4x64xf16
    // CHECK: %[[subview_0:.*]] = memref.subview %[[alloc]]{{\[}}0, 0] {{\[}}{{.*}}, {{.*}}] {{\[}}1, 1] : memref<4x64xf16,{{.*}}> to memref<?x1xf16, strided<[64, 1]>,{{.*}}>
    // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<4x64xf16
    // CHECK: %[[subview_1:.*]] = memref.subview %[[alloc_1]]{{\[}}0, 0] {{\[}}{{.*}}, {{.*}}] {{\[}}1, 1] : memref<4x64xf16,{{.*}}> to memref<?x1xf16, strided<[64, 1]>,{{.*}}>
    // CHECK: annotation.mark %[[subview_1]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
    // CHECK: hivm.hir.copy ins(%{{.*}} : memref<?x?xf16, strided<[8320, 130], offset: ?>, {{.*}}>) outs({{.*}} : memref<?x1xf16, strided<[64, 1]>,{{.*}}>)
    // CHECK: hivm.hir.vdeinterleave ins(%{{.*}} : memref<?x1xf16, strided<[64, 1]>, {{.*}}>) outs({{.*}} : memref<?x1xf16, strided<[64, 1]>,{{.*}}>) channel_num = 16 index_mode = <CHANNEL_0>
    %alloc = memref.alloc() : memref<4x64xf16, #hivm.address_space<ub>>
    annotation.mark %alloc : memref<4x64xf16, #hivm.address_space<ub>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%16, %17] [1, 1] : memref<4x64xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %alloc[0, 0] [%16, 1] [1, 1] : memref<4x64xf16, #hivm.address_space<ub>> to memref<?x1xf16, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%subview : memref<?x?xf16, strided<[8320, 130], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x1xf16, strided<[64, 1]>, #hivm.address_space<ub>>)
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
func.func @recognize_deinterleave_for_i8_tensor(%arg0: index, %arg1: memref<128xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>) {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<128xi8, #hivm.address_space<ub>>
  annotation.mark %alloc : memref<128xi8, #hivm.address_space<ub>>
  %subview = memref.subview %arg1[0] [%arg0] [1] : memref<128xi8, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>
  %subview_0 = memref.subview %alloc[0] [%arg0] [1] : memref<128xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%subview : memref<?xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i8 : i8 left_padding_num = %c0 : index init_out_buffer = false
  return
}
// CHECK:   hivm.hir.load ins(%subview : memref<?xi8, strided<[2], offset: ?>, #hivm.address_space<gm>>) outs(%[[VAL_8:.+]] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>)
// CHECK:   hivm.hir.vdeinterleave ins(%[[VAL_8]] : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_0 : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) channel_num = 32 index_mode = <CHANNEL_0>
