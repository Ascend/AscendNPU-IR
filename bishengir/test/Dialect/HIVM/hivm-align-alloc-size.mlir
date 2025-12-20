// RUN: bishengir-opt -hivm-align-alloc-size -split-input-file %s | FileCheck %s


// CHECK-LABEL: func @test_static_transpose
func.func @test_static_transpose() {
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16x16x16xf16
// CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC]][0, 0, 0] [2, 16, 15] [1, 1, 1]
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16x16x16xf16
// CHECK: %[[DST_SUBVIEW:.*]] = memref.subview %[[DST]][0, 0, 0] [15, 16, 2] [1, 1, 1]
// CHECK: hivm.hir.vtranspose ins(%[[SRC_SUBVIEW]]
// CHECK-SAME: %[[DST_SUBVIEW]]
  %src = memref.alloc() : memref<2x16x15xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<15x16x2xf16, #hivm.address_space<ub>>
  hivm.hir.vtranspose ins(%src : memref<2x16x15xf16, #hivm.address_space<ub>>)
                      outs(%dst : memref<15x16x2xf16, #hivm.address_space<ub>>)
                      permutation = [2, 1, 0]
  return
}

// -----

// CHECK: func @test_dyn_transpose(%[[INDEX0:.*]]: index, %[[INDEX1:.*]]: index)
func.func @test_dyn_transpose(%index0: index, %index1: index) {
// CHECK: %[[SRC:.*]] = memref.alloc(
// CHECK-SAME: memref<?x16x?xf16
// CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC]][0, 0, 0] [%[[INDEX0]], 16, %[[INDEX1]]] [1, 1, 1]
// CHECK: %[[DST:.*]] = memref.alloc(
// CHECK-SAME: memref<?x16x?xf16
// CHECK: %[[DST_SUBVIEW:.*]] = memref.subview %[[DST]][0, 0, 0] [%[[INDEX0]], 16, %[[INDEX1]]] [1, 1, 1]
// CHECK: hivm.hir.vtranspose ins(%[[SRC_SUBVIEW]]
// CHECK-SAME: %[[DST_SUBVIEW]]
  %src = memref.alloc(%index0, %index1) : memref<?x16x?xf16, #hivm.address_space<ub>>
  %dst = memref.alloc(%index0, %index1) : memref<?x16x?xf16, #hivm.address_space<ub>>
  hivm.hir.vtranspose ins(%src : memref<?x16x?xf16, #hivm.address_space<ub>>)
                      outs(%dst : memref<?x16x?xf16, #hivm.address_space<ub>>)
                      permutation = [2, 1, 0]
  return
}

// -----

func.func @test_cast_s322s8_2d() {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<32x16xi32, #hivm.address_space<ub>>
  // CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC:.*]][0, 0] [2, 16] [1, 1]
  // CHECK: %[[TMP:.*]] = memref.alloc() : memref<32x32xi8, #hivm.address_space<ub>>
  // CHECK: %[[TMP_SUBVIEW:.*]] = memref.subview %[[TMP:.*]][0, 0] [2, 16] [1, 1]
  // CHECK: hivm.hir.vcast ins(%[[SRC_SUBVIEW:.*]] : memref<2x16xi32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%[[TMP_SUBVIEW:.*]] : memref<2x16xi8, strided<[32, 1]>, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
  %s32 = memref.alloc() : memref<2x16xi32, #hivm.address_space<ub>>
  %s8 = memref.alloc() : memref<2x16xi8, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32, #hivm.address_space<ub>>)
                 outs(%s8 : memref<2x16xi8, #hivm.address_space<ub>>)
                 round_mode = #hivm.round_mode<truncwithoverflow>
  return
}

// -----

func.func @test_cast_s322s8_1d() {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<256xi32, #hivm.address_space<ub>>
  // CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC:.*]][0] [2] [1]
  // CHECK: %[[TMP:.*]] = memref.alloc() : memref<1024xi8, #hivm.address_space<ub>>
  // CHECK: %[[TMP_SUBVIEW:.*]] = memref.subview %[[TMP:.*]][0] [2] [1]
  // CHECK: hivm.hir.vcast ins(%[[SRC_SUBVIEW:.*]] : memref<2xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%[[TMP_SUBVIEW:.*]] : memref<2xi8, strided<[1]>, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
  %s32 = memref.alloc() : memref<2xi32, #hivm.address_space<ub>>
  %s8 = memref.alloc() : memref<2xi8, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%s32 : memref<2xi32, #hivm.address_space<ub>>)
                 outs(%s8 : memref<2xi8, #hivm.address_space<ub>>)
                 round_mode = #hivm.round_mode<truncwithoverflow>
  return
}

// -----

func.func @test_cast_s162s8_2d() {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<32x16xi16, #hivm.address_space<ub>>
  // CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview  %[[SRC:.*]][0, 0] [2, 16] [1, 1]
  // CHECK: %[[TMP:.*]] = memref.alloc() : memref<32x32xi8, #hivm.address_space<ub>>
  // CHECK: %[[TMP_SUBVIEW:.*]] = memref.subview %[[TMP:.*]][0, 0] [2, 16] [1, 1]
  // CHECK: hivm.hir.vcast ins(%[[SRC_SUBVIEW:.*]] : memref<2x16xi16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%[[TMP_SUBVIEW:.*]] : memref<2x16xi8, strided<[32, 1]>, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
  %s16 = memref.alloc() : memref<2x16xi16, #hivm.address_space<ub>>
  %s8 = memref.alloc() : memref<2x16xi8, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16, #hivm.address_space<ub>>)
                 outs(%s8 : memref<2x16xi8, #hivm.address_space<ub>>)
                 round_mode = #hivm.round_mode<truncwithoverflow>
  return
}

// -----

func.func @test_cast_s162s8_1d() {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<512xi16, #hivm.address_space<ub>>
  // CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC:.*]][0] [2] [1]
  // CHECK: %[[TMP:.*]] = memref.alloc() : memref<1024xi8, #hivm.address_space<ub>>
  // CHECK: %[[TMP_SUBVIEW:.*]] = memref.subview %[[TMP:.*]][0] [2] [1]
  // CHECK: hivm.hir.vcast ins(%[[SRC_SUBVIEW:.*]] : memref<2xi16, strided<[1]>, #hivm.address_space<ub>>) outs(%[[TMP_SUBVIEW:.*]] : memref<2xi8, strided<[1]>, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
  %s16 = memref.alloc() : memref<2xi16, #hivm.address_space<ub>>
  %s8 = memref.alloc() : memref<2xi8, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%s16 : memref<2xi16, #hivm.address_space<ub>>)
                 outs(%s8 : memref<2xi8, #hivm.address_space<ub>>)
                 round_mode = #hivm.round_mode<truncwithoverflow>
  return
}

// -----

func.func @test_sort_float() {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  // CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC:.*]][0] [8] [1]
  // CHECK: %[[DST_VALUE:.*]] = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  // CHECK: %[[DST_VALUE_SUBVIEW:.*]] = memref.subview %[[DST_VALUE:.*]][0] [8] [1]
  // CHECK: %[[DST_INDEX:.*]] = memref.alloc() : memref<32xi32, #hivm.address_space<ub>>
  // CHECK: %[[DST_INDEX_SUBVIEW:.*]] = memref.subview %[[DST_INDEX:.*]][0] [8] [1]
  %src = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %dst_value = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %dst_index = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
  hivm.hir.vsort ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                 outs(%dst_value: memref<8xf32, #hivm.address_space<ub>>)
                 descending = false
                 sort_axis = 0
  hivm.hir.vsort ins(%src : memref<8xf32, #hivm.address_space<ub>>)
                 outs(%dst_value, %dst_index : memref<8xf32, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>)
                 descending = false
                 sort_axis = 0
  return
}

// -----

func.func @test_sort_half() {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<32xf16, #hivm.address_space<ub>>
  // CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC:.*]][0] [8] [1]
  // CHECK: %[[DST_VALUE:.*]] = memref.alloc() : memref<32xf16, #hivm.address_space<ub>>
  // CHECK: %[[DST_VALUE_SUBVIEW:.*]] = memref.subview %[[DST_VALUE:.*]][0] [8] [1]
  // CHECK: %[[DST_INDEX:.*]] = memref.alloc() : memref<32xi32, #hivm.address_space<ub>>
  // CHECK: %[[DST_INDEX_SUBVIEW:.*]] = memref.subview %[[DST_INDEX:.*]][0] [8] [1]
  // CHECK: hivm.hir.vsort ins(%[[SRC_SUBVIEW:.*]] : memref<8xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%[[DST_VALUE_SUBVIEW:.*]] : memref<8xf16, strided<[1]>, #hivm.address_space<ub>>) descending = false sort_axis = 0
  // CHECK: hivm.hir.vsort ins(%[[SRC_SUBVIEW:.*]] : memref<8xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%[[DST_VALUE_SUBVIEW:.*]], %[[DST_INDEX_SUBVIEW:.*]] : memref<8xf16, strided<[1]>, #hivm.address_space<ub>>, memref<8xi32, strided<[1]>, #hivm.address_space<ub>>) descending = false sort_axis = 0
  %src = memref.alloc() : memref<8xf16, #hivm.address_space<ub>>
  %dst_value = memref.alloc() : memref<8xf16, #hivm.address_space<ub>>
  %dst_index = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
  hivm.hir.vsort ins(%src : memref<8xf16, #hivm.address_space<ub>>)
                 outs(%dst_value: memref<8xf16, #hivm.address_space<ub>>)
                 descending = false
                 sort_axis = 0
  hivm.hir.vsort ins(%src : memref<8xf16, #hivm.address_space<ub>>)
                 outs(%dst_value, %dst_index : memref<8xf16, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>)
                 descending = false
                 sort_axis = 0
  return
}

// -----

// CHECK:           %[[VAL_16:.*]] = memref.alloc() : memref<1x1x1x32x32xi8, #hivm.address_space<ub>>
// CHECK:           %[[VAL_17:.*]] = memref.subview %[[VAL_16]][0, 0, 0, 0, 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : memref<1x1x1x32x32xi8, #hivm.address_space<ub>> to  memref<1x1x1x1x32xi8, strided<[1024, 1024, 1024, 32, 1]>, #hivm.address_space<ub>>
// CHECK:           %[[VAL_20:.*]] = scf.for %[[VAL_21:.*]] = %[[VAL_9:.*]] to %[[VAL_10:.*]] step %[[VAL_9:.*]] iter_args(%[[VAL_22:.*]] = %[[VAL_17]]) -> (memref<1x1x1x1x32xi8, strided<[1024, 1024, 1024, 32, 1]>, #hivm.address_space<ub>>)  : i32 {
// CHECK:             %[[VAL_28:.*]] = memref.alloc() : memref<1x1x1x32x32xi8, #hivm.address_space<ub>>
// CHECK:             %[[VAL_29:.*]] = memref.subview %[[VAL_28]][0, 0, 0, 0, 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : memref<1x1x1x32x32xi8, #hivm.address_space<ub>> to  memref<1x1x1x1x32xi8, strided<[1024, 1024, 1024, 32, 1]>, #hivm.address_space<ub>>
// CHECK:             hivm.hir.vcast ins(%[[VAL_27:.*]] : memref<1x1x1x1x32xi32, strided<[1024, 1024, 1024, 32, 1]>, #hivm.address_space<ub>>) outs(%[[VAL_29]] : memref<1x1x1x1x32xi8, strided<[1024, 1024, 1024, 32, 1]>, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
// CHECK:             scf.yield %[[VAL_29]] : memref<1x1x1x1x32xi8, strided<[1024, 1024, 1024, 32, 1]>, #hivm.address_space<ub>>
// CHECK:           }
func.func @triton_yield_iter_args(%arg0: i64, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>>, %arg3: memref<?xi8, #hivm.address_space<gm>>, %arg4: memref<?xi8, #hivm.address_space<gm>>, %arg5: memref<?xi8, #hivm.address_space<gm>>, %arg6: i32,  %arg7: i32,  %arg8: i32){
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 1, 1, 1, 32], strides: [32, 32, 32, 32, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1x1x1x32xi8, strided<[32, 32, 32, 32, 1]>, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast : memref<1x1x1x1x32xi8, strided<[32, 32, 32, 32, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 1, 1, 1, 32], strides: [32, 32, 32, 32, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1x1x1x32xi8, strided<[32, 32, 32, 32, 1]>, #hivm.address_space<gm>>
  %alloc_1 = memref.alloc() : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<1x1x1x1x32xi8, strided<[32, 32, 32, 32, 1]>, #hivm.address_space<gm>>) outs(%alloc_1 : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x32xf16, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%alloc : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>) outs(%alloc_2 : memref<1x1x1x1x32xf16, #hivm.address_space<ub>>)
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%alloc_2 : memref<1x1x1x1x32xf16, #hivm.address_space<ub>>) outs(%alloc_3 : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>)
  %2 = scf.for %arg9 = %c1_i32 to %c3_i32 step %c1_i32 iter_args(%arg10 = %alloc_1) -> (memref<1x1x1x1x32xi8, #hivm.address_space<ub>>)  : i32 {
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x32xf16, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%arg10 : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>) outs(%alloc_5 : memref<1x1x1x1x32xf16, #hivm.address_space<ub>>)
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_5 : memref<1x1x1x1x32xf16, #hivm.address_space<ub>>) outs(%alloc_6 : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>)
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc_3, %alloc_6 : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>, memref<1x1x1x1x32xf32, #hivm.address_space<ub>>) outs(%alloc_7 : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>)
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x32xi32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_7 : memref<1x1x1x1x32xf32, #hivm.address_space<ub>>) outs(%alloc_8 : memref<1x1x1x1x32xi32, #hivm.address_space<ub>>) round_mode = <trunc>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_8 : memref<1x1x1x1x32xi32, #hivm.address_space<ub>>) outs(%alloc_9 : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
    scf.yield %alloc_9 : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>
  }
  %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1, 1, 1, 32], strides: [32, 32, 32, 32, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1x1x1x32xi8, strided<[32, 32, 32, 32, 1]>, #hivm.address_space<gm>>
  hivm.hir.store ins(%2 : memref<1x1x1x1x32xi8, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<1x1x1x1x32xi8, strided<[32, 32, 32, 32, 1]>, #hivm.address_space<gm>>)
  return
}