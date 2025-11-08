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

