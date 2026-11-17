// RUN: bishengir-opt %s -convert-hivm-to-std -cse | FileCheck %s --implicit-check-not=hivm.hir.l12bt
// A5 standalone copies: n is elements, padded storage is explicit.
// No allocation or synchronization is supplied by the library wrapper.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @half_to_float
  // CHECK-SAME: %[[SRC:.*]]: memref<32xf16, #hivm.address_space<cbuf>>, %[[DST:.*]]: memref<32xf32, #hivm.address_space<biasbuf>>
  func.func @half_to_float(%src: memref<32xf16, #hivm.address_space<cbuf>>, %dst: memref<32xf32, #hivm.address_space<biasbuf>>) {
    // Match the frontend's integer constant -> index_cast path.
    // CHECK-DAG: %[[N:.*]] = arith.constant 17 : i64
    %value = arith.constant 17 : i32
    %n = arith.index_cast %value : i32 to index
    // CHECK-DAG: %[[LIMIT:.*]] = arith.constant 3968 : i64
    // CHECK-DAG: %[[INVALID:.*]] = arith.constant -1 : i64
    // CHECK-DAG: %[[BASEIDX:.*]] = memref.extract_aligned_pointer_as_index %[[DST]]
    // CHECK-DAG: %[[BASE:.*]] = arith.index_castui %[[BASEIDX]] : index to i64
    // CHECK-DAG: %[[VALID:.*]] = arith.cmpi ule, %[[BASE]], %[[LIMIT]] : i64
    // CHECK-DAG: %[[ADDR:.*]] = arith.select %[[VALID]], %[[BASE]], %[[INVALID]] : i64
    // CHECK-DAG: %[[S:.*]] = memref.cast %[[SRC]]
    // CHECK: call @l12bt_a5_half_to_float(%[[S]], %[[ADDR]], %[[N]]) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64) -> ()
    hivm.hir.l12bt ins(%src : memref<32xf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<32xf32, #hivm.address_space<biasbuf>>) n = %n
    return
  }

  // Singleton parent dimensions may retain their strides. Copy a contiguous
  // row with src offset 16 (32 bytes), dst offset 16 (64 bytes).
  // CHECK-LABEL: func.func @bf16_parent_views
  func.func @bf16_parent_views(%src: memref<2x64xbf16, #hivm.address_space<cbuf>>, %dst: memref<1x32xf32, strided<[128, 1], offset: 16>, #hivm.address_space<biasbuf>>) {
    // Keep generated constants and view checks in one DAG group: CSE may
    // place constants before the view.
    // CHECK-DAG: %[[OFF:.*]] = arith.constant 64 : i64
    // CHECK-DAG: %[[LIM:.*]] = arith.constant 3904 : i64
    // CHECK-DAG: %[[NEG:.*]] = arith.constant -1 : i64
    // CHECK-DAG: %[[COUNT:.*]] = arith.constant 17 : i64
    // CHECK-DAG: %[[VIEW:.*]] = memref.subview
    %s = memref.subview %src[0, 16] [1, 32] [1, 1] : memref<2x64xbf16, #hivm.address_space<cbuf>> to memref<1x32xbf16, strided<[64, 1], offset: 16>, #hivm.address_space<cbuf>>
    %n = arith.constant 17 : index
    // CHECK-DAG: %[[COLLAPSED:.*]] = memref.collapse_shape %[[VIEW]] {{.*}} into memref<32xbf16, strided<[1], offset: 16>, #hivm.address_space<cbuf>>
    // CHECK-DAG: %[[DBASEIDX:.*]] = memref.extract_aligned_pointer_as_index
    // CHECK-DAG: %[[DBASE:.*]] = arith.index_castui %[[DBASEIDX]] : index to i64
    // CHECK-DAG: %[[OK:.*]] = arith.cmpi ule, %[[DBASE]], %[[LIM]] : i64
    // CHECK-DAG: %[[WITHOFF:.*]] = arith.addi %[[DBASE]], %[[OFF]] : i64
    // CHECK-DAG: %[[BT:.*]] = arith.select %[[OK]], %[[WITHOFF]], %[[NEG]] : i64
    // CHECK-DAG: %[[FLAT:.*]] = memref.cast %[[COLLAPSED]]
    // CHECK: call @l12bt_a5_bfloat16_t_to_float(%[[FLAT]], %[[BT]], %[[COUNT]])
    hivm.hir.l12bt ins(%s : memref<1x32xbf16, strided<[64, 1], offset: 16>, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1x32xf32, strided<[128, 1], offset: 16>, #hivm.address_space<biasbuf>>) n = %n
    return
  }

  // CHECK-LABEL: func.func @float_minimum
  func.func @float_minimum(%src: memref<1x1x1x16xf32, #hivm.address_space<cbuf>>, %dst: memref<16xf32, #hivm.address_space<biasbuf>>) {
    // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i64
    %n = arith.constant 1 : index
    // CHECK: call @l12bt_a5_float_to_float({{.*}}, {{.*}}, %[[ONE]])
    hivm.hir.l12bt ins(%src : memref<1x1x1x16xf32, #hivm.address_space<cbuf>>)
      outs(%dst : memref<16xf32, #hivm.address_space<biasbuf>>) n = %n
    return
  }

  // B32 n=8 still needs 16 elements due to even blockLen.
  // CHECK-LABEL: func.func @int32_even_burst
  func.func @int32_even_burst(%src: memref<1x1x16xi32, strided<[128, 64, 1]>, #hivm.address_space<cbuf>>, %dst: memref<1x1x1x16xi32, #hivm.address_space<biasbuf>>) {
    %n = arith.constant 8 : index
    // CHECK: call @l12bt_a5_int32_t_to_int32_t
    hivm.hir.l12bt ins(%src : memref<1x1x16xi32, strided<[128, 64, 1]>, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1x1x1x16xi32, #hivm.address_space<biasbuf>>) n = %n
    return
  }

  // CHECK-LABEL: func.func @maximum
  func.func @maximum(%src: memref<1024xf32, #hivm.address_space<cbuf>>, %dst: memref<1024xf32, #hivm.address_space<biasbuf>>) {
    %n = arith.constant 1024 : index
    // CHECK: call @l12bt_a5_float_to_float
    hivm.hir.l12bt ins(%src : memref<1024xf32, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1024xf32, #hivm.address_space<biasbuf>>) n = %n
    return
  }

  // Tail exactly fills the BT allocation after rounding.
  // CHECK-LABEL: func.func @maximum_tail
  func.func @maximum_tail(%src: memref<1024xf16, #hivm.address_space<cbuf>>, %dst: memref<1024xf32, #hivm.address_space<biasbuf>>) {
    %n = arith.constant 1009 : index
    // CHECK: call @l12bt_a5_half_to_float
    hivm.hir.l12bt ins(%src : memref<1024xf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1024xf32, #hivm.address_space<biasbuf>>) n = %n
    return
  }

  // Statically legal offsets at the ends of both memory spaces.
  // CHECK-LABEL: func.func @end_offsets
  func.func @end_offsets(%src: memref<16xf32, strided<[1], offset: 131056>, #hivm.address_space<cbuf>>, %dst: memref<16xf32, strided<[1], offset: 1008>, #hivm.address_space<biasbuf>>) {
    %n = arith.constant 16 : index
    // CHECK: call @l12bt_a5_float_to_float
    hivm.hir.l12bt ins(%src : memref<16xf32, strided<[1], offset: 131056>, #hivm.address_space<cbuf>>)
      outs(%dst : memref<16xf32, strided<[1], offset: 1008>, #hivm.address_space<biasbuf>>) n = %n
    return
  }
}
