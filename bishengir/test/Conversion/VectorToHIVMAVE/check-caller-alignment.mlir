// RUN: bishengir-opt %s -convert-vector-to-hivmave -allow-unregistered-dialect | FileCheck %s
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @callee_1x6_unaligned
  func.func @callee_1x6_unaligned(%arg0: memref<1x6xf32, #hivm.address_space<ub>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [6] : vector<64xi1>
    %subview = memref.subview %arg0[0, 0] [1, 6] [1, 1] : memref<1x6xf32, #hivm.address_space<ub>> to memref<6xf32, strided<[1]>, #hivm.address_space<ub>>
    // CHECK: ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access}
    %1 = vector.transfer_read %subview[%c0], %cst, %0 {in_bounds = [true]} : memref<6xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xf32>
    "some.use"(%1) : (vector<64xf32>) -> ()
    return
  }
  func.func @caller_1x6_unaligned(%arg0: memref<16x6xf32, #hivm.address_space<ub>>, %arg1: index) {
    %subview = memref.subview %arg0[%arg1, 0] [1, 6] [1, 1] : memref<16x6xf32, #hivm.address_space<ub>> to memref<1x6xf32, strided<[6, 1], offset: ?>, #hivm.address_space<ub>>
    %cast = memref.cast %subview {fold_offset_into_ptr} : memref<1x6xf32, strided<[6, 1], offset: ?>, #hivm.address_space<ub>> to memref<1x6xf32, #hivm.address_space<ub>>
    call @callee_1x6_unaligned(%cast) : (memref<1x6xf32, #hivm.address_space<ub>>) -> ()
    return
  }

  // The total size is 99 blocks, but the 99-element outer stride is not
  // 32-byte aligned.
  // CHECK-LABEL: func.func @callee_8x99_unaligned
  func.func @callee_8x99_unaligned(%arg0: memref<8x99xf32, #hivm.address_space<ub>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg0[0, 0] [1, 64] [1, 1] : memref<8x99xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    // CHECK: ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access}
    %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xf32>
    "some.use"(%0) : (vector<64xf32>) -> ()
    return
  }

  func.func @caller_8x99_unaligned(%arg0: memref<16x99xf32, #hivm.address_space<ub>>, %arg1: index) {
    %subview = memref.subview %arg0[%arg1, 0] [8, 99] [1, 1] : memref<16x99xf32, #hivm.address_space<ub>> to memref<8x99xf32, strided<[99, 1], offset: ?>, #hivm.address_space<ub>>
    %cast = memref.cast %subview {fold_offset_into_ptr} : memref<8x99xf32, strided<[99, 1], offset: ?>, #hivm.address_space<ub>> to memref<8x99xf32, #hivm.address_space<ub>>
    call @callee_8x99_unaligned(%cast) : (memref<8x99xf32, #hivm.address_space<ub>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @callee_rank1_unaligned
  func.func @callee_rank1_unaligned(%arg0: memref<6xf32, #hivm.address_space<ub>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    // CHECK: ave.hir.vload <NORM> %arg0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access}
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<6xf32, #hivm.address_space<ub>>, vector<64xf32>
    "some.use"(%0) : (vector<64xf32>) -> ()
    return
  }

  func.func @caller_rank1_unaligned(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: index) {
    %subview = memref.subview %arg0[%arg1] [6] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<6xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %cast = memref.cast %subview {fold_offset_into_ptr} : memref<6xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<6xf32, #hivm.address_space<ub>>
    call @callee_rank1_unaligned(%cast) : (memref<6xf32, #hivm.address_space<ub>>) -> ()
    return
  }
}
