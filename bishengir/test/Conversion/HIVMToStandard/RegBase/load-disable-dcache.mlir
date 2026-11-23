// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-mark-disable-load -convert-hivm-to-std -split-input-file | FileCheck %s

// Check cached/uncached symbol selection for direct calls and rank reduction.

// CHECK-LABEL: func.func @load_rank1
// CHECK: call @load_gm_to_ubuf_1d_int8_t_nocache(
// CHECK: call @load_gm_to_ubuf_1d_int8_t(
// CHECK-NOT: call @load_gm_to_ubuf_1d_int8_t_nocache
func.func @load_rank1(%rw: memref<2xi8, #hivm.address_space<gm>>, %ro: memref<2xi8, #hivm.address_space<gm>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i8
  %ub = memref.alloc() : memref<2xi8, strided<[2]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%rw : memref<2xi8, #hivm.address_space<gm>>) outs(%ub : memref<2xi8, strided<[2]>, #hivm.address_space<ub>>)
  memref.store %one, %rw[%c0] : memref<2xi8, #hivm.address_space<gm>>
  hivm.hir.load ins(%ro : memref<2xi8, #hivm.address_space<gm>>) outs(%ub : memref<2xi8, strided<[2]>, #hivm.address_space<ub>>)
  return
}

// -----

// Rank > 3 takes the separate rank-reduction lowering path.
// CHECK-LABEL: func.func @load_rank4
// CHECK: call @load_gm_to_ubuf_3d_int8_t_nocache(
// CHECK: call @load_gm_to_ubuf_3d_int8_t(
// CHECK-NOT: call @load_gm_to_ubuf_3d_int8_t_nocache
func.func @load_rank4(%rw: memref<2x2x2x2xi8, #hivm.address_space<gm>>, %ro: memref<2x2x2x2xi8, #hivm.address_space<gm>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i8
  %ub = memref.alloc() : memref<2x2x2x2xi8, strided<[16, 8, 4, 2]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%rw : memref<2x2x2x2xi8, #hivm.address_space<gm>>) outs(%ub : memref<2x2x2x2xi8, strided<[16, 8, 4, 2]>, #hivm.address_space<ub>>)
  memref.store %one, %rw[%c0, %c0, %c0, %c0] : memref<2x2x2x2xi8, #hivm.address_space<gm>>
  hivm.hir.load ins(%ro : memref<2x2x2x2xi8, #hivm.address_space<gm>>) outs(%ub : memref<2x2x2x2xi8, strided<[16, 8, 4, 2]>, #hivm.address_space<ub>>)
  return
}
