// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivm-to-std -convert-hivm-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s --check-prefix=ABI

// ABI: llvm.func @_mlir_ciface_tcb_pool_init(!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// ABI: llvm.func @_mlir_ciface_tcb_pool_alloc(!llvm.ptr, !llvm.ptr, i64, i32)
// ABI: llvm.func @_mlir_ciface_tcb_pool_release(!llvm.ptr)
// ABI: llvm.func @_mlir_ciface_tcb_pool_get_address(!llvm.ptr, !llvm.ptr, i64, i32)
module {
  // CHECK-LABEL: func.func @cube
  func.func @cube(%meta: memref<50xi32, #hivm.address_space<ssbuf>>) -> (i64, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %base = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%base) : memref<160xi8, #hivm.address_space<ub>>
    // CHECK: call @tcb_pool_init({{.*}}) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ssbuf>>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    hivm.hir.tcb_pool_init %meta {slot_bytes = 32 : i32, slot_count = 5 : i32, slots_per_buffer = array<i32: 2, 1>, search_begin = array<i32: 0, 4>, search_end = array<i32: 4, 5>, flag_mask = 31 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>
    // CHECK: call @tcb_pool_alloc(
    %a, %id = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> (i64, i32)
    // CHECK: call @tcb_pool_release(
    hivm.hir.tcb_pool_release %meta : memref<50xi32, #hivm.address_space<ssbuf>>
    return %a, %id : i64, i32
  }
  // CHECK-LABEL: func.func @vector
  func.func @vector(%meta: memref<50xi32, strided<[1], offset: ?>, #hivm.address_space<ssbuf>>) -> (i64, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %base = arith.constant 256 : i64
    %pool = hivm.hir.pointer_cast(%base) : memref<160xi8, #hivm.address_space<ub>>
    // CHECK: call @tcb_pool_get_address(
    %a, %id = hivm.hir.tcb_pool_get %meta, %pool {segment = 1 : i32} : memref<50xi32, strided<[1], offset: ?>, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> (i64, i32)
    return %a, %id : i64, i32
  }
}
