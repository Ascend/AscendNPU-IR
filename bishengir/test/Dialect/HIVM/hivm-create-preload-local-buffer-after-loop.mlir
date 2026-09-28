// RUN: bishengir-opt %s -create-preload | FileCheck %s

// A CV loop yields a preload-local UB pointer_cast that is stored after the
// loop. CreatePreload rematerializes that buffer in the body; post-loop users
// must see a clone after the loop, not the in-loop definition.
// CHECK-LABEL: func.func @yield_preload_local_used_after
// CHECK-DAG: %[[ADDR0:.*]] = arith.constant 16384 : i64
// CHECK-DAG: %[[ADDR1:.*]] = arith.constant 65536 : i64
// CHECK: %[[FOR:.*]] = scf.for
// CHECK: scf.yield %{{.*}} : memref<64x64xf32, {{.*}}ub{{.*}}>
// CHECK-NEXT: }
// CHECK: %[[CLONE:.*]] = hivm.hir.pointer_cast(%[[ADDR0]], %[[ADDR1]])
// CHECK-NEXT: %[[FLAT:.*]] = memref.collapse_shape %[[CLONE]]
// CHECK: hivm.hir.store ins(%[[FLAT]]
// CHECK: memref.collapse_shape %[[FOR]]
func.func @yield_preload_local_used_after(
    %arg0: memref<?xf32, #hivm.address_space<gm>>,
    %arg1: memref<?xf32, #hivm.address_space<gm>>,
    %ub: i32) {
  %c16384 = arith.constant 16384 : i64
  %c65536 = arith.constant 65536 : i64
  %c32768 = arith.constant 32768 : i64
  %c81920 = arith.constant 81920 : i64
  %c49152 = arith.constant 49152 : i64
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %init_tcb = hivm.hir.pointer_cast(%c16384, %c65536)
      : memref<64x64xf32, #hivm.address_space<ub>>
  %init_out = hivm.hir.pointer_cast(%c49152)
      : memref<64x64xf32, #hivm.address_space<ub>>
  %gm_in = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 64],
      strides: [64, 1]
      : memref<?xf32, #hivm.address_space<gm>>
      to memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
  %gm_out0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 64],
      strides: [64, 1]
      : memref<?xf32, #hivm.address_space<gm>>
      to memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
  %0:2 = scf.for %i = %c0_i32 to %ub step %c1_i32
      iter_args(%acc_tcb = %init_tcb, %acc_out = %init_out)
      -> (memref<64x64xf32, #hivm.address_space<ub>>,
          memref<64x64xf32, #hivm.address_space<ub>>) : i32 {
    %vec = hivm.hir.pointer_cast(%c32768, %c81920)
        : memref<64x64xf32, #hivm.address_space<ub>>
    annotation.mark %vec {hivm.multi_buffer = 2 : i32}
        : memref<64x64xf32, #hivm.address_space<ub>>
    %tcb = hivm.hir.pointer_cast(%c16384, %c65536)
        : memref<64x64xf32, #hivm.address_space<ub>>
    annotation.mark %tcb {hivm.multi_buffer = 2 : i32,
                          hivm.preload_local_buffer = 1 : i32}
        : memref<64x64xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      annotation.mark %tcb {effects = ["write", "read"],
          hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
          : memref<64x64xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
    %out = scope.scope : () -> memref<64x64xf32, #hivm.address_space<ub>> {
      %gm_flat = memref.collapse_shape %gm_in [[0, 1]]
          : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
          into memref<4096xf32, strided<[1]>, #hivm.address_space<gm>>
      %vec_flat = memref.collapse_shape %vec [[0, 1]]
          : memref<64x64xf32, #hivm.address_space<ub>>
          into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%gm_flat : memref<4096xf32, strided<[1]>,
                                      #hivm.address_space<gm>>)
                   outs(%vec_flat : memref<4096xf32, #hivm.address_space<ub>>)
          eviction_policy = <EvictFirst> core_type = <VECTOR>
      %produced = hivm.hir.pointer_cast(%c49152)
          : memref<64x64xf32, #hivm.address_space<ub>>
      scope.return %produced : memref<64x64xf32, #hivm.address_space<ub>>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %tcb, %out : memref<64x64xf32, #hivm.address_space<ub>>,
                           memref<64x64xf32, #hivm.address_space<ub>>
  } {hivm.cv_pipelined_loop}
  %tcb_flat = memref.collapse_shape %0#0 [[0, 1]]
      : memref<64x64xf32, #hivm.address_space<ub>>
      into memref<4096xf32, #hivm.address_space<ub>>
  %gm0_flat = memref.collapse_shape %gm_out0 [[0, 1]]
      : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
      into memref<4096xf32, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.store ins(%tcb_flat : memref<4096xf32, #hivm.address_space<ub>>)
                outs(%gm0_flat : memref<4096xf32, strided<[1]>,
                                     #hivm.address_space<gm>>)
  %out_flat = memref.collapse_shape %0#1 [[0, 1]]
      : memref<64x64xf32, #hivm.address_space<ub>>
      into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.store ins(%out_flat : memref<4096xf32, #hivm.address_space<ub>>)
                outs(%gm0_flat : memref<4096xf32, strided<[1]>,
                                     #hivm.address_space<gm>>)
  return
}
