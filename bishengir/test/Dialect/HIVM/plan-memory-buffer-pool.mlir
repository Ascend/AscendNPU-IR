// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory | FileCheck %s

// Four former double buffers share five physical slots, not twenty.
// CHECK-LABEL: func.func @four_loads_five_slots
func.func @four_loads_five_slots(%src: memref<1024xi8, #hivm.address_space<gm>>, %dst: memref<1024xi8, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // CHECK: %[[P0:.*]] = hivm.hir.pointer_cast(%[[S0:.*]], %[[S1:.*]], %[[S2:.*]], %[[S3:.*]], %[[S4:.*]]) : memref<1024xi8, #hivm.address_space<ub>>
    %b0 = memref.alloc() : memref<1024xi8, #hivm.address_space<ub>>
    annotation.mark %b0 {hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32, hivm.multi_buffer = 5 : i32, hivm.preload_local_buffer = 1 : i32, hivm.mem_unique} : memref<1024xi8, #hivm.address_space<ub>>
    // CHECK: %[[P1:.*]] = hivm.hir.pointer_cast(%[[S0]], %[[S1]], %[[S2]], %[[S3]], %[[S4]]) : memref<1024xi8, #hivm.address_space<ub>>
    %b1 = memref.alloc() : memref<1024xi8, #hivm.address_space<ub>>
    annotation.mark %b1 {hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32, hivm.multi_buffer = 5 : i32, hivm.preload_local_buffer = 1 : i32, hivm.mem_unique} : memref<1024xi8, #hivm.address_space<ub>>
    // CHECK: %[[P2:.*]] = hivm.hir.pointer_cast(%[[S0]], %[[S1]], %[[S2]], %[[S3]], %[[S4]]) : memref<1024xi8, #hivm.address_space<ub>>
    %b2 = memref.alloc() : memref<1024xi8, #hivm.address_space<ub>>
    annotation.mark %b2 {hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32, hivm.multi_buffer = 5 : i32, hivm.preload_local_buffer = 1 : i32, hivm.mem_unique} : memref<1024xi8, #hivm.address_space<ub>>
    // CHECK: %[[P3:.*]] = hivm.hir.pointer_cast(%[[S0]], %[[S1]], %[[S2]], %[[S3]], %[[S4]]) : memref<1024xi8, #hivm.address_space<ub>>
    %b3 = memref.alloc() : memref<1024xi8, #hivm.address_space<ub>>
    annotation.mark %b3 {hivm.mempool_id = 0 : i32, hivm.mempool_size = 5 : i32, hivm.multi_buffer = 5 : i32, hivm.preload_local_buffer = 1 : i32, hivm.mem_unique} : memref<1024xi8, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1024xi8, #hivm.address_space<gm>>) outs(%b0 : memref<1024xi8, #hivm.address_space<ub>>)
    hivm.hir.load ins(%src : memref<1024xi8, #hivm.address_space<gm>>) outs(%b1 : memref<1024xi8, #hivm.address_space<ub>>)
    hivm.hir.load ins(%src : memref<1024xi8, #hivm.address_space<gm>>) outs(%b2 : memref<1024xi8, #hivm.address_space<ub>>)
    hivm.hir.load ins(%src : memref<1024xi8, #hivm.address_space<gm>>) outs(%b3 : memref<1024xi8, #hivm.address_space<ub>>)
    hivm.hir.store ins(%b0 : memref<1024xi8, #hivm.address_space<ub>>) outs(%dst : memref<1024xi8, #hivm.address_space<gm>>)
    hivm.hir.store ins(%b1 : memref<1024xi8, #hivm.address_space<ub>>) outs(%dst : memref<1024xi8, #hivm.address_space<gm>>)
    hivm.hir.store ins(%b2 : memref<1024xi8, #hivm.address_space<ub>>) outs(%dst : memref<1024xi8, #hivm.address_space<gm>>)
    hivm.hir.store ins(%b3 : memref<1024xi8, #hivm.address_space<ub>>) outs(%dst : memref<1024xi8, #hivm.address_space<gm>>)
  }
  return
}
