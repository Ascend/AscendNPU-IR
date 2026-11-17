// RUN: bishengir-opt %s -hivm-bind-sub-block | FileCheck %s

// The producer writes a shared ring; the guarded consumer takes a private
// snapshot before destructive tensor operations. Both allocations must tile,
// while the copy stays in the original ready guard.
// CHECK-LABEL: func.func @snapshot_mix_aiv(
// CHECK-NOT: limit_sub_block_id0
// CHECK: scf.for
// CHECK: %[[RING:.*]] = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[RING]]
// CHECK: hivm.hir.copy ins(%{{.*}} : tensor<32xf32>) outs(%[[CAST]] : memref<32xf32>)
// CHECK: scf.if %{{.*}} {
// CHECK: %[[SNAP:.*]] = memref.alloc() : memref<32xf32>
// CHECK: hivm.hir.copy ins(%{{.*}} : memref<32xf32{{.*}}>) outs(%[[SNAP]] : memref<32xf32>) {tiled_op}
// CHECK: bufferization.to_tensor %[[SNAP]] restrict writable : memref<32xf32>
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.store
// CHECK: map_for_to_forall
// CHECK-NOT: limit_sub_block_id0
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">,
                   hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @snapshot_mix_aiv(%input: tensor<64xf32>, %output: memref<64xf32>,
                             %ready: i1) attributes {
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.core_ratio = #hivm.core_ratio<1, 2>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.part_of_mix, mix_mode = "mix"} {
    %ring = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %ring : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
    hivm.hir.copy ins(%input : tensor<64xf32>) outs(%cast : memref<64xf32>)
    scf.if %ready {
      %snapshot = memref.alloc() : memref<64xf32>
      hivm.hir.copy ins(%cast : memref<64xf32>) outs(%snapshot : memref<64xf32>)
      %tensor = bufferization.to_tensor %snapshot restrict writable : memref<64xf32>
      %empty = tensor.empty() : tensor<64xf32>
      %alpha = hivm.hir.vexp ins(%tensor : tensor<64xf32>) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
      hivm.hir.store ins(%alpha : tensor<64xf32>) outs(%output : memref<64xf32>)
    }
    return
  }
}
