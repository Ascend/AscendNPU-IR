// RUN: bishengir-opt %s --hivm-v2-pipeline="target=Ascend960" | FileCheck %s

// DHU2 uses 32 lanes of a 64-lane vector. The subtraction feeds the addition
// directly, and the final store still uses the tail mask.
// CHECK-LABEL: func.func @sub_add_fused_0(
// CHECK: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// CHECK: %[[DIFF:.*]] = arith.subf
// CHECK-NOT: vector.transfer_write
// CHECK: %[[SUM:.*]] = arith.addf %{{[^,]+}}, %[[DIFF]]
// CHECK: vector.transfer_write %[[SUM]], {{.*}}, %[[MASK]]
module {
  func.func @sub_add(%lhs: memref<64x32xf32>, %rhs: memref<64x32xf32>,
                     %carry: memref<64x32xf32>, %out: memref<64x32xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %lhs_tensor = bufferization.to_tensor %lhs restrict : memref<64x32xf32>
    %rhs_tensor = bufferization.to_tensor %rhs restrict : memref<64x32xf32>
    %carry_tensor = bufferization.to_tensor %carry restrict : memref<64x32xf32>
    %dest = tensor.empty() : tensor<64x32xf32>
    %diff = hivm.hir.vsub ins(%lhs_tensor, %rhs_tensor : tensor<64x32xf32>, tensor<64x32xf32>)
                          outs(%dest : tensor<64x32xf32>) -> tensor<64x32xf32>
    %sum = hivm.hir.vadd ins(%carry_tensor, %diff : tensor<64x32xf32>, tensor<64x32xf32>)
                         outs(%dest : tensor<64x32xf32>) -> tensor<64x32xf32>
    bufferization.materialize_in_destination %sum in writable %out : (tensor<64x32xf32>, memref<64x32xf32>) -> ()
    return
  }
}
