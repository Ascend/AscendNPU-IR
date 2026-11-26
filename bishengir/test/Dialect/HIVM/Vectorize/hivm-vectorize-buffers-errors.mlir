// RUN: not bishengir-opt %s --hivm-vectorize-buffers -split-input-file --mlir-print-ir-after-failure 2>&1 | FileCheck %s

// CHECK: error: paired buffer extent must be twice the channel extent
// CHECK: func.func @mismatched_extent
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.vinterleave
// CHECK-NOT: call @
module {
  func.func @mismatched_extent(%a: memref<4xf32>, %b: memref<4xf32>,
                               %dst: memref<7xf32>, %scalar: f32) {
    hivm.hir.vbrc ins(%scalar : f32) outs(%a : memref<4xf32>)
    hivm.hir.vinterleave ins(%a, %b : memref<4xf32>, memref<4xf32>)
        outs(%dst : memref<7xf32>) interleave_channel_nums = 2
    return
  }
}

// -----

// CHECK: error: channel lowering requires statically shaped rank-one buffers
module {
  func.func @rank_two_buffers(%src: memref<2x4xf32>, %dst: memref<2x2xf32>) {
    hivm.hir.vdeinterleave ins(%src : memref<2x4xf32>)
        outs(%dst : memref<2x2xf32>) index_mode = <CHANNEL_0>
    return
  }
}

// -----

// CHECK: error: channel lowering requires statically shaped rank-one buffers
module {
  func.func @dynamic_channel(%src: memref<?xf32>, %dst: memref<?xf32>) {
    hivm.hir.vdeinterleave ins(%src : memref<?xf32>)
        outs(%dst : memref<?xf32>) index_mode = <CHANNEL_1>
    return
  }
}

// -----

// CHECK: error: non-broadcast buffer dimensions must match
module {
  func.func @mismatched_broadcast(%src: memref<3x1xf32>, %dst: memref<4x5xf32>) {
    hivm.hir.vbrc ins(%src : memref<3x1xf32>)
        outs(%dst : memref<4x5xf32>) broadcast_dims = [1]
    return
  }
}
