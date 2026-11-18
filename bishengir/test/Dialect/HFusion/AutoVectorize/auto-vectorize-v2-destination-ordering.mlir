// RUN: bishengir-opt %s -split-input-file -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true emit-transform-sequence=true" | FileCheck %s
// RUN: bishengir-opt %s -split-input-file -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" | FileCheck %s

// The destination rule adds a forward edge from a memref.store to every user
// of its destination memref, oriented in block order and omitting self edges.
// The fusable writer P and reader Q below are connected only through the
// destination memref (store -> to_tensor), so no SSA edge represents this
// dependency: without the rule, planning could hoist the reader's loop above
// the writer's. The store must stay immediately before the to_tensor that
// observes it.
// CHECK-LABEL: func.func @destination_users
// CHECK: memref.store
// CHECK-NEXT: bufferization.to_tensor

#id = affine_map<(d0) -> (d0)>
module {
  func.func @destination_users(%a: tensor<128xf32>, %buf: memref<128xf32>) -> tensor<128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %e = tensor.empty() : tensor<128xf32>
    %c0 = arith.constant 0 : index
    %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.addf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    %s = tensor.extract %p[%c0] : tensor<128xf32>
    memref.store %s, %buf[%c0] : memref<128xf32>
    %t = bufferization.to_tensor %buf restrict : memref<128xf32>
    %scalar = tensor.extract %t[%c0] : tensor<128xf32>
    %q = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, #id], iterator_types = ["parallel"]} ins(%scalar : f32) outs(%e : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.mulf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    return %q : tensor<128xf32>
  }
}

// -----

// Two writers to the same destination keep their physical order (the rule must
// not introduce a reverse edge from a later writer to an earlier one), and both
// land before the reader's to_tensor.
// CHECK-LABEL: func.func @multiple_writers
// CHECK: memref.store
// CHECK: memref.store
// CHECK-NEXT: bufferization.to_tensor

#id = affine_map<(d0) -> (d0)>
module {
  func.func @multiple_writers(%a: tensor<128xf32>, %b: tensor<128xf32>, %buf: memref<128xf32>) -> tensor<128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %e1 = tensor.empty() : tensor<128xf32>
    %e2 = tensor.empty() : tensor<128xf32>
    %e3 = tensor.empty() : tensor<128xf32>
    %c0 = arith.constant 0 : index
    %p1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e1 : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.addf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    %s1 = tensor.extract %p1[%c0] : tensor<128xf32>
    memref.store %s1, %buf[%c0] : memref<128xf32>
    %p2 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%b : tensor<128xf32>) outs(%e2 : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.mulf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    %s2 = tensor.extract %p2[%c0] : tensor<128xf32>
    memref.store %s2, %buf[%c0] : memref<128xf32>
    %t = bufferization.to_tensor %buf restrict : memref<128xf32>
    %scalar = tensor.extract %t[%c0] : tensor<128xf32>
    %q = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, #id], iterator_types = ["parallel"]} ins(%scalar : f32) outs(%e3 : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.mulf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    return %q : tensor<128xf32>
  }
}

// -----

// The reader precedes the writer in the block. The rule orients the store edge
// in block order, so the writer's store must stay after the reader's read of
// the same destination rather than being hoisted above it.
// CHECK-LABEL: func.func @reader_before_writer
// CHECK: bufferization.to_tensor
// CHECK: memref.store
// CHECK: return

#id = affine_map<(d0) -> (d0)>
module {
  func.func @reader_before_writer(%a: tensor<128xf32>, %buf: memref<128xf32>) -> tensor<128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %e = tensor.empty() : tensor<128xf32>
    %c0 = arith.constant 0 : index
    %t = bufferization.to_tensor %buf restrict : memref<128xf32>
    %scalar = tensor.extract %t[%c0] : tensor<128xf32>
    %q = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, #id], iterator_types = ["parallel"]} ins(%scalar : f32) outs(%e : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.mulf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.addf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    %s = tensor.extract %p[%c0] : tensor<128xf32>
    memref.store %s, %buf[%c0] : memref<128xf32>
    return %q : tensor<128xf32>
  }
}
