// RUN: bishengir-opt %s -split-input-file -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true emit-transform-sequence=true" | FileCheck %s
// RUN: bishengir-opt %s -split-input-file -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" | FileCheck %s

// The materialize operation writes %src, which is then consumed as the copy
// source. This dependency is not represented by an SSA result of materialize.
// CHECK-LABEL: func.func @copy_source
// CHECK: bufferization.materialize_in_destination
// CHECK-NEXT: memref.copy
// CHECK-NEXT: bufferization.to_tensor

#id = affine_map<(d0) -> (d0)>
module {
  func.func @copy_source(%a: tensor<128xf32>, %src: memref<128xf32>, %dst: memref<128xf32>) -> tensor<128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %e = tensor.empty() : tensor<128xf32>
    %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.addf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    bufferization.materialize_in_destination %p in writable %src : (tensor<128xf32>, memref<128xf32>) -> ()
    memref.copy %src, %dst : memref<128xf32> to memref<128xf32>
    %t = bufferization.to_tensor %dst restrict : memref<128xf32>
    %c0 = arith.constant 0 : index
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

// A later source write must remain after the copy. The copy is itself a
// source user and must not add a self edge through the source-user rule.
// CHECK-LABEL: func.func @copy_before_source_write
// CHECK: memref.copy
// CHECK: bufferization.materialize_in_destination
// CHECK: return

#id = affine_map<(d0) -> (d0)>
module {
  func.func @copy_before_source_write(%a: tensor<128xf32>, %src: memref<128xf32>, %dst: memref<128xf32>) -> tensor<128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %e = tensor.empty() : tensor<128xf32>
    %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.addf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    memref.copy %src, %dst : memref<128xf32> to memref<128xf32>
    bufferization.materialize_in_destination %p in writable %src : (tensor<128xf32>, memref<128xf32>) -> ()
    %t = bufferization.to_tensor %dst restrict : memref<128xf32>
    %c0 = arith.constant 0 : index
    %scalar = tensor.extract %t[%c0] : tensor<128xf32>
    %q = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, #id], iterator_types = ["parallel"]} ins(%scalar : f32) outs(%e : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.mulf %v, %v : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    return %q : tensor<128xf32>
  }
}
