// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true emit-transform-sequence=true" -split-input-file | FileCheck %s

// U -> X -> P and Q -> D. P/Q (256 elements) fuse before U/R (128).
// Contracting Q into P must connect P's predecessor X to Q's successor D.
// Otherwise moving U next to R moves X and P/Q past D but leaves D behind,
// making D use Q before its definition. Check SSA order and both fusion groups.
// CHECK-LABEL: func.func @contraction
// CHECK: %[[U:.*]] = linalg.generic
// CHECK-SAME: {"hfusion-auto-vectorize-target-1"}
// CHECK: %[[R:.*]] = linalg.generic
// CHECK-SAME: {"hfusion-auto-vectorize-target-4"}
// CHECK: %[[X:.*]] = tensor.extract %[[U]]
// CHECK: %[[P:.*]] = linalg.generic
// CHECK-SAME: ins({{.*}}, %[[X]] :
// CHECK-SAME: {"hfusion-auto-vectorize-target-2"}
// CHECK: %[[Q:.*]] = linalg.generic
// CHECK-SAME: {"hfusion-auto-vectorize-target-3"}
// CHECK: %[[D:.*]] = tensor.extract %[[Q]]
// CHECK: return %[[P]], %[[R]], %[[D]]
// CHECK: transform.sequence
// CHECK: match attributes {"hfusion-auto-vectorize-target-1"}
// CHECK: match attributes {"hfusion-auto-vectorize-target-4"}
// CHECK: transform.loop.fuse_sibling
// CHECK: annotate {{.*}} "outlined-loop-target-1"
// CHECK: match attributes {"hfusion-auto-vectorize-target-2"}
// CHECK: match attributes {"hfusion-auto-vectorize-target-3"}
// CHECK: transform.loop.fuse_sibling
// CHECK: annotate {{.*}} "outlined-loop-target-2"

#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
module {
  func.func @contraction(%a: tensor<128xf32>, %b: tensor<256xf32>) -> (tensor<256xf32>, tensor<128xf32>, f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c0 = arith.constant 0 : index
    %e128 = tensor.empty() : tensor<128xf32>
    %e256 = tensor.empty() : tensor<256xf32>
    %u = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e128 : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %v2 = arith.addf %v, %v : f32
      linalg.yield %v2 : f32
    } -> tensor<128xf32>
    %x = tensor.extract %u[%c0] : tensor<128xf32>
    %p = linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel"]} ins(%b, %x : tensor<256xf32>, f32) outs(%e256 : tensor<256xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %v2 = arith.addf %v, %s : f32
      linalg.yield %v2 : f32
    } -> tensor<256xf32>
    %q = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%b : tensor<256xf32>) outs(%e256 : tensor<256xf32>) {
    ^bb0(%v: f32, %o: f32):
      %v2 = arith.mulf %v, %v : f32
      linalg.yield %v2 : f32
    } -> tensor<256xf32>
    %d = tensor.extract %q[%c0] : tensor<256xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e128 : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %v2 = arith.mulf %v, %v : f32
      linalg.yield %v2 : f32
    } -> tensor<128xf32>
    return %p, %r, %d : tensor<256xf32>, tensor<128xf32>, f32
  }
}

// -----

// Reverse the P/Q order: now P's predecessor X must reach the existing Q
// group's successor D when P is contracted into Q (the opposite cross term).
// CHECK-LABEL: func.func @reverse_contraction
// CHECK: %[[U:.*]] = linalg.generic
// CHECK-SAME: {"hfusion-auto-vectorize-target-1"}
// CHECK: %[[R:.*]] = linalg.generic
// CHECK-SAME: {"hfusion-auto-vectorize-target-4"}
// CHECK: %[[X:.*]] = tensor.extract %[[U]]
// CHECK: %[[Q:.*]] = linalg.generic
// CHECK-SAME: {"hfusion-auto-vectorize-target-2"}
// CHECK: %[[P:.*]] = linalg.generic
// CHECK-SAME: ins({{.*}}, %[[X]] :
// CHECK-SAME: {"hfusion-auto-vectorize-target-3"}
// CHECK: %[[D:.*]] = tensor.extract %[[Q]]
// CHECK: return %[[P]], %[[R]], %[[D]]
// CHECK: transform.sequence
// CHECK: match attributes {"hfusion-auto-vectorize-target-1"}
// CHECK: match attributes {"hfusion-auto-vectorize-target-4"}
// CHECK: transform.loop.fuse_sibling
// CHECK: annotate {{.*}} "outlined-loop-target-1"
// CHECK: match attributes {"hfusion-auto-vectorize-target-2"}
// CHECK: match attributes {"hfusion-auto-vectorize-target-3"}
// CHECK: transform.loop.fuse_sibling
// CHECK: annotate {{.*}} "outlined-loop-target-2"

#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
module {
  func.func @reverse_contraction(%a: tensor<128xf32>, %b: tensor<256xf32>) -> (tensor<256xf32>, tensor<128xf32>, f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c0 = arith.constant 0 : index
    %e128 = tensor.empty() : tensor<128xf32>
    %e256 = tensor.empty() : tensor<256xf32>
    %u = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e128 : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %v2 = arith.addf %v, %v : f32
      linalg.yield %v2 : f32
    } -> tensor<128xf32>
    %x = tensor.extract %u[%c0] : tensor<128xf32>
    %q = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%b : tensor<256xf32>) outs(%e256 : tensor<256xf32>) {
    ^bb0(%v: f32, %o: f32):
      %v2 = arith.mulf %v, %v : f32
      linalg.yield %v2 : f32
    } -> tensor<256xf32>
    %p = linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel"]} ins(%b, %x : tensor<256xf32>, f32) outs(%e256 : tensor<256xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %v2 = arith.addf %v, %s : f32
      linalg.yield %v2 : f32
    } -> tensor<256xf32>
    %d = tensor.extract %q[%c0] : tensor<256xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<128xf32>) outs(%e128 : tensor<128xf32>) {
    ^bb0(%v: f32, %o: f32):
      %v2 = arith.mulf %v, %v : f32
      linalg.yield %v2 : f32
    } -> tensor<128xf32>
    return %p, %r, %d : tensor<256xf32>, tensor<128xf32>, f32
  }
}
