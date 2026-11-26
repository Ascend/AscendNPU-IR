// RUN: bishengir-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  func.func private @use_index(index)
  func.func private @use(f32)
  func.func private @use_tile(tensor<4xf32>)

  // Applying patterns to the parent visits both branches and leaves the
  // loop-invariant sum in place.
  // CHECK-LABEL: func.func @branches(
  // CHECK-SAME: %{{[^:]+}}: i1, %[[N:[^:]+]]: index,
  // CHECK: scf.if
  // CHECK: %[[PRODUCT:.*]] = arith.mulf
  // CHECK-NOT: arith.mulf
  // CHECK: scf.for
  // CHECK-NOT: arith.mulf
  // CHECK: func.call @use(%[[PRODUCT]])
  // CHECK-NEXT: %[[SUM:.*]] = arith.addf
  // CHECK-NEXT: func.call @use(%[[SUM]])
  // CHECK: } else {
  // CHECK-NEXT: func.call @use_index(%[[N]])
  // CHECK-NEXT: %[[READ:.*]] = memref.load
  // CHECK-NEXT: func.call @use(%[[READ]])
  // CHECK-NEXT: func.call @use(%[[READ]])
  func.func @branches(%cond: i1, %n: index, %x: f32, %y: f32, %src: memref<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.if %cond {
      %a = arith.mulf %x, %y : f32
      func.call @use(%a) : (f32) -> ()
      scf.for %i = %c0 to %n step %c1 {
        %b = arith.mulf %x, %y : f32
        func.call @use(%b) : (f32) -> ()
        %sum = arith.addf %x, %y : f32
        func.call @use(%sum) : (f32) -> ()
      } {anchor}
    } else {
      %same = arith.addi %n, %c0 : index
      func.call @use_index(%same) : (index) -> ()
      %c = memref.load %src[] : memref<f32>
      %d = memref.load %src[] : memref<f32>
      func.call @use(%c) : (f32) -> ()
      func.call @use(%d) : (f32) -> ()
    }
    return
  }

  // Parent CSE reuses the extract dominating the loop in the same branch.
  // CHECK-LABEL: func.func @shared_slice
  // CHECK: %[[TILE:.*]] = tensor.extract_slice
  // CHECK: scf.for
  // CHECK-NOT: tensor.extract_slice
  // CHECK: func.call @use_tile(%[[TILE]])
  func.func @shared_slice(%cond: i1, %n: index, %src: tensor<8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.if %cond {
      %a = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
      func.call @use_tile(%a) : (tensor<4xf32>) -> ()
      scf.for %i = %c0 to %n step %c1 {
        %b = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
        func.call @use_tile(%b) : (tensor<4xf32>) -> ()
      } {anchor, "outlined-loop-target-1"}
    }
    return
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %anchor = transform.structured.match attributes {anchor} in %root : (!transform.any_op) -> !transform.any_op
    %parent = transform.get_parent_op %anchor : (!transform.any_op) -> !transform.op<"scf.if">
    transform.apply_patterns to %parent {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } {apply_cse, disable_patterns = ["SimplifyTrivialLoops"]} : !transform.op<"scf.if">
    transform.yield
  }
}
