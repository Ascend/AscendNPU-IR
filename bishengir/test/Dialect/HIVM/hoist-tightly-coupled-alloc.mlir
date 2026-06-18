// RUN: bishengir-opt %s -hivm-hoist-tightly-coupled-alloc -split-input-file | FileCheck %s

// A tightly-coupled UB alloc created in an inner scf.for, whose tensor view is
// yielded out of the inner loop and again out of the outer loop, is hoisted to
// the outermost region the value escapes to (the function body).
// CHECK-LABEL: func.func @hoist_for_yield
func.func @hoist_for_yield(%lb: i32, %ub: i32, %step: i32,
                           %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.for
  %outer = scf.for %i = %lb to %ub step %step iter_args(%oa = %init) -> (tensor<64x32xf32>) : i32 {
    %inner = scf.for %j = %lb to %ub step %step iter_args(%ia = %oa) -> (tensor<64x32xf32>) : i32 {
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    scf.yield %inner : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// When the alloc's view is NOT yielded out (only consumed in-loop), it stays in
// place.
// CHECK-LABEL: func.func @no_hoist_when_not_yielded
func.func @no_hoist_when_not_yielded(%lb: i32, %ub: i32, %step: i32,
                                     %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %outer = scf.for %i = %lb to %ub step %step iter_args(%oa = %init) -> (tensor<64x32xf32>) : i32 {
    // CHECK: scf.for
    // CHECK: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
    %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
    %add = hivm.hir.vadd ins(%t, %oa : tensor<64x32xf32>, tensor<64x32xf32>) outs(%oa : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %add : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// Multi-level scf.for: the inner for's result is not yielded out at all. The
// alloc can only be traced outward through the inner for init_arg, which is the
// outer for's iter_arg. Following the init_arg chain hoists the alloc across
// both for levels to the function body.
// CHECK-LABEL: func.func @hoist_nested_for_via_init_arg
func.func @hoist_nested_for_via_init_arg(%lb: i32, %ub: i32, %step: i32,
                                         %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.for
  %outer = scf.for %i = %lb to %ub step %step iter_args(%oa = %init) -> (tensor<64x32xf32>) : i32 {
    // CHECK: scf.for
    %inner = scf.for %j = %lb to %ub step %step iter_args(%ia = %oa) -> (tensor<64x32xf32>) : i32 {
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    %other = hivm.hir.vadd ins(%oa, %oa : tensor<64x32xf32>, tensor<64x32xf32>) outs(%oa : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %other : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// The value yielded by the inner for can be an outer loop iter_arg itself. Even
// though that iter_arg is not produced by another yield in the current block,
// tracing must jump to the outer loop's init_arg and continue outward.
// CHECK-LABEL: func.func @hoist_outer_for_iter_arg_to_init_arg
func.func @hoist_outer_for_iter_arg_to_init_arg(%lb: i32, %ub: i32, %step: i32,
                                                %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.for
  %outer = scf.for %i = %lb to %ub step %step iter_args(%oa = %init) -> (tensor<64x32xf32>) : i32 {
    // CHECK: scf.for
    %inner = scf.for %j = %lb to %ub step %step iter_args(%ia = %oa) -> (tensor<64x32xf32>) : i32 {
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    scf.yield %oa : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// Multi-level scf.for, but hoist only one level: the inner for yields the alloc
// value, and the outer for consumes that result through a non-view op. Since the
// same value is not yielded farther out, the alloc should be moved only to the
// outer for body, not to the function body.
// CHECK-LABEL: func.func @hoist_for_one_level
func.func @hoist_for_one_level(%lb: i32, %ub: i32, %step: i32,
                               %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scf.for
  %outer = scf.for %i = %lb to %ub step %step iter_args(%oa = %init) -> (tensor<64x32xf32>) : i32 {
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[ALLOC]]
    // CHECK: scf.for
    %seed = hivm.hir.vadd ins(%oa, %oa : tensor<64x32xf32>, tensor<64x32xf32>) outs(%oa : tensor<64x32xf32>) -> tensor<64x32xf32>
    %inner = scf.for %j = %lb to %ub step %step iter_args(%ia = %seed) -> (tensor<64x32xf32>) : i32 {
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    %next = hivm.hir.vadd ins(%inner, %oa : tensor<64x32xf32>, tensor<64x32xf32>) outs(%oa : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %next : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// scf.if: a tightly-coupled alloc yielded out of an scf.if is hoisted to the
// region the if-result escapes to.
// CHECK-LABEL: func.func @hoist_if_yield
func.func @hoist_if_yield(%cond: i1, %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.if
  %res = scf.if %cond -> (tensor<64x32xf32>) {
    // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
    %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
    scf.yield %t : tensor<64x32xf32>
  } else {
    scf.yield %init : tensor<64x32xf32>
  }
  return %res : tensor<64x32xf32>
}

// -----

// scf.while: the alloc lives in the "after" (do) region and is yielded back as
// the next iter arg. Because the "before" region's scf.condition forwards that
// same iter arg as the loop result, the value escapes the loop, so the alloc is
// hoisted out to the region the while-result escapes to (the function body).
// CHECK-LABEL: func.func @hoist_while_yield
func.func @hoist_while_yield(%cond: i1, %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.while
  %res = scf.while (%arg = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    scf.condition(%cond) %arg : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
    %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
    scf.yield %t : tensor<64x32xf32>
  }
  return %res : tensor<64x32xf32>
}

// -----

// scf.while: even if the "before" region forwards a freshly computed value
// (not the iter arg) through scf.condition, the alloc yielded from the "after"
// region still feeds the next iteration's init_arg. Trace that init_arg chain
// and hoist the alloc out to the while's parent region.
// CHECK-LABEL: func.func @hoist_while_cond_mismatch_via_init_arg
func.func @hoist_while_cond_mismatch_via_init_arg(%cond: i1, %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.while
  %res = scf.while (%arg = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    %other = hivm.hir.vadd ins(%arg, %arg : tensor<64x32xf32>, tensor<64x32xf32>) outs(%arg : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.condition(%cond) %other : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
    %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
    scf.yield %t : tensor<64x32xf32>
  }
  return %res : tensor<64x32xf32>
}

// -----

// Multi-level scf.while: the inner while's result is not yielded out at all,
// and its condition forwards a value other than the iter_arg. The only path to
// the outer loop is the inner while init_arg, which comes from the outer while
// after-region argument. Following init_arg hoists the alloc across both while
// levels to the function body.
// CHECK-LABEL: func.func @hoist_nested_while_via_init_arg
func.func @hoist_nested_while_via_init_arg(%outer_cond: i1, %inner_cond: i1,
                                           %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.while
  %outer = scf.while (%outer_arg = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    scf.condition(%outer_cond) %outer_arg : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    // CHECK: scf.while
    %inner = scf.while (%inner_arg = %a) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
      %other = hivm.hir.vadd ins(%inner_arg, %inner_arg : tensor<64x32xf32>, tensor<64x32xf32>) outs(%inner_arg : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.condition(%inner_cond) %other : tensor<64x32xf32>
    } do {
    ^bb1(%b: tensor<64x32xf32>):
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    scf.yield %a : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// Outer scf.while after-argument case: the inner for init_arg is the outer
// while do-region argument. The outer while yields a different value, so tracing
// must recognize that do-region argument as the outer while's iter_arg and
// continue from the outer while init_arg.
// CHECK-LABEL: func.func @hoist_outer_while_iter_arg_to_init_arg
func.func @hoist_outer_while_iter_arg_to_init_arg(%cond: i1, %lb: i32, %ub: i32,
                                                  %step: i32,
                                                  %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.while
  %outer = scf.while (%wa = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    scf.condition(%cond) %wa : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    // CHECK: scf.for
    %inner = scf.for %i = %lb to %ub step %step iter_args(%fa = %a) -> (tensor<64x32xf32>) : i32 {
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    %other = hivm.hir.vadd ins(%a, %a : tensor<64x32xf32>, tensor<64x32xf32>) outs(%a : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %other : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// Multi-level scf.while, but hoist only one level: the inner while yields the
// alloc value, and the outer while body consumes that result through a non-view
// op. Since the same value is not yielded through the outer while, the alloc is
// moved only to the outer while body.
// CHECK-LABEL: func.func @hoist_while_one_level
func.func @hoist_while_one_level(%outer_cond: i1, %inner_cond: i1,
                                 %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scf.while
  %outer = scf.while (%outer_arg = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    scf.condition(%outer_cond) %outer_arg : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[ALLOC]]
    // CHECK: scf.while
    %seed = hivm.hir.vadd ins(%a, %a : tensor<64x32xf32>, tensor<64x32xf32>) outs(%a : tensor<64x32xf32>) -> tensor<64x32xf32>
    %inner = scf.while (%inner_arg = %seed) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
      scf.condition(%inner_cond) %inner_arg : tensor<64x32xf32>
    } do {
    ^bb1(%b: tensor<64x32xf32>):
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    %next = hivm.hir.vadd ins(%inner, %a : tensor<64x32xf32>, tensor<64x32xf32>) outs(%a : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %next : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// scf.while nested in scf.for: the alloc in the while "after" region is carried
// through the while iter_arg (init_arg), escapes via the matching scf.condition
// as the while result, then is yielded out of the enclosing scf.for. It is
// hoisted across both levels up to the function body.
// CHECK-LABEL: func.func @hoist_while_in_for
func.func @hoist_while_in_for(%lb: i32, %ub: i32, %step: i32, %cond: i1,
                              %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.for
  %for_res = scf.for %i = %lb to %ub step %step iter_args(%fa = %init) -> (tensor<64x32xf32>) : i32 {
    // CHECK: scf.while
    %while_res = scf.while (%wa = %fa) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
      scf.condition(%cond) %wa : tensor<64x32xf32>
    } do {
    ^bb0(%a: tensor<64x32xf32>):
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    scf.yield %while_res : tensor<64x32xf32>
  }
  return %for_res : tensor<64x32xf32>
}

// -----

// scf.while "before" region: a tightly-coupled alloc created in the before
// region, whose view is forwarded out by scf.condition as the while result,
// escapes the loop and is hoisted to the region the while-result escapes to.
// CHECK-LABEL: func.func @hoist_while_before_region
func.func @hoist_while_before_region(%cond: i1, %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.while
  %res = scf.while (%arg = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
    %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
    scf.condition(%cond) %t : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    scf.yield %a : tensor<64x32xf32>
  }
  return %res : tensor<64x32xf32>
}

// -----

// scf.while "before" region nested in scf.for: the before-region alloc is
// forwarded out as the while result, which is then yielded out of the enclosing
// for, so the alloc is hoisted across both levels to the function body.
// CHECK-LABEL: func.func @hoist_while_before_region_in_for
func.func @hoist_while_before_region_in_for(%lb: i32, %ub: i32, %step: i32,
                                            %cond: i1,
                                            %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.for
  %for = scf.for %i = %lb to %ub step %step iter_args(%fa = %init) -> (tensor<64x32xf32>) : i32 {
    // CHECK: scf.while
    %w = scf.while (%wa = %fa) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.condition(%cond) %t : tensor<64x32xf32>
    } do {
    ^bb0(%a: tensor<64x32xf32>):
      scf.yield %a : tensor<64x32xf32>
    }
    scf.yield %w : tensor<64x32xf32>
  }
  return %for : tensor<64x32xf32>
}

// -----

// scf.for nested in scf.while "before" region: the inner for init_arg is the
// while before-region iter_arg, and scf.condition forwards that same iter_arg as
// the while result. Tracing the before-region iter_arg must follow both the
// init_arg and the (forwarded) result chain and hoist to the function body.
// CHECK-LABEL: func.func @hoist_for_in_while_before_region
func.func @hoist_for_in_while_before_region(%lb: i32, %ub: i32, %step: i32,
                                            %cond: i1,
                                            %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.while
  %res = scf.while (%wa = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    // CHECK: scf.for
    %inner = scf.for %i = %lb to %ub step %step iter_args(%fa = %wa) -> (tensor<64x32xf32>) : i32 {
      // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    scf.condition(%cond) %wa : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    scf.yield %a : tensor<64x32xf32>
  }
  return %res : tensor<64x32xf32>
}

// -----

// scf.while "before" region, but the alloc view is only consumed locally and
// the value scf.condition forwards is a freshly computed one, so the alloc does
// not escape and must stay in place.
// CHECK-LABEL: func.func @no_hoist_while_before_region_local
func.func @no_hoist_while_before_region_local(%cond: i1, %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %res = scf.while (%arg = %init) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
    // CHECK: scf.while
    // CHECK: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
    %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
    %use = hivm.hir.vadd ins(%t, %arg : tensor<64x32xf32>, tensor<64x32xf32>) outs(%arg : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.condition(%cond) %use : tensor<64x32xf32>
  } do {
  ^bb0(%a: tensor<64x32xf32>):
    scf.yield %a : tensor<64x32xf32>
  }
  return %res : tensor<64x32xf32>
}

// -----

// scf.while after-region argument: the before-region condition forwards a
// freshly computed value, so `conditionCheck` is false. The after-region yield
// forwards the after argument unchanged, so `yieldCheck` must connect the result
// chain back to the init_arg chain and hoist across the enclosing scf.for.
// CHECK-LABEL: func.func @hoist_while_after_arg_via_yield_check
func.func @hoist_while_after_arg_via_yield_check(%lb: i32, %ub: i32, %step: i32,
                                                 %cond: i1,
                                                 %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]]
  // CHECK: scf.for
  %outer = scf.for %i = %lb to %ub step %step iter_args(%oa = %init) -> (tensor<64x32xf32>) : i32 {
    // CHECK: scf.while
    %inner_while = scf.while (%wa = %oa) : (tensor<64x32xf32>) -> tensor<64x32xf32> {
      %other = hivm.hir.vadd ins(%wa, %wa : tensor<64x32xf32>, tensor<64x32xf32>) outs(%wa : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.condition(%cond) %other : tensor<64x32xf32>
    } do {
    ^bb0(%a: tensor<64x32xf32>):
      // CHECK: scf.for
      %inner_for = scf.for %j = %lb to %ub step %step iter_args(%fa = %a) -> (tensor<64x32xf32>) : i32 {
        // CHECK-NOT: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x32xf32, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
        %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
        scf.yield %t : tensor<64x32xf32>
      }
      scf.yield %a : tensor<64x32xf32>
    }
    scf.yield %oa : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}

// -----

// An alloc without the tightly_coupled_buffer mark is never hoisted.
// CHECK-LABEL: func.func @no_hoist_unmarked
func.func @no_hoist_unmarked(%lb: i32, %ub: i32, %step: i32,
                             %init: tensor<64x32xf32>) -> tensor<64x32xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %outer = scf.for %i = %lb to %ub step %step iter_args(%oa = %init) -> (tensor<64x32xf32>) : i32 {
    %inner = scf.for %j = %lb to %ub step %step iter_args(%ia = %oa) -> (tensor<64x32xf32>) : i32 {
      // CHECK: scf.for
      // CHECK: scf.for
      // CHECK: memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %alloc : memref<64x32xf32, #hivm.address_space<ub>> to memref<64x32xf32>
      %t = bufferization.to_tensor %cast restrict writable : memref<64x32xf32>
      scf.yield %t : tensor<64x32xf32>
    }
    scf.yield %inner : tensor<64x32xf32>
  }
  return %outer : tensor<64x32xf32>
}
