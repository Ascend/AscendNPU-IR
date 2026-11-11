// RUN: bishengir-opt -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// An align request that reaches an `arith.select` must be seeded onto both
// candidates, because `arith.select` carries
// `AllTypesMatch<["true_value", "false_value", "result"]>`: widening only one
// of them leaves the join with two different layouts, the request dies there,
// and the fallback hands every failing use a private buffer - splitting one
// logical storage into several.

// Boundary: when the request is attached directly to one allocation, nothing
// ever travels to the select, so the sibling candidate is not seeded and the
// join still cannot be satisfied. Seeding a sibling from below would need a
// downward closure over type-equality edges, which this change does not add.
// Behaviour here is therefore unchanged - the alignment is carried on the
// direct read and the select use still falls back to a private buffer. Pinned
// so the boundary is visible and cannot regress silently.
// CHECK-LABEL: func.func @mark_on_allocation_not_carried_to_select
// CHECK: %[[W:.*]] = memref.alloc() : memref<32x256x1xf16, #hivm.address_space<ub>>
// CHECK: %[[V:.*]] = memref.subview %[[W]]
// CHECK: %[[OTHER:.*]] = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
// CHECK: %[[FALLBACK:.*]] = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
// CHECK: hivm.hir.copy ins(%[[V]] : {{.*}}strided<[256, 1]>{{.*}}) outs(%[[FALLBACK]]
// CHECK: arith.select %{{.*}}, %[[FALLBACK]], %[[OTHER]] : memref<32x128xf16, #hivm.address_space<ub>>
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @mark_on_allocation_not_carried_to_select(%x: i1,
      %a: memref<32x128xf16, #hivm.address_space<ub>>,
      %c: memref<32x128xf16, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %b = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %b {hivm.stride_align_dims = array<i32: 1>,
                        hivm.stride_align_value_in_byte = array<i32: 512>}
      : memref<32x128xf16, #hivm.address_space<ub>>
    %other = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    %sel = arith.select %x, %b, %other : memref<32x128xf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%a : memref<32x128xf16, #hivm.address_space<ub>>)
                  outs(%sel : memref<32x128xf16, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%b : memref<32x128xf16, #hivm.address_space<ub>>)
                  outs(%c : memref<32x128xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Three rotating slots, the shape produced by a software-pipelined kernel. The
// request is attached to the read-side `scf.if` result, so it reaches the
// slots only through the if and through a nested select. All three slots must
// be widened to the same type; the write-side select chain, the if and the
// read-side select must all carry it, and no slot may be duplicated.
// CHECK-LABEL: func.func @rotating_slots_mark_on_if_result
// CHECK-COUNT-3: memref.alloc() : memref<32x256x1xf16, #hivm.address_space<ub>>
// CHECK-NOT: memref.alloc() : memref<32x256x1xf16, #hivm.address_space<ub>>
// CHECK-NOT: memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
// CHECK: scf.if %{{.*}} -> (memref<32x128xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
// CHECK-NOT: unrealized_conversion_cast
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @rotating_slots_mark_on_if_result(%c0: i1, %c1: i1,
      %a: memref<32x128xf16, #hivm.address_space<ub>>,
      %out: memref<32x128xf16, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %s0 = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    %s1 = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    %s2 = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>

    // write side: destination rotates among the three slots
    %w = arith.select %c1, %s1, %s2 : memref<32x128xf16, #hivm.address_space<ub>>
    %w2 = arith.select %c0, %s0, %w : memref<32x128xf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%a : memref<32x128xf16, #hivm.address_space<ub>>)
                  outs(%w2 : memref<32x128xf16, #hivm.address_space<ub>>)

    // read side: same rotation, written as if + select
    %r = scf.if %c0 -> (memref<32x128xf16, #hivm.address_space<ub>>) {
      scf.yield %s0 : memref<32x128xf16, #hivm.address_space<ub>>
    } else {
      %sel = arith.select %c1, %s1, %s2 : memref<32x128xf16, #hivm.address_space<ub>>
      scf.yield %sel : memref<32x128xf16, #hivm.address_space<ub>>
    }
    annotation.mark %r {hivm.stride_align_dims = array<i32: 1>,
                        hivm.stride_align_value_in_byte = array<i32: 512>}
      : memref<32x128xf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%r : memref<32x128xf16, #hivm.address_space<ub>>)
                  outs(%out : memref<32x128xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// The same rotation, but every slot already carries its own mark. The nested
// selects are visited while a sibling candidate is still a plain buffer, so
// the join fails on the first pass; retryDeferredPropagation must come back
// once all three allocations are widened. Without the retry this shape ends up
// with six fallback buffers instead of none.
// CHECK-LABEL: func.func @rotating_slots_all_marked
// CHECK-COUNT-3: memref.alloc() : memref<32x256x1xf16, #hivm.address_space<ub>>
// CHECK-NOT: memref.alloc() : memref<32x256x1xf16, #hivm.address_space<ub>>
// CHECK-NOT: memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
// CHECK: scf.if %{{.*}} -> (memref<32x128xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
// CHECK-NOT: unrealized_conversion_cast
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @rotating_slots_all_marked(%c0: i1, %c1: i1,
      %a: memref<32x128xf16, #hivm.address_space<ub>>,
      %out: memref<32x128xf16, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %s0 = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    %s1 = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    %s2 = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %s0 {hivm.stride_align_dims = array<i32: 1>,
                         hivm.stride_align_value_in_byte = array<i32: 512>}
      : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %s1 {hivm.stride_align_dims = array<i32: 1>,
                         hivm.stride_align_value_in_byte = array<i32: 512>}
      : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %s2 {hivm.stride_align_dims = array<i32: 1>,
                         hivm.stride_align_value_in_byte = array<i32: 512>}
      : memref<32x128xf16, #hivm.address_space<ub>>

    %w = arith.select %c1, %s1, %s2 : memref<32x128xf16, #hivm.address_space<ub>>
    %w2 = arith.select %c0, %s0, %w : memref<32x128xf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%a : memref<32x128xf16, #hivm.address_space<ub>>)
                  outs(%w2 : memref<32x128xf16, #hivm.address_space<ub>>)

    %r = scf.if %c0 -> (memref<32x128xf16, #hivm.address_space<ub>>) {
      scf.yield %s0 : memref<32x128xf16, #hivm.address_space<ub>>
    } else {
      %sel = arith.select %c1, %s1, %s2 : memref<32x128xf16, #hivm.address_space<ub>>
      scf.yield %sel : memref<32x128xf16, #hivm.address_space<ub>>
    }
    hivm.hir.copy ins(%r : memref<32x128xf16, #hivm.address_space<ub>>)
                  outs(%out : memref<32x128xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A candidate that cannot be widened at all - here a function argument, which
// PropagateAlignUpToRootAllocationPattern refuses to mark - makes the join
// unsatisfiable. Seeding only the local allocation would recreate the very
// split-storage shape this is meant to avoid, so the request is declined and
// the allocation keeps its original layout.
// CHECK-LABEL: func.func @select_partner_not_widenable
// CHECK: memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
// CHECK-NOT: memref.alloc() : memref<32x256x1xf16
// CHECK-NOT: memref.subview
// CHECK-NOT: unrealized_conversion_cast
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @select_partner_not_widenable(%x: i1,
      %incoming: memref<32x128xf16, #hivm.address_space<ub>>,
      %out: memref<32x128xf16, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %local = memref.alloc() : memref<32x128xf16, #hivm.address_space<ub>>
    %sel = arith.select %x, %local, %incoming : memref<32x128xf16, #hivm.address_space<ub>>
    annotation.mark %sel {hivm.stride_align_dims = array<i32: 1>,
                          hivm.stride_align_value_in_byte = array<i32: 512>}
      : memref<32x128xf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%sel : memref<32x128xf16, #hivm.address_space<ub>>)
                  outs(%out : memref<32x128xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Selects that do not forward memrefs must be left completely alone: a vector
// select names no storage, so there is nothing to align.
// CHECK-LABEL: func.func @non_memref_select_untouched
// CHECK: %[[W:.*]] = memref.alloc() : memref<32x256x1xf32, #hivm.address_space<ub>>
// CHECK: memref.subview %[[W]]
// CHECK: arith.select %{{.*}} : vector<128xi1>, vector<128xf32>
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @non_memref_select_untouched(%m: vector<128xi1>,
      %t: vector<128xf32>, %f: vector<128xf32>,
      %out: memref<32x128xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.stride_align_dims = array<i32: 1>,
                          hivm.stride_align_value_in_byte = array<i32: 1024>}
      : memref<32x128xf32, #hivm.address_space<ub>>
    %sel = arith.select %m, %t, %f : vector<128xi1>, vector<128xf32>
    vector.store %sel, %buf[%c0, %c0] : memref<32x128xf32, #hivm.address_space<ub>>, vector<128xf32>
    hivm.hir.copy ins(%buf : memref<32x128xf32, #hivm.address_space<ub>>)
                  outs(%out : memref<32x128xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Two candidates whose marks differ but whose widened types coincide: a 32x400
// f16 row is 800 bytes, and both a 512-byte and a 1024-byte request pad it to
// 1024 bytes. The join is satisfiable, so the request must be honoured rather
// than refused on a mark-text comparison.
// CHECK-LABEL: func.func @different_marks_same_widened_type
// CHECK-COUNT-2: memref.alloc() : memref<32x512x1xf16, #hivm.address_space<ub>>
// CHECK: arith.select %{{.*}} : memref<32x400xf16, strided<[512, 1]>, #hivm.address_space<ub>>
// CHECK-NOT: memref.alloc() : memref<32x400xf16, #hivm.address_space<ub>>
// CHECK-NOT: unrealized_conversion_cast
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @different_marks_same_widened_type(%x: i1,
      %a: memref<32x400xf16, #hivm.address_space<ub>>,
      %out: memref<32x400xf16, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %A = memref.alloc() : memref<32x400xf16, #hivm.address_space<ub>>
    annotation.mark %A {hivm.stride_align_dims = array<i32: 1>,
                        hivm.stride_align_value_in_byte = array<i32: 512>}
      : memref<32x400xf16, #hivm.address_space<ub>>
    %B = memref.alloc() : memref<32x400xf16, #hivm.address_space<ub>>
    annotation.mark %B {hivm.stride_align_dims = array<i32: 1>,
                        hivm.stride_align_value_in_byte = array<i32: 1024>}
      : memref<32x400xf16, #hivm.address_space<ub>>
    %sel = arith.select %x, %A, %B : memref<32x400xf16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%a : memref<32x400xf16, #hivm.address_space<ub>>)
                  outs(%sel : memref<32x400xf16, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%sel : memref<32x400xf16, #hivm.address_space<ub>>)
                  outs(%out : memref<32x400xf16, #hivm.address_space<ub>>)
    return
  }
}
