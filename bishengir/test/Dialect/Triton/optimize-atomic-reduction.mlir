// RUN: bishengir-opt -split-input-file %s -optimize-atomic-reduction | FileCheck %s

// A single select that picks between `%x` and a zero splat (the `fadd`
// identity).  The no-op lanes are folded into the atomic's mask and `%x` is
// forwarded as the value.
// CHECK-LABEL: tt.func @tensor_select_with_mask(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<f32>, %[[X:[^:]*]]: tensor<4xf32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
tt.func @tensor_select_with_mask(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %s = arith.select %cond, %x, %c0 : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  tt.return
}

// -----

// A nested select chain with no original mask: the rebuilt predicate becomes
// the whole mask.
// CHECK-LABEL: tt.func @tensor_nested_select_no_mask(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<f32>, %[[X:[^:]*]]: tensor<4xf32>, %{{[^:]*}}: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[P1:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[P2:.*]] = arith.select %{{.*}}, %[[P1]], %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %[[X]], %[[P2]] : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
tt.func @tensor_nested_select_no_mask(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %cond1: tensor<4xi1>, %cond2: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %s1 = arith.select %cond1, %x, %c0 : tensor<4xi1>, tensor<4xf32>
  %s2 = arith.select %cond2, %s1, %c0 : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %s2 : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> tensor<4xf32>
  tt.return
}

// -----

// A first-level select with the identity on one arm can be folded even when the
// other arm is itself a non-trivial select: forward that arm and AND the
// condition into the mask, without inspecting the other arm's leaves.
// CHECK-LABEL: tt.func @fold_first_level_identity(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<f32>, %[[X:[^:]*]]: tensor<4xf32>, %[[Y:[^:]*]]: tensor<4xf32>, %{{[^:]*}}: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[INNER:.*]] = arith.select %{{.*}}, %[[X]], %[[Y]] : tensor<4xi1>, tensor<4xf32>
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %[[INNER]], %[[PRED]] : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
tt.func @fold_first_level_identity(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %y: tensor<4xf32>, %cond0: tensor<4xi1>, %cond1: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %inner = arith.select %cond1, %x, %y : tensor<4xi1>, tensor<4xf32>
  %s = arith.select %cond0, %c0, %inner : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %s : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> tensor<4xf32>
  tt.return
}

// -----

// Scalar chain for the integer `add` reduction without an original mask.  The
// scalar `select(cond, true, false)` folds into the condition itself.
// CHECK-LABEL: tt.func @scalar_select_add(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[A:[^:]*]]: i32, %{{[^:]*}}: i1, %{{[^:]*}}: i1)
// CHECK: tt.atomic_rmw add, relaxed, gpu, %{{.*}}, %[[A]], %{{.*}} : (!tt.ptr<i32>, i32, i1) -> i32
tt.func @scalar_select_add(%ptr: !tt.ptr<i32>, %a: i32, %cond1: i1, %cond2: i1) {
  %c0 = arith.constant 0 : i32
  %s1 = arith.select %cond1, %a, %c0 : i32
  %s2 = arith.select %cond2, %s1, %c0 : i32
  %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %s2 : (!tt.ptr<i32>, i32) -> i32
  tt.return
}

// -----

// The non-zero leaf may sit on the false arm of the select.
// CHECK-LABEL: tt.func @nonzero_select_on_false(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<f32>, %[[X:[^:]*]]: tensor<4xf32>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %[[X]], %{{.*}} : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
tt.func @nonzero_select_on_false(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %s = arith.select %cond, %c0, %x : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %s : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> tensor<4xf32>
  tt.return
}

// -----

// An identity-valued `$val` makes the atomic an unconditional no-op.
// CHECK-LABEL: tt.func @constant_zero_val(
// CHECK-NOT: tt.atomic_rmw
// CHECK: tt.return
tt.func @constant_zero_val(%ptr: !tt.ptr<f32>, %mask: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %c0, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  tt.return
}

// -----

// The identity is nested on an arm that is itself a select, so the top-level
// select has two non-identity arms and the scan bails: the atomic is untouched.
// CHECK-LABEL: tt.func @nested_identity_not_on_arm(
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
// CHECK-NOT: arith.andi
tt.func @nested_identity_not_on_arm(%ptr: !tt.ptr<f32>, %a: tensor<4xf32>, %b: tensor<4xf32>, %mask: tensor<4xi1>, %cond1: tensor<4xi1>, %cond2: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %s1 = arith.select %cond1, %a, %c0 : tensor<4xi1>, tensor<4xf32>
  %s2 = arith.select %cond2, %b, %s1 : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %s2, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  tt.return
}

// -----

// The top-level select is between two non-identity arms, so the scan ends
// immediately and the atomic is left untouched.
// CHECK-LABEL: tt.func @no_identity_two_values(
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
// CHECK-NOT: arith.andi
tt.func @no_identity_two_values(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>, %mask: tensor<4xi1>, %cond0: tensor<4xi1>, %cond1: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %s1 = arith.select %cond1, %y, %z : tensor<4xi1>, tensor<4xf32>
  %s0 = arith.select %cond0, %x, %s1 : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %s0, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  tt.return
}

// -----

// If the returned old value is used, masked-off lanes would read as undefined,
// so the atomic must not be masked.
// CHECK-LABEL: tt.func @result_used(
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
// CHECK-NOT: arith.andi
tt.func @result_used(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %s = arith.select %cond, %x, %c0 : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  tt.store %ptrs, %old, %mask : tensor<4x!tt.ptr<f32>>
  tt.return
}

// -----

// `and` identity is all-ones for the element type.
// CHECK-LABEL: tt.func @tensor_select_and_all_ones(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw and, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_and_all_ones(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<-1> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw and, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// `or` identity is zero.
// CHECK-LABEL: tt.func @tensor_select_or_zero(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw or, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_or_zero(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<0> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw or, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// `xor` identity is zero.
// CHECK-LABEL: tt.func @tensor_select_xor_zero(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw xor, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_xor_zero(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<0> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw xor, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Signed `max` identity is INT_MIN.
// CHECK-LABEL: tt.func @tensor_select_max_signed_min(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw max, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_max_signed_min(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<-2147483648> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw max, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Signed `min` identity is INT_MAX.
// CHECK-LABEL: tt.func @tensor_select_min_signed_max(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw min, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_min_signed_max(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<2147483647> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw min, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Unsigned `umax` identity is zero.
// CHECK-LABEL: tt.func @tensor_select_umax_zero(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw umax, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_umax_zero(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<0> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw umax, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Unsigned `umin` identity is all-ones.
// CHECK-LABEL: tt.func @tensor_select_umin_all_ones(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw umin, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_umin_all_ones(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<-1> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw umin, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Scalar `umax` chain exercises the scalar identity path.
// CHECK-LABEL: tt.func @scalar_select_umax(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[A:[^:]*]]: i32, %{{[^:]*}}: i1, %{{[^:]*}}: i1)
// CHECK: tt.atomic_rmw umax, relaxed, gpu, %{{.*}}, %[[A]], %{{.*}} : (!tt.ptr<i32>, i32, i1) -> i32
tt.func @scalar_select_umax(%ptr: !tt.ptr<i32>, %a: i32, %cond1: i1, %cond2: i1) {
  %c0 = arith.constant 0 : i32
  %s1 = arith.select %cond1, %a, %c0 : i32
  %s2 = arith.select %cond2, %s1, %c0 : i32
  %old = tt.atomic_rmw umax, relaxed, gpu, %ptr, %s2 : (!tt.ptr<i32>, i32) -> i32
  tt.return
}

// -----

// A constant identity value makes `min` an unconditional no-op.
// CHECK-LABEL: tt.func @constant_min_identity_erased(
// CHECK-NOT: tt.atomic_rmw
// CHECK: tt.return
tt.func @constant_min_identity_erased(%ptr: !tt.ptr<i32>, %mask: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<2147483647> : tensor<4xi32>
  %old = tt.atomic_rmw min, relaxed, gpu, %ptrs, %c, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Zero is not the identity of `max`, so the atomic must be left untouched.
// CHECK-LABEL: tt.func @non_identity_max_not_masked(
// CHECK-NOT: arith.andi
// CHECK: tt.atomic_rmw max, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @non_identity_max_not_masked(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<0> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw max, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Zero is not the identity of `and`, so the atomic must be left untouched.
// CHECK-LABEL: tt.func @non_identity_and_not_masked(
// CHECK-NOT: arith.andi
// CHECK: tt.atomic_rmw and, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @non_identity_and_not_masked(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<0> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw and, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Floating-point `max` identity is -inf.
// CHECK-LABEL: tt.func @tensor_select_fmax_neg_inf(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<f32>, %[[X:[^:]*]]: tensor<4xf32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw max, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
tt.func @tensor_select_fmax_neg_inf(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c = arith.constant dense<0xFF800000> : tensor<4xf32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw max, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  tt.return
}

// -----

// Floating-point `min` identity is +inf.
// CHECK-LABEL: tt.func @tensor_select_fmin_pos_inf(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<f32>, %[[X:[^:]*]]: tensor<4xf32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw min, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
tt.func @tensor_select_fmin_pos_inf(%ptr: !tt.ptr<f32>, %x: tensor<4xf32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %c = arith.constant dense<0x7F800000> : tensor<4xf32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xf32>
  %old = tt.atomic_rmw min, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  tt.return
}

// -----

// `and` with the identity on the true arm exercises condition inversion for a
// non-add reduction.
// CHECK-LABEL: tt.func @reverse_select_and(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw and, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @reverse_select_and(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<-1> : tensor<4xi32>
  %s = arith.select %cond, %c, %x : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw and, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}

// -----

// Nested `max` select chain without an original mask; INT_MIN is the identity.
// CHECK-LABEL: tt.func @tensor_nested_select_max(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %{{[^:]*}}: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[P1:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[P2:.*]] = arith.select %{{.*}}, %[[P1]], %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: tt.atomic_rmw max, relaxed, gpu, %{{.*}}, %[[X]], %[[P2]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_nested_select_max(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %cond1: tensor<4xi1>, %cond2: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<-2147483648> : tensor<4xi32>
  %s1 = arith.select %cond1, %x, %c : tensor<4xi1>, tensor<4xi32>
  %s2 = arith.select %cond2, %s1, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw max, relaxed, gpu, %ptrs, %s2 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return
}

// -----

// `umin` without an original mask; all-ones is the identity and the rebuilt
// predicate becomes the whole mask.
// CHECK-LABEL: tt.func @tensor_select_umin_no_mask(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: tt.atomic_rmw umin, relaxed, gpu, %{{.*}}, %[[X]], %[[PRED]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_umin_no_mask(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c = arith.constant dense<-1> : tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw umin, relaxed, gpu, %ptrs, %s : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  tt.return
}

// -----

// A scalar zero hidden behind `tt.splat` is still the `umax` identity.
// CHECK-LABEL: tt.func @tensor_select_umax_splat(
// CHECK-SAME: %{{[^:]*}}: !tt.ptr<i32>, %[[X:[^:]*]]: tensor<4xi32>, %[[MASK:[^:]*]]: tensor<4xi1>, %{{[^:]*}}: tensor<4xi1>)
// CHECK: %[[PRED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi1>
// CHECK: %[[NEWMASK:.*]] = arith.andi %[[MASK]], %[[PRED]] : tensor<4xi1>
// CHECK: tt.atomic_rmw umax, relaxed, gpu, %{{.*}}, %[[X]], %[[NEWMASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @tensor_select_umax_splat(%ptr: !tt.ptr<i32>, %x: tensor<4xi32>, %mask: tensor<4xi1>, %cond: tensor<4xi1>) {
  %offs = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %sptr = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %ptrs = tt.addptr %sptr, %offs : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %c0 = arith.constant 0 : i32
  %c = tt.splat %c0 : i32 -> tensor<4xi32>
  %s = arith.select %cond, %x, %c : tensor<4xi1>, tensor<4xi32>
  %old = tt.atomic_rmw umax, relaxed, gpu, %ptrs, %s, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  tt.return
}
