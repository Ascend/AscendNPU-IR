// RUN: bishengir-opt %s --optimize-simt-expressions | FileCheck %s

// CHECK-LABEL: tt.func @pair_axis_last
// CHECK-NOT: "tt.reduce"
// CHECK: tt.split
// CHECK: arith.addf
// CHECK: tt.join
// CHECK-NOT: "tt.reduce"
// CHECK: tt.return
tt.func @pair_axis_last(%x: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<4x2xf32>) -> tensor<4xf32>
  %e = tt.expand_dims %r {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
  %b = tt.broadcast %e : tensor<4x1xf32> -> tensor<4x2xf32>
  %out = arith.addf %b, %x : tensor<4x2xf32>
  tt.return %out : tensor<4x2xf32>
}

// CHECK-LABEL: tt.func @pair_axis_first
// CHECK-NOT: "tt.reduce"
// CHECK: tt.split
// CHECK: arith.addf
// CHECK: tt.join
// CHECK-NOT: "tt.reduce"
// CHECK: tt.return
tt.func @pair_axis_first(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<2x4xf32>) -> tensor<4xf32>
  %e = tt.expand_dims %r {axis = 0 : i32} : tensor<4xf32> -> tensor<1x4xf32>
  %b = tt.broadcast %e : tensor<1x4xf32> -> tensor<2x4xf32>
  %out = arith.addf %b, %x : tensor<2x4xf32>
  tt.return %out : tensor<2x4xf32>
}

// CHECK-LABEL: tt.func @pair_axis_middle
// CHECK-NOT: "tt.reduce"
// CHECK: tt.split
// CHECK: arith.addf
// CHECK: tt.join
// CHECK-NOT: "tt.reduce"
// CHECK: tt.return
tt.func @pair_axis_middle(%x: tensor<2x2x4xf32>) -> tensor<2x2x4xf32> {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<2x2x4xf32>) -> tensor<2x4xf32>
  %e = tt.expand_dims %r {axis = 1 : i32} : tensor<2x4xf32> -> tensor<2x1x4xf32>
  %b = tt.broadcast %e : tensor<2x1x4xf32> -> tensor<2x2x4xf32>
  %out = arith.addf %b, %x : tensor<2x2x4xf32>
  tt.return %out : tensor<2x2x4xf32>
}

// Multiplication by zero must remain: NaN/Inf and signed zero are observable.
// CHECK-LABEL: tt.func @preserve_float_zero_product
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tt.return
tt.func @preserve_float_zero_product(%x: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %mask = arith.constant dense<[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]> : tensor<4x2xf32>
  %m = arith.mulf %x, %mask : tensor<4x2xf32>
  %r = "tt.reduce"(%m) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<4x2xf32>) -> tensor<4xf32>
  %e = tt.expand_dims %r {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
  %b = tt.broadcast %e : tensor<4x1xf32> -> tensor<4x2xf32>
  %out = arith.addf %b, %x : tensor<4x2xf32>
  tt.return %out : tensor<4x2xf32>
}

// A mixed four-element reduction without shared pointwise work stays opaque.
// CHECK-LABEL: tt.func @non_pair
// CHECK: "tt.reduce"
// CHECK-NOT: tt.split
// CHECK: tt.return
tt.func @non_pair(%x: tensor<2x4xi32>) -> tensor<2x4xi32> {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %s = arith.addi %a, %b : i32
    tt.reduce.return %s : i32
  }) : (tensor<2x4xi32>) -> tensor<2xi32>
  %e = tt.expand_dims %r {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %b = tt.broadcast %e : tensor<2x1xi32> -> tensor<2x4xi32>
  %out = arith.addi %b, %x : tensor<2x4xi32>
  tt.return %out : tensor<2x4xi32>
}

// CHECK-LABEL: tt.func @conditional_xor_0(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi32>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi32>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[X]], %[[Y]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_0(%c: tensor<8xi1>, %x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %zero = arith.constant dense<0> : tensor<8xi32>
  %delta = arith.xori %y, %x : tensor<8xi32>
  %update = arith.select %c, %zero, %delta : tensor<8xi1>, tensor<8xi32>
  %out = arith.xori %update, %x : tensor<8xi32>
  tt.return %out : tensor<8xi32>
}

// CHECK-LABEL: tt.func @conditional_xor_1(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi64>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi64>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[X]], %[[Y]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_1(%c: tensor<8xi1>, %x: tensor<8xi64>, %y: tensor<8xi64>) -> tensor<8xi64> {
  %zero = arith.constant dense<0> : tensor<8xi64>
  %delta = arith.xori %x, %y : tensor<8xi64>
  %update = arith.select %c, %zero, %delta : tensor<8xi1>, tensor<8xi64>
  %out = arith.xori %update, %x : tensor<8xi64>
  tt.return %out : tensor<8xi64>
}

// CHECK-LABEL: tt.func @conditional_xor_2(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi32>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi32>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[Y]], %[[X]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_2(%c: tensor<8xi1>, %x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %zero = arith.constant dense<0> : tensor<8xi32>
  %delta = arith.xori %y, %x : tensor<8xi32>
  %update = arith.select %c, %delta, %zero : tensor<8xi1>, tensor<8xi32>
  %out = arith.xori %update, %x : tensor<8xi32>
  tt.return %out : tensor<8xi32>
}

// CHECK-LABEL: tt.func @conditional_xor_3(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi64>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi64>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[Y]], %[[X]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_3(%c: tensor<8xi1>, %x: tensor<8xi64>, %y: tensor<8xi64>) -> tensor<8xi64> {
  %zero = arith.constant dense<0> : tensor<8xi64>
  %delta = arith.xori %x, %y : tensor<8xi64>
  %update = arith.select %c, %delta, %zero : tensor<8xi1>, tensor<8xi64>
  %out = arith.xori %update, %x : tensor<8xi64>
  tt.return %out : tensor<8xi64>
}

// CHECK-LABEL: tt.func @conditional_xor_4(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi32>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi32>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[X]], %[[Y]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_4(%c: tensor<8xi1>, %x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %zero = arith.constant dense<0> : tensor<8xi32>
  %delta = arith.xori %y, %x : tensor<8xi32>
  %update = arith.select %c, %zero, %delta : tensor<8xi1>, tensor<8xi32>
  %out = arith.xori %x, %update : tensor<8xi32>
  tt.return %out : tensor<8xi32>
}

// CHECK-LABEL: tt.func @conditional_xor_5(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi64>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi64>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[X]], %[[Y]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_5(%c: tensor<8xi1>, %x: tensor<8xi64>, %y: tensor<8xi64>) -> tensor<8xi64> {
  %zero = arith.constant dense<0> : tensor<8xi64>
  %delta = arith.xori %x, %y : tensor<8xi64>
  %update = arith.select %c, %zero, %delta : tensor<8xi1>, tensor<8xi64>
  %out = arith.xori %x, %update : tensor<8xi64>
  tt.return %out : tensor<8xi64>
}

// CHECK-LABEL: tt.func @conditional_xor_6(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi32>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi32>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[Y]], %[[X]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_6(%c: tensor<8xi1>, %x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %zero = arith.constant dense<0> : tensor<8xi32>
  %delta = arith.xori %y, %x : tensor<8xi32>
  %update = arith.select %c, %delta, %zero : tensor<8xi1>, tensor<8xi32>
  %out = arith.xori %x, %update : tensor<8xi32>
  tt.return %out : tensor<8xi32>
}

// CHECK-LABEL: tt.func @conditional_xor_7(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: tensor<8xi1>, %[[X:[a-zA-Z0-9_]+]]: tensor<8xi64>, %[[Y:[a-zA-Z0-9_]+]]: tensor<8xi64>
// CHECK-NOT: arith.xori
// CHECK: arith.select %[[C]], %[[Y]], %[[X]]
// CHECK-NOT: arith.xori
// CHECK: tt.return
tt.func @conditional_xor_7(%c: tensor<8xi1>, %x: tensor<8xi64>, %y: tensor<8xi64>) -> tensor<8xi64> {
  %zero = arith.constant dense<0> : tensor<8xi64>
  %delta = arith.xori %x, %y : tensor<8xi64>
  %update = arith.select %c, %delta, %zero : tensor<8xi1>, tensor<8xi64>
  %out = arith.xori %x, %update : tensor<8xi64>
  tt.return %out : tensor<8xi64>
}

// A nonzero inactive branch cannot use conditional XOR cancellation.
// CHECK-LABEL: tt.func @nonzero_inactive
// CHECK: arith.xori
// CHECK: arith.select
// CHECK: arith.xori
// CHECK: tt.return
tt.func @nonzero_inactive(%c: tensor<8xi1>, %x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %one = arith.constant dense<1> : tensor<8xi32>
  %delta = arith.xori %x, %y : tensor<8xi32>
  %update = arith.select %c, %delta, %one : tensor<8xi1>, tensor<8xi32>
  %out = arith.xori %x, %update : tensor<8xi32>
  tt.return %out : tensor<8xi32>
}

// Ordinary broadcasts need neither a reduction nor a compare-and-swap.
// The uniform path supports the same factors as mixed partitioning.
// CHECK-LABEL: tt.func @uniform_factor_four
// CHECK-NOT: tt.split
// CHECK: arith.addf {{.*}} : tensor<8xf32>
// CHECK: arith.mulf {{.*}} : tensor<8xf32>
// CHECK: tt.broadcast {{.*}} -> tensor<2x4x4xf32>
// CHECK: tt.return
tt.func @uniform_factor_four(%a: tensor<2x1x4xf32>, %b: tensor<2x1x4xf32>) -> tensor<2x4x4xf32> {
  %a_full = tt.broadcast %a : tensor<2x1x4xf32> -> tensor<2x4x4xf32>
  %b_full = tt.broadcast %b : tensor<2x1x4xf32> -> tensor<2x4x4xf32>
  %sum = arith.addf %a_full, %b_full : tensor<2x4x4xf32>
  %out = arith.mulf %sum, %a_full : tensor<2x4x4xf32>
  tt.return %out : tensor<2x4x4xf32>
}

// Unsupported factors remain unchanged, including uniform expressions.
// CHECK-LABEL: tt.func @uniform_factor_three
// CHECK-NOT: tt.split
// CHECK: tt.broadcast {{.*}} -> tensor<2x3x4xf32>
// CHECK: arith.addf {{.*}} : tensor<2x3x4xf32>
// CHECK: arith.mulf {{.*}} : tensor<2x3x4xf32>
// CHECK-NOT: tt.join
// CHECK: tt.return
tt.func @uniform_factor_three(%a: tensor<2x1x4xf32>, %b: tensor<2x1x4xf32>) -> tensor<2x3x4xf32> {
  %a_full = tt.broadcast %a : tensor<2x1x4xf32> -> tensor<2x3x4xf32>
  %b_full = tt.broadcast %b : tensor<2x1x4xf32> -> tensor<2x3x4xf32>
  %sum = arith.addf %a_full, %b_full : tensor<2x3x4xf32>
  %out = arith.mulf %sum, %a_full : tensor<2x3x4xf32>
  tt.return %out : tensor<2x3x4xf32>
}

// Unsupported factors remain unchanged, including uniform expressions.
// CHECK-LABEL: tt.func @uniform_factor_eight
// CHECK-NOT: tt.split
// CHECK: tt.broadcast {{.*}} -> tensor<2x8x4xf32>
// CHECK: arith.addf {{.*}} : tensor<2x8x4xf32>
// CHECK: arith.mulf {{.*}} : tensor<2x8x4xf32>
// CHECK-NOT: tt.join
// CHECK: tt.return
tt.func @uniform_factor_eight(%a: tensor<2x1x4xf32>, %b: tensor<2x1x4xf32>) -> tensor<2x8x4xf32> {
  %a_full = tt.broadcast %a : tensor<2x1x4xf32> -> tensor<2x8x4xf32>
  %b_full = tt.broadcast %b : tensor<2x1x4xf32> -> tensor<2x8x4xf32>
  %sum = arith.addf %a_full, %b_full : tensor<2x8x4xf32>
  %out = arith.mulf %sum, %a_full : tensor<2x8x4xf32>
  tt.return %out : tensor<2x8x4xf32>
}

// A reduction of more than two terms remains intact, while the following
// uniform elementwise work may still move before its broadcast.
// CHECK-LABEL: tt.func @opaque_reduction
// CHECK: "tt.reduce"
// CHECK-NOT: tt.split
// CHECK: arith.mulf {{.*}} : tensor<2xf32>
// CHECK: tt.broadcast {{.*}} -> tensor<2x4xf32>
// CHECK: tt.return
tt.func @opaque_reduction(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<2x4xf32>) -> tensor<2xf32>
  %e = tt.expand_dims %r {axis = 1 : i32} : tensor<2xf32> -> tensor<2x1xf32>
  %b = tt.broadcast %e : tensor<2x1xf32> -> tensor<2x4xf32>
  %out = arith.mulf %b, %b : tensor<2x4xf32>
  tt.return %out : tensor<2x4xf32>
}

// Sharing a broadcast expression is useful even with unrelated full-size data.
// CHECK-LABEL: tt.func @mixed_factor_four
// CHECK: arith.muli {{.*}} : tensor<4xi32>
// CHECK: tt.split
// CHECK: arith.addi {{.*}} : tensor<4xi32>
// CHECK: tt.join
// CHECK: tt.return
tt.func @mixed_factor_four(%a: tensor<2x1x2xi32>, %b: tensor<2x1x2xi32>, %x: tensor<2x4x2xi32>) -> tensor<2x4x2xi32> {
  %a_full = tt.broadcast %a : tensor<2x1x2xi32> -> tensor<2x4x2xi32>
  %b_full = tt.broadcast %b : tensor<2x1x2xi32> -> tensor<2x4x2xi32>
  %product = arith.muli %a_full, %b_full : tensor<2x4x2xi32>
  %out = arith.addi %product, %x : tensor<2x4x2xi32>
  tt.return %out : tensor<2x4x2xi32>
}

// Large mixed replication must not expand code just to share one operation.
// CHECK-LABEL: tt.func @mixed_factor_sixteen
// CHECK-NOT: tt.split
// CHECK: arith.muli {{.*}} : tensor<2x16xi32>
// CHECK: arith.addi {{.*}} : tensor<2x16xi32>
// CHECK-NOT: tt.join
// CHECK: tt.return
tt.func @mixed_factor_sixteen(%a: tensor<2x1xi32>, %b: tensor<2x1xi32>, %x: tensor<2x16xi32>) -> tensor<2x16xi32> {
  %a_full = tt.broadcast %a : tensor<2x1xi32> -> tensor<2x16xi32>
  %b_full = tt.broadcast %b : tensor<2x1xi32> -> tensor<2x16xi32>
  %product = arith.muli %a_full, %b_full : tensor<2x16xi32>
  %out = arith.addi %product, %x : tensor<2x16xi32>
  tt.return %out : tensor<2x16xi32>
}

// Large mixed replication must not expand code just to share one operation.
// CHECK-LABEL: tt.func @mixed_factor_eight
// CHECK-NOT: tt.split
// CHECK: arith.muli {{.*}} : tensor<2x8xi32>
// CHECK: arith.addi {{.*}} : tensor<2x8xi32>
// CHECK-NOT: tt.join
// CHECK: tt.return
tt.func @mixed_factor_eight(%a: tensor<2x1xi32>, %b: tensor<2x1xi32>, %x: tensor<2x8xi32>) -> tensor<2x8xi32> {
  %a_full = tt.broadcast %a : tensor<2x1xi32> -> tensor<2x8xi32>
  %b_full = tt.broadcast %b : tensor<2x1xi32> -> tensor<2x8xi32>
  %product = arith.muli %a_full, %b_full : tensor<2x8xi32>
  %out = arith.addi %product, %x : tensor<2x8xi32>
  tt.return %out : tensor<2x8xi32>
}

// More than one expanded axis is not modeled by the single-axis plan.
// CHECK-LABEL: tt.func @multiple_axes
// CHECK: tt.broadcast
// CHECK: arith.addf {{.*}} : tensor<2x4xf32>
// CHECK-NOT: tt.split
// CHECK: tt.return
tt.func @multiple_axes(%a: tensor<1x1xf32>) -> tensor<2x4xf32> {
  %full = tt.broadcast %a : tensor<1x1xf32> -> tensor<2x4xf32>
  %out = arith.addf %full, %full : tensor<2x4xf32>
  tt.return %out : tensor<2x4xf32>
}

// An unprofitable first broadcast must not hide a useful later direction.
// CHECK-LABEL: tt.func @later_broadcast_direction
// CHECK: tt.split
// CHECK: arith.mulf {{.*}} : tensor<16xf32>
// CHECK: arith.addf {{.*}} : tensor<16xf32>
// CHECK: tt.join
// CHECK: tt.return
tt.func @later_broadcast_direction(%c: tensor<1x2xf32>, %a: tensor<16x1xf32>, %b: tensor<16x1xf32>) -> tensor<16x2xf32> {
  %c_full = tt.broadcast %c : tensor<1x2xf32> -> tensor<16x2xf32>
  %a_full = tt.broadcast %a : tensor<16x1xf32> -> tensor<16x2xf32>
  %b_full = tt.broadcast %b : tensor<16x1xf32> -> tensor<16x2xf32>
  %product = arith.mulf %a_full, %b_full : tensor<16x2xf32>
  %out = arith.addf %c_full, %product : tensor<16x2xf32>
  tt.return %out : tensor<16x2xf32>
}

// A rejected candidate must discard every operation in its detached block.
// CHECK-LABEL: tt.func @no_shared_work
// CHECK-NOT: tt.split
// CHECK: arith.addi {{.*}} : tensor<2x2xi32>
// CHECK-NOT: tt.join
// CHECK: tt.return
tt.func @no_shared_work(%a: tensor<2x1xi32>, %x: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %full = tt.broadcast %a : tensor<2x1xi32> -> tensor<2x2xi32>
  %out = arith.addi %full, %x : tensor<2x2xi32>
  tt.return %out : tensor<2x2xi32>
}

// The shared product cannot justify expanding an oversized producer tree.
// CHECK-LABEL: tt.func @reject_oversized_candidate
// CHECK-NOT: tt.split
// CHECK: arith.mulf {{.*}} : tensor<2x4xf32>
// CHECK-NOT: tt.split
// CHECK-NOT: tt.join
// CHECK: tt.return
tt.func @reject_oversized_candidate(
    %a: tensor<2x1xf32>, %b: tensor<2x1xf32>,
    %x0: tensor<2x4xf32>,
    %x1: tensor<2x4xf32>,
    %x2: tensor<2x4xf32>,
    %x3: tensor<2x4xf32>,
    %x4: tensor<2x4xf32>,
    %x5: tensor<2x4xf32>,
    %x6: tensor<2x4xf32>,
    %x7: tensor<2x4xf32>,
    %x8: tensor<2x4xf32>,
    %x9: tensor<2x4xf32>,
    %x10: tensor<2x4xf32>,
    %x11: tensor<2x4xf32>,
    %x12: tensor<2x4xf32>,
    %x13: tensor<2x4xf32>,
    %x14: tensor<2x4xf32>,
    %x15: tensor<2x4xf32>
) -> tensor<2x4xf32> {
  %af = tt.broadcast %a : tensor<2x1xf32> -> tensor<2x4xf32>
  %bf = tt.broadcast %b : tensor<2x1xf32> -> tensor<2x4xf32>
  %shared = arith.mulf %af, %bf : tensor<2x4xf32>
  %sum0 = arith.addf %x0, %x1 : tensor<2x4xf32>
  %sum1 = arith.addf %x2, %x3 : tensor<2x4xf32>
  %sum2 = arith.addf %x4, %x5 : tensor<2x4xf32>
  %sum3 = arith.addf %x6, %x7 : tensor<2x4xf32>
  %sum4 = arith.addf %x8, %x9 : tensor<2x4xf32>
  %sum5 = arith.addf %x10, %x11 : tensor<2x4xf32>
  %sum6 = arith.addf %x12, %x13 : tensor<2x4xf32>
  %sum7 = arith.addf %x14, %x15 : tensor<2x4xf32>
  %sum8 = arith.addf %sum0, %sum1 : tensor<2x4xf32>
  %sum9 = arith.addf %sum2, %sum3 : tensor<2x4xf32>
  %sum10 = arith.addf %sum4, %sum5 : tensor<2x4xf32>
  %sum11 = arith.addf %sum6, %sum7 : tensor<2x4xf32>
  %sum12 = arith.addf %sum8, %sum9 : tensor<2x4xf32>
  %sum13 = arith.addf %sum10, %sum11 : tensor<2x4xf32>
  %sum14 = arith.addf %sum12, %sum13 : tensor<2x4xf32>
  %out = arith.addf %shared, %sum14 : tensor<2x4xf32>
  tt.return %out : tensor<2x4xf32>
}
