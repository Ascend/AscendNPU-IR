// RUN: bishengir-opt %s -hivm-normalize-matmul -split-input-file | FileCheck %s

// ============================================================================
// scf.for consecutive with scf.for
// ============================================================================
// CHECK-LABEL: func.func @ccf_for_for
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: arith.andi
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vbrc
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_for_for(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    %r2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %r1) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.for consecutive with scf.while
// ============================================================================
// CHECK-LABEL: func.func @ccf_for_while
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: arith.andi
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vbrc
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_for_while(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    %r2:2 = scf.while (%acc = %r1, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    return %r2#0 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.for consecutive with scf.if
// ============================================================================
// CHECK-LABEL: func.func @ccf_for_if
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: arith.andi
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vbrc
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_for_if(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    %r2 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %r1 : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.for consecutive with bare mmad
// ============================================================================
// CHECK-LABEL: func.func @ccf_for_mmad
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: memref.load
// CHECK: arith.cmpi
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: fallback_not_exec
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_for_mmad(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    %r2 = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1 : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.while consecutive with scf.for
// ============================================================================
// CHECK-LABEL: func.func @ccf_while_for
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: arith.andi
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vbrc
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_while_for(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1:2 = scf.while (%acc = %acc0, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    %r2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %r1#0) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.while consecutive with scf.while
// ============================================================================
// CHECK-LABEL: func.func @ccf_while_while
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: arith.andi
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vbrc
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_while_while(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1:2 = scf.while (%acc = %acc0, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    %r2:2 = scf.while (%acc = %r1#0, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    return %r2#0 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.while consecutive with scf.if
// ============================================================================
// CHECK-LABEL: func.func @ccf_while_if
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: arith.andi
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vbrc
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_while_if(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1:2 = scf.while (%acc = %acc0, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    %r2 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1#0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %r1#0 : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.while consecutive with bare mmad
// ============================================================================
// CHECK-LABEL: func.func @ccf_while_mmad
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: hivm.hir.mmadL1
// CHECK: memref.load
// CHECK: arith.cmpi
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: fallback_not_exec
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_while_mmad(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1:2 = scf.while (%acc = %acc0, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    %r2 = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1#0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.if consecutive with scf.for
// ============================================================================
// CHECK-LABEL: func.func @ccf_if_for
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vadd
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_if_for(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %acc0 : tensor<64x32xf32>
    }
    %r2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %r1) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.if consecutive with scf.while
// ============================================================================
// CHECK-LABEL: func.func @ccf_if_while
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vadd
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_if_while(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %acc0 : tensor<64x32xf32>
    }
    %r2:2 = scf.while (%acc = %r1, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
    } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    return %r2#0 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.if consecutive with scf.if
// ============================================================================
// CHECK-LABEL: func.func @ccf_if_if
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vadd
// CHECK: fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_if_if(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %acc0 : tensor<64x32xf32>
    }
    %r2 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %r1 : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// scf.if consecutive with bare mmad
// ============================================================================
// CHECK-LABEL: func.func @ccf_if_mmad
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.vadd
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_if_mmad(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %acc0 : tensor<64x32xf32>
    }
    %r2 = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1 : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// bare mmad consecutive with scf.for
// ============================================================================
// CHECK-LABEL: func.func @ccf_mmad_for
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: fallback_not_exec
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_mmad_for(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %r1) -> (tensor<64x32xf32>) : i32 {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// bare mmad consecutive with scf.while
// ============================================================================
// CHECK-LABEL: func.func @ccf_mmad_while
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.while
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: fallback_not_exec
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_mmad_while(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r2:2 = scf.while (%acc = %r1, %i = %c0_i32) : (tensor<64x32xf32>, i32) -> (tensor<64x32xf32>, i32) {
      %cnd = arith.cmpi slt, %i, %n : i32
      scf.condition(%cnd) %acc, %i : tensor<64x32xf32>, i32
  } do {
    ^bb0(%acc: tensor<64x32xf32>, %i: i32):
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      %in = arith.addi %i, %c1_i32 : i32
      scf.yield %mm, %in : tensor<64x32xf32>, i32
    }
    return %r2#0 : tensor<64x32xf32>
  }
}

// ============================================================================
// bare mmad consecutive with scf.if
// ============================================================================
// CHECK-LABEL: func.func @ccf_mmad_if
// CHECK: hivm.hir.mmadL1
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: fallback_not_exec
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_mmad_if(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r2 = scf.if %cond -> (tensor<64x32xf32>) {
      %mm = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mm : tensor<64x32xf32>
    } else {
      scf.yield %r1 : tensor<64x32xf32>
    }
    return %r2 : tensor<64x32xf32>
  }
}

// ============================================================================
// bare mmad consecutive with bare mmad
// ============================================================================
// CHECK-LABEL: func.func @ccf_mmad_mmad
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: memref.alloca() {normalize_matmul_counter
// CHECK-NOT: memref.alloca() {normalize_matmul_counter
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: fallback_not_exec
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @ccf_mmad_mmad(%a: tensor<64x32xf16>, %b1: tensor<32x32xf16>, %b2: tensor<32x32xf16>, %cond: i1, %n: i32) -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<64x32xf32>
    %acc0 = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r1 = hivm.hir.mmadL1 ins(%a, %b1, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%acc0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %r2 = hivm.hir.mmadL1 ins(%a, %b2, %false, %c16, %c16, %c16 : tensor<64x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%r1 : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %r2 : tensor<64x32xf32>
  }
}
