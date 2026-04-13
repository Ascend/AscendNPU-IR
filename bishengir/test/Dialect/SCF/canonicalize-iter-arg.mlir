// RUN: bishengir-opt -scf-canonicalize-iter-arg -allow-unregistered-dialect -split-input-file %s | FileCheck %s

func.func @test() -> (tensor<?xi8>, tensor<?xi8>) {
  %size = "some_op"() : () -> index
  %e = tensor.empty(%size) : tensor<?xi8>
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %ub = arith.constant 16 : index
  %cond = "some_op"() : () -> i1
  // CHECK: iter_args(
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]] = [[INIT:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]] = [[INIT]]
  %res:2 = scf.for %i = %lb to %ub  step %step iter_args(%arg0 = %e, %arg1 = %e) -> (tensor<?xi8>, tensor<?xi8>) {
    // CHECK: iter_args(
    // CHECK-SAME: = [[INIT]]
    %inner = scf.for %j = %lb to %ub step %step iter_args(%iarg = %arg0) -> tensor<?xi8> {
      "some_op"(): ()-> ()
      scf.yield %iarg : tensor<?xi8>
    }
    // CHECK: scf.if
    %inner2 = scf.if %cond -> tensor<?xi8> {
      "some_op"(): ()-> ()
      // CHECK: yield [[INIT]]
      scf.yield %e : tensor<?xi8>
    } else {
      // CHECK: yield [[INIT]]
      scf.yield %arg1 : tensor<?xi8>
    }

    // CHECK: yield [[INIT]], [[INIT]]
    scf.yield %inner, %inner2 : tensor<?xi8>, tensor<?xi8>
  }
  // CHECK: return [[INIT]], [[INIT]]
  return %res#0, %res#1 : tensor<?xi8>, tensor<?xi8>
}

// -----
func.func @remove_two_iter_args(
    %arg0: index, %arg1: index,
    %arg2: tensor<32xi64>, %arg3: tensor<32xi32>) -> (index, index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %lb = arith.constant 0 : index
        %ub = arith.addi %arg0, %c32 : index
        %step = arith.constant 1 : index
        %res:4 = scf.for %iv = %lb to %ub step %step
        iter_args(%it0 = %arg2, %it1 = %arg3, %it2 = %arg0, %it3 = %c0)
        -> (tensor<32xi64>, tensor<32xi32>, index, index) {
        %y0 = arith.addi %it0, %it0 : tensor<32xi64>
        %y1 = arith.addi %it1, %it1 : tensor<32xi32>

        %y2 = arith.addi %it2, %c32 : index
        %y3 = arith.addi %it3, %c32 : index
        scf.yield %y0, %y1, %y2, %y3
        : tensor<32xi64>, tensor<32xi32>, index, index
        }
        // CHECK-LABEL: func.func @remove_two_iter_args
        // CHECK: iter_args(%[[IT2:.*]] = %arg0, %[[IT3:.*]] = %c0) -> (index, index) {
        // CHECK-NOT: arith.addi %{{.*}} : tensor<32xi64>
        // CHECK-NOT: arith.addi %{{.*}} : tensor<32xi32>
        // CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
        // CHECK: return
        return %res#2, %res#3 : index, index
}

// -----
func.func @remove_only_one_iter_args(
    %arg0: index, %arg1: index,
    %arg2: tensor<32xi64>, %arg3: tensor<32xi32>) -> (index, index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %lb = arith.constant 0 : index
        %ub = arith.addi %arg0, %c32 : index
        %step = arith.constant 1 : index
        %res:4 = scf.for %iv = %lb to %ub step %step
        iter_args(%it0 = %arg2, %it1 = %arg3, %it2 = %arg0, %it3 = %c0)
        -> (tensor<32xi64>, tensor<32xi32>, index, index) {
        %y0 = arith.addi %it0, %it0 : tensor<32xi64>
        %y1 = arith.addi %it1, %it1 : tensor<32xi32>
        // store has memory effect so this iter arg should not be removed
        %stored = memref.alloc() : memref<32xi32>
        hivm.hir.store ins(%y1 : tensor<32xi32>) outs(%stored: memref<32xi32>)
        %y2 = arith.addi %it2, %c32 : index
        %y3 = arith.addi %it3, %c32 : index
        scf.yield %y0, %y1, %y2, %y3
        : tensor<32xi64>, tensor<32xi32>, index, index
        }
        // CHECK-LABEL: func.func @remove_only_one_iter_args
        // CHECK: iter_args(%[[IT1:.*]] = %arg3, %[[IT2:.*]] = %arg0, %[[IT3:.*]] = %c0) -> (tensor<32xi32>, index, index) {
        // CHECK-NOT: arith.addi %{{.*}} : tensor<32xi64>
        // CHECK: arith.addi %{{.*}} : tensor<32xi32>
        // CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}} : tensor<32xi32>, index, index
        // CHECK: return
        return %res#2, %res#3 : index, index
}

// -----
func.func @induction_var_yield_iter_arg() -> index {
  %c0  = arith.constant 0  : index
  %c10 = arith.constant 10 : index
  %c1  = arith.constant 1  : index

  %r = scf.for %iv = %c0 to %c10 step %c1 iter_args(%x = %c0) -> (index) {
    // The yielded value for an iter_arg is the induction variable (block arg #0)
    scf.yield %iv : index
  }

  return %r : index
}

// -----
// CHECK-LABEL: func.func @test_simplify_for_op
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK-SAME:    %[[COND:[a-zA-Z0-9_]+]]: i1
func.func @test_simplify_for_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %cond: i1) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %c0  = arith.constant 0  : index
  %c1  = arith.constant 1  : index
  %c10 = arith.constant 10 : index
  // CHECK: %[[RESULT:.+]]:3 = scf.for
  %result:3 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%iter0 = %arg0, %iter1 = %arg1, %iter2 = %arg0) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    // CHECK: %[[INNER:.+]] = tensor.empty
    %inner = tensor.empty() : tensor<4xf32>
    // CHECK: %[[NESTED:.+]] = scf.if
    %nested = scf.if %cond -> tensor<4xf32> {
      %t = tensor.empty() : tensor<4xf32>
      scf.yield %t : tensor<4xf32>
    } else {
      %t = tensor.empty() : tensor<4xf32>
      scf.yield %t : tensor<4xf32>
    }
    // CHECK: scf.yield %[[ARG0]], %[[INNER]], %[[NESTED]]
    scf.yield %arg0, %inner, %nested : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
  // CHECK: return %[[ARG0]], %[[RESULT]]#1, %[[RESULT]]#2
  func.return %result#0, %result#1, %result#2: tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @remove_dead_select_iter_arg
// CHECK: scf.for
// CHECK-SAME: iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK-NOT: tensor<32xi32>
// CHECK: scf.yield {{.*}} : i32
func.func @remove_dead_select_iter_arg(%ub: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %empty = tensor.empty() : tensor<32xi32>
  %res:2 = scf.for %iv = %c0 to %ub step %c1
      iter_args(%a = %c0, %b = %empty) -> (i32, tensor<32xi32>) : i32 {
    %a_next = arith.addi %a, %c1 : i32
    %cmp = arith.cmpi eq, %a, %ub : i32
    %b_next = arith.select %cmp, %empty, %b : tensor<32xi32>
    scf.yield %a_next, %b_next : i32, tensor<32xi32>
  }
  return %res#0 : i32
}
