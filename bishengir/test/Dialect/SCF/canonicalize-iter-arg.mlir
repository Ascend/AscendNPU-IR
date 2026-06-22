// RUN: bishengir-opt -scf-canonicalize-iter-arg -allow-unregistered-dialect %s | FileCheck %s

func.func @test() -> (tensor<?xi8>, tensor<?xi8>) {
  %size = "some_op"() : () -> index
  %e = tensor.empty(%size) : tensor<?xi8>
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %ub = arith.constant 16 : index
  %cond = "some_op"() : () -> i1
  // CHECK: %[[SIZE:.*]] = "some_op"() : () -> index
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[SIZE]])
  %res:2 = scf.for %i = %lb to %ub  step %step iter_args(%arg0 = %e, %arg1 = %e) -> (tensor<?xi8>, tensor<?xi8>) {
    %inner = scf.for %j = %lb to %ub step %step iter_args(%iarg = %arg0) -> tensor<?xi8> {
      "some_op"(): ()-> ()
      scf.yield %iarg : tensor<?xi8>
    }
    // CHECK: scf.if
    %inner2 = scf.if %cond -> tensor<?xi8> {
      "some_op"(): ()-> ()
      // CHECK: yield %[[INIT]]
      scf.yield %e : tensor<?xi8>
    } else {
      // CHECK: yield %[[INIT]]
      scf.yield %arg1 : tensor<?xi8>
    }

    scf.yield %inner, %inner2 : tensor<?xi8>, tensor<?xi8>
  }
  // CHECK: return %[[INIT]], %[[INIT]]
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
func.func @nested_loop_iter_arg(
    %arg0: index, %arg1: index,
    %arg2: tensor<32xi64>, %arg3: tensor<32xi32>) -> (index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %lb_outer = arith.constant 0 : index
  %ub_outer = arith.constant 10 : index
  %step_outer = arith.constant 2 : index
  
  %outer_res:2 = scf.for %iv_outer = %lb_outer to %ub_outer step %step_outer
      iter_args(%outer_it0 = %arg0, %outer_it1 = %c0) -> (index, index) {

    %lb_inner = arith.constant 0 : index
    %ub_inner = arith.addi %outer_it0, %c32 : index
    %step_inner = arith.constant 1 : index
    
    %res:4 = scf.for %iv_inner = %lb_inner to %ub_inner step %step_inner
        iter_args(%it0 = %arg2, %it1 = %arg3, %it2 = %outer_it0, %it3 = %outer_it1)
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
    // Use results from inner loop
    %new_outer_it0 = arith.addi %c1, %ub_inner : index
    %new_outer_it1 = arith.addi %res#3, %c1 : index
    scf.yield %new_outer_it0, %new_outer_it1 : index, index
  }
  
  // CHECK-LABEL: func.func @nested_loop_iter_arg
  // CHECK: scf.for %[[IV_OUTER:.*]] =
  // CHECK:   iter_args(%[[IT1:.*]] = %arg3, %[[IT2:.*]] = %[[OUTER_IT0:.*]])
  // CHECK-NOT: arith.addi %{{.*}} : tensor<32xi64>
  // CHECK:   arith.addi %[[IT1]], %[[IT1]] : tensor<32xi32>
  // CHECK:   hivm.hir.store
  // CHECK:   arith.addi %[[IT2]], %c32 : index
  // CHECK:   scf.yield %{{.*}}, %[[Y2:.*]] : tensor<32xi32>, index
  // CHECK:   scf.yield %{{.*}}, %{{.*}} : index, index
  // CHECK: return
  return %outer_res#0 : index
}

// -----
func.func @nested_loop_iter_arg_removed(
    %arg0: index, %arg1: index,
    %arg2: tensor<32xi64>, %arg3: tensor<32xi32>) -> (index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %lb_outer = arith.constant 0 : index
  %ub_outer = arith.constant 10 : index
  %step_outer = arith.constant 2 : index
  
  %outer_res:2 = scf.for %iv_outer = %lb_outer to %ub_outer step %step_outer
      iter_args(%outer_it0 = %arg0, %outer_it1 = %c0) -> (index, index) {
    // Outer loop body - we'll have the inner loop inside here
    %lb_inner = arith.constant 0 : index
    %ub_inner = arith.addi %outer_it0, %c32 : index
    %step_inner = arith.constant 1 : index
    
    %res:4 = scf.for %iv_inner = %lb_inner to %ub_inner step %step_inner
        iter_args(%it0 = %arg2, %it1 = %arg3, %it2 = %outer_it0, %it3 = %outer_it1)
        -> (tensor<32xi64>, tensor<32xi32>, index, index) {
      %y0 = arith.addi %it0, %it0 : tensor<32xi64>
      %y1 = arith.addi %it1, %it1 : tensor<32xi32>
      // IF there is no memory effect, this loop is not needed
      // %stored = memref.alloc() : memref<32xi32>
      // hivm.hir.store ins(%y1 : tensor<32xi32>) outs(%stored: memref<32xi32>)
      %y2 = arith.addi %it2, %c32 : index
      %y3 = arith.addi %it3, %c32 : index
      scf.yield %y0, %y1, %y2, %y3
      : tensor<32xi64>, tensor<32xi32>, index, index
    }
    // Use results from inner loop
    %new_outer_it0 = arith.addi %c1, %ub_inner : index
    %new_outer_it1 = arith.addi %res#3, %c1 : index
    scf.yield %new_outer_it0, %new_outer_it1 : index, index
  }
  
  // CHECK-LABEL: func.func @nested_loop_iter_arg_removed
  // CHECK: %[[LoopRes:.*]] = scf.for %[[IV_OUTER:.*]] =
  // CHECK-NOT: scf.for %[[IV_INNER:.*]] =
  // CHECK:   scf.yield %{{.*}} : index
  // CHECK: return %[[LoopRes]]
  return %outer_res#0 : index
}

// -----
// CHECK-LABEL: func.func @remove_redundant_iter_arg_select
// CHECK: %[[VAL_0:.*]]:2 = scf.for %[[ITER:.*]] = %[[START:.*]] to %[[END:.*]] step %[[STEP:.*]] iter_args(%[[ARG0:.*]] = %[[VAL_1:.*]], %[[ARG1:.*]] = %[[VAL_2:.*]]) -> (tensor<2x128xf32>, tensor<3x128xf32>)  : i32 {
// CHECK:   %[[VAL_3:.*]] = arith.select %[[CONDITION:.*]], %[[ARG0]], %[[EMPTY1:.*]] : tensor<2x128xf32>
// CHECK:   %[[VAL_4:.*]] = hivm.hir.vadd ins(%[[ARG1]], %[[EMPTY2:.*]] : tensor<3x128xf32>, tensor<3x128xf32>) outs(%[[EMPTY3:.*]] : tensor<3x128xf32>) -> tensor<3x128xf32>
// CHECK:   scf.yield %[[VAL_3]], %[[VAL_4]] : tensor<2x128xf32>, tensor<3x128xf32>
// CHECK: }
// CHECK: return %[[VAL_0]]#0, %[[VAL_0]]#1 : tensor<2x128xf32>, tensor<3x128xf32>
func.func @remove_redundant_iter_arg_select(%arg0: tensor<1x128xf32>, %arg1: tensor<2x128xf32>, %arg2: tensor<3x128xf32>, %arg3: tensor<4x128xf32>, %arg4: memref<1xi1>) ->
  (tensor<2x128xf32>, tensor<3x128xf32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %0 = tensor.empty() : tensor<1x128xf32>
  %1 = tensor.empty() : tensor<2x128xf32>
  %2 = tensor.empty() : tensor<3x128xf32>
  %3 = tensor.empty() : tensor<4x128xf32>
  %cond = memref.load %arg4[%c0] : memref<1xi1>
  %4:4 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2, %arg9 = %arg3) ->
    (tensor<1x128xf32>, tensor<2x128xf32>, tensor<3x128xf32>, tensor<4x128xf32>) : i32 {
    %5 = arith.select %cond, %arg6, %0 : tensor<1x128xf32>
    %6 = arith.select %cond, %arg7, %1 : tensor<2x128xf32>
    %7 = hivm.hir.vadd ins(%arg8, %arg2 : tensor<3x128xf32>, tensor<3x128xf32>) outs(%2 : tensor<3x128xf32>) -> tensor<3x128xf32>
    %8 = arith.select %cond, %arg9, %3 : tensor<4x128xf32>
    scf.yield %5, %6, %7, %8 : tensor<1x128xf32>, tensor<2x128xf32>, tensor<3x128xf32>, tensor<4x128xf32>
  }
  return %4#1, %4#2 : tensor<2x128xf32>, tensor<3x128xf32>
}
