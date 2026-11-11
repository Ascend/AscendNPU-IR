// RUN: bishengir-opt %s --allow-unregistered-dialect --hfusion-optimize-scalar-transfers --symbol-dce --cse --split-input-file | FileCheck %s

// Positive:
// One of the arguments of the addf operation is produced by vector unit,
// and then through DU chain consumed in the vector context again
// CHECK-LABEL: func.func @lift_scalar(
// CHECK-NOT: tensor.extract
// CHECK: %[[LIFT_CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[LIFT_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[LIFT_CST_TENSOR:.*]] = linalg.fill ins(%[[LIFT_CST]] : f32) outs(%[[LIFT_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[LIFT_EXPANDED:.*]] = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[LIFT_ADD:.*]] = arith.addf %[[LIFT_EXPANDED]], %[[LIFT_CST_TENSOR]] : tensor<1xf32>
// CHECK-NOT: tensor.insert %[[LIFT_ADD]]
// CHECK: %[[LIFT_RESULT:.*]] = math.sqrt %[[LIFT_ADD]] : tensor<1xf32>
// CHECK: return %[[LIFT_RESULT]] : tensor<1xf32>
module {
  func.func @lift_scalar(%input: tensor<f32>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %input[] : tensor<f32>
    %add = arith.addf %extracted, %cst : f32
    %inserted = tensor.insert %add into %empty[%c0] : tensor<1xf32>
    %result = math.sqrt %inserted : tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// -----

// Positive: an intermediate scalar result is carried through both if branches
// as a singleton tensor and extracted after the if.
// CHECK-LABEL: func.func @lift_if(
// CHECK-NOT: tensor.extract
// CHECK: %[[IF_RESULT:.*]] = scf.if {{.*}} -> (tensor<1xf32>)
// CHECK: %[[IF_THEN_ADD:.*]] = arith.addf %arg0, %arg0 : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.yield {{.*}} : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: return %[[IF_RESULT]] : tensor<1xf32>
module {
  func.func @lift_if(%arg: tensor<1xf32>, %cond: i1) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %init = tensor.extract %arg[%c0] : tensor<1xf32>
    %result = scf.if %cond -> (f32) {
      %then = arith.addf %init, %init : f32
      scf.yield %then : f32
    } else {
      scf.yield %init : f32
    }
    %empty = tensor.empty() : tensor<1xf32>
    %output = tensor.insert %result into %empty[%c0] : tensor<1xf32>
    return %output : tensor<1xf32>
  }
}

// -----

// Positive: fillOp hoisted to unlock value numbering
// hoisted above the scf.if.
// CHECK-LABEL: func.func @lift_if_with_external_scalar_yield(
// CHECK: linalg.fill {{.*}} : tensor<1xf32>
// CHECK: %[[IF_OUTER:.*]] = scf.if {{.*}} -> (tensor<1xf32>) {
// CHECK: arith.addf {{.*}} : tensor<1xf32>
// CHECK: scf.yield {{.*}} : tensor<1xf32>
// CHECK: } else {
// CHECK: scf.yield {{.*}} : tensor<1xf32>
// CHECK: }
// CHECK-NOT: tensor.insert
// CHECK: return %[[IF_OUTER]] : tensor<1xf32>
module {
  func.func @lift_if_with_external_scalar_yield(%arg: tensor<1xf32>, %cst: f32, %cond: i1) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %init = tensor.extract %arg[%c0] : tensor<1xf32>
    %result = scf.if %cond -> (f32) {
      %then = arith.addf %init, %init : f32
      scf.yield %then : f32
    } else {
      scf.yield %cst : f32
    }
    %empty = tensor.empty() : tensor<1xf32>
    %output = tensor.insert %result into %empty[%c0] : tensor<1xf32>
    return %output : tensor<1xf32>
  }
}

// -----

// Negative: an if whose scalar result is not consumed in a vector context is
// not normalized.
// CHECK-LABEL: func.func @leave_if_scalar(
// CHECK: %[[SCALAR_IF:.*]] = scf.if {{.*}} -> (f32)
// CHECK-NOT: tensor.insert
// CHECK: return {{.*}} : f32
module {
  func.func @leave_if_scalar(%arg: tensor<1xf32>, %cond: i1) -> f32 attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %init = tensor.extract %arg[%c0] : tensor<1xf32>
    %result = scf.if %cond -> (f32) {
      %then = arith.addf %init, %init : f32
      scf.yield %then : f32
    } else {
      scf.yield %init : f32
    }
    %output = arith.addf %result, %result : f32
    return %output : f32
  }
}

// -----

// Positive: only the intermediate result of a mixed multi-result if is
// normalized; the second result remains scalar.
// CHECK-LABEL: func.func @lift_if_with_vector_scalar_results(
// CHECK-NOT: tensor.insert
// CHECK: %[[MIXED_IF:.*]]:2 = scf.if {{.*}} -> (tensor<1xf32>, f32)
// CHECK-NOT: tensor.extract %[[MIXED_IF]]#0
// CHECK-NOT: tensor.insert
// CHECK: return %[[MIXED_IF]]#0 : tensor<1xf32>
module {
  func.func @lift_if_with_vector_scalar_results(%arg: tensor<1xf32>, %cond: i1) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %init = tensor.extract %arg[%c0] : tensor<1xf32>
    %result, %plain = scf.if %cond -> (f32, f32) {
      %then = arith.addf %init, %init : f32
      scf.yield %then, %init : f32, f32
    } else {
      scf.yield %init, %init : f32, f32
    }
    %empty = tensor.empty() : tensor<1xf32>
    %output = tensor.insert %result into %empty[%c0] : tensor<1xf32>
    %unused = arith.addf %plain, %plain : f32
    return %output : tensor<1xf32>
  }
}

// -----

// Positive: all results of a multi-result if can be normalized independently.
// CHECK-LABEL: func.func @normalize_if_all_results(
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[ALL_IF:.*]]:2 = scf.if {{.*}} -> (tensor<1xf32>, tensor<1xf32>)
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[ALL_RESULT:.*]] = arith.addf %[[ALL_IF]]#0, %[[ALL_IF]]#1 : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: return %[[ALL_RESULT]] : tensor<1xf32>
module {
  func.func @normalize_if_all_results(%arg: tensor<1xf32>, %cond: i1) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %init = tensor.extract %arg[%c0] : tensor<1xf32>
    %first, %second = scf.if %cond -> (f32, f32) {
      %then = arith.addf %init, %init : f32
      scf.yield %then, %init : f32, f32
    } else {
      scf.yield %init, %init : f32, f32
    }
    %empty = tensor.empty() : tensor<1xf32>
    %first_tensor = tensor.insert %first into %empty[%c0] : tensor<1xf32>
    %second_tensor = tensor.insert %second into %empty[%c0] : tensor<1xf32>
    %output = arith.addf %first_tensor, %second_tensor : tensor<1xf32>
    return %output : tensor<1xf32>
  }
}

// -----

// Positive: nested if operations are normalized by the same preprocessing
// traversal without special-case nested-control-flow handling.
// CHECK-LABEL: func.func @lift_nested_ifs(
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.if {{.*}} -> (tensor<1xf32>)
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.if {{.*}} -> (tensor<1xf32>)
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: return {{.*}} : tensor<1xf32>
module {
  func.func @lift_nested_ifs(%arg: tensor<1xf32>, %outer_cond: i1,
                                 %inner_cond: i1) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %init = tensor.extract %arg[%c0] : tensor<1xf32>
    %result = scf.if %outer_cond -> (f32) {
      %inner = scf.if %inner_cond -> (f32) {
        %then = arith.addf %init, %init : f32
        scf.yield %then : f32
      } else {
        scf.yield %init : f32
      }
      scf.yield %inner : f32
    } else {
      scf.yield %init : f32
    }
    %empty = tensor.empty() : tensor<1xf32>
    %output = tensor.insert %result into %empty[%c0] : tensor<1xf32>
    return %output : tensor<1xf32>
  }
}

// -----

// Positive: an if nested in a loop is normalized after loop preprocessing.
// CHECK-LABEL: func.func @lift_if_nested_in_for(
// CHECK: %[[NESTED_FOR:.*]] = scf.for {{.*}} iter_args(%[[NESTED_ITER:.*]] = %arg0) -> (tensor<1xf32>) {
// CHECK: %[[NESTED_IF:.*]] = scf.if {{.*}} -> (tensor<1xf32>)
// CHECK: %[[NESTED_ADD:.*]] = arith.addf %[[NESTED_ITER]], %[[NESTED_ITER]] : tensor<1xf32>
// CHECK: scf.yield %[[NESTED_ADD]] : tensor<1xf32>
// CHECK: } else {
// CHECK: scf.yield %[[NESTED_ITER]] : tensor<1xf32>
// CHECK: }
// CHECK: %[[NESTED_SUM:.*]] = arith.addf %[[NESTED_IF]], %arg1 : tensor<1xf32>
// CHECK: scf.yield %[[NESTED_SUM]] : tensor<1xf32>
// CHECK: }
// CHECK: return %[[NESTED_FOR]] : tensor<1xf32>
module {
  func.func @lift_if_nested_in_for(%arg: tensor<1xf32>, %val: tensor<1xf32>, %cond: i1,
                                 %lb: index, %ub: index, %step: index) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %init = tensor.extract %arg[%c0] : tensor<1xf32>
    %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
      %selected = scf.if %cond -> (f32) {
        %then = arith.addf %acc, %acc : f32
        scf.yield %then : f32
      } else {
        scf.yield %acc : f32
      }
      %empty_inner = tensor.empty() : tensor<1xf32>
      %inserted = tensor.insert %selected into %empty_inner[%c0] : tensor<1xf32>
      %sum = arith.addf %inserted, %val : tensor<1xf32>
      %extracted = tensor.extract %sum[%c0] : tensor<1xf32>
      scf.yield %extracted : f32
    }
    %empty = tensor.empty() : tensor<1xf32>
    %output = tensor.insert %result into %empty[%c0] : tensor<1xf32>
    return %output : tensor<1xf32>
  }
}

// -----

// Positive: the chain of transfers detected and transformed
// CHECK-LABEL: func.func @lift_scalar_chain(
// CHECK-NOT: tensor.extract
// CHECK: %[[CHAIN_CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CHAIN_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[CHAIN_CST_TENSOR:.*]] = linalg.fill ins(%[[CHAIN_CST]] : f32) outs(%[[CHAIN_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[CHAIN_CST1:.*]] = arith.constant 4.200000e+01 : f32
// CHECK: %[[CHAIN_CST1_TENSOR:.*]] = linalg.fill ins(%[[CHAIN_CST1]] : f32) outs(%[[CHAIN_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[CHAIN_EXPANDED:.*]] = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[CHAIN_ADD0:.*]] = arith.addf %[[CHAIN_EXPANDED]], %[[CHAIN_CST_TENSOR]] : tensor<1xf32>
// CHECK: %[[CHAIN_ADD1:.*]] = arith.addf %[[CHAIN_ADD0]], %[[CHAIN_CST1_TENSOR]] : tensor<1xf32>
// CHECK: %[[CHAIN_MUL:.*]] = arith.mulf %[[CHAIN_ADD1]], %[[CHAIN_ADD0]] : tensor<1xf32>
// CHECK: %[[CHAIN_RESULT:.*]] = math.sqrt %[[CHAIN_MUL]] : tensor<1xf32>
// CHECK: return %[[CHAIN_RESULT]] : tensor<1xf32>
module {
  func.func @lift_scalar_chain(%input: tensor<f32>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.0 : f32
    %cst1 = arith.constant 42.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %input[] : tensor<f32>
    %add0 = arith.addf %extracted, %cst : f32
    %add1 = arith.addf %add0, %cst1 : f32
    %mul = arith.mulf %add1, %add0 : f32
    %inserted = tensor.insert %mul into %empty[%c0] : tensor<1xf32>
    %result = math.sqrt %inserted : tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// -----

// CHECK-LABEL: func.func @lift_scalar_chain_with_multiple_consumers(
// CHECK: %[[MULTI_CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[MULTI_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[MULTI_CST_TENSOR:.*]] = linalg.fill ins(%[[MULTI_CST]] : f32) outs(%[[MULTI_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK: %[[MULTI_EXPANDED1:.*]] = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[MULTI_EXPANDED2:.*]] = tensor.expand_shape %arg1 [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK: %[[MULTI_ADD0:.*]] = arith.addf %[[MULTI_EXPANDED1]], %[[MULTI_CST_TENSOR]] : tensor<1xf32>
// CHECK: %[[MULTI_ADD1:.*]] = arith.addf %[[MULTI_ADD0]], %[[MULTI_EXPANDED2]] : tensor<1xf32>
// CHECK: %[[MULTI_MUL:.*]] = arith.mulf %[[MULTI_ADD1]], %[[MULTI_ADD0]] : tensor<1xf32>
// CHECK-NOT: tensor.insert
// CHECK: %[[MULTI_RESULT:.*]] = arith.addf %[[MULTI_MUL]], %[[MULTI_ADD1]] : tensor<1xf32>
// CHECK-NOT: tensor.insert
// CHECK: return %[[MULTI_RESULT]] : tensor<1xf32>
module {
  func.func @lift_scalar_chain_with_multiple_consumers(%input1: tensor<f32>, %input2: tensor<f32>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.0 : f32
    %cst1 = arith.constant 42.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted1 = tensor.extract %input1[] : tensor<f32>
    %extracted2 = tensor.extract %input2[] : tensor<f32>
    %add0 = arith.addf %extracted1, %cst : f32
    %add1 = arith.addf %add0, %extracted2 : f32
    %mul = arith.mulf %add1, %add0 : f32
    %inserted_mul = tensor.insert %mul   into %empty[%c0] : tensor<1xf32>
    %inserted_add = tensor.insert %add1 into %empty[%c0] : tensor<1xf32>
    %result = arith.addf %inserted_mul, %inserted_add: tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// -----

// Positive: a scalar loop-carried value is lifted to a singleton tensor
// CHECK-LABEL: func.func @lift_loop_carried(
// CHECK: %[[A_CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[A_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[A_C0:.*]] = arith.constant 0 : index
// CHECK: %[[A_INSERT:.*]] = tensor.insert %[[A_CST]] into %[[A_EMPTY]][%[[A_C0]]] : tensor<1xf32>
// CHECK: %[[A_FOR:.*]] = scf.for {{.*}} iter_args(%[[A_ITER:.*]] = %[[A_INSERT]]) -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[A_ADD:.*]] = arith.addf %[[A_ITER]], {{.*}} : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.yield %[[A_ADD]] : tensor<1xf32>
// CHECK: }
// CHECK: %[[A_EXTRACT:.*]] = tensor.extract %[[A_FOR]][%[[A_C0]]] : tensor<1xf32>
// CHECK: %[[A_RESULT:.*]] = arith.addf %[[A_EXTRACT]], %[[A_EXTRACT]] : f32
// CHECK: return %[[A_RESULT]] : f32
module {
  func.func @lift_loop_carried(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> f32 attributes { mix_mode = "aiv" } {
    %cst = arith.constant 1.000000e+00 : f32
    %result = scf.for %i = %lb to %ub step %step
              iter_args(%acc = %cst) -> (f32) {
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 1.000000e+00 : f32
      %empty = tensor.empty() : tensor<1xf32>
      %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
      %sum = arith.addf %inserted, %val : tensor<1xf32>
      %extracted = tensor.extract %sum[%c0] : tensor<1xf32>
      scf.yield %extracted : f32
    }
    %output = arith.addf %result, %result : f32
    return %output : f32
  }
}

// -----

// Positive: nested loops, the inner loop result depends on its iter_arg
// CHECK-LABEL: func.func @lift_nested_loops(
// CHECK: %[[B_CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[B_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[B_C0:.*]] = arith.constant 0 : index
// CHECK: %[[B_INSERT:.*]] = tensor.insert %[[B_CST]] into %[[B_EMPTY]][%[[B_C0]]] : tensor<1xf32>
// CHECK: %[[B_FOR:.*]] = scf.for {{.*}} iter_args(%[[B_ITER:.*]] = %[[B_INSERT]]) -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[B_ADD:.*]] = arith.addf %[[B_ITER]], {{.*}} : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[B_FOR2:.*]] = scf.for {{.*}} iter_args(%[[B_ITER2:.*]] = %[[B_ADD]]) -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[B_ADD2:.*]] = arith.addf %[[B_ITER2]], {{.*}} : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.yield %[[B_ADD2]] : tensor<1xf32>
// CHECK: }
// CHECK: scf.yield %[[B_FOR2]] : tensor<1xf32>
// CHECK: }
// CHECK-NOT: tensor.extract %[[B_FOR2]]
// CHECK-NOT: tensor.insert
// CHECK: %[[B_EX2:.*]] = tensor.extract %[[B_FOR]][{{.*}}] : tensor<1xf32>
// CHECK: %[[B_RESULT:.*]] = arith.addf %[[B_EX2]], %[[B_EX2]] : f32
// CHECK: return %[[B_RESULT]] : f32
module {
  func.func @lift_nested_loops(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> f32 attributes { mix_mode = "aiv" } {
    %cst = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %result = scf.for %i = %lb to %ub step %step
              iter_args(%acc = %cst) -> (f32) {
      %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
      %sum = arith.addf %inserted, %val : tensor<1xf32>
      %extracted_sum = tensor.extract %sum[%c0] : tensor<1xf32>
      %result_2 = scf.for %j = %lb to %ub step %step
              iter_args(%acc_2 = %extracted_sum) -> (f32) {
        %inserted_2 = tensor.insert %acc_2 into %empty[%c0] : tensor<1xf32>
        %sum_2 = arith.addf %inserted_2, %val : tensor<1xf32>
        %extracted_2 = tensor.extract %sum_2[%c0] : tensor<1xf32>
        scf.yield %extracted_2 : f32
      }
      scf.yield %result_2 : f32
    }

    %output = arith.addf %result, %result : f32
    return %output : f32
  }
}

// -----

// Positive: nested loops, the inner loop is invariant (yields the outer add)
// CHECK-LABEL: func.func @lift_for_with_dead_iter_arg(
// CHECK: %[[C_INSERT:.*]] = tensor.insert {{.*}} : tensor<1xf32>
// CHECK: %[[C_FOR:.*]] = scf.for {{.*}} iter_args(%[[C_ITER:.*]] = %[[C_INSERT]]) -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[C_ADD:.*]] = arith.addf %[[C_ITER]], {{.*}} : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[C_FOR2:.*]] = scf.for {{.*}} iter_args(%[[C_ITER2:.*]] = %[[C_ADD]]) -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.yield %[[C_ADD]] : tensor<1xf32>
// CHECK: }
// CHECK: scf.yield %[[C_FOR2]] : tensor<1xf32>
// CHECK: }
// CHECK-NOT: tensor.extract %[[C_FOR2]]
// CHECK-NOT: tensor.insert
// CHECK: %[[C_EX2:.*]] = tensor.extract %[[C_FOR]][{{.*}}] : tensor<1xf32>
// CHECK: %[[C_RESULT:.*]] = arith.addf %[[C_EX2]], %[[C_EX2]] : f32
// CHECK: return %[[C_RESULT]] : f32
module {
  func.func @lift_for_with_dead_iter_arg(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> f32 attributes { mix_mode = "aiv" } {
    %cst = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %result = scf.for %i = %lb to %ub step %step
              iter_args(%acc = %cst) -> (f32) {
      %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
      %sum = arith.addf %inserted, %val : tensor<1xf32>
      %extracted_sum = tensor.extract %sum[%c0] : tensor<1xf32>
      %result_2 = scf.for %j = %lb to %ub step %step
              iter_args(%acc_2 = %extracted_sum) -> (f32) {
        %inserted_2 = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
        %sum_2 = arith.addf %inserted, %val : tensor<1xf32>
        %extracted_2 = tensor.extract %sum[%c0] : tensor<1xf32>
        scf.yield %extracted_2 : f32
      }
      scf.yield %result_2 : f32
    }

    %output = arith.addf %result, %result : f32
    return %output : f32
  }
}

// -----

// Positive: extract lifted to identity, scalar add lifted back to tensor
// CHECK-LABEL: func.func @lift_whole_loop(
// CHECK: %[[D_CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[D_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[D_INSERT:.*]] = linalg.fill ins(%[[D_CST]] : f32) outs(%[[D_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[D_FOR:.*]] = scf.for {{.*}} iter_args(%[[D_ITER:.*]] = %arg1) -> (tensor<1xf32>) {
// CHECK: %[[D_ADD:.*]] = arith.addf %[[D_ITER]], %[[D_INSERT]] : tensor<1xf32>
// CHECK: scf.yield %[[D_ADD]] : tensor<1xf32>
// CHECK: }
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: return %[[D_FOR]] : tensor<1xf32>
module {
  func.func @lift_whole_loop(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %val[%c0] : tensor<1xf32>
    %result = scf.for %i = %lb to %ub step %step
              iter_args(%acc = %extracted) -> (f32) {
      %add = arith.addf %acc, %cst_0 : f32
      scf.yield %add : f32
    }
    %inserted = tensor.insert %result into %empty[%c0] : tensor<1xf32>
    return %inserted : tensor<1xf32>
  }
}


// -----

// Positive: an intermediate chain with a non-intermediate operand adapts that
// operand into a singleton tensor instead of lifting it.
// CHECK-LABEL: func.func @intermediate_with_non_intermediate_operand(
// CHECK: %[[OPERAND_CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[OPERAND_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[OPERAND_CST_TENSOR:.*]] = linalg.fill ins(%[[OPERAND_CST]] : f32) outs(%[[OPERAND_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[OPERAND_EXPANDED:.*]] = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[OPERAND_ADD:.*]] = arith.addf %[[OPERAND_EXPANDED]], %[[OPERAND_CST_TENSOR]] : tensor<1xf32>
// CHECK: %[[OPERAND_MUL:.*]] = arith.mulf %[[OPERAND_ADD]], %[[OPERAND_ADD]] : tensor<1xf32>
// CHECK-NOT: tensor.insert
// CHECK: return %[[OPERAND_MUL]] : tensor<1xf32>
module {
  func.func @intermediate_with_non_intermediate_operand(%input: tensor<f32>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %input[] : tensor<f32>
    %add = arith.addf %extracted, %cst : f32
    %mul = arith.mulf %add, %add : f32
    %inserted = tensor.insert %mul into %empty[%c0] : tensor<1xf32>
    return %inserted : tensor<1xf32>
  }
}

// -----

// Positive: an intermediate chain that also feeds a scalar (non-vector) use is
// still lifted; the scalar use reads the lifted value through an extract.
// CHECK-LABEL: func.func @intermediate_with_non_intermediate_use(
// CHECK: %[[USE_CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[USE_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[USE_CST_TENSOR:.*]] = linalg.fill ins(%[[USE_CST]] : f32) outs(%[[USE_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[USE_EXPANDED:.*]] = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[USE_ADD:.*]] = arith.addf %[[USE_EXPANDED]], %[[USE_CST_TENSOR]] : tensor<1xf32>
// CHECK: %[[USE_C0:.*]] = arith.constant 0 : index
// CHECK: %[[USE_EXTRACT:.*]] = tensor.extract %[[USE_ADD]][%[[USE_C0]]] : tensor<1xf32>
// CHECK: %[[USE_SIDE:.*]] = arith.mulf %[[USE_EXTRACT]], %[[USE_EXTRACT]] : f32
// CHECK: return %[[USE_ADD]], %[[USE_SIDE]] : tensor<1xf32>, f32
module {
  func.func @intermediate_with_non_intermediate_use(%input: tensor<f32>) -> (tensor<1xf32>, f32) attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %input[] : tensor<f32>
    %add = arith.addf %extracted, %cst : f32
    %inserted = tensor.insert %add into %empty[%c0] : tensor<1xf32>
    %side = arith.mulf %add, %add : f32
    return %inserted, %side : tensor<1xf32>, f32
  }
}

// -----

// Positive: a static rank-one linalg.fill consumer drains the intermediate
// scalar into the fill through a single extract.
// CHECK-LABEL: func.func @lift_static_rank_one_fill(
// CHECK: %[[FILL_CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[FILL_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[FILL_CST_TENSOR:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[FILL_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[FILL_EXPANDED:.*]] = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[FILL_ADD:.*]] = arith.addf %[[FILL_EXPANDED]], %[[FILL_CST_TENSOR]] : tensor<1xf32>
// CHECK: %[[FILL_C0:.*]] = arith.constant 0 : index
// CHECK: %[[FILL_EXTRACT:.*]] = tensor.extract %[[FILL_ADD]][%[[FILL_C0]]] : tensor<1xf32>
// CHECK: %[[FILL_RESULT:.*]] = linalg.fill ins(%[[FILL_EXTRACT]] : f32) outs(%[[FILL_EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[FILL_RESULT]] : tensor<1xf32>
module {
  func.func @lift_static_rank_one_fill(%input: tensor<f32>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %input[] : tensor<f32>
    %add = arith.addf %extracted, %cst : f32
    %fill = linalg.fill ins(%add : f32) outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    return %fill : tensor<1xf32>
  }
}

// -----

// Negative: a fill whose output tensor has a dynamic dimension is not a
// scalar-transfer consumer, so the chain stays scalar.
// CHECK-LABEL: func.func @no_lift_dynamic_rank_one_fill(
// CHECK: %[[DYNF_EMPTY:.*]] = tensor.empty(%arg1) : tensor<?xf32>
// CHECK: %[[DYNF_EXTRACT:.*]] = tensor.extract %arg0[] : tensor<f32>
// CHECK: %[[DYNF_ADD:.*]] = arith.addf %[[DYNF_EXTRACT]], %{{.*}} : f32
// CHECK: %[[DYNF_FILL:.*]] = linalg.fill ins(%[[DYNF_ADD]] : f32) outs(%[[DYNF_EMPTY]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK: return %[[DYNF_FILL]] : tensor<?xf32>
module {
  func.func @no_lift_dynamic_rank_one_fill(%input: tensor<f32>, %n: index) -> tensor<?xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.0 : f32
    %empty = tensor.empty(%n) : tensor<?xf32>
    %extracted = tensor.extract %input[] : tensor<f32>
    %add = arith.addf %extracted, %cst : f32
    %fill = linalg.fill ins(%add : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    return %fill : tensor<?xf32>
  }
}

// -----

// Positive: an intermediate channel produced before an outer loop is carried
// through both the outer and the inner loop as a singleton tensor.
// CHECK-LABEL: func.func @channel_crossing_outer_loop(
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[CROSS_OUTER:.*]] = scf.for {{.*}} iter_args(%[[CROSS_OUTER_ITER:.*]] = %arg0) -> (tensor<1xf32>) {
// CHECK: %[[CROSS_OUTER_ADD:.*]] = arith.addf %[[CROSS_OUTER_ITER]], %arg1 : tensor<1xf32>
// CHECK: %[[CROSS_INNER:.*]] = scf.for {{.*}} iter_args(%[[CROSS_INNER_ITER:.*]] = %[[CROSS_OUTER_ADD]]) -> (tensor<1xf32>) {
// CHECK: %[[CROSS_INNER_ADD:.*]] = arith.addf %[[CROSS_INNER_ITER]], %arg1 : tensor<1xf32>
// CHECK: scf.yield %[[CROSS_INNER_ADD]] : tensor<1xf32>
// CHECK: }
// CHECK: %[[CROSS_OUTER_ADD2:.*]] = arith.addf %[[CROSS_INNER]], %arg1 : tensor<1xf32>
// CHECK: scf.yield %[[CROSS_OUTER_ADD2]] : tensor<1xf32>
// CHECK: }
// CHECK: return %[[CROSS_OUTER]] : tensor<1xf32>
module {
  func.func @channel_crossing_outer_loop(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %arg[%c0] : tensor<1xf32>
    %result = scf.for %i = %lb to %ub step %step
              iter_args(%acc = %extracted) -> (f32) {
      %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
      %sum = arith.addf %inserted, %val : tensor<1xf32>
      %extracted_sum = tensor.extract %sum[%c0] : tensor<1xf32>
      %inner = scf.for %j = %lb to %ub step %step
              iter_args(%acc2 = %extracted_sum) -> (f32) {
        %inserted2 = tensor.insert %acc2 into %empty[%c0] : tensor<1xf32>
        %sum2 = arith.addf %inserted2, %val : tensor<1xf32>
        %extracted2 = tensor.extract %sum2[%c0] : tensor<1xf32>
        scf.yield %extracted2 : f32
      }
      %outer_carry = tensor.insert %inner into %empty[%c0] : tensor<1xf32>
      %outer_sum = arith.addf %outer_carry, %val : tensor<1xf32>
      %outer_extracted = tensor.extract %outer_sum[%c0] : tensor<1xf32>
      scf.yield %outer_extracted : f32
    }
    %out = tensor.insert %result into %empty[%c0] : tensor<1xf32>
    return %out : tensor<1xf32>
  }
}

// -----

// Positive: when the non-intermediate channel precedes the intermediate one,
// only the intermediate channel is lifted and the plain channel keeps its
// scalar representation and position.
// CHECK-LABEL: func.func @mixed_loop_channels_plain_first(
// CHECK: %[[MIXP_FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[MIXP_PLAIN:.*]] = %{{.*}}, %[[MIXP_ACC:.*]] = %arg0) -> (f32, tensor<1xf32>) {
// CHECK: %[[MIXP_SUM:.*]] = arith.addf %[[MIXP_ACC]], %arg1 : tensor<1xf32>
// CHECK: %[[MIXP_PLAIN2:.*]] = arith.addf %[[MIXP_PLAIN]], %{{.*}} : f32
// CHECK: scf.yield %[[MIXP_PLAIN2]], %[[MIXP_SUM]] : f32, tensor<1xf32>
// CHECK: }
// CHECK: %[[MIXP_EXTRACT:.*]] = tensor.extract %[[MIXP_FOR]]#1[{{.*}}] : tensor<1xf32>
// CHECK: return %[[MIXP_FOR]]#0, %[[MIXP_EXTRACT]] : f32, f32
module {
  func.func @mixed_loop_channels_plain_first(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> (f32, f32) attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %arg[%c0] : tensor<1xf32>
    %r:2 = scf.for %i = %lb to %ub step %step
              iter_args(%plain = %cst, %acc = %extracted) -> (f32, f32) {
      %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
      %sum = arith.addf %inserted, %val : tensor<1xf32>
      %extracted_sum = tensor.extract %sum[%c0] : tensor<1xf32>
      %plain2 = arith.addf %plain, %cst : f32
      scf.yield %plain2, %extracted_sum : f32, f32
    }
    return %r#0, %r#1 : f32, f32
  }
}

// -----

// Positive: a loop whose intermediate channel is followed by a non-intermediate
// channel lifts only the intermediate channel while preserving the plain one.
// CHECK-LABEL: func.func @mixed_loop_channels_intermediate_first(
// CHECK-NOT: tensor.insert
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}, %[[MIXI_PLAIN:.*]] = {{.*}}) -> (tensor<1xf32>, f32) {
// CHECK: arith.addf {{.*}}, %arg1 : tensor<1xf32>
// CHECK: %[[MIXI_PLAIN2:.*]] = arith.addf %[[MIXI_PLAIN]], %{{.*}} : f32
// CHECK: scf.yield {{.*}} : tensor<1xf32>, f32
// CHECK: }
// CHECK: tensor.extract {{.*}} : tensor<1xf32>
// CHECK: return {{.*}} : f32, f32
module {
  func.func @mixed_loop_channels_intermediate_first(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> (f32, f32) attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %arg[%c0] : tensor<1xf32>
    %r:2 = scf.for %i = %lb to %ub step %step
              iter_args(%acc = %extracted, %plain = %cst) -> (f32, f32) {
      %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
      %sum = arith.addf %inserted, %val : tensor<1xf32>
      %extracted_sum = tensor.extract %sum[%c0] : tensor<1xf32>
      %plain2 = arith.addf %plain, %cst : f32
      scf.yield %extracted_sum, %plain2 : f32, f32
    }
    return %r#0, %r#1 : f32, f32
  }
}

// -----

// Positive: multiple selected channels interspersed with a non-intermediate
// channel are all lifted while the plain channel keeps its type and position.
// CHECK-LABEL: func.func @mixed_loop_channels_two_selected(
// CHECK-NOT: tensor.insert
// CHECK: %[[TS_FOR:.*]]:3 = scf.for {{.*}} iter_args({{.*}}, %[[TS_PLAIN:.*]] = {{.*}}, {{.*}}) -> (tensor<1xf32>, f32, tensor<1xf32>) {
// CHECK: arith.addf {{.*}} : tensor<1xf32>
// CHECK: %[[TS_PLAIN2:.*]] = arith.addf %[[TS_PLAIN]], {{.*}} : f32
// CHECK: arith.addf {{.*}} : tensor<1xf32>
// CHECK: scf.yield {{.*}}, %[[TS_PLAIN2]], {{.*}} : tensor<1xf32>, f32, tensor<1xf32>
// CHECK: }
// CHECK-DAG: %[[TS_EX0:.*]] = tensor.extract %[[TS_FOR]]#0[
// CHECK-DAG: %[[TS_EX2:.*]] = tensor.extract %[[TS_FOR]]#2[
// CHECK: return %[[TS_EX0]], %[[TS_FOR]]#1, %[[TS_EX2]] : f32, f32, f32
module {
  func.func @mixed_loop_channels_two_selected(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> (f32, f32, f32) attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %extracted0 = tensor.extract %arg[%c0] : tensor<1xf32>
    %extracted1 = tensor.extract %arg[%c0] : tensor<1xf32>
    %r:3 = scf.for %i = %lb to %ub step %step
              iter_args(%acc0 = %extracted0, %plain = %cst, %acc1 = %extracted1) -> (f32, f32, f32) {
      %inserted0 = tensor.insert %acc0 into %empty[%c0] : tensor<1xf32>
      %sum0 = arith.addf %inserted0, %val : tensor<1xf32>
      %extracted_sum0 = tensor.extract %sum0[%c0] : tensor<1xf32>
      %plain2 = arith.addf %plain, %cst : f32
      %inserted1 = tensor.insert %acc1 into %empty[%c0] : tensor<1xf32>
      %sum1 = arith.addf %inserted1, %val : tensor<1xf32>
      %extracted_sum1 = tensor.extract %sum1[%c0] : tensor<1xf32>
      scf.yield %extracted_sum0, %plain2, %extracted_sum1 : f32, f32, f32
    }
    return %r#0, %r#1, %r#2 : f32, f32, f32
  }
}

// -----

// Positive: three nested loops, the transfer chain is lifted through every
// level so all loops carry a singleton tensor.
// CHECK-LABEL: func.func @lift_triple_nested_loops(
// CHECK: scf.for {{.*}} -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.for {{.*}} -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.for {{.*}} -> (tensor<1xf32>) {
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: scf.yield {{.*}} : tensor<1xf32>
// CHECK: scf.yield {{.*}} : tensor<1xf32>
// CHECK: scf.yield {{.*}} : tensor<1xf32>
// CHECK: return {{.*}} : f32
module {
  func.func @lift_triple_nested_loops(%arg: tensor<1xf32>, %val: tensor<1xf32>,
                 %lb: index, %ub: index, %step: index) -> f32 attributes { mix_mode = "aiv" } {
    %cst = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %c0 = arith.constant 0 : index
    %result = scf.for %i = %lb to %ub step %step
              iter_args(%acc0 = %cst) -> (f32) {
      %r1 = scf.for %j = %lb to %ub step %step
                iter_args(%acc1 = %acc0) -> (f32) {
        %r2 = scf.for %k = %lb to %ub step %step
                  iter_args(%acc = %acc1) -> (f32) {
          %inserted = tensor.insert %acc into %empty[%c0] : tensor<1xf32>
          %sum = arith.addf %inserted, %val : tensor<1xf32>
          %extracted = tensor.extract %sum[%c0] : tensor<1xf32>
          scf.yield %extracted : f32
        }
        scf.yield %r2 : f32
      }
      scf.yield %r1 : f32
    }
    %output = arith.addf %result, %result : f32
    return %output : f32
  }
}

// -----

// Positive: a cast with different operand and result types (`arith.extf`
// f16 -> f32) is lifted, so the operand and result are both lifted to
// singleton tensors with their respective element types.
// CHECK-LABEL: func.func @lift_extf(
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: %[[EXP:.*]] = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f16> into tensor<1xf16>
// CHECK: %[[EXT:.*]] = arith.extf %[[EXP]] : tensor<1xf16> to tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: return %[[EXT]] : tensor<1xf32>
module {
  func.func @lift_extf(%input: tensor<f16>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<1xf32>
    %extracted = tensor.extract %input[] : tensor<f16>
    %ext = arith.extf %extracted : f16 to f32
    %inserted = tensor.insert %ext into %empty[%c0] : tensor<1xf32>
    return %inserted : tensor<1xf32>
  }
}

// -----

// Negative: a cast with an explicit rounding mode is not lifted, so the
// operand and result stay as scalar values.
// CHECK-LABEL: func.func @dont_lift_truncf_with_rounding_mode(
// CHECK: tensor.extract
// CHECK: arith.truncf %{{.*}} toward_zero : f32 to f16
// CHECK: tensor.insert
module {
  func.func @dont_lift_truncf_with_rounding_mode(%input: tensor<f32>) -> tensor<1xf16> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<1xf16>
    %extracted = tensor.extract %input[] : tensor<f32>
    %trunc = arith.truncf %extracted toward_zero : f32 to f16
    %inserted = tensor.insert %trunc into %empty[%c0] : tensor<1xf16>
    return %inserted : tensor<1xf16>
  }
}


// -----

// Positive: a cast without an explicit rounding mode is lifted
// CHECK-LABEL: func.func @lift_truncf_without_rounding_mode(
// CHECK: arith.truncf %{{.*}} : tensor<1xf32> to tensor<1xf16>
module {
  func.func @lift_truncf_without_rounding_mode(%input: tensor<f32>) -> tensor<1xf16> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<1xf16>
    %extracted = tensor.extract %input[] : tensor<f32>
    %trunc = arith.truncf %extracted : f32 to f16
    %inserted = tensor.insert %trunc into %empty[%c0] : tensor<1xf16>
    return %inserted : tensor<1xf16>
  }
}
