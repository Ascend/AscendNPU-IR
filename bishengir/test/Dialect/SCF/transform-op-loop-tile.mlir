// RUN: bishengir-opt --split-input-file -transform-interpreter %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  // CHECK: scf.for %[[IIV:.*]] =
  // CHECK: %[[UPPERBOUND:.*]] = arith.addi
  // CHECK: scf.for %[[OUTERIV:.*]] = %[[IIV]] to %[[UPPERBOUND]] step %[[OUTSTEP:.*]]
  // CHECK: %[[INNERUPPER:.*]] = affine.min
  // CHECK: scf.for %[[INNERIV:.*]] =
  // CHECK-SAME: to %[[INNERUPPER]] step
  func.func @loop_tile(%IN1 : memref<256xf32>, %IN2 : memref<256xf32>, %OUT : memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c256 step %c128 {
      %ub = arith.addi %i, %c128 : index
      scf.for %j = %i to %ub step %c1 {
        %0 = memref.load %IN1[%j] : memref<256xf32>
        %1 = memref.load %IN2[%j] : memref<256xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %OUT[%j] : memref<256xf32>
      } {loop_1}
    } {loop_2}
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {  
    %0 = transform.structured.match ops{["scf.for"]} attributes{loop_1} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2:2 = transform.loop.tile %0 [10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
  // CHECK: %[[CONST256:.*]] = arith.constant 256 : index
  // CHECK: %[[CONST128:.*]] = arith.constant 128 : index
  // CHECK: scf.for %[[OUTIV:.*]] = %[[CONST0]] to %[[CONST256]] step %[[OUTERSTEP:.*]]
  // CHECK: %[[INNERUPPER:.*]] = affine.min
  // CHECK: scf.for
  // CHECK-SAME to %[[INNERUPPER]] step %[[CONST128]]
  func.func @loop_tile_with_step(%IN1 : memref<256xf32>, %IN2 : memref<256xf32>, %OUT : memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c256 step %c128 {
      %ub = arith.addi %i, %c128 : index
      scf.for %j = %i to %ub step %c1 {
        %0 = memref.load %IN1[%j] : memref<256xf32>
        %1 = memref.load %IN2[%j] : memref<256xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %OUT[%j] : memref<256xf32>
      } {loop_1}
    } {loop_2}
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {  
    %0 = transform.structured.match ops{["scf.for"]} attributes{loop_2} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2:2 = transform.loop.tile %0 [1280] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK: %[[RES:.*]] = scf.for
  // CHECK-SAME: iter_args(%[[ARG:.*]] =
  // CHECK: %[[INNERRES:.*]] = scf.for
  // CHECK-SAME: iter_args(%[[INNERARG:.*]] = %[[ARG]]
  // CHECK: scf.yield %[[INNERRES]]
  // CHECK: return %[[RES]]
  func.func @loop_tile_with_yield(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = tensor.empty() : tensor<256xf32>
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %1 = scf.for %arg2 = %c0 to %c256 step %c2 iter_args(%arg3 = %0) -> (tensor<256xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [2] [1] : tensor<256xf32> to tensor<2xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg2] [2] [1] : tensor<256xf32> to tensor<2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg2] [2] [1] : tensor<256xf32> to tensor<2xf32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 : tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_1 : tensor<2xf32>) -> tensor<2xf32>
      %inserted_slice = tensor.insert_slice %2 into %arg3[%arg2] [2] [1] : tensor<2xf32> into tensor<256xf32>
      scf.yield %inserted_slice : tensor<256xf32>
    }
    return %1 : tensor<256xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {  
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2:2 = transform.loop.tile %0 [20] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK: scf.for %[[IIV:.*]] =
  // CHECK: %[[UPPERBOUND:.*]] = arith.addi
  // CHECK: scf.for %[[OUTERIV:.*]] = %[[IIV]] to %[[UPPERBOUND]] step %[[OUTSTEP:.*]]
  // CHECK: %[[INNERUPPER:.*]] = affine.min
  // CHECK: scf.for %[[INNERIV:.*]] =
  // CHECK-SAME: to %[[INNERUPPER]] step
  func.func @loop_tile_dyn(%IN1 : memref<256xf32>, %IN2 : memref<256xf32>, %OUT : memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %sz = func.call @get_dynamic_tile_size() : () -> index
    scf.for %i = %c0 to %c256 step %c128 {
      %ub = arith.addi %i, %c128 : index
      scf.for %j = %i to %ub step %c1 {
        %0 = memref.load %IN1[%j] : memref<256xf32>
        %1 = memref.load %IN2[%j] : memref<256xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %OUT[%j] : memref<256xf32>
      } {loop_1}
    } {loop_2}
    return
  }

  func.func private @get_dynamic_tile_size() -> index

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {  
    %1 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["scf.for"]} attributes{loop_1} in %arg0 : (!transform.any_op) -> !transform.any_op
    %4:2 = transform.loop.tile %3 [%1] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK: scf.for %[[IIV:.*]] =
  // CHECK: %[[UPPERBOUND:.*]] = arith.addi
  // CHECK: scf.for %[[OUTERIV:.*]] = %[[IIV]] to %[[UPPERBOUND]] step %[[OUTSTEP:.*]]
  // CHECK: %[[INNERUPPER:.*]] = affine.min
  // CHECK: scf.for %[[INNERIV:.*]] =
  // CHECK-SAME: to %[[INNERUPPER]] step
  func.func @loop_tile_dyn(%IN1 : memref<256xf32>, %IN2 : memref<256xf32>, %OUT : memref<256xf32>, %DYN_TILE_SIZE : index) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c256 step %c128 {
      %ub = arith.addi %i, %c128 : index
      scf.for %j = %i to %ub step %c1 {
        %0 = memref.load %IN1[%j] : memref<256xf32>
        %1 = memref.load %IN2[%j] : memref<256xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %OUT[%j] : memref<256xf32>
      } {loop_1}
    } {loop_2}
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["scf.for"]} attributes{loop_1} in %arg0 : (!transform.any_op) -> !transform.any_op

    %8 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %9 = transform.func.get_func_argument %8[3] : (!transform.any_op) -> !transform.any_value

    %4:2 = transform.loop.tile %3 [%9] : (!transform.any_op, !transform.any_value) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK-LABEL: func.func @loop_tile_with_factor_reorder(
  // CHECK: %[[CONST0_0:.*]] = arith.constant 0 : index
  // CHECK: %[[CONST256:.*]] = arith.constant 256 : index
  // CHECK: %[[CONST20:.*]] = arith.constant 20 : index
  // CHECK: scf.for
  // CHECK-SAME: %[[OUTVAR:.*]] = %[[CONST0_0]] to %[[CONST20]] step %c1
  // CHECK-SAME: iter_args(%[[INNERARG0:.*]] = %[[ARG0:.*]]
  // CHECK: %[[CONST0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[INNER_UPPER_BOUND:.*]] = affine.apply
  // CHECK-SAME: [%[[CONST256]], %[[OUTVAR]]
  // CHECK: scf.for %[[INNERVAR:.*]] = %[[CONST0_1]] to
  // CHECK-SAME: iter_args(%[[INNERARG1:.*]] = %[[INNERARG0]]
  // CHECK-NEXT: arith.addi %[[OUTVAR]], %[[INNERVAR]] : index
  func.func @loop_tile_with_factor_reorder(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = tensor.empty() : tensor<256xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %1 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %0) -> (tensor<256xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [2] [1] : tensor<256xf32> to tensor<2xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg2] [2] [1] : tensor<256xf32> to tensor<2xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg2] [2] [1] : tensor<256xf32> to tensor<2xf32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 : tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_1 : tensor<2xf32>) -> tensor<2xf32>
      %inserted_slice = tensor.insert_slice %2 into %arg3[%arg2] [2] [1] : tensor<2xf32> into tensor<256xf32>
      scf.yield %inserted_slice : tensor<256xf32>
    }
    return %1 : tensor<256xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {  
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2:2 = transform.loop.tile %0 [20] {is_reorder_mode = true} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s0 - s1 * (s0 ceildiv s2), s0 ceildiv s2)>
  // CHECK-DAG: #[[MAP1:.*]] = affine_map<() -> (13)>
  // CHECK: func.func @loop_tile_with_yield(
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: scf.for %[[ARG:.*]] = {{.*}} to %[[C20]] step %[[C1]]
  // CHECK: %[[APPLY0:.*]] = affine.min #[[MAP0]]()
  func.func @loop_tile_with_yield(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = tensor.empty() : tensor<256xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %1 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %0) -> (tensor<256xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [1] [1] : tensor<256xf32> to tensor<1xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg2] [1] [1] : tensor<256xf32> to tensor<1xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg2] [1] [1] : tensor<256xf32> to tensor<1xf32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 : tensor<1xf32>, tensor<1xf32>) outs(%extracted_slice_1 : tensor<1xf32>) -> tensor<1xf32>
      %inserted_slice = tensor.insert_slice %2 into %arg3[%arg2] [1] [1] : tensor<1xf32> into tensor<256xf32>
      scf.yield %inserted_slice : tensor<256xf32>
    }
    return %1 : tensor<256xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {  
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2:2 = transform.loop.tile %0 [20] {is_npart_mode = true} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
