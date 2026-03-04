// RUN: bishengir-opt -transform-interpreter -split-input-file -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> (tensor<?xf16>, tensor<?xf16>)  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    %1 = tensor.empty(%dim0) : tensor<?xf16>
    // CHECK-NOT: scf.for
    // CHECK: scf.forall
    // CHECK: scf.forall.in_parallel
    // CHECK: tensor.parallel_insert_slice
    // CHECK: tensor.parallel_insert_slice
    %2:2 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<?xf16>, tensor<?xf16>) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg3[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      %inserted_slice1 = tensor.insert_slice %ret into %arg4[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice, %inserted_slice1 : tensor<?xf16>, tensor<?xf16>
    }
    return %2#0, %2#1 : tensor<?xf16>, tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall_with_mapping(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> tensor<?xf16>  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    // CHECK: scf.forall
    // CHECK: mapping = [#hivm.block]
    %1 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg1 = %0) -> (tensor<?xf16>) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg1[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice : tensor<?xf16>
    }
    return %1 : tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 {mapping = [#hivm.block]} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> tensor<?xf16>  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    %some_index = "some_index"() : () -> (index)
    // expected-error @+1 {{the target loop can only yield tensor.insert_slices!}}
    %1:2 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg3 = %0, %arg4 = %some_index) -> (tensor<?xf16>, index) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg3[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice, %offset : tensor<?xf16>, index
    }
    return %1 : tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @test_for_to_forall_with_mapping(%arg0: tensor<?xf16>, %lb: index, %ub: index, %step: index) -> tensor<?xf16>  {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf16>
    %0 = tensor.empty(%dim0) : tensor<?xf16>
    // CHECK-NOT: scf.forall
    %1 = scf.for %arg2 = %lb to %ub step %step iter_args(%arg1 = %0) -> (tensor<?xf16>) {
      %offset = "offset_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %size = "size_compute"(%arg2, %lb, %ub, %dim0) : (index, index, index, index) -> (index)
      %ret = "some_use"() : () -> (tensor<?xf16>)
      %inserted_slice = tensor.insert_slice %ret into %arg1[%offset] [%size] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice : tensor<?xf16>
    // CHECK: map_for_to_forall, mapping = [#hivm.block]
    }
    return %1 : tensor<?xf16>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.for_to_forall %1 {mapping = [#hivm.block], annotate_only = true} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func @test_fors_to_forall_with_mapping(%arg0: tensor<4x64x4xf32>, %arg1: tensor<4x32x4xi32>) {
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x32x4xf32>
    // CHECK: %1 = scf.forall (%arg2, %arg3, %arg4) in (%c4, %c32, %c4) shared_outs(%arg5 = %0) -> (tensor<4x32x4xf32>) {
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x32x4xf32>) {
      %2 = scf.for %arg4 = %c0 to %c32 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x32x4xf32>) {
        %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x32x4xf32>) {
          %extracted = tensor.extract %arg1[%arg2, %arg4, %arg6] : tensor<4x32x4xi32>
          %4 = arith.index_cast %extracted : i32 to index
          %extracted_0 = tensor.extract %arg0[%arg2, %4, %arg6] : tensor<4x64x4xf32>
          %5 = tensor.empty() : tensor<1x1x1xf32>
          %inserted = tensor.insert %extracted_0 into %5[%c0, %c0, %c0] : tensor<1x1x1xf32>
          %inserted_slice = tensor.insert_slice %inserted into %arg7[%arg2, %arg4, %arg6] [1, 1, 1] [1, 1, 1] : tensor<1x1x1xf32> into tensor<4x32x4xf32>
          scf.yield %inserted_slice : tensor<4x32x4xf32>
        } {map_for_to_forall, mapping = [#hivm.block]}
        scf.yield %3 : tensor<4x32x4xf32>
      } {map_for_to_forall, mapping = [#hivm.block]}
      scf.yield %2 : tensor<4x32x4xf32>
    } {map_for_to_forall, mapping = [#hivm.block]}
    // CHECK: mapping = [#hivm.block, #hivm.block, #hivm.block]
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1:3 = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.loop.for_to_forall %1#2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

