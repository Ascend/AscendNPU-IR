// RUN: bishengir-opt -allow-unregistered-dialect -optimize-dps-op-with-yielded-insert-slice %s | FileCheck %s
// RUN: bishengir-opt -allow-unregistered-dialect \
// RUN:   -optimize-dps-op-with-yielded-insert-slice \
// RUN:   -one-shot-bufferize=allow-unknown-ops -cse -canonicalize-ext %s | FileCheck %s -check-prefix=CHECK-ONE-SHOT

func.func @optimize_dps_inits(%lb: index, %ub: index, %step: index) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %init = "some_value"() : () -> (tensor<64xf32>)
  %add_src0 = "some_value"() : () -> (tensor<16xf32>)
  %add_src1 = "some_value"() : () -> (tensor<16xf32>)
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<64xf32>) {
    // CHECK: %[[OFFSET:.*]] = "some_calculation"
    %offset = "some_calculation"(%lb, %ub, %step) : (index, index, index) -> (index)
    %empty = tensor.empty() : tensor<16xf32>
    // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG:.*]][%[[OFFSET]]] [16] [1] : tensor<64xf32> to tensor<16xf32>
    // CHECK: linalg.add ins({{.*}}) outs(%[[SLICE]] : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT: linalg.add
    %1 = linalg.add ins(%add_src0, %add_src1 : tensor<16xf32>, tensor<16xf32>) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT-NOT: memref.copy
    // CHECK: tensor.insert_slice {{.*}} into %[[ARG]][%[[OFFSET]]] [16] [1] : tensor<16xf32> into tensor<64xf32>
    %inserted_slice = tensor.insert_slice %1 into %arg1[%offset] [16] [1] : tensor<16xf32> into tensor<64xf32>
    scf.yield %inserted_slice : tensor<64xf32>
  }
  return %res : tensor<64xf32>
}

// -----

func.func @dominance_issue(%lb: index, %ub: index, %step: index) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %init = "some_value"() : () -> (tensor<64xf32>)
  %add_src0 = "some_value"() : () -> (tensor<16xf32>)
  %add_src1 = "some_value"() : () -> (tensor<16xf32>) 
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<64xf32>) {
    %value = "some_calculation"(%lb, %ub, %step) : (index, index, index) -> (index)
    // CHECK-NOT: tensor.extract_slice
    %empty = tensor.empty() : tensor<16xf32>
    %1 = linalg.add ins(%add_src0, %add_src1 : tensor<16xf32>, tensor<16xf32>) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT: memref.copy
    %offset = "some_calculation"(%value) : (index) -> (index)
    %inserted_slice = tensor.insert_slice %1 into %arg1[%offset] [16] [1] : tensor<16xf32> into tensor<64xf32>
    scf.yield %inserted_slice : tensor<64xf32>
  }
  return %res : tensor<64xf32>
}

// -----

func.func @init_not_empty(%lb: index, %ub: index, %step: index) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %init = "some_value"() : () -> (tensor<64xf32>)
  %add_src0 = "some_value"() : () -> (tensor<16xf32>)
  %add_src1 = "some_value"() : () -> (tensor<16xf32>) 
  %add_dst = "some_value"() : () -> (tensor<16xf32>) 
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<64xf32>) {
    %value = "some_calculation"(%lb, %ub, %step) : (index, index, index) -> (index)
    // CHECK-NOT: tensor.extract_slice
    %1 = linalg.add ins(%add_src0, %add_src1 : tensor<16xf32>, tensor<16xf32>) outs(%add_dst : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT: memref.copy
    %offset = "some_calculation"(%value) : (index) -> (index)
    %inserted_slice = tensor.insert_slice %1 into %arg1[%offset] [16] [1] : tensor<16xf32> into tensor<64xf32>
    scf.yield %inserted_slice : tensor<64xf32>
  }
  return %res : tensor<64xf32>
}