// RUN: bishengir-opt %s -hfusion-pull-slice-into-vector-function | FileCheck %s

module {
  // CHECK-LABEL: func.func @result_only_vf(
  // CHECK-SAME: %[[SLICE_ARG:arg[0-9]+]]: tensor<?xf32>
  // CHECK-SAME: %[[FULL_ARG:arg[0-9]+]]: tensor<256xf32>
  // CHECK-SAME: %[[OFFSET:arg[0-9]+]]: index
  // CHECK-SAME: %[[SIZE:arg[0-9]+]]: index
  // CHECK-SAME: %[[STRIDE:arg[0-9]+]]: index
  // CHECK-SAME: -> tensor<256xf32>
  // CHECK: %[[COMPUTED:.*]] = math.exp %[[SLICE_ARG]] : tensor<?xf32>
  // CHECK: %[[INSERTED:.*]] = tensor.insert_slice %[[COMPUTED]] into %[[FULL_ARG]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]] : tensor<?xf32> into tensor<256xf32>
  // CHECK: return %[[INSERTED]] : tensor<256xf32>
  func.func @result_only_vf(%arg0: tensor<?xf32>) -> tensor<?xf32>
      attributes {hivm.vector_function} {
    %0 = math.exp %arg0 : tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  // CHECK-LABEL: func.func @caller(
  // CHECK: %[[CALL:.*]] = call @result_only_vf(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {hivm.vector_function} : (tensor<?xf32>, tensor<256xf32>, index, index, index) -> tensor<256xf32>
  // CHECK-NEXT: return %[[CALL]] : tensor<256xf32>
  func.func @caller(%arg0: tensor<256xf32>, %arg1: tensor<?xf32>)
      -> tensor<256xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %0 = func.call @result_only_vf(%arg1) {hivm.vector_function}
        : (tensor<?xf32>) -> tensor<?xf32>
    %1 = tensor.insert_slice %0 into %arg0[%c2] [%c64] [%c1]
        : tensor<?xf32> into tensor<256xf32>
    return %1 : tensor<256xf32>
  }

  func.func @result_only_vf_exec(%arg0: tensor<?xf32>) -> tensor<?xf32>
      attributes {hivm.vector_function} {
    %0 = math.exp %arg0 : tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  func.func @consume_slice(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }

  func.func @result_only_vf_direct_extra_user(%arg0: tensor<?xf32>)
      -> tensor<?xf32> attributes {hivm.vector_function} {
    %0 = math.exp %arg0 : tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  // CHECK-LABEL: func.func @caller_direct_extra_user(
  // CHECK: %[[CALL:.*]] = call @result_only_vf_direct_extra_user(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {hivm.vector_function} : (tensor<?xf32>, tensor<256xf32>, index, index, index) -> tensor<256xf32>
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[CALL]][%{{.*}}] [%{{.*}}] [%{{.*}}] : tensor<256xf32> to tensor<?xf32>
  // CHECK: call @consume_slice(%[[SLICE]]) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK-NEXT: return %[[CALL]] : tensor<256xf32>
  func.func @caller_direct_extra_user(%arg0: tensor<256xf32>,
                                      %arg1: tensor<?xf32>)
      -> tensor<256xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %0 = func.call @result_only_vf_direct_extra_user(%arg1)
        {hivm.vector_function} : (tensor<?xf32>) -> tensor<?xf32>
    %1 = func.call @consume_slice(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %2 = tensor.insert_slice %0 into %arg0[%c2] [%c64] [%c1]
        : tensor<?xf32> into tensor<256xf32>
    return %2 : tensor<256xf32>
  }

  // CHECK-LABEL: func.func @caller_execute_region_extra_user(
  // CHECK: %[[REGION:.*]] = scf.execute_region -> tensor<256xf32>
  // CHECK:   %[[CALL:.*]] = func.call @result_only_vf_exec(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {hivm.vector_function} : (tensor<?xf32>, tensor<256xf32>, index, index, index) -> tensor<256xf32>
  // CHECK:   scf.yield %[[CALL]] : tensor<256xf32>
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[REGION]][%{{.*}}] [%{{.*}}] [%{{.*}}] : tensor<256xf32> to tensor<?xf32>
  // CHECK: call @consume_slice(%[[SLICE]]) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: return %[[REGION]] : tensor<256xf32>
  func.func @caller_execute_region_extra_user(%arg0: tensor<256xf32>,
                                              %arg1: tensor<?xf32>)
      -> tensor<256xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %0 = scf.execute_region -> tensor<?xf32> {
      %1 = func.call @result_only_vf_exec(%arg1) {hivm.vector_function}
          : (tensor<?xf32>) -> tensor<?xf32>
      scf.yield %1 : tensor<?xf32>
    }
    %1 = func.call @consume_slice(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %2 = tensor.insert_slice %0 into %arg0[%c2] [%c64] [%c1]
        : tensor<?xf32> into tensor<256xf32>
    return %2 : tensor<256xf32>
  }

  func.func @result_only_vf_late_dest(%arg0: tensor<?xf32>) -> tensor<?xf32>
      attributes {hivm.vector_function} {
    %0 = math.exp %arg0 : tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  // The insert_slice destination is defined after the execute_region. Pulling
  // it into the call would make the call use a value that does not dominate it.
  // CHECK-LABEL: func.func @caller_execute_region_late_dest(
  // CHECK: %[[REGION:.*]] = scf.execute_region -> tensor<?xf32>
  // CHECK:   %[[CALL:.*]] = func.call @result_only_vf_late_dest(%{{.*}}) {hivm.vector_function} : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK:   scf.yield %[[CALL]] : tensor<?xf32>
  // CHECK: %[[FULL:.*]] = tensor.empty() : tensor<256xf32>
  // CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[REGION]] into %[[FULL]][%{{.*}}] [%{{.*}}] [%{{.*}}] : tensor<?xf32> into tensor<256xf32>
  // CHECK: return %[[INSERT]] : tensor<256xf32>
  func.func @caller_execute_region_late_dest(%arg0: tensor<?xf32>)
      -> tensor<256xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %0 = scf.execute_region -> tensor<?xf32> {
      %2 = func.call @result_only_vf_late_dest(%arg0) {hivm.vector_function}
          : (tensor<?xf32>) -> tensor<?xf32>
      scf.yield %2 : tensor<?xf32>
    }
    %1 = tensor.empty() : tensor<256xf32>
    %2 = tensor.insert_slice %0 into %1[%c2] [%c64] [%c1]
        : tensor<?xf32> into tensor<256xf32>
    return %2 : tensor<256xf32>
  }

  func.func @result_only_vf_extract_user(%arg0: tensor<256xf32>)
      -> tensor<256xf32> attributes {hivm.vector_function} {
    return %arg0 : tensor<256xf32>
  }

  // Result-only matching must not look through tensor.extract_slice. The
  // extracted value is the insert source type, not the raw call result type.
  // CHECK-LABEL: func.func @caller_result_only_extract_user(
  // CHECK: %[[CALL:.*]] = call @result_only_vf_extract_user(%{{.*}}) {hivm.vector_function} : (tensor<256xf32>) -> tensor<256xf32>
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[CALL]][%{{.*}}] [%{{.*}}] [%{{.*}}] : tensor<256xf32> to tensor<?xf32>
  // CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[SLICE]] into %{{.*}}[%{{.*}}] [%{{.*}}] [%{{.*}}] : tensor<?xf32> into tensor<256xf32>
  // CHECK: return %[[INSERT]] : tensor<256xf32>
  func.func @caller_result_only_extract_user(%arg0: tensor<256xf32>,
                                             %arg1: tensor<256xf32>)
      -> tensor<256xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %0 = func.call @result_only_vf_extract_user(%arg1) {hivm.vector_function}
        : (tensor<256xf32>) -> tensor<256xf32>
    %1 = tensor.extract_slice %0[%c2] [%c64] [%c1]
        : tensor<256xf32> to tensor<?xf32>
    %2 = tensor.insert_slice %1 into %arg0[%c2] [%c64] [%c1]
        : tensor<?xf32> into tensor<256xf32>
    return %2 : tensor<256xf32>
  }
}
