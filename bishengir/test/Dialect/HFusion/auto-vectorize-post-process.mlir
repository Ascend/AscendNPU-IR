// RUN: bishengir-opt %s -hfusion-auto-vectorize -hfusion-pull-slice-into-vector-function | FileCheck %s

module {
  // CHECK-LABEL: func @test_vectorization_post_process_outlined_vf_0(
  // CHECK:       tensor.extract_slice
  // CHECK:       tensor.insert_slice
  func.func @test_vectorization_post_process_outlined_vf_0(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hivm.vector_function} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = scf.for %arg1 = %c0 to %c3 step %c1 iter_args(%arg2 = %arg0) -> (tensor<?x?xf32>) {
      %1 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
        %2 = math.exp %arg4 : tensor<?x?xf32>
        scf.yield %2 : tensor<?x?xf32>
      }
      scf.yield %1 : tensor<?x?xf32>
    }
    return %0 : tensor<?x?xf32>
  }

  // CHECK-LABEL: func @test_vectorization_post_process(
  // CHECK: func.call @test_vectorization_post_process_outlined_vf_0(
  // CHECK: tensor.extract_slice
  func.func @test_vectorization_post_process(%arg0: tensor<256x256xf32>) {
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %extracted_slice = tensor.extract_slice %arg0[%c6, %c5] [%c4, %c3] [%c2, %c1] : tensor<256x256xf32> to tensor<?x?xf32>
    %0 = scf.execute_region -> tensor<?x?xf32> {
      %1 = func.call @test_vectorization_post_process_outlined_vf_0(%extracted_slice) {hivm.vector_function} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      scf.yield %1 : tensor<?x?xf32>
    }
    return
  }
}