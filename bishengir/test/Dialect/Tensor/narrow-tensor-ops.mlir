// RUN: bishengir-opt -allow-unregistered-dialect -narrow-tensor-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test_empty
// CHECK: scf.for
// CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty
// CHECK: "use1"(%[[EMPTY]])
// CHECK: "use2"(%[[EMPTY]])
func.func @test_empty() {
  %1 = tensor.empty() : tensor<128xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %arg = %c0 to %c10 step %c1 {
    "use1"(%1) : (tensor<128xf32>) -> ()
    "use2"(%1) : (tensor<128xf32>) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @test_extract_slice
// CHECK: scf.for
// CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty
// CHECK-NEXT: %[[SLICE:.*]] = tensor.extract_slice %[[EMPTY]]
// CHECK: "use1"(%[[SLICE]])
// CHECK: "use2"(%[[SLICE]])
func.func @test_extract_slice() {
  %1 = tensor.empty() : tensor<128xf32>
  %2 = tensor.extract_slice %1[0] [16] [1] : tensor<128xf32> to tensor<16xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %arg = %c0 to %c10 step %c1 {
    "use1"(%2) : (tensor<16xf32>) -> ()
    "use2"(%2) : (tensor<16xf32>) -> ()
  }
  return
}
