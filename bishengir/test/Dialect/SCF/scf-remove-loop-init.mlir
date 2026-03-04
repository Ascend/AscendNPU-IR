// RUN: bishengir-opt %s -scf-remove-redundant-loop-init -allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func @test_redundant_init
func.func @test_redundant_init(%in0 : tensor<29x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[EMPTYOP:.*]] = tensor.empty()
  // CHECK: scf.for
  // CHECK-SAME: %[[EMPTYOP]]
  %5 = scf.for %arg0 = %c0_i32 to %c768_i32 step %c256_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0) : (tensor<29x128xf16>) -> (tensor<29x256xf32>)
    %inserted_slice = tensor.insert_slice %3 into %arg1[0, %2] [29, 256] [1, 1] : tensor<29x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  return %5 : tensor<29x768xf32>
}

// -----

// CHECK-LABEL: func @test_not_yield_insert_slice
func.func @test_not_yield_insert_slice(%in0 : tensor<29x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c669_i32 = arith.constant 669 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: tensor.empty()
  %4 = scf.for %arg0 = %c0_i32 to %c669_i32 step %c256_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0) : (tensor<29x128xf16>) -> (tensor<29x768xf32>)
    scf.yield %3 : tensor<29x768xf32>
  }
  return %4 : tensor<29x768xf32>
}

// -----

// CHECK-LABEL: func @test_init_used_in_loop
func.func @test_init_used_in_loop(%in0 : tensor<29x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: tensor.empty()
  %5 = scf.for %arg0 = %c0_i32 to %c768_i32 step %c256_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0, %1) : (tensor<29x128xf16>, tensor<29x768xf32>) -> (tensor<29x256xf32>)
    %inserted_slice = tensor.insert_slice %3 into %arg1[0, %2] [29, 256] [1, 1] : tensor<29x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  return %5 : tensor<29x768xf32>
}

// -----

// CHECK-LABEL: func @test_iter_arg_used_not_insert_slice
func.func @test_iter_arg_used_not_insert_slice(%in0 : tensor<29x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: tensor.empty()
  %5 = scf.for %arg0 = %c0_i32 to %c768_i32 step %c256_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0, %arg1) : (tensor<29x128xf16>, tensor<29x768xf32>) -> (tensor<29x256xf32>)
    %inserted_slice = tensor.insert_slice %3 into %arg1[0, %2] [29, 256] [1, 1] : tensor<29x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  return %5 : tensor<29x768xf32>
}

// -----

// CHECK-LABEL: func @test_not_unique_slice
func.func @test_not_unique_slice(%in0 : tensor<27x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: tensor.empty()
  %5 = scf.for %arg0 = %c0_i32 to %c768_i32 step %c256_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0, %arg1) : (tensor<27x128xf16>, tensor<29x768xf32>) -> (tensor<27x256xf32>)
    %inserted_slice = tensor.insert_slice %3 into %arg1[0, %2] [27, 256] [1, 1] : tensor<27x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  return %5 : tensor<29x768xf32>
}

// -----

// CHECK-LABEL: func @test_not_full_coverage_mismatch_low_bound
func.func @test_not_full_coverage_mismatch_low_bound(%in0 : tensor<29x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: tensor.empty()
  %5 = scf.for %arg0 = %c1_i32 to %c768_i32 step %c256_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0) : (tensor<29x128xf16>) -> (tensor<29x256xf32>)
    %inserted_slice = tensor.insert_slice %3 into %arg1[0, %2] [29, 256] [1, 1] : tensor<29x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  return %5 : tensor<29x768xf32>
}

// -----

// CHECK-LABEL: func @test_not_full_coverage_mismatch_upper_bound
func.func @test_not_full_coverage_mismatch_upper_bound(%in0 : tensor<29x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c669_i32 = arith.constant 669 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: tensor.empty()
  %5 = scf.for %arg0 = %c0_i32 to %c669_i32 step %c256_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0) : (tensor<29x128xf16>) -> (tensor<29x256xf32>)
    %inserted_slice = tensor.insert_slice %3 into %arg1[0, %2] [29, 256] [1, 1] : tensor<29x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  return %5 : tensor<29x768xf32>
}

// -----

// CHECK-LABEL: func @test_not_full_coverage_mismatch_step
func.func @test_not_full_coverage_mismatch_step(%in0 : tensor<29x128xf16>) -> tensor<29x768xf32> {
  %1 = "some_use"(): () -> (tensor<29x768xf32>)
  %c255_i32 = arith.constant 255 : i32
  %c768_i32 = arith.constant 768 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: tensor.empty()
  %5 = scf.for %arg0 = %c0_i32 to %c768_i32 step %c255_i32 iter_args(%arg1 = %1) -> (tensor<29x768xf32>)  : i32 {
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = "some_use"(%in0) : (tensor<29x128xf16>) -> (tensor<29x256xf32>)
    %inserted_slice = tensor.insert_slice %3 into %arg1[0, %2] [29, 256] [1, 1] : tensor<29x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  return %5 : tensor<29x768xf32>
}