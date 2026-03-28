// RUN: bishengir-opt --hfusion-pre-vectorization-fusion --debug-only="hfusion-utils" -split-input-file %s | FileCheck %s

// Fill directly used by matmul - should NOT be converted to generic
// CHECK-LABEL: func @test_direct_matmul
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @test_direct_matmul() -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = tensor.empty() : tensor<16x16xf32>
  %matmul = linalg.matmul ins(%fill, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  return %matmul : tensor<16x16xf32>
}

// -----

// Fill directly used by batch_matmul - should NOT be converted to generic
// CHECK-LABEL: func @test_batch_matmul
// CHECK: linalg.fill
// CHECK: linalg.batch_matmul
func.func @test_batch_matmul() -> tensor<4x16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
  
  %1 = tensor.empty() : tensor<4x16x16xf32>
  %2 = tensor.empty() : tensor<4x16x16xf32>
  %matmul = linalg.batch_matmul ins(%fill, %1 : tensor<4x16x16xf32>, tensor<4x16x16xf32>) outs(%2 : tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
  
  return %matmul : tensor<4x16x16xf32>
}

// -----

// Fill directly used by MatMulMx - should NOT be converted to generic
// CHECK-LABEL: func @test_matmul_mx
// CHECK: linalg.fill
// CHECK: hfusion.matmul_mx
func.func @test_matmul_mx() -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  
  %1 = tensor.empty() : tensor<64x64xf8E5M2>
  %2 = tensor.empty() : tensor<64x64xf8E5M2>
  %3 = tensor.empty() : tensor<64x2xi8>
  %4 = tensor.empty() : tensor<64x2xi8>
  %matmul = hfusion.matmul_mx ins(%1, %2, %3, %4 : tensor<64x64xf8E5M2>, tensor<64x64xf8E5M2>, tensor<64x2xi8>, tensor<64x2xi8>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  
  return %matmul : tensor<64x64xf32>
}

// -----

// Fill used by generic op whose first input traces back to matmul - should NOT be converted to generic
// CHECK-LABEL: func @test_generic_trace_matmul
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.generic
func.func @test_generic_trace_matmul() -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = tensor.empty() : tensor<16x16xf32>
  %matmul = linalg.matmul ins(%fill, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %3 = tensor.empty() : tensor<16x16xf32>
  %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%matmul : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<16x16xf32>
  
  return %generic : tensor<16x16xf32>
}

// -----

// Test 5: Fill used by matmul through for loop - should NOT be converted to generic
// CHECK-LABEL: func @test_through_for_loop
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: linalg.matmul
func.func @test_through_for_loop() -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg0 = %fill) -> (tensor<16x16xf32>) {
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = tensor.empty() : tensor<16x16xf32>
    %matmul = linalg.matmul ins(%arg0, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    
    scf.yield %matmul : tensor<16x16xf32>
  }
  
  return %result : tensor<16x16xf32>
}

// -----

// Test 6: Fill used by matmul through nested for loops - should NOT be converted to generic
// CHECK-LABEL: func @test_nested_for_loops
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.matmul
func.func @test_nested_for_loops() -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg0 = %fill) -> (tensor<16x16xf32>) {
    %inner_result = scf.for %j = %c0 to %c1 step %c1 iter_args(%arg1 = %arg0) -> (tensor<16x16xf32>) {
      %1 = tensor.empty() : tensor<16x16xf32>
      %2 = tensor.empty() : tensor<16x16xf32>
      %matmul = linalg.matmul ins(%arg1, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
      
      scf.yield %matmul : tensor<16x16xf32>
    }
    
    scf.yield %inner_result : tensor<16x16xf32>
  }
  
  return %result : tensor<16x16xf32>
}

// -----

// Test 7: Fill not reaching matmul - should be converted to generic
// CHECK-LABEL: func @test_no_reach
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic
func.func @test_no_reach() -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %1 = tensor.empty() : tensor<16x16xf32>
  %add = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%fill, %fill : tensor<16x16xf32>, tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %sum = arith.addf %in1, %in2 : f32
    linalg.yield %sum : f32
  } -> tensor<16x16xf32>
  
  return %add : tensor<16x16xf32>
}

// -----

// Test 8: Fill passed through for loop iter_arg but not used - should be converted to generic
// CHECK-LABEL: func @test_empty_use
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic
// CHECK: scf.for
func.func @test_empty_use() -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst2 = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  // Fill is passed as iter_arg but not actually used in the loop body
  // The loop yields a different value created inside
  %result = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg0 = %fill) -> (tensor<16x16xf32>) {
    // Create a different tensor inside the loop
    %1 = tensor.empty() : tensor<16x16xf32>
    %inner_fill = linalg.fill ins(%cst2 : f32) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    
    // Yield the inner fill, not the iter_arg
    scf.yield %inner_fill : tensor<16x16xf32>
  }
  
  return %result : tensor<16x16xf32>
}

// -----

// Test 9: Fill with multiple users, some reaching matmul - should NOT be converted to generic
// CHECK-LABEL: func @test_multiple_users_some_reach
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @test_multiple_users_some_reach() -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %1 = tensor.empty() : tensor<16x16xf32>
  %add = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%fill, %fill : tensor<16x16xf32>, tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %sum = arith.addf %in1, %in2 : f32
    linalg.yield %sum : f32
  } -> tensor<16x16xf32>
  
  %2 = tensor.empty() : tensor<16x16xf32>
  %3 = tensor.empty() : tensor<16x16xf32>
  %matmul = linalg.matmul ins(%fill, %2 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  return %matmul, %add : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

// Test 10: Fill with multiple users, none reaching matmul - should be converted to generic
// CHECK-LABEL: func @test_multiple_users_none_reach
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic
func.func @test_multiple_users_none_reach() -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  
  %1 = tensor.empty() : tensor<16x16xf32>
  %add = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%fill, %fill : tensor<16x16xf32>, tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %sum = arith.addf %in1, %in2 : f32
    linalg.yield %sum : f32
  } -> tensor<16x16xf32>
  
  %2 = tensor.empty() : tensor<16x16xf32>
  %sub = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%fill, %fill : tensor<16x16xf32>, tensor<16x16xf32>) outs(%2 : tensor<16x16xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %diff = arith.subf %in1, %in2 : f32
    linalg.yield %diff : f32
  } -> tensor<16x16xf32>
  
  return %add : tensor<16x16xf32>
}
