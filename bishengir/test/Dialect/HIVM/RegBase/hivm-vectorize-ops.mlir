// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops --cse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @test_hivm_vadd_vectorize_0(
// CHECK-NOT: vector.create_mask
// CHECK: %[[VAL_7:.*]] = vector.transfer_read %{{.*}}{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true]} : tensor<16x16xf32>, vector<4x8xf32>
// CHECK: %[[VAL_8:.*]] = vector.transfer_read %{{.*}}{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true]} : tensor<16x16xf32>, vector<4x8xf32>
// CHECK: %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : vector<4x8xf32>
// CHECK: %[[VAL_10:.*]] = vector.transfer_write %[[VAL_9]], %{{.*}}{{\[}}%{{.*}},  %{{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, tensor<16x16xf32>
func.func @test_hivm_vadd_vectorize_0(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> 
attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}
