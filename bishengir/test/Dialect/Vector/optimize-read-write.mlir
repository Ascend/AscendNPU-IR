// RUN: bishengir-opt %s -canonicalize -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func @test_forward_scalar_broadcast_simple
// CHECK-SAME:  (%[[VAL:.*]]: f32, 
// CHECK: %[[RES:.*]] = vector.broadcast %[[VAL]] : f32 to vector<1xf32>
func.func @test_forward_scalar_broadcast_simple(%arg0: f32, %arg1: tensor<1x64xf32>) -> vector<1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
  %1 = tensor.empty() : tensor<1xf32>
  %2 = vector.constant_mask [1] : vector<64xi1>
  %3 = vector.broadcast %arg0 : f32 to vector<64xf32>
  %4 = vector.transfer_write %3, %1[%c0], %2 {in_bounds = [true]} : vector<64xf32>, tensor<1xf32>
  %5 = vector.transfer_read %4[%c0], %cst {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
  %6 = vector.multi_reduction <add>, %0, %5 [1] : vector<1x64xf32> to vector<1xf32>
  return %6 : vector<1xf32>
}

// -----

// Pattern should fail if mask is not unit
// CHECK-LABEL: func @test_forward_scalar_broadcast_non_unit_fail
// CHECK: %[[WRITE:.*]] = vector.transfer_write {{.*}} vector<64xf32>, tensor<10xf32>
// CHECK: vector.transfer_read %[[WRITE]]{{.*}} tensor<10xf32>, vector<1xf32>
func.func @test_forward_scalar_broadcast_non_unit_fail(%arg0: f32, %arg1: tensor<1x64xf32>) -> vector<1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
  %1 = tensor.empty() : tensor<10xf32>
  %2 = vector.constant_mask [10] : vector<64xi1>
  %3 = vector.broadcast %arg0 : f32 to vector<64xf32>
  %4 = vector.transfer_write %3, %1[%c0], %2 {in_bounds = [true]} : vector<64xf32>, tensor<10xf32>
  %5 = vector.transfer_read %4[%c0], %cst {in_bounds = [true]} : tensor<10xf32>, vector<1xf32>
  %6 = vector.multi_reduction <add>, %0, %5 [1] : vector<1x64xf32> to vector<1xf32>
  return %6 : vector<1xf32>
}

// -----

// CHECK-LABEL: func @test_forward_write_empty_to_read
// CHECK: %[[cmp:.*]] = arith.cmpi {{.*}} : vector<1x256xi8>
// CHECK: %[[ucc:.*]] = builtin.unrealized_conversion_cast %[[cmp]] : vector<1x256xi1> to vector<1x64xi1>
// CHECK: return %[[ucc]] : vector<1x64xi1>
func.func @test_forward_write_empty_to_read(%arg0: tensor<1x64xi8>, %arg1: tensor<1x64xi8>) -> vector<1x64xi1> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %false = arith.constant false
  %0 = tensor.empty() : tensor<1x64xi1>
  %1 = vector.create_mask %c1, %c64 : vector<1x256xi1>
  %2 = vector.transfer_read %arg0[%c0, %c0], %c0_i8, %1 {in_bounds = [true, true]} : tensor<1x64xi8>, vector<1x256xi8>
  %3 = vector.transfer_read %arg1[%c0, %c0], %c0_i8, %1 {in_bounds = [true, true]} : tensor<1x64xi8>, vector<1x256xi8>
  %4 = arith.cmpi ne, %2, %3 : vector<1x256xi8>
  %5 = vector.transfer_write %4, %0[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x256xi1>, tensor<1x64xi1>
  %6 = vector.transfer_read %5[%c0, %c0], %false {in_bounds = [true, true]} : tensor<1x64xi1>, vector<1x64xi1>
  return %6 : vector<1x64xi1>
}