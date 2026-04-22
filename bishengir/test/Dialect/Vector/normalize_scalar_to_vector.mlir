// RUN: bishengir-opt %s -normalize-vector | FileCheck %s

func.func @test_scalar_to_vector(%arg3: f32, %arg4: f32, %arg5: f32, %arg14: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK: vector.broadcast {{.*}} f32 to vector<64xf32>
  // CHECK: math.absf {{.*}} vector<64xf32>
  // CHECK: vector.broadcast {{.*}} vector<64xf32>
  // CHECK: mathExt.divfhp {{.*}} vector<64xf32>
  // CHECK: math.round {{.*}} vector<64xf32>
  %1 = math.absf %arg3 : f32
  %2 = mathExt.divfhp %1, %arg4 : f32
  %3 = math.round %2 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : f32
  %4 = arith.mulf %3, %arg4 : f32
  %5 = arith.subf %1, %4 : f32
  %6 = arith.mulf %5, %arg5 : f32
  %7 = vector.broadcast %6 : f32 to vector<64xf32>
  vector.transfer_write %7, %arg14[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return 
}

// -----
func.func @test_unary_followed_by_broadcast(%arg0: f32, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK-LABEL: @test_unary_followed_by_broadcast
  // CHECK: vector.broadcast %arg0 : f32 to vector<64xf32>
  // CHECK: math.absf {{.*}} : vector<64xf32>
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK-NOT: vector.broadcast {{.*}} vector<64xf32>
  %0 = math.absf %arg0 : f32
  %1 = vector.broadcast %0 : f32 to vector<64xf32>
  vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// -----
func.func @test_multiple_broadcast_users(%arg0: f32, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK-LABEL: @test_multiple_broadcast_users
  // CHECK: vector.broadcast %arg0 : f32 to vector<64xf32>
  // CHECK: math.log {{.*}} : vector<64xf32>
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK-NOT: vector.broadcast {{.*}} vector<64xf32>
  %0 = math.log %arg0 : f32
  %1 = vector.broadcast %0 : f32 to vector<64xf32>
  %2 = vector.broadcast %0 : f32 to vector<64xf32>
  vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  vector.transfer_write %2, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// -----
func.func @test_round_with_attributes(%arg0: f32, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK-LABEL: @test_round_with_attributes
  // CHECK: vector.broadcast %arg0 : f32 to vector<64xf32>
  // CHECK: math.round {{.*}} {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : vector<64xf32>
  %0 = math.round %arg0 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : f32
  %1 = vector.broadcast %0 : f32 to vector<64xf32>
  vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}
