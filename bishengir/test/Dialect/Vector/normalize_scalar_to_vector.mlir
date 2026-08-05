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

// -----
func.func @test_fma_scalar_to_vector(%arg0: f32, %arg1: f32, %arg2: f32, %arg14: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK: vector.broadcast {{.*}} f32 to vector<64xf32>
  // CHECK: math.absf {{.*}} : vector<64xf32>
  // CHECK: vector.broadcast {{.*}} f32 to vector<64xf32>
  // CHECK: vector.broadcast {{.*}} f32 to vector<64xf32>
  // CHECK: math.fma {{.*}} {{.*}} {{.*}} : vector<64xf32>
  %0 = math.absf %arg1 : f32
  %1 = math.fma %arg0, %0, %arg2 : f32
  %2 = vector.broadcast %1 : f32 to vector<64xf32>
  vector.transfer_write %2, %arg14[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// -----

// Scalar bit-twiddling chain (Triton absf lowering) ending in a
// `vector.broadcast` sink. The chain must be vectorized as a whole so the
// scalar `arith.bitcast f32 <-> i32` does not survive to LLVM lowering
// (the AIC vec backend cannot select `i32 = bitcast f32`).
// CHECK-LABEL: @test_scalar_bitcast_chain_broadcast
// CHECK: vector.broadcast %arg0 : f32 to vector<64xf32>
// CHECK: arith.bitcast {{.*}} : vector<64xf32> to vector<64xi32>
// CHECK: arith.andi {{.*}} : vector<64xi32>
// CHECK: arith.addi {{.*}} : vector<64xi32>
// CHECK: arith.bitcast {{.*}} : vector<64xi32> to vector<64xf32>
// CHECK: math.absf {{.*}} : vector<64xf32>
// CHECK: arith.bitcast {{.*}} : vector<64xf32> to vector<64xi32>
// CHECK: arith.minsi {{.*}} : vector<64xi32>
// CHECK-NOT: arith.bitcast {{.*}} f32 to i32
// CHECK-NOT: arith.bitcast {{.*}} i32 to f32
func.func @test_scalar_bitcast_chain_broadcast(%arg0: f32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<64xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %0 = arith.bitcast %arg0 : f32 to i32
  %1 = arith.andi %0, %arg1 : i32
  %2 = arith.addi %1, %arg2 : i32
  %3 = arith.bitcast %2 : i32 to f32
  %4 = math.absf %3 : f32
  %5 = arith.bitcast %4 : f32 to i32
  %6 = arith.minsi %5, %arg3 : i32
  %7 = vector.broadcast %6 : i32 to vector<64xi32>
  vector.transfer_write %7, %arg4[%c0] {in_bounds = [true]} : vector<64xi32>, memref<64xi32, #hivm.address_space<ub>>
  return
}
