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