// RUN: bishengir-opt %s -normalize-vector -vector-transfer-lowering -convert-vector-to-hivmave -convert-arith-to-hivmave --split-input-file | FileCheck %s

// CHECK-LABEL: @test_transfer_read_64xf32
func.func @test_transfer_read_64xf32(%arg: memref<?xf32, #hivm.address_space<ub>>) -> vector<64xf32> {
  // CHECK: %[[CONST:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.vload <NORM> %{{.*}}[%[[CONST]]] : memref<?xf32, #hivm.address_space<ub>> into vector<64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %1 = vector.transfer_read %arg[%c0], %cst {in_bounds = [true]} : memref<?xf32, #hivm.address_space<ub>>, vector<64xf32>
  return %1 : vector<64xf32>
}

// -----

// : @test_transfer_read_1xf32
func.func @test_transfer_read_1xf32(%arg: memref<1xf32, #hivm.address_space<ub>>) -> vector<1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %1 = ave.hir.scalar_broadcast %0 : f32 -> vector<f32>
  %0 = vector.transfer_read %arg[%c0], %cst {in_bounds = [true]} : memref<1xf32, #hivm.address_space<ub>>, vector<1xf32>
  return %0 : vector<1xf32>
}

// -----

// CHECK-LABEL: @test_transfer_write_64xf32
func.func @test_transfer_write_64xf32(%arg: memref<?xf32, #hivm.address_space<ub>>, %vector: vector<64xf32>) {
  // CHECK: %[[CONST:.*]] = arith.constant 0 : index
  // CHECK: %[[MASK:.*]] = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: ave.hir.masked_store <NORM_B32> %{{.*}}[%[[CONST]]], %[[MASK]], %{{.*}} : memref<?xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %arg[%c0] {in_bounds = [true]} : vector<64xf32>, memref<?xf32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: @test_transfer_write_1xf32
func.func @test_transfer_write_1xf32(%arg: memref<1xf32, #hivm.address_space<ub>>, %vector: vector<1xf32>) {
  // TODO: Support this pattern issue2499
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %arg[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: @test_transfer_read_64xf16_before_extf
func.func @test_transfer_read_64xf16_before_extf(%arg: memref<64xf16, #hivm.address_space<ub>>) -> vector<64xf32> {
  // CHECK: %[[CONST:.*]] = arith.constant 0 : index
  // CHECK: %[[LOAD:.*]] = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf16, #hivm.address_space<ub>> into vector<64xf16>
  // CHECK: %{{.*}} = ave.hir.vextf %[[LOAD]], <part_even>, %{{.*}} : vector<64xf16>, vector<64xf32>
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %1 = vector.transfer_read %arg[%c0], %cst {in_bounds = [true]} : memref<64xf16, #hivm.address_space<ub>>, vector<64xf16>
  %2 = arith.extf %1 : vector<64xf16> to vector<64xf32>
  return %2 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_transfer_write_64xf16_after_truncf
func.func @test_transfer_write_64xf16_after_truncf(%arg: memref<64xf16, #hivm.address_space<ub>>, %vector: vector<64xf32>) {
  // TODO: Support this pattern issue2498
  %0 = arith.truncf %vector : vector<64xf32> to vector<64xf16>
  %c0 = arith.constant 0 : index
  vector.transfer_write %0, %arg[%c0] {in_bounds = [true]} : vector<64xf16>, memref<64xf16, #hivm.address_space<ub>>
  return
}


