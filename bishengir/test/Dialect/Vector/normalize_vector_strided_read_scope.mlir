// RUN: bishengir-opt %s -normalize-vector | FileCheck %s

// The first version extends only 1-D identity reads. Keep the 2-D read at this
// stage: its segmented index materialization requires a separate backend fix.
// This checks the scope boundary, not correctness of the existing 2-D backend.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @keep_2d_identity
  // CHECK-NOT: vector.gather
  // CHECK: vector.transfer_read {{.*}} : memref<2x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>, vector<2x32xf32>
  // CHECK-NOT: vector.gather
  // CHECK: return
  func.func @keep_2d_identity(%src: memref<2x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>) -> vector<2x32xf32> {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>, vector<2x32xf32>
    return %v : vector<2x32xf32>
  }
}
