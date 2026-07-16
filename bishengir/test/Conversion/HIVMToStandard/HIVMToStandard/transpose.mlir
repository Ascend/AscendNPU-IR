// RUN: bishengir-opt -convert-hivm-to-std  -split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_last_axis_transpose_2d(%0 : memref<32x8xf32>, %1: memref<8x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @transpose_2d_with_last_axis_float(%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}})
    hivm.hir.vtranspose ins(%0 :  memref<32x8xf32>) outs(%1 : memref<8x32xf32>) permutation = [1, 0]
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_last_axis_transpose_021(%0 : memref<16x32x8xf32>, %1: memref<16x8x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: scf.for %{{.*}} = %c0 to %c16 step %c1
    // CHECK: %[[src:.*]] = {{.*}} : memref<32x8xf32, strided<[8, 1], offset: ?>>
    // CHECK: %[[dst:.*]] = {{.*}} : memref<8x32xf32, strided<[32, 1], offset: ?>>
    // CHECK: call @transpose_2d_with_last_axis_float(%[[src]], %[[dst]])
    hivm.hir.vtranspose ins(%0 :  memref<16x32x8xf32>) outs(%1 : memref<16x8x32xf32>) permutation = [0, 2, 1]
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_last_axis_transpose_0132(%0 : memref<6x16x32x8xf32>, %1: memref<6x16x8x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: scf.for %{{.*}} = %c0 to %c6 step %c1
    // CHECK: scf.for %{{.*}} = %c0 to %c16 step %c1
    // CHECK: %[[src:.*]] = {{.*}} : memref<32x8xf32, strided<[8, 1], offset: ?>>
    // CHECK: %[[dst:.*]] = {{.*}} : memref<8x32xf32, strided<[32, 1], offset: ?>>
    // CHECK: call @transpose_2d_with_last_axis_float(%[[src]], %[[dst]])
    hivm.hir.vtranspose ins(%0 :  memref<6x16x32x8xf32>) outs(%1 : memref<6x16x8x32xf32>) permutation = [0, 1, 3, 2]
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_last_axis_transpose_0321(%0 : memref<6x16x32x8xf32>, %1: memref<6x8x32x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: scf.for %{{.*}} = %c0 to %c6 step %c1
    // CHECK: %[[src:.*]] = memref.cast {{.*}} : memref<16x32x8xf32, strided<[256, 8, 1], offset: ?>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    // CHECK: %[[dst:.*]] = memref.cast {{.*}} : memref<8x32x16xf32, strided<[512, 16, 1], offset: ?>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    // CHECK: call @transpose_3d_with_last_axis_float(%[[src]], %[[dst]])
    hivm.hir.vtranspose ins(%0 :  memref<6x16x32x8xf32>) outs(%1 : memref<6x8x32x16xf32>) permutation = [0, 3, 2, 1]
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_non_last_transpose_3d(%0 : memref<32x8x16xf32>, %1: memref<8x32x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @transpose_3d_without_last_axis_float(%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}})
    hivm.hir.vtranspose ins(%0 :  memref<32x8x16xf32>)
                        outs(%1 : memref<8x32x16xf32>)
                        permutation = [1, 0, 2]
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_non_last_transpose_0213(%0 : memref<4x32x8x16xf32>, %1: memref<4x8x32x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: scf.for %{{.*}} = %c0 to %c4 step %c1
    // CHECK: %[[src:.*]] = {{.*}} : memref<32x8x16xf32, strided<[128, 16, 1], offset: ?>>
    // CHECK: %[[dst:.*]] = {{.*}} : memref<8x32x16xf32, strided<[512, 16, 1], offset: ?>>
    // CHECK: call @transpose_3d_without_last_axis_float(%[[src]], %[[dst]])
    hivm.hir.vtranspose ins(%0 :  memref<4x32x8x16xf32>)
                        outs(%1 : memref<4x8x32x16xf32>)
                        permutation = [0, 2, 1, 3]

    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_non_last_transpose_032145(%0 : memref<2x3x4x32x8x16xf32>, %1: memref<2x32x4x3x8x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: scf.for %{{.*}} = %c0 to %c2 step %c1
    // CHECK: scf.for %{{.*}} = %c0 to %c4 step %c1
    // CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1
    // CHECK: %[[src:.*]] = {{.*}} : memref<3x32x16xf32, strided<[16384, 128, 1], offset: ?>>
    // CHECK: %[[dst:.*]] = {{.*}} : memref<32x3x16xf32, strided<[1536, 128, 1], offset: ?>>
    // CHECK: call @transpose_3d_without_last_axis_float(%[[src]], %[[dst]])
    hivm.hir.vtranspose ins(%0 :  memref<2x3x4x32x8x16xf32>)
                        outs(%1 : memref<2x32x4x3x8x16xf32>)
                        permutation = [0, 3, 2, 1, 4, 5]

    return
  }
}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func @test_vtranspose_3d_uint64
  // CHECK: scf.for
  // CHECK: transpose_2d_with_last_axis_uint64_t(%[[SRC:.*]], %[[DST:.*]], %[[TMP:.*]])
  func.func @test_vtranspose_3d_uint64() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src = memref.alloc() : memref<37x5x3xui64, strided<[32, 4, 1]>, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<37x3x5xui64, strided<[32, 8, 1]>, #hivm.address_space<ub>>
    %tmp_buf = memref.alloc() : memref<512xui64, #hivm.address_space<ub>>
    hivm.hir.vtranspose ins(%src : memref<37x5x3xui64, strided<[32, 4, 1]>, #hivm.address_space<ub>>) 
                        outs(%dst : memref<37x3x5xui64, strided<[32, 8, 1]>, #hivm.address_space<ub>>) 
                        temp_buffer(%tmp_buf : memref<512xui64, #hivm.address_space<ub>>) 
                        permutation = [0, 2, 1]
    return
  }
}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func @test_vtranspose_3d_210_b64
  // CHECK: transpose_3d_with_last_axis_uint64_t(%[[SRC:.*]], %[[DST:.*]], %[[TMP:.*]])
  func.func @test_vtranspose_3d_210_b64() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src = memref.alloc() : memref<37x5x3xui64, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<3x5x37xui64, #hivm.address_space<ub>>
    %tmp_buf = memref.alloc() : memref<512xui64, #hivm.address_space<ub>>
    hivm.hir.vtranspose ins(%src : memref<37x5x3xui64, #hivm.address_space<ub>>) 
                        outs(%dst : memref<3x5x37xui64, #hivm.address_space<ub>>) 
                        temp_buffer(%tmp_buf : memref<512xui64, #hivm.address_space<ub>>) 
                        permutation = [2, 1, 0]
    return
  }
}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func @test_vtranspose_3d_210_b16
  // CHECK: transpose_3d_with_last_axis_int16_t(%[[SRC:.*]], %[[DST:.*]])
  func.func @test_vtranspose_3d_210_b16() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src = memref.alloc() : memref<15x7x27xi16, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<27x7x15xi16, #hivm.address_space<ub>>
    hivm.hir.vtranspose ins(%src : memref<15x7x27xi16, #hivm.address_space<ub>>) 
                        outs(%dst : memref<27x7x15xi16, #hivm.address_space<ub>>) 
                        permutation = [2, 1, 0]
    return
  }
}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_last_axis_transpose_0321(%0 : memref<6x16x3x32x8xf32>, %1: memref<6x8x3x32x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: scf.for %{{.*}} = %c0 to %c6 step %c1
    // CHECK: scf.for %{{.*}} = %c0 to %c3 step %c1
    // CHECK: %[[src:.*]] = memref.cast {{.*}} : memref<16x32x8xf32, strided<[768, 8, 1], offset: ?>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    // CHECK: %[[dst:.*]] = memref.cast {{.*}} : memref<8x32x16xf32, strided<[1536, 16, 1], offset: ?>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    // CHECK: call @transpose_3d_with_last_axis_float(%[[src]], %[[dst]])
    hivm.hir.vtranspose ins(%0 :  memref<6x16x3x32x8xf32>) outs(%1 : memref<6x8x3x32x16xf32>) permutation = [0, 4, 2, 3, 1]
    return
  }
}
