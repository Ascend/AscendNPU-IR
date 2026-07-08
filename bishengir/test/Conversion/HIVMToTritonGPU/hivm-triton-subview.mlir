// RUN: bishengir-opt %s --convert-hivm-to-tritongpu -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: tt.func @subview_dynamic_offset_load
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<i64>,
// CHECK:       %[[ROWRANGE:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK:       %[[ROWCOL:.*]] = tt.reshape %[[ROWRANGE]] : tensor<32xi32> -> tensor<32x1xi32>
// CHECK:       %[[ROWSTRIDE:.*]] = arith.constant dense<16> : tensor<32x1xi32>
// CHECK:       %{{.*}} = arith.muli %[[ROWCOL]], %[[ROWSTRIDE]] : tensor<32x1xi32>
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : tensor<32x1xi32>
// CHECK:       %[[BASEPTRS:.*]] = tt.splat %[[ARG0]] : !tt.ptr<i64> -> tensor<32x1x!tt.ptr<i64>>
// CHECK:       %[[ROWPTRS:.*]] = tt.addptr %[[BASEPTRS]], %{{.*}}
// CHECK:       %{{.*}} = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:       %{{.*}} = tt.reshape %{{.*}} : tensor<16xi32> -> tensor<1x16xi32>
// CHECK:       %[[BCASTPTRS:.*]] = tt.broadcast %[[ROWPTRS]] : tensor<32x1x!tt.ptr<i64>> -> tensor<32x16x!tt.ptr<i64>>
// CHECK:       %[[BCASTCOL:.*]] = tt.broadcast %{{.*}} : tensor<1x16xi32> -> tensor<32x16xi32>
// CHECK:       %[[TILEPTRS:.*]] = tt.addptr %[[BCASTPTRS]], %[[BCASTCOL]] : tensor<32x16x!tt.ptr<i64>>, tensor<32x16xi32>
// CHECK:       tt.load %[[TILEPTRS]] evictionPolicy = evict_first : tensor<32x16x!tt.ptr<i64>>
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @subview_dynamic_offset_load(%arg0: memref<?xi64>, %arg1: index, %arg2: memref<32x16xi64>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<?xi64> to memref<64x16xi64, strided<[16, 1]>>
    %0 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg1]
    %subview = memref.subview %reinterpret_cast[%0, 0] [32, 16] [1, 1] : memref<64x16xi64, strided<[16, 1]>> to memref<32x16xi64, strided<[16, 1], offset: ?>>
    hivm.hir.load ins(%subview : memref<32x16xi64, strided<[16, 1], offset: ?>>) outs(%arg2 : memref<32x16xi64>) eviction_policy = <EvictFirst>
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_static_offset_load
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f32>,
// CHECK:       %{{.*}} = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:       %{{.*}} = tt.reshape %{{.*}} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.constant dense<16> : tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.muli %{{.*}}, %{{.*}} : tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.constant dense<256> : tensor<16x1xi32>
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : tensor<16x1xi32>
// CHECK:       %{{.*}} = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK:       tt.load {{.*}} evictionPolicy = evict_first

module {
  func.func @subview_static_offset_load(%arg0: memref<?xf32>, %arg1: memref<16x16xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<?xf32> to memref<64x16xf32, strided<[16, 1]>>
    %subview = memref.subview %reinterpret_cast[16, 0] [16, 16] [1, 1] : memref<64x16xf32, strided<[16, 1]>> to memref<16x16xf32, strided<[16, 1], offset: 256>>
    hivm.hir.load ins(%subview : memref<16x16xf32, strided<[16, 1], offset: 256>>) outs(%arg1 : memref<16x16xf32>) eviction_policy = <EvictFirst>
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_hivm_address_space_local_load
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f32, 6>,
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.load {{.*}} : tensor<16x16x!tt.ptr<f32, 6>>

module {
  func.func @subview_hivm_address_space_local_load(%arg0: memref<4x16x16xf32, #hivm.address_space<ub>>, %arg1: index) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %subview = memref.subview %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = hivm.hir.local_load ins(%subview : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>) -> tensor<16x16xf32>
    return
  }
}

// -----

// CHECK-LABEL: tt.func @subview_hivm_address_space_local_store
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f32, 6>,
// CHECK-NOT:   memref.subview
// CHECK-NOT:   memref.reinterpret_cast
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.store {{.*}} : tensor<16x16x!tt.ptr<f32, 6>>

module {
  func.func @subview_hivm_address_space_local_store(%arg0: memref<4x16x16xf32, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<16x16xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = bufferization.to_tensor %arg2 restrict writable : memref<16x16xf32>
    %subview = memref.subview %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%subview : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %0 : tensor<16x16xf32>)
    return
  }
}
