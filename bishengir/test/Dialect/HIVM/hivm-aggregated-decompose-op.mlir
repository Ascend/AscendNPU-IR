// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=before-hivm-align" -split-input-file -verify-diagnostics | FileCheck %s  --check-prefix=BEFOREALIGN
// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=after-hivm-align" -split-input-file -verify-diagnostics | FileCheck %s  --check-prefix=AFTERALIGN
// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=after-infer-hivm-data-layout" -split-input-file -verify-diagnostics | FileCheck %s  --check-prefix=AFTERLAYOUT
// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=after-hivm-recognize-deinterleave" -split-input-file -verify-diagnostics | FileCheck %s  --check-prefix=AFTERDEINTERLEAVE
// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=after-hivm-recognize-broadcast" -split-input-file -verify-diagnostics | FileCheck %s  --check-prefix=AFTERBROADCAST

// AFTERALIGN-LABEL: func @test_static_concat_last_dim
func.func @test_static_concat_last_dim() -> memref<136x4096xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<136x2048xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<136x2048xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<136x4096xf32>
  // CHECK:  %[[SUBVIEW:.*]] = memref.subview %alloc_1[0, 0] [136, 2048] [1, 1] : memref<136x4096xf32> to memref<136x2048xf32, strided<[4096, 1]>>
  // CHECK:  hivm.hir.copy ins(%[[alloc:.*]] : memref<136x2048xf32>) outs(%[[subview:.*]] : memref<136x2048xf32, strided<[4096, 1]>>)
  // CHECK:  %[[SUBVIEW_2:.*]] = memref.subview %alloc_1[0, 2048] [136, 2048] [1, 1] : memref<136x4096xf32> to memref<136x2048xf32, strided<[4096, 1], offset: 2048>>
  // CHECK:  hivm.hir.copy ins(%[[alloc_0:.*]] : memref<136x2048xf32>) outs(%[[subview_2:.*]] : memref<136x2048xf32, strided<[4096, 1], offset: 2048>>)
  hivm.hir.vconcat dim(1) ins(%alloc, %alloc_0 : memref<136x2048xf32>, memref<136x2048xf32>) outs(%alloc_1 : memref<136x4096xf32>)
  return %alloc_1 : memref<136x4096xf32>
}

// -----
// AFTERALIGN-LABEL: func @test_static_concat_unlast_dim
func.func @test_static_concat_unlast_dim() -> memref<256x2048xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x2048xf32> 
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128x2048xf32> 
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x2048xf32>
  // CHECK:  %[[SUBVIEW:.*]] = memref.subview %alloc_1[0, 0] [128, 2048] [1, 1] : memref<256x2048xf32> to memref<128x2048xf32, strided<[2048, 1]>>
  // CHECK:  hivm.hir.copy ins(%[[alloc:.*]] : memref<128x2048xf32>) outs(%[[subview:.*]] : memref<128x2048xf32, strided<[2048, 1]>>)
  // CHECK:  %[[SUBVIEW_2:.*]] = memref.subview %alloc_1[128, 0] [128, 2048] [1, 1] : memref<256x2048xf32> to memref<128x2048xf32, strided<[2048, 1], offset: 262144>>
  // CHECK:  hivm.hir.copy ins(%[[alloc_0:.*]] : memref<128x2048xf32>) outs(%[[subview_2:.*]] : memref<128x2048xf32, strided<[2048, 1], offset: 262144>>)
  hivm.hir.vconcat dim(0) ins(%alloc, %alloc_0 : memref<128x2048xf32>, memref<128x2048xf32>) outs(%alloc_1 : memref<256x2048xf32>)
  return %alloc_1 : memref<256x2048xf32>
}

// -----
// AFTERALIGN-LABEL: func @test_static_concat_middle_dim
func.func @test_static_concat_middle_dim() -> memref<128x768x2048xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128x2048xf32> 
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128x256x2048xf32> 
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<128x384x2048xf32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x768x2048xf32>
  // CHECK:  %[[SUBVIEW:.*]] = memref.subview %alloc_2[0, 0, 0] [128, 128, 2048] [1, 1, 1] : memref<128x768x2048xf32> to memref<128x128x2048xf32, strided<[1572864, 2048, 1]>>
  // CHECK:  hivm.hir.copy ins(%[[alloc:.*]] : memref<128x128x2048xf32>) outs(%[[subview:.*]] : memref<128x128x2048xf32, strided<[1572864, 2048, 1]>>)
  // CHECK:  %[[SUBVIEW_3:.*]] = memref.subview %alloc_2[0, 128, 0] [128, 256, 2048] [1, 1, 1] : memref<128x768x2048xf32> to memref<128x256x2048xf32, strided<[1572864, 2048, 1], offset: 262144>>
  // CHECK:  hivm.hir.copy ins(%[[alloc_0:.*]] : memref<128x256x2048xf32>) outs(%[[subview_3:.*]] : memref<128x256x2048xf32, strided<[1572864, 2048, 1], offset: 262144>>)
  // CHECK:  %[[SUBVIEW_4:.*]] = memref.subview %alloc_2[0, 384, 0] [128, 384, 2048] [1, 1, 1] : memref<128x768x2048xf32> to memref<128x384x2048xf32, strided<[1572864, 2048, 1], offset: 786432>>
  // CHECK:  hivm.hir.copy ins(%[[alloc_1:.*]] : memref<128x384x2048xf32>) outs(%[[subview_4:.*]] : memref<128x384x2048xf32, strided<[1572864, 2048, 1], offset: 786432>>)
  hivm.hir.vconcat dim(1) ins(%alloc, %alloc_0, %alloc_1 : memref<128x128x2048xf32>, memref<128x256x2048xf32>, memref<128x384x2048xf32>) outs(%alloc_2 : memref<128x768x2048xf32>)
  return %alloc_2 : memref<128x768x2048xf32>
}

// -----
// AFTERALIGN-LABEL: func @test_dyn_concat
func.func @test_dyn_concat(%dim0 : index, %dim1: index) attributes { enable_auto_mark_buffer_size } {
  %alloc = memref.alloc(%dim0) {alignment = 64 : i64} : memref<?x2048xf32> 
  %alloc_0 = memref.alloc(%dim0) {alignment = 64 : i64} : memref<?x2048xf32> 
  %alloc_1 = memref.alloc(%dim1) {alignment = 64 : i64} : memref<?x2048xf32>
  %alloc_2 = memref.alloc(%dim0) {alignment = 64 : i64} : memref<?x2048xf32>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %alloc_2[0, 0] [%arg0, 2048] [1, 1] : memref<?x2048xf32> to memref<?x2048xf32, strided<[2048, 1]>>
  // CHECK: hivm.hir.copy ins(%[[alloc:.*]] : memref<?x2048xf32>) outs(%[[subview:.*]] : memref<?x2048xf32, strided<[2048, 1]>>)
  // CHECK: %[[SUBVIEW_3:.*]] = memref.subview %alloc_2[%arg0, 0] [%arg0, 2048] [1, 1] : memref<?x2048xf32> to memref<?x2048xf32, strided<[2048, 1], offset: ?>>
  // CHECK: hivm.hir.copy ins(%[[alloc_0:.*]] : memref<?x2048xf32>) outs(%[[alloc_3:.*]] : memref<?x2048xf32, strided<[2048, 1], offset: ?>>)
  // CHECK: %[[SUBVIEW_4:.*]] = memref.subview %alloc_2[%0, 0] [%arg1, 2048] [1, 1] : memref<?x2048xf32> to memref<?x2048xf32, strided<[2048, 1], offset: ?>>
  // CHECK: hivm.hir.copy ins(%[[SUBVIEW_1:.*]] : memref<?x2048xf32>) outs(%[[SUBVIEW_4:.*]] : memref<?x2048xf32, strided<[2048, 1], offset: ?>>)
  hivm.hir.vconcat dim(0) ins(%alloc, %alloc_0, %alloc_1 : memref<?x2048xf32>, memref<?x2048xf32>, memref<?x2048xf32>) outs(%alloc_2 : memref<?x2048xf32>)
  return 
}

// -----
#map = affine_map<()[s0, s1] -> ((s0 + 15) floordiv 16)>
#map1 = affine_map<()[s0, s1] -> ((s1 + 7) floordiv 8)>
#map2 = affine_map<()[s0, s1] -> (s0 floordiv 16)>
#map3 = affine_map<()[s0, s1] -> (s0 mod 16)>
#map4 = affine_map<()[s0, s1] -> (s1 floordiv 8)>
#map5 = affine_map<()[s0, s1] -> (s1 mod 8)>
#map6 = affine_map<()[s0, s1] -> ((s1 + 15) floordiv 16)>
// AFTERLAYOUT-LABEL: func @mm_nd2nz_trans_brc_nd2nz
func.func @mm_nd2nz_trans_brc_nd2nz(%arg0: memref<256x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, %arg1: memref<128x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, %arg2: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "mix", enable_auto_mark_buffer_size } {
  %cst = arith.constant 2.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %0 = arith.index_cast %arg2 : i32 to index
  %c256 = arith.constant 256 : index
  %c128_0 = arith.constant 128 : index
  %1 = affine.apply #map()[%c256, %c128_0]
  %2 = affine.apply #map1()[%c256, %c128_0]
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %alloc = memref.alloc(%2, %1, %c16, %c8) : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
  %c128_1 = arith.constant 128 : index
  %c128_2 = arith.constant 128 : index
  %3 = affine.apply #map()[%c128_1, %c128_2]
  %4 = affine.apply #map1()[%c128_1, %c128_2]
  %c8_3 = arith.constant 8 : index
  %c16_4 = arith.constant 16 : index
  %alloc_5 = memref.alloc(%4, %3, %c16_4, %c8_3) : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
  %subview = memref.subview %arg0[0, 0] [256, %0] [1, 1] : memref<256x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<256x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %c256_6 = arith.constant 256 : index
  %5 = affine.apply #map()[%c256_6, %0]
  %6 = affine.apply #map1()[%c256_6, %0]
  %c8_7 = arith.constant 8 : index
  %c16_8 = arith.constant 16 : index
  %c0_9 = arith.constant 0 : index
  %c0_10 = arith.constant 0 : index
  %7 = affine.apply #map2()[%c0_9, %c0_10]
  %8 = affine.apply #map3()[%c0_9, %c0_10]
  %9 = affine.apply #map4()[%c0_9, %c0_10]
  %10 = affine.apply #map5()[%c0_9, %c0_10]
  %subview_11 = memref.subview %alloc[%9, %7, %8, %10] [%6, %5, %c16_8, %c8_7] [1, 1, 1, 1] : memref<?x?x?x?xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
  // AFTERLAYOUT: hivm.hir.vbrc ins({{.*}}: f32)
  // AFTERLAYOUT: hivm.hir.nd2nz {dst_continuous} ins({{.*}} : memref<256x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs({{.*}} : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<256x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = %cst : f32
  %11 = arith.minsi %0, %c128 : index
  %subview_12 = memref.subview %arg1[0, 0] [%11, 128] [1, 1] : memref<128x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %c128_13 = arith.constant 128 : index
  %12 = affine.apply #map()[%11, %c128_13]
  %13 = affine.apply #map1()[%11, %c128_13]
  %c8_14 = arith.constant 8 : index
  %c16_15 = arith.constant 16 : index
  %c0_16 = arith.constant 0 : index
  %c0_17 = arith.constant 0 : index
  %14 = affine.apply #map2()[%c0_16, %c0_17]
  %15 = affine.apply #map3()[%c0_16, %c0_17]
  %16 = affine.apply #map4()[%c0_16, %c0_17]
  %17 = affine.apply #map5()[%c0_16, %c0_17]
  %subview_18 = memref.subview %alloc_5[%16, %14, %15, %17] [%13, %12, %c16_15, %c8_14] [1, 1, 1, 1] : memref<?x?x?x?xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
  // AFTERLAYOUT: hivm.hir.vbrc ins({{.*}}: f32)
  // AFTERLAYOUT: hivm.hir.nd2nz {dst_continuous} ins({{.*}} : memref<?x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs({{.*}} : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.nd2nz {dst_continuous} ins(%subview_12 : memref<?x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_18 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = %cst : f32
  %c256_19 = arith.constant 256 : index
  %c128_20 = arith.constant 128 : index
  %18 = affine.apply #map()[%c256_19, %c128_20]
  %19 = affine.apply #map6()[%c256_19, %c128_20]
  %c16_21 = arith.constant 16 : index
  %c16_22 = arith.constant 16 : index
  %alloc_23 = memref.alloc(%19, %18, %c16_22, %c16_21) {alignment = 64 : i64} : memref<?x?x?x?xf32, #hivm.address_space<cc>>
  hivm.hir.mmadL1 ins(%alloc, %alloc_5, %true, %c0, %c0, %c0 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  return
}

// -----
// BEFOREALIGN-LABEL: func @test_load
func.func @test_load(%arg0: memref<256x128xf32, #hivm.address_space<gm>>) {
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<256x128xf32, #hivm.address_space<ub>>
  // BEFOREALIGN: hivm.hir.vbrc
  // BEFOREALIGN: hivm.hir.load ins({{.*}} : memref<256x128xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<256x128xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = {{.*}} : f32
  hivm.hir.load ins(%arg0: memref<256x128xf32, #hivm.address_space<gm>>) outs(%alloc: memref<256x128xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = true
  return 
}

//===----------------------------------------------------------------------===//
// Test VReduceOp Decompose Buffer static and dynamic
//===----------------------------------------------------------------------===//

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_memref_01
func.func @test_reduce_op_multi_memref_01() {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<16x8x32xf32>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x1x32xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<16x8x32xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[alloc_1]][0, 0, 0] [1, 8, 32] [1, 1, 1] : memref<16x8x32xf32> to memref<1x8x32xf32, strided<[256, 32, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[alloc]] : memref<16x8x32xf32>) outs(%[[subview]] : memref<1x8x32xf32, strided<[256, 32, 1]>>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview]] : memref<1x8x32xf32, strided<[256, 32, 1]>>) outs(%[[alloc_0]] : memref<1x1x32xf32>) reduce_dims = [1]

  %src = memref.alloc() : memref<16x8x32xf32>
  %dst = memref.alloc() : memref<1x1x32xf32>
  hivm.hir.vreduce <sum> ins(%src : memref<16x8x32xf32>) outs(%dst : memref<1x1x32xf32>) reduce_dims = [0, 1]
  return
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_memref_01_args
func.func @test_reduce_op_multi_memref_01_args(%src : memref<16x8x32xf32>, %dst : memref<1x1x32xf32>) {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<16x8x32xf32>
  // CHECK: %[[subview:.*]] = memref.subview %alloc[0, 0, 0] [1, 8, 32] [1, 1, 1] : memref<16x8x32xf32> to memref<1x8x32xf32, strided<[256, 32, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[arg0]] : memref<16x8x32xf32>) outs(%[[subview]] : memref<1x8x32xf32, strided<[256, 32, 1]>>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview]] : memref<1x8x32xf32, strided<[256, 32, 1]>>) outs(%[[arg1]] : memref<1x1x32xf32>) reduce_dims = [1]

  hivm.hir.vreduce <sum> ins(%src : memref<16x8x32xf32>) outs(%dst : memref<1x1x32xf32>) reduce_dims = [0, 1]
  return
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_memref_01_dyn
func.func @test_reduce_op_multi_memref_01_dyn(%d : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[alloc:.*]] =   memref.alloc(%[[arg0]]) : memref<16x8x?xf32>
  // CHECK: %[[alloc_0:.*]] = memref.alloc(%[[arg0]]) : memref<1x1x?xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc(%[[arg0]]) : memref<16x8x?xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[alloc_1]][0, 0, 0] [1, 8, %arg0] [1, 1, 1] : memref<16x8x?xf32> to memref<1x8x?xf32, strided<[?, ?, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[alloc]] : memref<16x8x?xf32>) outs(%[[subview]] : memref<1x8x?xf32, strided<[?, ?, 1]>>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview]] : memref<1x8x?xf32, strided<[?, ?, 1]>>) outs(%[[alloc_0]] : memref<1x1x?xf32>) reduce_dims = [1]

  %src = memref.alloc(%d) : memref<16x8x?xf32>
  %dst = memref.alloc(%d) : memref<1x1x?xf32>
  hivm.hir.vreduce <sum> ins(%src : memref<16x8x?xf32>) outs(%dst : memref<1x1x?xf32>) reduce_dims = [0, 1]
  return
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_memref_01_dyn2
func.func @test_reduce_op_multi_memref_01_dyn2(%d : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[alloc:.*]] = memref.alloc(%[[arg0]]) : memref<16x?x32xf32>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x1x32xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc(%[[arg0]]) : memref<16x?x32xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[alloc_1]][0, 0, 0] [1, %arg0, 32] [1, 1, 1] : memref<16x?x32xf32> to memref<1x?x32xf32, strided<[?, 32, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[alloc:.*]] : memref<16x?x32xf32>) outs(%[[subview]] : memref<1x?x32xf32, strided<[?, 32, 1]>>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview:.*]] : memref<1x?x32xf32, strided<[?, 32, 1]>>) outs(%[[alloc_0]] : memref<1x1x32xf32>) reduce_dims = [1]

  %src = memref.alloc(%d) : memref<16x?x32xf32>
  %dst = memref.alloc() : memref<1x1x32xf32>
  hivm.hir.vreduce <sum> ins(%src : memref<16x?x32xf32>) outs(%dst : memref<1x1x32xf32>) reduce_dims = [0, 1]
  return
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_memref_013
func.func @test_reduce_op_multi_memref_013() {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<16x8x10x32xf32>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x1x10x1xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<16x8x10x32xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[alloc_0]][0, 0, 0, 0] [1, 8, 10, 32] [1, 1, 1, 1] : memref<1x1x10x1xf32> to memref<1x8x10x32xf32, strided<[10, 10, 1, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[alloc]] : memref<16x8x10x32xf32>) outs(%[[subview]] : memref<1x8x10x32xf32, strided<[10, 10, 1, 1]>>) reduce_dims = [0]
  // CHECK: %[[subview_2:.*]] = memref.subview %alloc_1[0, 0, 0, 0] [1, 1, 10, 32] [1, 1, 1, 1] : memref<16x8x10x32xf32> to memref<1x1x10x32xf32, strided<[2560, 320, 32, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview]] : memref<1x8x10x32xf32, strided<[10, 10, 1, 1]>>) outs(%[[subview_2]] : memref<1x1x10x32xf32, strided<[2560, 320, 32, 1]>>) reduce_dims = [1]
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview_2]] : memref<1x1x10x32xf32, strided<[2560, 320, 32, 1]>>) outs(%[[alloc_0]] : memref<1x1x10x1xf32>) reduce_dims = [3]

  %src = memref.alloc() : memref<16x8x10x32xf32>
  %dst = memref.alloc() : memref<1x1x10x1xf32>
  hivm.hir.vreduce <sum> ins(%src : memref<16x8x10x32xf32>) outs(%dst : memref<1x1x10x1xf32>) reduce_dims = [0, 1, 3]
  return
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_memref_023
func.func @test_reduce_op_multi_memref_023() {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<16x8x10x32xf32>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x8x1x1xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<16x8x10x32xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[alloc_0]][0, 0, 0, 0] [1, 8, 10, 32] [1, 1, 1, 1] : memref<1x8x1x1xf32> to memref<1x8x10x32xf32, strided<[8, 1, 1, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[alloc]] : memref<16x8x10x32xf32>) outs(%[[subview]] : memref<1x8x10x32xf32, strided<[8, 1, 1, 1]>>) reduce_dims = [0]
  // CHECK: %[[subview_2:.*]] = memref.subview %[[alloc_1]][0, 0, 0, 0] [1, 8, 1, 32] [1, 1, 1, 1] : memref<16x8x10x32xf32> to memref<1x8x1x32xf32, strided<[2560, 320, 32, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview]] : memref<1x8x10x32xf32, strided<[8, 1, 1, 1]>>) outs(%[[subview_2]] : memref<1x8x1x32xf32, strided<[2560, 320, 32, 1]>>) reduce_dims = [2]
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview_2]] : memref<1x8x1x32xf32, strided<[2560, 320, 32, 1]>>) outs(%[[alloc_0]] : memref<1x8x1x1xf32>) reduce_dims = [3]

  %src = memref.alloc() : memref<16x8x10x32xf32>
  %dst = memref.alloc() : memref<1x8x1x1xf32>
  hivm.hir.vreduce <sum> ins(%src : memref<16x8x10x32xf32>) outs(%dst : memref<1x8x1x1xf32>) reduce_dims = [0, 2, 3]
  return
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_memref_0234
func.func @test_reduce_op_multi_memref_0234() {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<16x8x10x32x16x16xf32>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x8x1x1x1x16xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<16x8x10x32x16x16xf32>
  // CHECK: %[[subview:.*]] = memref.subview %alloc_1[0, 0, 0, 0, 0, 0] [1, 8, 10, 32, 16, 16] [1, 1, 1, 1, 1, 1] : memref<16x8x10x32x16x16xf32> to memref<1x8x10x32x16x16xf32, strided<[655360, 81920, 8192, 256, 16, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[alloc]] : memref<16x8x10x32x16x16xf32>) outs(%[[subview]] : memref<1x8x10x32x16x16xf32, strided<[655360, 81920, 8192, 256, 16, 1]>>) reduce_dims = [0]
  // CHECK: %[[subview_2:.*]] = memref.subview %[[alloc_0]][0, 0, 0, 0, 0, 0] [1, 8, 1, 32, 16, 16] [1, 1, 1, 1, 1, 1] : memref<1x8x1x1x1x16xf32> to memref<1x8x1x32x16x16xf32, strided<[128, 16, 16, 16, 16, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview]] : memref<1x8x10x32x16x16xf32, strided<[655360, 81920, 8192, 256, 16, 1]>>) outs(%[[subview_2]] : memref<1x8x1x32x16x16xf32, strided<[128, 16, 16, 16, 16, 1]>>) reduce_dims = [2]
  // CHECK: %[[subview_3:.*]] = memref.subview %[[alloc_1]][0, 0, 0, 0, 0, 0] [1, 8, 1, 1, 16, 16] [1, 1, 1, 1, 1, 1] : memref<16x8x10x32x16x16xf32> to memref<1x8x1x1x16x16xf32, strided<[655360, 81920, 8192, 256, 16, 1]>>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview_2]] : memref<1x8x1x32x16x16xf32, strided<[128, 16, 16, 16, 16, 1]>>) outs(%[[subview_3]] : memref<1x8x1x1x16x16xf32, strided<[655360, 81920, 8192, 256, 16, 1]>>) reduce_dims = [3]
  // CHECK: hivm.hir.vreduce <sum> ins(%[[subview_3]] : memref<1x8x1x1x16x16xf32, strided<[655360, 81920, 8192, 256, 16, 1]>>) outs(%[[alloc_0]] : memref<1x8x1x1x1x16xf32>) reduce_dims = [4]

  %src = memref.alloc() : memref<16x8x10x32x16x16xf32>
  %dst = memref.alloc() : memref<1x8x1x1x1x16xf32>
  hivm.hir.vreduce <sum> ins(%src : memref<16x8x10x32x16x16xf32>) outs(%dst : memref<1x8x1x1x1x16xf32>) reduce_dims = [0, 2, 3, 4]
  return
}

//===----------------------------------------------------------------------===//
// Test VReduceOp Decompose Tensor static and dynamic
//===----------------------------------------------------------------------===//

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_tensor_01
func.func @test_reduce_op_multi_tensor_01() -> tensor<1x1x32xf32> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<16x8x32xf32>
  // CHECK: %[[T1:.*]] = tensor.empty() : tensor<1x1x32xf32>
  // CHECK: %[[T2:.*]] = tensor.empty() : tensor<16x8x32xf32>
  // CHECK: %[[extracted_slice:.*]] = tensor.extract_slice %[[T2]][0, 0, 0] [1, 8, 32] [1, 1, 1] : tensor<16x8x32xf32> to tensor<1x8x32xf32>
  // CHECK: %[[T3:.*]] = hivm.hir.vreduce <sum> ins(%[[T0]] : tensor<16x8x32xf32>) outs(%[[extracted_slice]] : tensor<1x8x32xf32>) reduce_dims = [0] -> tensor<1x8x32xf32>
  // CHECK: %[[T4:.*]] = hivm.hir.vreduce <sum> ins(%[[T3]] : tensor<1x8x32xf32>) outs(%[[T1]] : tensor<1x1x32xf32>) reduce_dims = [1] -> tensor<1x1x32xf32>
  %src = tensor.empty() : tensor<16x8x32xf32>
  %dst = tensor.empty() : tensor<1x1x32xf32>
  %res = hivm.hir.vreduce <sum> ins(%src : tensor<16x8x32xf32>) outs(%dst : tensor<1x1x32xf32>) reduce_dims = [0, 1] -> tensor<1x1x32xf32>
  return %res : tensor<1x1x32xf32>
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_tensor_01_args
func.func @test_reduce_op_multi_tensor_01_args(%src : tensor<16x8x32xf32>, %dst : tensor<1x1x32xf32>) -> tensor<1x1x32xf32> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<16x8x32xf32>
  // CHECK: %[[extracted_slice:.*]] = tensor.extract_slice %[[T0]][0, 0, 0] [1, 8, 32] [1, 1, 1] : tensor<16x8x32xf32> to tensor<1x8x32xf32>
  // CHECK: %[[T1:.*]] = hivm.hir.vreduce <sum> ins(%[[arg0]] : tensor<16x8x32xf32>) outs(%[[extracted_slice]] : tensor<1x8x32xf32>) reduce_dims = [0] -> tensor<1x8x32xf32>
  // CHECK: %[[T2:.*]] = hivm.hir.vreduce <sum> ins(%[[T1]] : tensor<1x8x32xf32>) outs(%[[arg1]] : tensor<1x1x32xf32>) reduce_dims = [1] -> tensor<1x1x32xf32>
  %res = hivm.hir.vreduce <sum> ins(%src : tensor<16x8x32xf32>) outs(%dst : tensor<1x1x32xf32>) reduce_dims = [0, 1] -> tensor<1x1x32xf32>
  return %res : tensor<1x1x32xf32>
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_tensor_01_dyn
func.func @test_reduce_op_multi_tensor_01_dyn(%d : index) -> tensor<1x1x?xf32> {
  // CHECK: %[[c2:.*]] = arith.constant 2 : index
  // CHECK: %[[T0:.*]] = tensor.empty(%[[arg0]]) : tensor<16x8x?xf32>
  // CHECK: %[[T1:.*]] = tensor.empty(%[[arg0]]) : tensor<1x1x?xf32>
  // CHECK: %[[dim:.*]] = tensor.dim %[[T0]], %[[c2]] : tensor<16x8x?xf32>
  // CHECK: %[[T2:.*]] = tensor.empty(%[[dim]]) : tensor<16x8x?xf32>
  // CHECK: %[[dim_0:.*]] = tensor.dim %[[T0]], %[[c2]] : tensor<16x8x?xf32>
  // CHECK: %[[extracted_slice:.*]] = tensor.extract_slice %[[T2]][0, 0, 0] [1, 8, %dim_0] [1, 1, 1] : tensor<16x8x?xf32> to tensor<1x8x?xf32>
  // CHECK: %[[T3:.*]] = hivm.hir.vreduce <sum> ins(%[[T0]] : tensor<16x8x?xf32>) outs(%[[extracted_slice]] : tensor<1x8x?xf32>) reduce_dims = [0] -> tensor<1x8x?xf32>
  // CHECK: %[[T4:.*]] = hivm.hir.vreduce <sum> ins(%[[T3]] : tensor<1x8x?xf32>) outs(%[[T1]] : tensor<1x1x?xf32>) reduce_dims = [1] -> tensor<1x1x?xf32>

  %src = tensor.empty(%d) : tensor<16x8x?xf32>
  %dst = tensor.empty(%d) : tensor<1x1x?xf32>
  %res = hivm.hir.vreduce <sum> ins(%src : tensor<16x8x?xf32>) outs(%dst : tensor<1x1x?xf32>) reduce_dims = [0, 1] -> tensor<1x1x?xf32>
  return %res : tensor<1x1x?xf32>
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_tensor_01_dyn2
func.func @test_reduce_op_multi_tensor_01_dyn2(%d : index) -> tensor<1x1x32xf32> {
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: %[[T0:.*]] = tensor.empty(%[[arg0]]) : tensor<16x?x32xf32>
  // CHECK: %[[T1:.*]] = tensor.empty() : tensor<1x1x32xf32>
  // CHECK: %[[dim:.*]] = tensor.dim %[[T0]], %[[c1]] : tensor<16x?x32xf32>
  // CHECK: %[[T2:.*]] = tensor.empty(%[[dim]]) : tensor<16x?x32xf32>
  // CHECK: %[[dim_0:.*]] = tensor.dim %[[T0]], %[[c1]] : tensor<16x?x32xf32>
  // CHECK: %[[extracted_slice:.*]] = tensor.extract_slice %[[T2]][0, 0, 0] [1, %dim_0, 32] [1, 1, 1] : tensor<16x?x32xf32> to tensor<1x?x32xf32>
  // CHECK: %[[T3:.*]] = hivm.hir.vreduce <sum> ins(%[[T0]] : tensor<16x?x32xf32>) outs(%[[extracted_slice]] : tensor<1x?x32xf32>) reduce_dims = [0] -> tensor<1x?x32xf32>
  // CHECK: %[[T4:.*]] = hivm.hir.vreduce <sum> ins(%[[T3]] : tensor<1x?x32xf32>) outs(%[[T1]] : tensor<1x1x32xf32>) reduce_dims = [1] -> tensor<1x1x32xf32>

  %src = tensor.empty(%d) : tensor<16x?x32xf32>
  %dst = tensor.empty() : tensor<1x1x32xf32>
  %res = hivm.hir.vreduce <sum> ins(%src : tensor<16x?x32xf32>) outs(%dst : tensor<1x1x32xf32>) reduce_dims = [0, 1] -> tensor<1x1x32xf32>
  return %res : tensor<1x1x32xf32>
}

// -----
// BEFOREALIGN-LABEL: func.func @test_reduce_op_multi_tensor_013
func.func @test_reduce_op_multi_tensor_013() -> tensor<1x1x10x1xf32> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<16x8x10x32xf32>
  // CHECK: %[[T1:.*]] = tensor.empty() : tensor<1x1x10x1xf32>
  // CHECK: %[[T2:.*]] = tensor.empty() : tensor<16x8x10x32xf32>
  // CHECK: %[[extracted_slice:.*]] = tensor.extract_slice %[[T1]][0, 0, 0, 0] [1, 8, 10, 32] [1, 1, 1, 1] : tensor<1x1x10x1xf32> to tensor<1x8x10x32xf32>
  // CHECK: %[[T3:.*]] = hivm.hir.vreduce <sum> ins(%[[T0]] : tensor<16x8x10x32xf32>) outs(%[[extracted_slice]] : tensor<1x8x10x32xf32>) reduce_dims = [0] -> tensor<1x8x10x32xf32>
  // CHECK: %[[extracted_slice_0:.*]] = tensor.extract_slice %[[T2]][0, 0, 0, 0] [1, 1, 10, 32] [1, 1, 1, 1] : tensor<16x8x10x32xf32> to tensor<1x1x10x32xf32>
  // CHECK: %[[T4:.*]] = hivm.hir.vreduce <sum> ins(%[[T3]] : tensor<1x8x10x32xf32>) outs(%[[extracted_slice_0]] : tensor<1x1x10x32xf32>) reduce_dims = [1] -> tensor<1x1x10x32xf32>
  // CHECK: %[[T5:.*]] = hivm.hir.vreduce <sum> ins(%[[T4]] : tensor<1x1x10x32xf32>) outs(%[[T1]] : tensor<1x1x10x1xf32>) reduce_dims = [3] -> tensor<1x1x10x1xf32>

  %src = tensor.empty() : tensor<16x8x10x32xf32>
  %dst = tensor.empty() : tensor<1x1x10x1xf32>
  %res = hivm.hir.vreduce <sum> ins(%src : tensor<16x8x10x32xf32>) outs(%dst : tensor<1x1x10x1xf32>) reduce_dims = [0, 1, 3] -> tensor<1x1x10x1xf32>
  return %res : tensor<1x1x10x1xf32>
}

// -----
// AFTERLAYOUT-LABEL: func @mm_nd2nz_trans_if_brc_nd2nz
func.func @mm_nd2nz_trans_if_brc_nd2nz(%arg0: memref<?x?x?x?xf32, #hivm.address_space<gm>>, %arg1: i1) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>
  // AFTERLAYOUT: scf.if
  // AFTERLAYOUT: hivm.hir.vbrc
  // AFTERLAYOUT: } {hivm.unlikely_condition}
  // AFTERLAYOUT: hivm.hir.nd2nz {dst_continuous} ins({{.*}} : memref<?x?x?x?xf32, #hivm.address_space<gm>>) outs({{.*}} : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.nd2nz {dst_continuous}  ins(%arg0 : memref<?x?x?x?xf32, #hivm.address_space<gm>>) outs(%alloc : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = %cst : f32 init_condition = %arg1 : i1
  return
}

// -----
// AFTERDEINTERLEAVE-LABEL:   @test_i8_vdeinterleave_memref
// CHECK-SAME:    %[[VAL_0:.+]]: memref<?xi8>,
// CHECK-SAME:    %[[VAL_1:.+]]: memref<?xi8>)
func.func @test_i8_vdeinterleave_memref(%arg0: memref<?xi8>, %arg1: memref<?xi8>) attributes { enable_auto_mark_buffer_size } {
  // CHECK:       %[[VAL_2:.+]] = arith.constant 0 : index
  // CHECK:       %[[VAL_3:.+]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?xi8>
  // CHECK:       %[[VAL_4:.+]] = memref.alloc(%[[VAL_3]]) : memref<?xf16>
  // CHECK:       hivm.hir.vcast ins(%[[VAL_0]] : memref<?xi8>) outs(%[[VAL_4]] : memref<?xf16>)
  // CHECK:       %[[VAL_5:.+]] = memref.dim %[[VAL_1]], %[[VAL_2]] : memref<?xi8>
  // CHECK:       %[[VAL_6:.+]] = memref.alloc(%[[VAL_5]]) : memref<?xf16>
  // CHECK:       hivm.hir.vdeinterleave ins(%[[VAL_4]] : memref<?xf16>) outs(%[[VAL_6]] : memref<?xf16>) channel_num = 32 index_mode = <CHANNEL_0>
  // CHECK:       hivm.hir.vcast ins(%[[VAL_6]] : memref<?xf16>) outs(%[[VAL_1]] : memref<?xi8>)
  // CHECK:       return
  hivm.hir.vdeinterleave ins(%arg0 : memref<?xi8>) outs(%arg1 : memref<?xi8>) channel_num = 32 index_mode = <CHANNEL_0>
  return
}

// -----
// AFTERDEINTERLEAVE-LABEL:   @test_i8_vdeinterleave_tensor
// CHECK-SAME:    %[[VAL_0:.+]]: tensor<?xi8>
func.func @test_i8_vdeinterleave_tensor(%arg0: tensor<?xi8>) -> tensor<?xi8> attributes { enable_auto_mark_buffer_size } {
  // CHECK:       %[[VAL_1:.+]] = arith.constant 0 : index
  // CHECK:       %[[VAL_2:.+]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?xi8>
  // CHECK:       %[[VAL_3:.+]] = tensor.empty(%[[VAL_2]]) : tensor<?xi8>
  // CHECK:       %[[VAL_4:.+]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?xi8>
  // CHECK:       %[[VAL_5:.+]] = tensor.empty(%[[VAL_4]]) : tensor<?xf16>
  // CHECK:       %[[VAL_6:.+]] = hivm.hir.vcast ins(%[[VAL_0]] : tensor<?xi8>) outs(%[[VAL_5]] : tensor<?xf16>) -> tensor<?xf16>
  // CHECK:       %[[VAL_7:.+]] = tensor.dim %[[VAL_3]], %[[VAL_1]] : tensor<?xi8>
  // CHECK:       %[[VAL_8:.+]] = tensor.empty(%[[VAL_7]]) : tensor<?xf16>
  // CHECK:       %[[VAL_9:.+]] = hivm.hir.vdeinterleave ins(%[[VAL_6]] : tensor<?xf16>) outs(%[[VAL_8]] : tensor<?xf16>) channel_num = 32 index_mode = <CHANNEL_0> -> tensor<?xf16>
  // CHECK:       %[[VAL_10:.+]] = hivm.hir.vcast ins(%[[VAL_9]] : tensor<?xf16>) outs(%[[VAL_3]] : tensor<?xi8>) -> tensor<?xi8>
  // CHECK:       return %[[VAL_10]] : tensor<?xi8>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xi8>
  %alloc = tensor.empty(%dim) : tensor<?xi8>
  %ret = hivm.hir.vdeinterleave ins(%arg0 : tensor<?xi8>) outs(%alloc : tensor<?xi8>) channel_num = 32 index_mode = <CHANNEL_0> -> tensor<?xi8>
  return %ret: tensor<?xi8>
}

// -----
// AFTERDEINTERLEAVE-LABEL:   @test_i8_vdeinterleave_multi_outs
// CHECK-SAME:    %[[VAL_0:.+]]: memref<32xi8>,
// CHECK-SAME:    %[[VAL_1:.+]]: memref<16xi8>,
// CHECK-SAME:    %[[VAL_2:.+]]: memref<16xi8>)
func.func @test_i8_vdeinterleave_multi_outs(%src: memref<32xi8>, %dst_even: memref<16xi8>,
                                          %dst_odd: memref<16xi8>) {
  // CHECK:       %[[VAL_3:.+]] = memref.alloc() : memref<32xf16>
  // CHECK:       hivm.hir.vcast ins(%[[VAL_0]] : memref<32xi8>) outs(%[[VAL_3]] : memref<32xf16>)
  // CHECK:       %[[VAL_4:.+]] = memref.alloc() : memref<16xf16>
  // CHECK:       %[[VAL_5:.+]] = memref.alloc() : memref<16xf16>
  // CHECK:       hivm.hir.vdeinterleave ins(%[[VAL_3]] : memref<32xf16>) outs(%[[VAL_4]], %[[VAL_5]] : memref<16xf16>, memref<16xf16>) index_mode = <ALL_CHANNELS>
  // CHECK:       hivm.hir.vcast ins(%[[VAL_4]] : memref<16xf16>) outs(%[[VAL_1]] : memref<16xi8>)
  // CHECK:       hivm.hir.vcast ins(%[[VAL_5]] : memref<16xf16>) outs(%[[VAL_2]] : memref<16xi8>)
  // CHECK:       return
  hivm.hir.vdeinterleave ins(%src: memref<32xi8>)
                        outs(%dst_even, %dst_odd: memref<16xi8>, memref<16xi8>)
                        index_mode = <ALL_CHANNELS>
  return
}

// -----
// AFTERDEINTERLEAVE-LABEL:   func.func @test_i8_vdeinterleave_multi_outs_tensor
// CHECK-SAME:    %[[VAL_0:.+]]: tensor<32xi8>
func.func @test_i8_vdeinterleave_multi_outs_tensor(%src: tensor<32xi8>) -> (tensor<16xi8>, tensor<16xi8>) {
  // CHECK:       %[[VAL_1:.+]] = tensor.empty() : tensor<16xi8>
  // CHECK:       %[[VAL_2:.+]] = tensor.empty() : tensor<16xi8>
  // CHECK:       %[[VAL_3:.+]] = tensor.empty() : tensor<32xf16>
  // CHECK:       %[[VAL_4:.+]] = hivm.hir.vcast ins(%[[VAL_0]] : tensor<32xi8>) outs(%[[VAL_3]] : tensor<32xf16>) -> tensor<32xf16>
  // CHECK:       %[[VAL_5:.+]] = tensor.empty() : tensor<16xf16>
  // CHECK:       %[[VAL_6:.+]] = tensor.empty() : tensor<16xf16>
  // CHECK:       %[[VAL_7:.+]]:2 = hivm.hir.vdeinterleave ins(%[[VAL_4]] : tensor<32xf16>) outs(%[[VAL_5]], %[[VAL_6]] : tensor<16xf16>, tensor<16xf16>) index_mode = <ALL_CHANNELS> -> tensor<16xf16>, tensor<16xf16>
  // CHECK:       %[[VAL_8:.+]] = hivm.hir.vcast ins(%[[VAL_7]]#0 : tensor<16xf16>) outs(%[[VAL_1]] : tensor<16xi8>) -> tensor<16xi8>
  // CHECK:       %[[VAL_9:.+]] = hivm.hir.vcast ins(%[[VAL_7]]#1 : tensor<16xf16>) outs(%[[VAL_2]] : tensor<16xi8>) -> tensor<16xi8>
  // CHECK:       return %[[VAL_8]], %[[VAL_9]] : tensor<16xi8>, tensor<16xi8>
  %alloc0 = tensor.empty() : tensor<16xi8>
  %alloc1 = tensor.empty() : tensor<16xi8>
  %0:2 = hivm.hir.vdeinterleave ins(%src: tensor<32xi8>)
                        outs(%alloc0, %alloc1: tensor<16xi8>, tensor<16xi8>)
                        index_mode = <ALL_CHANNELS> -> tensor<16xi8>, tensor<16xi8>
  return %0#0, %0#1: tensor<16xi8>, tensor<16xi8>
}

// -----
// AFTERBROADCAST-LABEL:   @tensor_i1_vbrc
// CHECK-SAME:                              %[[VAL_0:.*]]: memref<2x3x1xi1>,
// CHECK-SAME:                              %[[VAL_1:.*]]: memref<2x3x32xi1>) {
func.func @tensor_i1_vbrc(%arg0: memref<2x3x1xi1>, %arg1: memref<2x3x32xi1>) {
  // CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<2x3x1xf16>
  // CHECK:           hivm.hir.vcast ins(%[[VAL_0]] : memref<2x3x1xi1>) outs(%[[VAL_3]] : memref<2x3x1xf16>) round_mode = <trunc>
  // CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<2x3x32xf16>
  // CHECK:           hivm.hir.vbrc ins(%[[VAL_3]] : memref<2x3x1xf16>) outs(%[[VAL_4]] : memref<2x3x32xf16>) broadcast_dims = [2]
  // CHECK:           hivm.hir.vcmp ins(%[[VAL_4]], %[[VAL_2]] : memref<2x3x32xf16>, f16) outs(%[[VAL_1]] : memref<2x3x32xi1>) compare_mode = <ne>
  hivm.hir.vbrc ins(%arg0 : memref<2x3x1xi1>) outs(%arg1 : memref<2x3x32xi1>) broadcast_dims = [2]
  return
}

// -----
// AFTERBROADCAST-LABEL:   @tensor_i8_vbrc
// CHECK-SAME:                              %[[VAL_0:.*]]: memref<2x3x1xi8>,
// CHECK-SAME:                              %[[VAL_1:.*]]: memref<2x3x32xi8>) {
func.func @tensor_i8_vbrc(%arg0: memref<2x3x1xi8>, %arg1: memref<2x3x32xi8>) {
  // CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<2x3x1xf16>
  // CHECK:           hivm.hir.vcast ins(%[[VAL_0]] : memref<2x3x1xi8>) outs(%[[VAL_2]] : memref<2x3x1xf16>)
  // CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<2x3x32xf16>
  // CHECK:           hivm.hir.vbrc ins(%[[VAL_2]] : memref<2x3x1xf16>) outs(%[[VAL_3]] : memref<2x3x32xf16>) broadcast_dims = [2]
  // CHECK:           hivm.hir.vcast ins(%[[VAL_3]] : memref<2x3x32xf16>) outs(%[[VAL_1]] : memref<2x3x32xi8>) round_mode = <trunc>
  hivm.hir.vbrc ins(%arg0 : memref<2x3x1xi8>) outs(%arg1 : memref<2x3x32xi8>) broadcast_dims = [2]
  return
}

// -----
// BEFOREALIGN: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 + d1 + 2048, 0)>
// BEFOREALIGN: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0, 0)>
// BEFOREALIGN: #[[$ATTR_2:.+]] = affine_map<()[s0, s1] -> (s0, s1)>
// BEFOREALIGN: #[[$ATTR_3:.+]] = affine_map<()[s0] -> (s0 + 2048, 0)>
// BEFOREALIGN: #[[$ATTR_4:.+]] = affine_map<()[s0] -> (-s0, 0)>
// BEFOREALIGN: #[[$ATTR_5:.+]] = affine_map<()[s0] -> (s0, 2048)>
// BEFOREALIGN: #[[$ATTR_6:.+]] = affine_map<()[s0, s1] -> (s0 - s1)>
func.func @test_dynamic_pad(%arg0: index, %arg1: index, %in: memref<128x2048xf32>, %cst: f32) attributes { enable_auto_mark_buffer_size } {
  %0 = affine.max affine_map<(d0, d1) -> (d0 + d1 + 2048, 0)>(%arg0, %arg1)
  %out = memref.alloc(%0): memref<128x?xf32>
  hivm.hir.vpad ins(%in: memref<128x2048xf32>) outs(%out : memref<128x?xf32>) low[0, %arg0] high[0, %arg1] pad_value %cst : f32
  return
}
// BEFOREALIGN-LABEL:   func.func @test_dynamic_pad
// BEFOREALIGN-SAME:                               (%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: memref<128x2048xf32>, %[[VAL_3:.*]]: f32)
// BEFOREALIGN:           %[[VAL_4:.*]] = affine.max #[[$ATTR_0]](%[[VAL_0]], %[[VAL_1]])
// BEFOREALIGN:           %[[VAL_5:.*]] = memref.alloc(%[[VAL_4]]) : memref<128x?xf32>
// BEFOREALIGN:           %[[VAL_6:.*]] = affine.max #[[$ATTR_1]](){{\[}}%[[VAL_0]]]
// BEFOREALIGN:           %[[VAL_7:.*]] = affine.min #[[$ATTR_2]](){{\[}}%[[VAL_6]], %[[VAL_4]]]
// BEFOREALIGN:           %[[VAL_8:.*]] = memref.subview %[[VAL_5]][0, 0] [128, %[[VAL_7]]] [1, 1] : memref<128x?xf32> to memref<128x?xf32, strided<[?, 1]>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_3]] : f32) outs(%[[VAL_8]] : memref<128x?xf32, strided<[?, 1]>>)
// BEFOREALIGN:           %[[VAL_9:.*]] = affine.max #[[$ATTR_3]](){{\[}}%[[VAL_0]]]
// BEFOREALIGN:           %[[VAL_10:.*]] = affine.min #[[$ATTR_2]](){{\[}}%[[VAL_9]], %[[VAL_4]]]
// BEFOREALIGN:           %[[VAL_11:.*]] = affine.max #[[$ATTR_1]](){{\[}}%[[VAL_1]]]
// BEFOREALIGN:           %[[VAL_12:.*]] = affine.min #[[$ATTR_2]](){{\[}}%[[VAL_11]], %[[VAL_4]]]
// BEFOREALIGN:           %[[VAL_13:.*]] = memref.subview %[[VAL_5]][0, %[[VAL_10]]] [128, %[[VAL_12]]] [1, 1] : memref<128x?xf32> to memref<128x?xf32, strided<[?, 1], offset: ?>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_3]] : f32) outs(%[[VAL_13]] : memref<128x?xf32, strided<[?, 1], offset: ?>>)
// BEFOREALIGN:           %[[VAL_14:.*]] = affine.max #[[$ATTR_4]](){{\[}}%[[VAL_0]]]
// BEFOREALIGN:           %[[VAL_15:.*]] = affine.min #[[$ATTR_5]](){{\[}}%[[VAL_14]]]
// BEFOREALIGN:           %[[VAL_16:.*]] = affine.max #[[$ATTR_3]](){{\[}}%[[VAL_1]]]
// BEFOREALIGN:           %[[VAL_17:.*]] = affine.min #[[$ATTR_5]](){{\[}}%[[VAL_16]]]
// BEFOREALIGN:           %[[VAL_18:.*]] = affine.apply #[[$ATTR_6]](){{\[}}%[[VAL_17]], %[[VAL_15]]]
// BEFOREALIGN:           %[[VAL_19:.*]] = memref.subview %[[VAL_2]][0, %[[VAL_15]]] [128, %[[VAL_18]]] [1, 1] : memref<128x2048xf32> to memref<128x?xf32, strided<[2048, 1], offset: ?>>
// BEFOREALIGN:           %[[VAL_20:.*]] = affine.max #[[$ATTR_1]](){{\[}}%[[VAL_0]]]
// BEFOREALIGN:           %[[VAL_21:.*]] = affine.min #[[$ATTR_2]](){{\[}}%[[VAL_20]], %[[VAL_4]]]
// BEFOREALIGN:           %[[VAL_22:.*]] = memref.subview %[[VAL_5]][0, %[[VAL_21]]] [128, %[[VAL_18]]] [1, 1] : memref<128x?xf32> to memref<128x?xf32, strided<[?, 1], offset: ?>>
// BEFOREALIGN:           hivm.hir.copy ins(%[[VAL_19]] : memref<128x?xf32, strided<[2048, 1], offset: ?>>) outs(%[[VAL_22]] : memref<128x?xf32, strided<[?, 1], offset: ?>>)
// BEFOREALIGN:           return

// -----

func.func @test_static_pad(%in: memref<128x2048xf32>, %out: memref<128x2053xf32>, %cst: f32) {
  hivm.hir.vpad ins(%in: memref<128x2048xf32>) outs(%out : memref<128x2053xf32>) low[0, 4] high[0, 1] pad_value %cst : f32
  return
}
// BEFOREALIGN-LABEL:   func.func @test_static_pad(
// BEFOREALIGN-SAME:                               %[[VAL_0:.*]]: memref<128x2048xf32>,
// BEFOREALIGN-SAME:                               %[[VAL_1:.*]]: memref<128x2053xf32>,
// BEFOREALIGN-SAME:                               %[[VAL_2:.*]]: f32) {
// BEFOREALIGN:           %[[VAL_3:.*]] = memref.subview %[[VAL_1]][0, 0] [128, 4] [1, 1] : memref<128x2053xf32> to memref<128x4xf32, strided<[2053, 1]>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_3]] : memref<128x4xf32, strided<[2053, 1]>>)
// BEFOREALIGN:           %[[VAL_4:.*]] = memref.subview %[[VAL_1]][0, 2052] [128, 1] [1, 1] : memref<128x2053xf32> to memref<128x1xf32, strided<[2053, 1], offset: 2052>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_4]] : memref<128x1xf32, strided<[2053, 1], offset: 2052>>)
// BEFOREALIGN:           %[[VAL_5:.*]] = memref.subview %[[VAL_0]][0, 0] [128, 2048] [1, 1] : memref<128x2048xf32> to memref<128x2048xf32, strided<[2048, 1]>>
// BEFOREALIGN:           %[[VAL_6:.*]] = memref.subview %[[VAL_1]][0, 4] [128, 2048] [1, 1] : memref<128x2053xf32> to memref<128x2048xf32, strided<[2053, 1], offset: 4>>
// BEFOREALIGN:           hivm.hir.copy ins(%[[VAL_5]] : memref<128x2048xf32, strided<[2048, 1]>>) outs(%[[VAL_6]] : memref<128x2048xf32, strided<[2053, 1], offset: 4>>)
// BEFOREALIGN:           return
// BEFOREALIGN:         }

// -----

func.func @test_static_pad_neg(%in: memref<128x2048xf32>, %out: memref<128x2043xf32>, %cst: f32) {
  hivm.hir.vpad ins(%in: memref<128x2048xf32>) outs(%out : memref<128x2043xf32>) low[0, -4] high[0, -1] pad_value %cst : f32
  return
}
// BEFOREALIGN-LABEL:   func.func @test_static_pad_neg(
// BEFOREALIGN-SAME:                                   %[[VAL_0:.*]]: memref<128x2048xf32>,
// BEFOREALIGN-SAME:                                   %[[VAL_1:.*]]: memref<128x2043xf32>,
// BEFOREALIGN-SAME:                                   %[[VAL_2:.*]]: f32) {
// BEFOREALIGN:           %[[VAL_3:.*]] = memref.subview %[[VAL_1]][0, 0] [128, 0] [1, 1] : memref<128x2043xf32> to memref<128x0xf32, strided<[2043, 1]>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_3]] : memref<128x0xf32, strided<[2043, 1]>>)
// BEFOREALIGN:           %[[VAL_4:.*]] = memref.subview %[[VAL_1]][0, 2043] [128, 0] [1, 1] : memref<128x2043xf32> to memref<128x0xf32, strided<[2043, 1], offset: 2043>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_4]] : memref<128x0xf32, strided<[2043, 1], offset: 2043>>)
// BEFOREALIGN:           %[[VAL_5:.*]] = memref.subview %[[VAL_0]][0, 4] [128, 2043] [1, 1] : memref<128x2048xf32> to memref<128x2043xf32, strided<[2048, 1], offset: 4>>
// BEFOREALIGN:           %[[VAL_6:.*]] = memref.subview %[[VAL_1]][0, 0] [128, 2043] [1, 1] : memref<128x2043xf32> to memref<128x2043xf32, strided<[2043, 1]>>
// BEFOREALIGN:           hivm.hir.copy ins(%[[VAL_5]] : memref<128x2043xf32, strided<[2048, 1], offset: 4>>) outs(%[[VAL_6]] : memref<128x2043xf32, strided<[2043, 1]>>)
// BEFOREALIGN:           return
// BEFOREALIGN:         }

// -----
func.func @test_zero_mid_high(%in: memref<128x2048xf32>, %out: memref<16x2048xf32>, %cst: f32) {
  hivm.hir.vpad ins(%in: memref<128x2048xf32>) outs(%out : memref<16x2048xf32>) low[-130, 0] high[18, 0] pad_value %cst : f32
  return
}
// BEFOREALIGN-LABEL:   func.func @test_zero_mid_high(
// BEFOREALIGN-SAME:                                  %[[VAL_0:.*]]: memref<128x2048xf32>,
// BEFOREALIGN-SAME:                                  %[[VAL_1:.*]]: memref<16x2048xf32>,
// BEFOREALIGN-SAME:                                  %[[VAL_2:.*]]: f32) {
// BEFOREALIGN:           %[[VAL_3:.*]] = memref.subview %[[VAL_1]][0, 0] [0, 2048] [1, 1] : memref<16x2048xf32> to memref<0x2048xf32, strided<[2048, 1]>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_3]] : memref<0x2048xf32, strided<[2048, 1]>>)
// BEFOREALIGN:           %[[VAL_4:.*]] = memref.subview %[[VAL_1]][0, 0] [16, 2048] [1, 1] : memref<16x2048xf32> to memref<16x2048xf32, strided<[2048, 1]>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_4]] : memref<16x2048xf32, strided<[2048, 1]>>)
// BEFOREALIGN:           %[[VAL_5:.*]] = memref.subview %[[VAL_0]][128, 0] [0, 2048] [1, 1] : memref<128x2048xf32> to memref<0x2048xf32, strided<[2048, 1], offset: 262144>>
// BEFOREALIGN:           %[[VAL_6:.*]] = memref.subview %[[VAL_1]][0, 0] [0, 2048] [1, 1] : memref<16x2048xf32> to memref<0x2048xf32, strided<[2048, 1]>>
// BEFOREALIGN:           hivm.hir.copy ins(%[[VAL_5]] : memref<0x2048xf32, strided<[2048, 1], offset: 262144>>) outs(%[[VAL_6]] : memref<0x2048xf32, strided<[2048, 1]>>)
// BEFOREALIGN:           return
// BEFOREALIGN:         }

// -----
func.func @test_zero_mid_low(%in: memref<128x2048xf32>, %out: memref<16x2048xf32>, %cst: f32) {
  hivm.hir.vpad ins(%in: memref<128x2048xf32>) outs(%out : memref<16x2048xf32>) low[18, 0] high[-130, 0] pad_value %cst : f32
  return
}
// BEFOREALIGN-LABEL:   func.func @test_zero_mid_low(
// BEFOREALIGN-SAME:                                 %[[VAL_0:.*]]: memref<128x2048xf32>,
// BEFOREALIGN-SAME:                                 %[[VAL_1:.*]]: memref<16x2048xf32>,
// BEFOREALIGN-SAME:                                 %[[VAL_2:.*]]: f32) {
// BEFOREALIGN:           %[[VAL_3:.*]] = memref.subview %[[VAL_1]][0, 0] [16, 2048] [1, 1] : memref<16x2048xf32> to memref<16x2048xf32, strided<[2048, 1]>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_3]] : memref<16x2048xf32, strided<[2048, 1]>>)
// BEFOREALIGN:           %[[VAL_4:.*]] = memref.subview %[[VAL_1]][16, 0] [0, 2048] [1, 1] : memref<16x2048xf32> to memref<0x2048xf32, strided<[2048, 1], offset: 32768>>
// BEFOREALIGN:           hivm.hir.vbrc ins(%[[VAL_2]] : f32) outs(%[[VAL_4]] : memref<0x2048xf32, strided<[2048, 1], offset: 32768>>)
// BEFOREALIGN:           %[[VAL_5:.*]] = memref.subview %[[VAL_0]][0, 0] [0, 2048] [1, 1] : memref<128x2048xf32> to memref<0x2048xf32, strided<[2048, 1]>>
// BEFOREALIGN:           %[[VAL_6:.*]] = memref.subview %[[VAL_1]][16, 0] [0, 2048] [1, 1] : memref<16x2048xf32> to memref<0x2048xf32, strided<[2048, 1], offset: 32768>>
// BEFOREALIGN:           hivm.hir.copy ins(%[[VAL_5]] : memref<0x2048xf32, strided<[2048, 1]>>) outs(%[[VAL_6]] : memref<0x2048xf32, strided<[2048, 1], offset: 32768>>)
// BEFOREALIGN:           return
// BEFOREALIGN:         }

// -----
func.func @test_all_zero_pad(%in: memref<128x2048xf32>, %out: memref<128x2048xf32>, %cst: f32) {
  hivm.hir.vpad ins(%in: memref<128x2048xf32>) outs(%out : memref<128x2048xf32>) low[0, 0] high[0, 0] pad_value %cst : f32
  return
}
// BEFOREALIGN-LABEL:   func.func @test_all_zero_pad
// BEFOREALIGN-SAME:                                 (%[[VAL_0:.*]]: memref<128x2048xf32>, %[[VAL_1:.*]]: memref<128x2048xf32>, %[[VAL_2:.*]]: f32) {
// BEFOREALIGN:           hivm.hir.copy ins(%[[VAL_0]] : memref<128x2048xf32>) outs(%[[VAL_1]] : memref<128x2048xf32>)
// BEFOREALIGN:           return
// BEFOREALIGN:         }

//===----------------------------------------------------------------------===//
// Test VTransposeOp Unaligned Decompose 
//===----------------------------------------------------------------------===//

// -----
// AFTERALIGN-LABEL: func @test_vtranspose_disable_align
func.func @test_vtranspose_disable_align() -> memref<2x3x256xf32, #hivm.address_space<ub>> {
  // CHECK: %[[C256:.*]] = arith.constant 256 : i64
  // CHECK: %[[C3:.*]] = arith.constant 3 : i64
  // CHECK: %[[C96:.*]] = arith.constant 96 : i64
  // CHECK: %[[C8:.*]] = arith.constant 8 : i64
  // CHECK: %[[C24:.*]] = arith.constant 24 : i64
  // CHECK: %[[C2IDX:.*]] = arith.constant 2 : index
  // CHECK: %[[C32:.*]] = arith.constant 32 : i64
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : i64
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x256x3xf32, #hivm.address_space<ub>>
  // CHECK: %[[A0:.*]] = memref.alloc() : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C2]], %[[A0]][%[[C0]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C32]], %[[A0]][%[[C1]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C24]], %[[A0]][%[[C2IDX]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: %[[R:.*]] = memref.reshape %[[A]](%[[A0]]) : (memref<2x256x3xf32, #hivm.address_space<ub>>, memref<3xi64, #hivm.address_space<ub>>) -> memref<2x32x24xf32, #hivm.address_space<ub>>
  // CHECK: %[[A1:.*]] = memref.alloc() : memref<2x24x32xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vtranspose ins(%[[R]] : memref<2x32x24xf32, #hivm.address_space<ub>>) outs(%[[A1]] : memref<2x24x32xf32, #hivm.address_space<ub>>) permutation = [0, 2, 1]
  // CHECK: %[[A2:.*]] = memref.alloc() : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C2]], %[[A2]][%[[C0]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C8]], %[[A2]][%[[C1]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C96]], %[[A2]][%[[C2IDX]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: %[[R3:.*]] = memref.reshape %[[A1]](%[[A2]]) : (memref<2x24x32xf32, #hivm.address_space<ub>>, memref<3xi64, #hivm.address_space<ub>>) -> memref<2x8x96xf32, #hivm.address_space<ub>>
  // CHECK: %[[A4:.*]] = memref.alloc() : memref<2x96x8xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vtranspose ins(%[[R3]] : memref<2x8x96xf32, #hivm.address_space<ub>>) outs(%[[A4]] : memref<2x96x8xf32, #hivm.address_space<ub>>) permutation = [0, 2, 1]
  // CHECK: %[[A5:.*]] = memref.alloc() : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C2]], %[[A5]][%[[C0]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C3]], %[[A5]][%[[C1]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: memref.store %[[C256]], %[[A5]][%[[C2IDX]]] : memref<3xi64, #hivm.address_space<ub>>
  // CHECK: %[[R6:.*]] = memref.reshape %[[A4]](%[[A5]]) : (memref<2x96x8xf32, #hivm.address_space<ub>>, memref<3xi64, #hivm.address_space<ub>>) -> memref<2x3x256xf32, #hivm.address_space<ub>>
  // CHECK: return %[[R6]] : memref<2x3x256xf32, #hivm.address_space<ub>>
  %src = memref.alloc() : memref<2x256x3xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<2x3x256xf32, #hivm.address_space<ub>>

  hivm.hir.vtranspose ins(%src : memref<2x256x3xf32, #hivm.address_space<ub>>) outs(%dst : memref<2x3x256xf32, #hivm.address_space<ub>>) permutation = [0, 2, 1] disable_align = true
  return %dst : memref<2x3x256xf32, #hivm.address_space<ub>>
}

