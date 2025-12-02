// RUN: bishengir-opt %s -convert-to-hivm-op -split-input-file -allow-unregistered-dialect | FileCheck %s


// -----
func.func @copy_with_pad(%arg0: memref<?xi8>, %arg1: memref<?xi8>) attributes {global_kernel = "local"} {
  %c100_i8 = arith.constant 100 : i8
  %2 = arith.constant 4 : i32
  %4 = arith.constant 0 : index
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%4], sizes: [128, 32], strides: [32, 1] : memref<?xi8> to memref<128x32xi8, strided<[32, 1], offset: ?>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [128, 32], strides: [65, 1] : memref<?xi8> to memref<128x32xi8, strided<[65, 1], offset: ?>>
  %alloc = memref.alloc() : memref<128x32xi8>
  %22 = arith.constant 128 : index
  %23 = arith.constant 32 : index
  hivm.hir.vbrc ins(%c100_i8 : i8) outs(%alloc : memref<128x32xi8>)
  %subview = memref.subview %reinterpret_cast_0[0, 0] [%22, %23] [1, 1] : memref<128x32xi8, strided<[65, 1], offset: ?>> to memref<?x?xi8, strided<[65, 1], offset: ?>>
  %subview_1 = memref.subview %alloc[0, 0] [%22, %23] [1, 1] : memref<128x32xi8> to memref<?x?xi8, strided<[32, 1]>>
  // CHECK: hivm.hir.load ins({{.*}} : memref<?x?xi8, strided<[65, 1], offset: ?>>) outs({{.*}} : memref<?x?xi8, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = {{.*}} : i8
  memref.copy %subview, %subview_1 : memref<?x?xi8, strided<[65, 1], offset: ?>> to memref<?x?xi8, strided<[32, 1]>>
  %27 = bufferization.to_tensor %alloc restrict writable : memref<128x32xi8>
  // CHECK: hivm.hir.store ins({{.*}} : tensor<128x32xi8>) outs(%{{.*}} : memref<128x32xi8, strided<[32, 1], offset: ?>>)
  bufferization.materialize_in_destination %27 in writable %reinterpret_cast : (tensor<128x32xi8>, memref<128x32xi8, strided<[32, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_not_convert_host_to_hivm
// CHECK: memref.copy
// CHECK-NOT: hivm.hir.copy
func.func @test_not_convert_host_to_hivm(%arg0: tensor<768x256xf32>) -> memref<768x256xf32, #hivm.address_space<gm>>
attributes {hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<UNKNOWN>} {
    %0 = bufferization.to_memref %arg0 : memref<768x256xf32, strided<[?, ?], offset: ?>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<768x256xf32, #hivm.address_space<gm>>
    memref.copy %0, %alloc_1 : memref<768x256xf32, strided<[?, ?], offset: ?>> to memref<768x256xf32, #hivm.address_space<gm>>
    return %alloc_1 : memref<768x256xf32, #hivm.address_space<gm>>
}


// -----

// CHECK-LABEL: @test_hivm_load_with_pad_val
// CHECK: hivm.hir.load
func.func @test_hivm_load_with_pad_val(%arg0: memref<32xf32>) {
    %subview_1 = memref.subview %arg0[8] [16] [1] : memref<32xf32> to memref<16xf32, strided<[1], offset: 8>>
    %alloc = memref.alloc() : memref<32xf32>
    %subview_2 = memref.subview %alloc[8] [16] [1] : memref<32xf32> to memref<16xf32, strided<[1], offset: 8>>
    memref.copy %subview_1, %subview_2 : memref<16xf32, strided<[1], offset: 8>> to memref<16xf32, strided<[1], offset: 8>>
    return
}

// -----
// CHECK-LABEL: @do_not_set_init_out_buffer
func.func @do_not_set_init_out_buffer(%arg0: memref<256x128xf32, strided<[?, ?], offset: ?>>, %arg1: i32, %arg2: i1) -> tensor<256x128xf32> {
  %cst = arith.constant 2.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %alloc = memref.alloc() : memref<256x128xf32>
  scf.if %arg2 {
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<256x128xf32>)
    hivm.hir.vtanh ins(%alloc : memref<256x128xf32>) outs(%alloc : memref<256x128xf32>)
  }
  %subview = memref.subview %arg0[0, 0] [256, %0] [1, 1] : memref<256x128xf32, strided<[?, ?], offset: ?>> to memref<256x?xf32, strided<[?, ?], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [256, %0] [1, 1] : memref<256x128xf32> to memref<256x?xf32, strided<[128, 1]>>
  // CHECK: hivm.hir.load ins({{.*}} : memref<256x?xf32, strided<[?, ?], offset: ?>>) outs({{.*}} : memref<256x?xf32, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = {{.*}} : f32 left_padding_num = {{.*}} : index init_out_buffer = false
  memref.copy %subview, %subview_0 : memref<256x?xf32, strided<[?, ?], offset: ?>> to memref<256x?xf32, strided<[128, 1]>>
  %1 = bufferization.to_tensor %alloc restrict writable : memref<256x128xf32>
  return %1 : tensor<256x128xf32>
}

// -----
// CHECK-LABEL: @mm_set_init_out_buffer
func.func @mm_set_init_out_buffer(%arg0: memref<256x128xf32, strided<[?, ?], offset: ?>>, %arg1: memref<128x128xf32, strided<[?, ?], offset: ?>>, %arg2: i32, %arg3: i1) -> tensor<256x128xf32> {
  %cst = arith.constant 2.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %0 = arith.index_cast %arg2 : i32 to index
  %1 = tensor.empty() : tensor<256x128xf32>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<256x128xf32>
  %alloc = memref.alloc() : memref<256x128xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<128x128xf32>
  %alloc_0 = memref.alloc() : memref<128x128xf32>
  scf.if %arg3 {
    // CHECK-NOT: hivm.hir.vbrc
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<256x128xf32>)
  }
  %subview = memref.subview %arg0[0, 0] [256, %0] [1, 1] : memref<256x128xf32, strided<[?, ?], offset: ?>> to memref<256x?xf32, strided<[?, ?], offset: ?>>
  %subview_1 = memref.subview %alloc[0, 0] [256, %0] [1, 1] : memref<256x128xf32> to memref<256x?xf32, strided<[128, 1]>>
  // CHECK: hivm.hir.load ins({{.*}} : memref<256x?xf32, strided<[?, ?], offset: ?>>) outs({{.*}} : memref<256x?xf32, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = {{.*}} : f32 left_padding_num ={{.*}} : index init_out_buffer = true init_condition = {{.*}}
  memref.copy %subview, %subview_1 : memref<256x?xf32, strided<[?, ?], offset: ?>> to memref<256x?xf32, strided<[128, 1]>>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<256x128xf32>
  // CHECK-NOT: hivm.hir.vbrc
  hivm.hir.vbrc ins(%cst : f32) outs(%alloc_0 : memref<128x128xf32>)
  %3 = arith.minsi %0, %c128 : index
  %subview_2 = memref.subview %arg1[0, 0] [%3, 128] [1, 1] : memref<128x128xf32, strided<[?, ?], offset: ?>> to memref<?x128xf32, strided<[?, ?], offset: ?>>
  %subview_3 = memref.subview %alloc_0[0, 0] [%3, 128] [1, 1] : memref<128x128xf32> to memref<?x128xf32, strided<[128, 1]>>
  memref.copy %subview_2, %subview_3 : memref<?x128xf32, strided<[?, ?], offset: ?>> to memref<?x128xf32, strided<[128, 1]>>
  // CHECK: hivm.hir.load ins({{.*}} : memref<?x128xf32, strided<[?, ?], offset: ?>>) outs({{.*}} : memref<?x128xf32, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = {{.*}} : f32 left_padding_num = {{.*}} : index init_out_buffer = true
  %4 = bufferization.to_tensor %alloc_0 restrict writable : memref<128x128xf32>
  %5 = hivm.hir.mmadL1 ins(%2, %4, %true, %c0, %c0, %c0 : tensor<256x128xf32>, tensor<128x128xf32>, i1, index, index, index) outs(%1 : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %5 : tensor<256x128xf32>
}

// -----
// CHECK-LABEL: @test_dynamic_offset_memref_copy
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: memref<1024xi64, strided<[1]>>)
// CHECK: hivm.hir.load ins(%subview : memref<?xi64, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xi64, strided<[1], offset: ?>>) left_padding_num = %[[arg0:.*]] : index init_out_buffer = false
func.func @test_dynamic_offset_memref_copy(%arg0 : index, %arg1 : memref<1024xi64, strided<[1]>>) -> tensor<1024xi64> {
  %c0 = arith.constant 0 :index
  %alloc = memref.alloc() : memref<1024xi64>
  %subview = memref.subview %arg1[%arg0] [%c0] [1] : memref<1024xi64, strided<[1]>> to memref<?xi64, strided<[1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg0] [%c0] [1] : memref<1024xi64> to memref<?xi64, strided<[1], offset: ?>>
  memref.copy %subview, %subview_0 : memref<?xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1], offset: ?>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xi64>
  return %0 : tensor<1024xi64>
}

// -----
// CHECK-LABEL: @test_load_with_mask_other_memref_collapse_shape
func.func @test_load_with_mask_other_memref_collapse_shape(
  %arg0 : index, %arg1 : memref<1024xi64, strided<[1]>>, %arg2 : i1) -> tensor<1024x1xi64> {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1024x1xi64>
  %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<1024x1xi64> into memref<1024xi64>
  %c-1_i64 = arith.constant -1 : i64
  scf.if %arg2 {
    hivm.hir.vbrc ins(%c-1_i64 : i64) outs(%collapse_shape : memref<1024xi64>)
  }
  %subview = memref.subview %arg1[%arg0] [%c0] [1] : memref<1024xi64, strided<[1], offset: 0>> to memref<?xi64, strided<[1], offset: ?>>
  %subview_0 = memref.subview %collapse_shape[%arg0] [%c0] [1] : memref<1024xi64> to memref<?xi64, strided<[1], offset: ?>>
  // CHECK: hivm.hir.load ins(%subview : memref<?xi64, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xi64, strided<[1], offset: ?>>) pad_mode = <PadValue> pad_value = %c-1_i64 : i64 left_padding_num = %arg0 : index init_out_buffer = true init_condition = %arg2 : i1 may_implicit_transpose_with_last_axis = false
  memref.copy %subview, %subview_0 : memref<?xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1], offset: ?>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1024x1xi64>
  return %0 : tensor<1024x1xi64>
}

// -----
// CHECK-LABEL: @test_load_with_mask_other_memref_expand_shape
func.func @test_load_with_mask_other_memref_expand_shape(
  %arg0 : index, %arg1 : memref<1024x1xi64, strided<[1, 1]>>, %arg2 : i1) -> tensor<1024xi64> {
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc() : memref<1024xi64>
  %expand_shape = memref.expand_shape %alloc [[0, 1]] output_shape [1024, 1] : memref<1024xi64> into memref<1024x1xi64>
  %c-1_i64 = arith.constant -1 : i64
  scf.if %arg2 {
    hivm.hir.vbrc ins(%c-1_i64 : i64) outs(%expand_shape : memref<1024x1xi64>)
  }
  %subview = memref.subview %arg1[%arg0, 0] [%c10, 1] [1, 1] : memref<1024x1xi64, strided<[1, 1], offset: 0>> to memref<?x1xi64, strided<[1, 1], offset: ?>>
  %subview_0 = memref.subview %expand_shape[%arg0, 0] [%c10, 1] [1, 1] : memref<1024x1xi64> to memref<?x1xi64, strided<[1, 1], offset: ?>>
  // CHECK: hivm.hir.load ins(%subview : memref<?x1xi64, strided<[1, 1], offset: ?>>) outs(%subview_0 : memref<?x1xi64, strided<[1, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %c-1_i64 : i64 left_padding_num = %c0 : index init_out_buffer = true init_condition = %arg2 : i1 may_implicit_transpose_with_last_axis = false
  memref.copy %subview, %subview_0 : memref<?x1xi64, strided<[1, 1], offset: ?>> to memref<?x1xi64, strided<[1, 1], offset: ?>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xi64>
  return %0 : tensor<1024xi64>
}
