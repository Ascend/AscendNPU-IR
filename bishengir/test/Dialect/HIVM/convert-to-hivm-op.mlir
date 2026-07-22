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
func.func @triton_load_store_one_grid_outlined_vf_0(%arg0: memref<128x32xi8>) attributes {hivm.vector_function} {
    %cst = arith.constant dense<1> : vector<128x32xi8>
    %c0 = arith.constant 0 : index
    return
}
func.func @copy_with_pad_new(%arg0: memref<?xi8>, %arg1: memref<?xi8>) attributes {global_kernel = "local"} {
  %c100_i8 = arith.constant 100 : i8
  %2 = arith.constant 4 : i32
  %4 = arith.constant 0 : index
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%4], sizes: [128, 32], strides: [32, 1] : memref<?xi8> to memref<128x32xi8, strided<[32, 1], offset: ?>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [128, 32], strides: [65, 1] : memref<?xi8> to memref<128x32xi8, strided<[65, 1], offset: ?>>
  %alloc = memref.alloc() : memref<128x32xi8>
  %22 = arith.constant 128 : index
  %23 = arith.constant 32 : index
  // CHECK-NOT: annotation.mark %alloc keys = ["pad_const"] values = [%c100_i8 : i8] : memref<128x32xi8>
  annotation.mark %alloc keys = ["pad_const"] values = [%c100_i8 : i8] : memref<128x32xi8>
  func.call @triton_load_store_one_grid_outlined_vf_0(%alloc) {hivm.vector_function} : (memref<128x32xi8>) -> ()
  %subview = memref.subview %reinterpret_cast_0[0, 0] [%22, %23] [1, 1] : memref<128x32xi8, strided<[65, 1], offset: ?>> to memref<?x?xi8, strided<[65, 1], offset: ?>>
  %subview_1 = memref.subview %alloc[0, 0] [%22, %23] [1, 1] : memref<128x32xi8> to memref<?x?xi8, strided<[32, 1]>>
  // CHECK: hivm.hir.load ins({{.*}} : memref<?x?xi8, strided<[65, 1], offset: ?>>) outs({{.*}} : memref<?x?xi8, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = %c100_i8 : i8
  memref.copy %subview, %subview_1 : memref<?x?xi8, strided<[65, 1], offset: ?>> to memref<?x?xi8, strided<[32, 1]>>
  %27 = bufferization.to_tensor %alloc restrict writable : memref<128x32xi8>
  bufferization.materialize_in_destination %27 in writable %reinterpret_cast : (tensor<128x32xi8>, memref<128x32xi8, strided<[32, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_RemovePadConstAnnotation
func.func @test_RemovePadConstAnnotation(%arg0: memref<?xi8>) attributes {global_kernel = "local"} {
  %c100_i8 = arith.constant 100 : i8
  %alloc = memref.alloc() : memref<128x32xi8>
  // CHECK-NOT: annotation.mark %alloc keys = ["pad_const"] values = [%c100_i8 : i8] : memref<128x32xi8>
  annotation.mark %alloc keys = ["pad_const"] values = [%c100_i8 : i8] : memref<128x32xi8>
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
  // CHECK: hivm.hir.load ins({{.*}} : memref<256x?xf32, strided<[?, ?], offset: ?>>) outs({{.*}} : memref<256x?xf32, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = {{.*}} : f32 left_padding_num = {{.*}} : index
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
// Master-ahead: dynamic offset left_padding_num is remui(offset, num_per_block).
// CHECK-LABEL: @test_dynamic_offset_memref_copy
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: memref<1024xi64, strided<[1]>>)
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[LEFTPAD:.*]] = arith.remui %[[arg0]], %[[C4]] : index
// CHECK: hivm.hir.load ins(%subview : memref<?xi64, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xi64, strided<[1], offset: ?>>) left_padding_num = %[[LEFTPAD]] : index
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
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[LEFTPAD:.*]] = arith.remui %arg0, %[[C4]] : index
  // CHECK: hivm.hir.load {{.*}} pad_mode = <PadValue> pad_value = %c-1_i64 : i64 left_padding_num = %[[LEFTPAD]] : index init_out_buffer = true init_condition = %arg2 : i1
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
  // CHECK: hivm.hir.load {{.*}} pad_mode = <PadValue> pad_value = %c-1_i64 : i64 left_padding_num = %{{.*}} : index init_out_buffer = true init_condition = %arg2 : i1
  memref.copy %subview, %subview_0 : memref<?x1xi64, strided<[1, 1], offset: ?>> to memref<?x1xi64, strided<[1, 1], offset: ?>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1024xi64>
  return %0 : tensor<1024xi64>
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  memref.global "private" @tbl : memref<16xf32, #hivm.address_space<gm>> = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> {alignment = 32 : i64}

  func.func @test_memref_global(%arg2: memref<?xf32>) {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<16xf32>
    //CHECK: hivm.hir.load ins(%reinterpret_cast : memref<16xf32, strided<[1]>>) outs(%alloc : memref<16xf32>)
    memref.copy %reinterpret_cast, %alloc : memref<16xf32, strided<[1]>> to memref<16xf32>
    %global_ref = memref.get_global @tbl : memref<16xf32, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %global_ref to offset: [0], sizes: [16], strides: [1] : memref<16xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>, #hivm.address_space<gm>>
    %alloc_1 = memref.alloc() : memref<16xf32>
    //CHECK: hivm.hir.load ins(%reinterpret_cast_0 : memref<16xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16xf32>)
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<16xf32, strided<[1]>, #hivm.address_space<gm>> to memref<16xf32>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %tensor1 = bufferization.to_tensor %alloc restrict writable : memref<16xf32>
    %tensor2 = bufferization.to_tensor %alloc_1 restrict writable : memref<16xf32>
    %sum_tensor = arith.addf %tensor1, %tensor2 : tensor<16xf32>
    %result = bufferization.to_memref %sum_tensor : memref<16xf32>
    //CHECK: hivm.hir.store
    memref.copy %result, %reinterpret_cast_0 : memref<16xf32> to memref<16xf32, strided<[1]>, #hivm.address_space<gm>>
    return
  }
}
// -----
// CHECK-LABEL: @triton_load_store_one_grid
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @triton_load_store_one_grid_outlined_vf_0(%arg0: memref<16xi32>) attributes {hivm.vector_function} {
    %cst = arith.constant dense<1> : vector<64xi32>
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    vector.transfer_write %cst, %arg0[%c0], %0 {in_bounds = [true]} : vector<64xi32>, memref<16xi32>
    return
  }
  func.func @triton_load_store_one_grid(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %c1_i32 = arith.constant 1 : i32
    %4 = arith.divsi %3, %c1_i32 : i32
    %5 = arith.remsi %4, %arg8 : i32
    %6 = arith.muli %c1_i32, %arg8 : i32
    %7 = arith.divsi %3, %6 : i32
    %8 = arith.remsi %7, %arg7 : i32
    %9 = arith.muli %6, %arg7 : i32
    %10 = arith.divsi %3, %9 : i32
    %11 = arith.remsi %10, %arg6 : i32
    %c1_i32_0 = arith.constant 1 : i32
    %c22 = arith.constant 22 : index
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    scf.for %arg9 = %c0_i32 to %arg5 step %c16_i32  : i32 {
      %12 = arith.index_cast %arg9 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%12], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<16xi32>
      %13 = arith.addi %12, %c16 : index
      %14 = arith.maxsi %12, %c22 : index
      %15 = arith.minsi %13, %14 : index
      %16 = arith.subi %15, %12 : index
      %17 = arith.cmpi slt, %16, %c16 : index
      scf.if %17 {
        annotation.mark %alloc keys = ["pad_const"] values = [%c1_i32_0 : i32] : memref<16xi32>
        func.call @triton_load_store_one_grid_outlined_vf_0(%alloc) {hivm.vector_function} : (memref<16xi32>) -> ()
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0] [%16] [1] : memref<16xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %alloc[0] [%16] [1] : memref<16xi32> to memref<?xi32, strided<[1]>>
      memref.copy %subview, %subview_1 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
      // CHECK: hivm.hir.load {{.*}} pad_mode = <PadValue> pad_value = {{.*}} : i32 {{.*}} eviction_policy = <EvictFirst>
      %18 = bufferization.to_tensor %alloc restrict writable : memref<16xi32>
      // CHECK-NOT: annotation.mark %{{.*}} {eviction_policy
      annotation.mark %alloc {eviction_policy = "evict_first"} : memref<16xi32>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%12], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %18 in writable %reinterpret_cast_2 : (tensor<16xi32>, memref<16xi32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

// -----
// CHECK-LABEL: @memref_copy_dyn_offset_non_const
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[LEFTPADNUM:.*]] = arith.remui %[[RAWOFFSET:.*]], %[[C8]] : index
// CHECK: hivm.hir.load {{.*}} left_padding_num = %[[LEFTPADNUM]] : index eviction_policy = <EvictFirst>
func.func @memref_copy_dyn_offset_non_const(%arg2: memref<?xf32>, %offset0: index, %offset1: index, %size: index) {
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%offset0], sizes: [9598], strides: [1] : memref<?xf32> to memref<9598xf32, strided<[1], offset: ?>>
  %alloc_4 = memref.alloc() : memref<9598xf32>
  %subview_5 = memref.subview %reinterpret_cast_3[%offset1] [%size] [1] : memref<9598xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  %subview_6 = memref.subview %alloc_4[%offset1] [%size] [1] : memref<9598xf32> to memref<?xf32, strided<[1], offset: ?>>
  memref.copy %subview_5, %subview_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  return
}

// -----
// CHECK-LABEL: @do_not_capture_loop_subview_offset
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%[[ALLOC]] : memref<16xf32>)
// CHECK-NOT: left_padding_num
func.func @do_not_capture_loop_subview_offset(%arg0: memref<16xf32>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<16xf32>
  memref.copy %arg0, %alloc : memref<16xf32> to memref<16xf32>
  scf.for %iv = %c0 to %c16 step %c1 {
    %subview = memref.subview %alloc[%iv] [1] [1] : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
    %val = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>>, vector<1xf32>
    vector.transfer_write %val, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
  }
  return
}


// -----
// CHECK-LABEL: @check_dominance_legal
// CHECK: hivm.hir.load ins(%arg0 : memref<32xbf16>) outs(%alloc : memref<32xbf16>) pad_mode = <PadValue> pad_value = %{{.*}}
module {
func.func private @some_vf_call() -> bf16
func.func @check_dominance_legal(%arg0: memref<32xbf16>, %arg1: memref<32xbf16>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<32xbf16>
  scf.if %cond {
    memref.copy %arg0, %alloc : memref<32xbf16> to memref<32xbf16>
    %res = func.call @some_vf_call() : () -> bf16
    annotation.mark %alloc keys = ["pad_const"] values = [%res : bf16] : memref<32xbf16>
  }
  return
}
}

// -----
// CHECK-LABEL: @triton_V_C_kernel_backup
func.func @triton_V_C_kernel_backup(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: i32, %arg7: i32, %arg8: i32) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<1x3xf32>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<1x3xf32>
  %3 = tensor.empty() : tensor<1xf32>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f32) outs({{.*}} : tensor<1xf32>) -> tensor<1xf32>
  %4 = hivm.hir.vbrc ins(%cst : f32) outs(%3 : tensor<1xf32>) -> tensor<1xf32>
  %expanded = tensor.expand_shape %4 [[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
  %5 = hivm.hir.vreduce <sum> ins(%2 : tensor<1x3xf32>) outs(%expanded : tensor<1x1xf32>) reduce_dims = [1] -> tensor<1x1xf32>
  %collapsed = tensor.collapse_shape %5 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
  %6 = tensor.empty() : tensor<1x3xf32>
  %extracted = tensor.extract %collapsed[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f32) outs({{.*}} : tensor<1x3xf32>) -> tensor<1x3xf32>
  %7 = hivm.hir.vbrc ins(%extracted : f32) outs(%6 : tensor<1x3xf32>) -> tensor<1x3xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 3], strides: [3, 1] : memref<?xf32> to memref<1x3xf32, strided<[3, 1]>>
  hivm.hir.store ins(%7 : tensor<1x3xf32>) outs(%reinterpret_cast_0 : memref<1x3xf32, strided<[3, 1]>>)
  return
}

// -----
// i1 element type must not divide-by-zero in getNumPerBlock when computing
// left_padding_num from a subview offset.
// CHECK-LABEL: @load_i1_with_subview_offset
// CHECK: hivm.hir.load
// CHECK-SAME: left_padding_num = {{.*}} : index
func.func @load_i1_with_subview_offset(%arg0: memref<256xi1>) {
  %subview_1 = memref.subview %arg0[8] [16] [1] : memref<256xi1> to memref<16xi1, strided<[1], offset: 8>>
  %alloc = memref.alloc() : memref<256xi1>
  %subview_2 = memref.subview %alloc[8] [16] [1] : memref<256xi1> to memref<16xi1, strided<[1], offset: 8>>
  memref.copy %subview_1, %subview_2 : memref<16xi1, strided<[1], offset: 8>> to memref<16xi1, strided<[1], offset: 8>>
  return
}

// -----
// CHECK-LABEL: @skip_materialize_in_vector_function
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: hivm.hir.store
func.func @skip_materialize_in_vector_function(%arg0: tensor<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function} {
  bufferization.materialize_in_destination %arg0 in writable %arg1 : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----
// CHECK-LABEL: @skip_materialize_in_forall
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: hivm.hir.store
func.func @skip_materialize_in_forall(%arg0: tensor<16xf32>, %arg1: memref<16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  scf.forall (%i) in (1) {
    bufferization.materialize_in_destination %arg0 in writable %arg1 : (tensor<16xf32>, memref<16xf32>) -> ()
  }
  return
}

// -----
// CHECK-LABEL: @skip_memref_copy_in_forall
// CHECK: memref.copy
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.store
// CHECK-NOT: hivm.hir.copy
func.func @skip_memref_copy_in_forall(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  scf.forall (%i) in (1) {
    memref.copy %arg0, %arg1 : memref<16xf32> to memref<16xf32>
  }
  return
}
