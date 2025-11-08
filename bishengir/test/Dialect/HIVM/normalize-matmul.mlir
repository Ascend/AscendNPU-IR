// RUN: bishengir-opt -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_MmadL1_Normalize_Mkn(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK: %[[VAL_1:.*]] = arith.constant true
// CHECK: %[[VAL_2:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_4:.*]] = bufferization.to_tensor %[[VAL_3]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_5:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_6:.*]] = bufferization.to_tensor %[[VAL_5]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: %[[VAL_8:.*]] = arith.constant 16 : index
// CHECK: %[[VAL_9:.*]] = arith.constant 16 : index
// CHECK: %[[VAL_10:.*]] = arith.constant 16 : index
// CHECK: %[[VAL_11:.*]] = hivm.hir.mmadL1 ins(%[[VAL_4]], %[[VAL_6]], %[[VAL_1]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_2]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf32>) outs(%[[VAL_7]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: return %[[VAL_11]] : tensor<16x16xf32>
// CHECK: }

func.func @test_MmadL1_Normalize_Mkn(%arg0: memref<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %alloc_0 = memref.alloc() : memref<16x16xf16>
    %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x16xf16>
    %true = arith.constant true
    %3 = tensor.empty() : tensor<16x16xf32>
    %c0 = arith.constant 0 : index
    %4 = hivm.hir.mmadL1 ins(%1, %2, %true, %c0, %c0, %c0, %0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %4 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL:   func.func @test_MmadL1_Normalize_decompose_matmul(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK:           %[[VAL_4:.*]] = bufferization.to_tensor %[[VAL_3]] restrict writable : memref<16x16xf16>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK:           %[[VAL_6:.*]] = bufferization.to_tensor %[[VAL_5]] restrict writable : memref<16x16xf16>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[VAL_8:.*]] = hivm.hir.load ins(%[[VAL_2]] : tensor<16x16xf32>) outs(%[[VAL_7]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[VAL_10:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_13:.*]] = hivm.hir.mmadL1 ins(%[[VAL_4]], %[[VAL_6]], %[[VAL_1]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%[[VAL_9]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[VAL_15:.*]] = hivm.hir.vadd ins(%[[VAL_13]], %[[VAL_8]] : tensor<16x16xf32>, tensor<16x16xf32>) outs(%[[VAL_14]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           return %[[VAL_15]] : tensor<16x16xf32>
// CHECK:         }

func.func @test_MmadL1_Normalize_decompose_matmul(%arg0: memref<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %alloc_0 = memref.alloc() : memref<16x16xf16>
    %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x16xf16>
    %false = arith.constant false
    %3 = tensor.empty() : tensor<16x16xf32>
    %c0 = arith.constant 0 : index
    %5 = hivm.hir.load ins(%0 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = hivm.hir.mmadL1 ins(%1, %2, %false, %c0, %c0, %c0: tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %4 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL:   func.func @test_madL1_normal_PerChannelAdd(
func.func @test_madL1_normal_PerChannelAdd(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32>) {
  %false = arith.constant false
  %c29_i32 = arith.constant 29 : i32
  %c128 = arith.constant 128 : index
  %c768 = arith.constant 768 : index
  %c29 = arith.constant 29 : index
  %c86 = arith.constant 86 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [29, 128], strides: [128, 1] : memref<?xf16> to memref<29x128xf16, strided<[128, 1]>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<?xf16> to memref<128x768xf16, strided<[768, 1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 768], strides: [768, 1] : memref<?xf32> to memref<1x768xf32, strided<[768, 1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [29, 768], strides: [768, 1] : memref<?xf16> to memref<29x768xf16, strided<[768, 1]>>
  %alloc = memref.alloc() : memref<29x128xf16>
  hivm.hir.load ins(%reinterpret_cast : memref<29x128xf16, strided<[128, 1]>>) outs(%alloc : memref<29x128xf16>)
  %9 = bufferization.to_tensor %alloc restrict writable : memref<29x128xf16>
  %alloc_3 = memref.alloc() : memref<128x768xf16>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<128x768xf16, strided<[768, 1]>>) outs(%alloc_3 : memref<128x768xf16>)
  %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x768xf16>
  %alloc_4 = memref.alloc() : memref<1x768xf32>
  hivm.hir.load ins(%reinterpret_cast_1 : memref<1x768xf32, strided<[768, 1]>>) outs(%alloc_4 : memref<1x768xf32>)
  // CHECK: %[[INIT_TRUE:.*]] = arith.constant true
  // CHECK: %[[VAL_2:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<1x768xf32>
  // CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<29x768xf32>
  // CHECK: %[[VAL_4:.*]] = arith.constant 29 : index
  // CHECK: %[[VAL_5:.*]] = arith.constant 128 : index
  // CHECK: %[[VAL_6:.*]] = arith.constant 768 : index
  // CHECK: %[[VAL_7:.*]] = hivm.hir.mmadL1 ins({{.*}}, {{.*}}, %[[INIT_TRUE]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_2]] : tensor<29x128xf16>, tensor<128x768xf16>, i1, index, index, index, tensor<1x768xf32>) outs(%[[VAL_3]] : tensor<29x768xf32>) -> tensor<29x768xf32>
  %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x768xf32>
  %12 = tensor.empty() : tensor<29x768xf32>
  %13 = hivm.hir.vbrc ins(%11 : tensor<1x768xf32>) outs(%12 : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
  %14 = hivm.hir.mmadL1 ins(%9, %10, %false, %c29, %c128, %c768 : tensor<29x128xf16>, tensor<128x768xf16>, i1, index, index, index)
        outs(%13 : tensor<29x768xf32>) -> tensor<29x768xf32>
  %15 = tensor.empty() : tensor<29x768xf16>
  %16 = hivm.hir.vcast ins(%14 : tensor<29x768xf32>) outs(%15 : tensor<29x768xf16>) round_mode = <rint> -> tensor<29x768xf16>
  hivm.hir.store ins(%16 : tensor<29x768xf16>) outs(%reinterpret_cast_2 : memref<29x768xf16, strided<[768, 1]>>)
  return
}

// -----
// CHECK-LABEL:   func.func @test_madL1_with_perChannelAdd_withSplitKAdd(
func.func @test_madL1_with_perChannelAdd_withSplitKAdd(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32> , %arg6: i32, %arg7: i32, %arg8: i32)  {
  %c5_i32 = arith.constant 5 : i32
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  %c2480_i32 = arith.constant 2480 : i32
  %c128_i32 = arith.constant 128 : i32
  %c16_i32 = arith.constant 16 : i32
  %c2480 = arith.constant 2480 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c16 = arith.constant 16 : index
  %c65536 = arith.constant 65536 : index
  %c32 = arith.constant 32 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %arg8, %arg7 : i32
  %3 = arith.divsi %1, %2 : i32
  %4 = arith.remsi %3, %arg6 : i32
  hivm.hir.set_mask_norm
  %5 = tensor.empty() : tensor<16x128xf32>
  %6 = arith.subi %c2_i32, %4 : i32
  %7 = arith.minsi %6, %c1_i32 : i32
  %8 = arith.remsi %c0_i32, %7 : i32
  %9 = arith.addi %4, %8 : i32
  %10 = arith.divsi %c0_i32, %7 : i32
  %11 = arith.muli %9, %c16_i32 : i32
  %12 = arith.muli %10, %c128_i32 : i32
  %13 = arith.index_cast %11 : i32 to index
  %14 = arith.muli %13, %c2480 : index
  %15 = arith.index_cast %12 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [1, 128], strides: [128, 1] : memref<?xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
  %alloc = memref.alloc() : memref<1x128xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<1x128xf32, strided<[128, 1], offset: ?>>) outs(%alloc : memref<1x128xf32>)
  // CHECK: %[[VAL_2:.*]] = bufferization.to_tensor %alloc restrict writable : memref<1x128xf32>
  %16 = bufferization.to_tensor %alloc restrict writable : memref<1x128xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [16, 512], strides: [2480, 1] : memref<?xf16> to memref<16x512xf16, strided<[2480, 1], offset: ?>>
  %cast = memref.cast %reinterpret_cast_0 : memref<16x512xf16, strided<[2480, 1], offset: ?>> to memref<16x512xf16, strided<[?, ?], offset: ?>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%15], sizes: [512, 128], strides: [128, 1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1], offset: ?>>
  %cast_2 = memref.cast %reinterpret_cast_1 : memref<512x128xf16, strided<[128, 1], offset: ?>> to memref<512x128xf16, strided<[?, ?], offset: ?>>
  %17 = tensor.empty() : tensor<16x128xf32>
  %18:7 = scf.for %arg9 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg10 = %17, %arg11 = %cast, %arg12 = %cast_2, %arg13 = %14, %arg14 = %c0, %arg15 = %15, %arg16 = %c0) -> (tensor<16x128xf32>, memref<16x512xf16, strided<[?, ?], offset: ?>>, memref<512x128xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
    %35 = arith.muli %arg9, %c512_i32 : i32
    %36 = arith.subi %c2480_i32, %35 : i32
    %alloc_4 = memref.alloc() : memref<16x512xf16>
    %37 = arith.index_cast %36 : i32 to index
    %38 = arith.maxsi %37, %c0 : index
    %39 = arith.minsi %38, %c512 : index
    %subview_5 = memref.subview %arg11[0, 0] [16, %39] [1, 1] : memref<16x512xf16, strided<[?, ?], offset: ?>> to memref<16x?xf16, strided<[?, ?], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0, 0] [16, %39] [1, 1] : memref<16x512xf16> to memref<16x?xf16, strided<[512, 1]>>
    hivm.hir.load ins(%subview_5 : memref<16x?xf16, strided<[?, ?], offset: ?>>) outs(%subview_6 : memref<16x?xf16, strided<[512, 1]>>) left_padding_num = %c0 : index
    %40 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x512xf16>
    %alloc_7 = memref.alloc() : memref<512x128xf16>
    %subview_8 = memref.subview %arg12[0, 0] [%39, 128] [1, 1] : memref<512x128xf16, strided<[?, ?], offset: ?>> to memref<?x128xf16, strided<[?, ?], offset: ?>>
    %subview_9 = memref.subview %alloc_7[0, 0] [%39, 128] [1, 1] : memref<512x128xf16> to memref<?x128xf16, strided<[128, 1]>>
    hivm.hir.load ins(%subview_8 : memref<?x128xf16, strided<[?, ?], offset: ?>>) outs(%subview_9 : memref<?x128xf16, strided<[128, 1]>>) left_padding_num = %c0 : index
    %41 = bufferization.to_tensor %alloc_7 restrict writable : memref<512x128xf16>
    %42 = arith.cmpi eq, %arg9, %c0_i32 : i32
    // CHECK: %[[VAL_3:.*]] = arith.constant 16 : index
    // CHECK: %[[VAL_4:.*]] = arith.constant 128 : index
    // CHECK: %[[VAL_5:.*]] = hivm.hir.mmadL1 ins({{.*}}, {{.*}}, {{.*}}, %[[VAL_3]], {{.*}}, %[[VAL_4]], %[[VAL_2]] : tensor<16x512xf16>, tensor<512x128xf16>, i1, index, index, index, tensor<1x128xf32>) outs({{.*}} : tensor<16x128xf32>) -> tensor<16x128xf32>
    %43 = hivm.hir.mmadL1 ins(%40, %41, %42, %c16, %39, %c128 : tensor<16x512xf16>, tensor<512x128xf16>, i1, index, index, index) outs(%arg10 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %44 = arith.addi %arg13, %c512 : index
    %45 = arith.addi %44, %arg14 : index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [16, 512], strides: [2480, 1] : memref<?xf16> to memref<16x512xf16, strided<[2480, 1], offset: ?>>
    %cast_11 = memref.cast %reinterpret_cast_10 : memref<16x512xf16, strided<[2480, 1], offset: ?>> to memref<16x512xf16, strided<[?, ?], offset: ?>>
    %46 = arith.addi %arg15, %c65536 : index
    %47 = arith.addi %46, %arg16 : index
    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%47], sizes: [512, 128], strides: [128, 1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1], offset: ?>>
    %cast_13 = memref.cast %reinterpret_cast_12 : memref<512x128xf16, strided<[128, 1], offset: ?>> to memref<512x128xf16, strided<[?, ?], offset: ?>>
    scf.yield %43, %cast_11, %cast_13, %45, %c0, %47, %c0 : tensor<16x128xf32>, memref<16x512xf16, strided<[?, ?], offset: ?>>, memref<512x128xf16, strided<[?, ?], offset: ?>>, index, index, index, index
  }
  // CHECK-NOT: hivm.hir.vbrc
  %19 = hivm.hir.vbrc ins(%16 : tensor<1x128xf32>) outs(%5 : tensor<16x128xf32>) broadcast_dims = [0] -> tensor<16x128xf32>
  // CHECK-NOT: hivm.hir.vadd
  %20 = hivm.hir.vadd ins(%18#0, %19 : tensor<16x128xf32>, tensor<16x128xf32>) outs(%5 : tensor<16x128xf32>) -> tensor<16x128xf32>
  %21 = tensor.empty() : tensor<16x128xf16>
  %22 = hivm.hir.vcast ins(%20 : tensor<16x128xf32>) outs(%21 : tensor<16x128xf16>) round_mode = <rint> -> tensor<16x128xf16>
  %23 = arith.muli %13, %c128 : index
  %24 = arith.addi %23, %15 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [16, 128], strides: [128, 1] : memref<?xf16> to memref<16x128xf16, strided<[128, 1], offset: ?>>
  %25 = arith.addi %13, %c16 : index
  %26 = arith.maxsi %13, %c32 : index
  %27 = arith.minsi %25, %26 : index
  %28 = arith.subi %27, %13 : index
  %29 = arith.addi %15, %c128 : index
  %30 = arith.maxsi %15, %c128 : index
  %31 = arith.minsi %29, %30 : index
  %32 = arith.subi %31, %15 : index
  %33 = arith.minsi %28, %c16 : index
  %34 = arith.minsi %32, %c128 : index
  %extracted_slice = tensor.extract_slice %22[0, 0] [%33, %34] [1, 1] : tensor<16x128xf16> to tensor<?x?xf16>
  %subview = memref.subview %reinterpret_cast_3[0, 0] [%33, %34] [1, 1] : memref<16x128xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
  hivm.hir.store ins(%extracted_slice : tensor<?x?xf16>) outs(%subview : memref<?x?xf16, strided<[128, 1], offset: ?>>)
  return
}

// -----
// CHECK-LABEL: func.func @dot_with_gm_bias(
module {
  func.func @dot_with_gm_bias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %false = arith.constant false
    hivm.hir.set_mask_norm
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi8> to memref<16x16xi8, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xi8>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xi8, strided<[16, 1]>>) outs(%alloc : memref<16x16xi8>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi8> to memref<16x16xi8, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xi8>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xi8, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xi8>)
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi32> to memref<16x16xi32, strided<[16, 1]>>
    %alloc_3 = memref.alloc() : memref<16x16xi32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<16x16xi32, strided<[16, 1]>>) outs(%alloc_3 : memref<16x16xi32>)
    %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xi32>
    %c16 = arith.constant 16 : index
    %c16_4 = arith.constant 16 : index
    %c16_5 = arith.constant 16 : index
    %3 = hivm.hir.mmadL1 ins(%0, %1, %false, %c16, %c16_4, %c16_5 : tensor<16x16xi8>, tensor<16x16xi8>, i1, index, index, index) outs(%2 : tensor<16x16xi32>) -> tensor<16x16xi32>
    // CHECK: hivm.hir.vadd
    %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi32> to memref<16x16xi32, strided<[16, 1]>>
    hivm.hir.store ins(%3 : tensor<16x16xi32>) outs(%reinterpret_cast_6 : memref<16x16xi32, strided<[16, 1]>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @loop_matmul_with_normal_perChannel
  func.func @loop_matmul_with_normal_perChannel(%arg4: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %c5_i32 = arith.constant 5 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %b = memref.alloc() : memref<256x64xf16>
    %bTensor = bufferization.to_tensor %b restrict writable : memref<256x64xf16>
    // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
    %bias = arith.constant dense<1.000000e+00> : tensor<1x64xf32>
    %1 = tensor.empty() : tensor<128x64xf32>
    %bias_brc = hivm.hir.vbrc ins(%bias : tensor<1x64xf32>) outs(%1 : tensor<128x64xf32>) broadcast_dims = [0] -> tensor<128x64xf32>
    scf.for %arg = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
      %2 = arith.muli %arg, %c128_i32 : i32
      %3 = arith.index_cast %2 : i32 to index
      %a = memref.alloc() : memref<128x256xf16>
      %aTensor = bufferization.to_tensor %a restrict writable : memref<128x256xf16>
      // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<128x64xf32>
      // CHECK: hivm.hir.mmadL1 ins
      // CHECK-SAME: %[[TRUE]]
      // CHECK-SAME: %[[BIAS]]
      // CHECK-SAME: outs(%[[EMPTY]]
      %mad = hivm.hir.mmadL1 ins(%aTensor, %bTensor, %false, %c0, %c0, %c0 : tensor<128x256xf16>, tensor<256x64xf16>, i1, index, index, index) outs(%bias_brc : tensor<128x64xf32>) -> tensor<128x64xf32>
      %4 = arith.muli %3, %c64 : index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%4], sizes: [128, 64], strides: [64, 1] : memref<?xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
      hivm.hir.store ins(%mad : tensor<128x64xf32>) outs(%reinterpret_cast : memref<128x64xf32, strided<[64, 1], offset: ?>>)
    }
    return
  }
}

// -----
module {
  func.func @only_pad_k(%gm_a : memref<?x?xi8>, %gm_b : memref<?x?xi8>,
                        %tile_m : index, %tile_k : index, %tile_n : index,
                        %real_m : index, %real_k : index, %real_n : index) {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %init_cond = arith.constant 1 : i1

    %gm_subview_a = memref.subview %gm_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    %alloc_a = memref.alloc(%tile_m, %tile_k) : memref<?x?xi8>
    %subview_a = memref.subview %alloc_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    // CHECK: init_out_buffer = false
    // CHECK-NOT: init_condition
    hivm.hir.load ins(%gm_subview_a : memref<?x?xi8, strided<[?, 1]>>) outs(%subview_a : memref<?x?xi8, strided<[?, 1]>>)
      pad_mode = <PadValue>
      pad_value = %c0_i8 : i8
      left_padding_num = %c0 : index
      init_out_buffer = true
      init_condition = %init_cond : i1

    %tensor_a = bufferization.to_tensor %alloc_a restrict writable : memref<?x?xi8>
    annotation.mark %tensor_a {dot_pad_only_k} : tensor<?x?xi8>

    %gm_subview_b = memref.subview %gm_b[0, 0] [%real_k, %real_n] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    %alloc_b = memref.alloc(%tile_k, %tile_n) : memref<?x?xi8>
    %subview_b = memref.subview %alloc_b[0, 0] [%real_k, %real_n] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    // CHECK: init_out_buffer = false
    // CHECK-NOT: init_condition
    hivm.hir.load ins(%gm_subview_b : memref<?x?xi8, strided<[?, 1]>>) outs(%subview_b : memref<?x?xi8, strided<[?, 1]>>)
      pad_mode = <PadValue>
      pad_value = %c0_i8 : i8
      left_padding_num = %c0 : index
      init_out_buffer = true
      init_condition = %init_cond : i1

    %tensor_b = bufferization.to_tensor %alloc_b restrict writable : memref<?x?xi8>
    annotation.mark %tensor_b {dot_pad_only_k} : tensor<?x?xi8>

    %empty = tensor.empty(%tile_m, %tile_n) : tensor<?x?xi32>
    %tensor_c = hivm.hir.mmadL1 ins(%tensor_a, %tensor_b, %init_cond, %c0, %c0, %c0 : tensor<?x?xi8>, tensor<?x?xi8>, i1, index, index, index) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
    "some_use"(%tensor_c) : (tensor<?x?xi32>) -> ()
    return
  }
}

// -----
module {
  func.func @only_pad_k_none_zero(%gm_a : memref<?x?xi8>, %gm_b : memref<?x?xi8>,
                                  %tile_m : index, %tile_k : index, %tile_n : index,
                                  %real_m : index, %real_k : index, %real_n : index) {
    %c0 = arith.constant 0 : index
    %c100_i8 = arith.constant 100: i8
    %init_cond = arith.constant 1 : i1

    %gm_subview_a = memref.subview %gm_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    %alloc_a = memref.alloc(%tile_m, %tile_k) : memref<?x?xi8>
    %subview_a = memref.subview %alloc_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    // CHECK: init_out_buffer = true
    hivm.hir.load ins(%gm_subview_a : memref<?x?xi8, strided<[?, 1]>>) outs(%subview_a : memref<?x?xi8, strided<[?, 1]>>)
      pad_mode = <PadValue>
      pad_value = %c100_i8 : i8
      left_padding_num = %c0 : index
      init_out_buffer = true
      init_condition = %init_cond : i1

    return
  }
}

// -----
module {
  func.func @triton_dot_perChannel_implicit_brc(%arg3: memref<?xf32>) {
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<100x100xf16>
    %0 = bufferization.to_tensor %a restrict writable : memref<100x100xf16>
    %b = memref.alloc() : memref<100x100xf16>
    %1 = bufferization.to_tensor %b restrict writable : memref<100x100xf16>
    // CHECK: %[[BIAS:.*]] = tensor.empty() : tensor<100xf16>
    %2 = tensor.empty() : tensor<100xf16>
    %3 = tensor.empty() : tensor<100xf32>
    %4 = hivm.hir.vcast ins(%2 : tensor<100xf16>) outs(%3 : tensor<100xf32>) -> tensor<100xf32>
    %5 = tensor.empty() : tensor<100x100xf32>
    // CHECK-NOT: tensor.expand_shape
    %expanded = tensor.expand_shape %4 [[0, 1]] output_shape [1, 100] : tensor<100xf32> into tensor<1x100xf32>
    %6 = hivm.hir.vbrc ins(%expanded : tensor<1x100xf32>) outs(%5 : tensor<100x100xf32>) broadcast_dims = [0] -> tensor<100x100xf32>
    // CHECK: hivm.hir.mmadL1
    // CHECK-SAME: %[[BIAS]]
    %7 = hivm.hir.mmadL1 ins(%0, %1, %false, %c0, %c0, %c0 : tensor<100x100xf16>, tensor<100x100xf16>, i1, index, index, index) outs(%6 : tensor<100x100xf32>) -> tensor<100x100xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [100, 100], strides: [100, 1] : memref<?xf32> to memref<100x100xf32, strided<[100, 1]>>
    hivm.hir.store ins(%7 : tensor<100x100xf32>) outs(%reinterpret_cast_4 : memref<100x100xf32, strided<[100, 1]>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @triton_no_perChannel_with_ifop
  func.func @triton_no_perChannel_with_ifop(%arg0: memref<?xf32>, %arg1: i1, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32xi32>
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = hivm.hir.vbrc ins(%cst : f32) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = hivm.hir.varange offset[%c0] strides[%c1] outs(%0 : tensor<32xi32>) -> tensor<32xi32>
    %4 = scf.if %arg1 -> (tensor<32x32xf32>) {
      %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
      %8 = tensor.empty() : tensor<1x32xi32>
      %9 = hivm.hir.vadd ins(%expanded, %arg2 : tensor<1x32xi32>, i32) outs(%8 : tensor<1x32xi32>) -> tensor<1x32xi32>
      %10 = tensor.empty() : tensor<1x32xf32>
      %11 = hivm.hir.vcast ins(%9 : tensor<1x32xi32>) outs(%10 : tensor<1x32xf32>) -> tensor<1x32xf32>
      %12 = hivm.hir.vbrc ins(%11 : tensor<1x32xf32>) outs(%1 : tensor<32x32xf32>) broadcast_dims = [0] -> tensor<32x32xf32>
      scf.yield %12 : tensor<32x32xf32>
    } else {
      scf.yield %2 : tensor<32x32xf32>
    }
    %false = arith.constant false
    %alloc = memref.alloc() : memref<32x64xf16>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16>
    %alloc_0 = memref.alloc() : memref<64x32xf16>
    %6 = bufferization.to_tensor %alloc_0 restrict writable : memref<64x32xf16>
    %7 = hivm.hir.mmadL1 ins(%5, %6, %false, %c0, %c0, %c0 : tensor<32x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: hivm.hir.mmadL1
    // CHECK: hivm.hir.vadd
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    hivm.hir.store ins(%7 : tensor<32x32xf32>) outs(%reinterpret_cast : memref<32x32xf32, strided<[32, 1]>>)
    return
  }
}