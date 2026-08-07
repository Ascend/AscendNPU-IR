// RUN: bishengir-opt -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// A3 / mem-based (Ascend910B4)
//===----------------------------------------------------------------------===//

// -----
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @only_pad_k(%gm_a : memref<?x?xi8>, %gm_b : memref<?x?xi8>,
                        %tile_m : index, %tile_k : index, %tile_n : index,
                        %real_m : index, %real_k : index, %real_n : index) {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %init_cond = arith.constant 1 : i1

    %gm_subview_a = memref.subview %gm_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    %alloc_a = memref.alloc(%tile_m, %tile_k) : memref<?x?xi8>
    %subview_a = memref.subview %alloc_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    // CHECK-NOT: init_out_buffer
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
    // CHECK-NOT: init_out_buffer
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
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
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

//===----------------------------------------------------------------------===//
// A5 / reg-based (Ascend950PR_9589)
//===----------------------------------------------------------------------===//

// -----
// CHECK-LABEL: func.func @test_MmadL1_Normalize_Mkn(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK-DAG: %[[VAL_1:.*]] = arith.constant true
// CHECK-DAG: %[[VAL_2:.*]] = arith.constant 16 : index
// CHECK: %[[VAL_3:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK: %[[VAL_4:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: %[[VAL_9:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins(%[[VAL_5]], %[[VAL_7]], %[[VAL_1]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_3]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf32>) outs(%[[VAL_8]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: return %[[VAL_9]] : tensor<16x16xf32>
// CHECK: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
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
}

// -----
// CHECK-LABEL: func.func @test_MmadL1_Normalize_Mkn_init_false(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK-DAG: %[[VAL_1:.*]] = arith.constant true
// CHECK-DAG: %[[VAL_2:.*]] = arith.constant 16 : index
// CHECK: %[[VAL_3:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK: %[[VAL_4:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK-NOT: %[[VAL_9:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins(%[[VAL_5]], %[[VAL_7]], %[[VAL_1]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_3]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf32>) outs(%[[VAL_8]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_MmadL1_Normalize_Mkn_init_false(%arg0: memref<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %alloc_0 = memref.alloc() : memref<16x16xf16>
    %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x16xf16>
    %false = arith.constant false
    %3 = tensor.empty() : tensor<16x16xf32>
    %c0 = arith.constant 0 : index
    %4 = hivm.hir.mmadL1 ins(%1, %2, %false, %c0, %c0, %c0, %0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %4 : tensor<16x16xf32>
}
}

// -----
// CHECK-LABEL: func.func @test_MmadL1_Normalize_Mkn_init_false_vbrc(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK-DAG: %[[VAL_1:.*]] = arith.constant true
// CHECK-DAG: %[[VAL_2:.*]] = arith.constant 16 : index
// CHECK: %[[VAL_3:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK: %[[VAL_4:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<16x16xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK-NOT:  %[[VAL_9:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins(%[[VAL_5]], %[[VAL_7]], %[[VAL_1]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]], %[[VAL_3]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf32>) outs(%[[VAL_8]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK--: return %[[VAL_9]] : tensor<16x16xf32>
// CHECK: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_MmadL1_Normalize_Mkn_init_false_vbrc(%arg0: memref<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %alloc_0 = memref.alloc() : memref<16x16xf16>
    %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x16xf16>
    %false = arith.constant false
    %3 = tensor.empty() : tensor<16x16xf32>
    %4 = hivm.hir.vbrc ins(%cst : f32) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %c0 = arith.constant 0 : index
    %5 = hivm.hir.mmadL1 ins(%1, %2, %false, %c0, %c0, %c0, %0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %5 : tensor<16x16xf32>
}
}

// -----
// CHECK-LABEL:   func.func @test_MmadL1_Normalize_decompose_matmul(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_3:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK:           %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] restrict writable : memref<16x16xf16>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<16x16xf16>
// CHECK:           %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<16x16xf16>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.load ins(%[[VAL_3]] : tensor<16x16xf32>) outs(%[[VAL_8]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[VAL_11:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%[[VAL_5]], %[[VAL_7]], %[[VAL_1]], %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%[[VAL_10]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[VAL_13:.*]] = hivm.hir.vadd ins(%[[VAL_11]], %[[VAL_9]] : tensor<16x16xf32>, tensor<16x16xf32>) outs(%[[VAL_12]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           return %[[VAL_13]] : tensor<16x16xf32>
// CHECK:         }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
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
}

// -----
// CHECK-LABEL:   func.func @test_madL1_normal_PerChannelAdd(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_madL1_normal_PerChannelAdd(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32>) {
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 29 : index
  // CHECK-DAG: %[[VAL_5:.*]] = arith.constant 128 : index
  // CHECK-DAG: %[[VAL_6:.*]] = arith.constant 768 : index
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
  // CHECK-DAG: %[[INIT_TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[VAL_2:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<1x768xf32>
  // CHECK-DAG: %[[VAL_3:.*]] = tensor.empty() : tensor<29x768xf32>
  // CHECK-NOT: %[[VAL_7:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, %[[INIT_TRUE]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_2]] : tensor<29x128xf16>, tensor<128x768xf16>, i1, index, index, index, tensor<1x768xf32>) outs(%[[VAL_3]] : tensor<29x768xf32>) -> tensor<29x768xf32>
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
}

// -----
// CHECK-LABEL:   func.func @test_madL1_normal_PerChannelAdd_init_true(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_madL1_normal_PerChannelAdd_init_true(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32>) {
  // CHECK-DAG: %[[INIT_TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 29 : index
  // CHECK-DAG: %[[VAL_5:.*]] = arith.constant 128 : index
  // CHECK-DAG: %[[VAL_6:.*]] = arith.constant 768 : index
  %true = arith.constant true
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
  // CHECK-DAG: %[[VAL_2:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<1x768xf32>
  // CHECK-DAG: %[[VAL_3:.*]] = tensor.empty() : tensor<29x768xf32>
  // CHECK-DAG: %[[VAL_13:.*]] = hivm.hir.vbrc ins(%[[VAL_2]] : tensor<1x768xf32>) outs(%[[VAL_3]] : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
  // CHECK: %[[VAL_7:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, %[[INIT_TRUE]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : tensor<29x128xf16>, tensor<128x768xf16>, i1, index, index, index) outs(%[[VAL_13]] : tensor<29x768xf32>) -> tensor<29x768xf32>
  %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x768xf32>
  %12 = tensor.empty() : tensor<29x768xf32>
  %13 = hivm.hir.vbrc ins(%11 : tensor<1x768xf32>) outs(%12 : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
  %14 = hivm.hir.mmadL1 ins(%9, %10, %true, %c29, %c128, %c768 : tensor<29x128xf16>, tensor<128x768xf16>, i1, index, index, index)
        outs(%13 : tensor<29x768xf32>) -> tensor<29x768xf32>
  %15 = tensor.empty() : tensor<29x768xf16>
  %16 = hivm.hir.vcast ins(%14 : tensor<29x768xf32>) outs(%15 : tensor<29x768xf16>) round_mode = <rint> -> tensor<29x768xf16>
  hivm.hir.store ins(%16 : tensor<29x768xf16>) outs(%reinterpret_cast_2 : memref<29x768xf16, strided<[768, 1]>>)
  return
}
}

// -----
// CHECK-LABEL:   func.func @test_madL1_with_perChannelAdd_withSplitKAdd(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_madL1_with_perChannelAdd_withSplitKAdd(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32> , %arg6: i32, %arg7: i32, %arg8: i32)  {
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 128 : index
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
    // CHECK: %[[VAL_5:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[VAL_3]], {{.*}}, %[[VAL_4]], %[[VAL_2]] : tensor<16x512xf16>, tensor<512x128xf16>, i1, index, index, index, tensor<1x128xf32>) outs({{.*}} : tensor<16x128xf32>) -> tensor<16x128xf32>
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
}

// -----
// CHECK-LABEL:   func.func @test_madL1_with_postPerChannelAdd_withSplitKAdd(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_madL1_with_postPerChannelAdd_withSplitKAdd(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32> , %arg6: i32, %arg7: i32, %arg8: i32)  {
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 128 : index
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
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<128xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<128xf32, strided<[1], offset: ?>>) outs(%alloc : memref<128xf32>)
  // CHECK: %[[VAL_2:.*]] = bufferization.to_tensor %alloc restrict writable : memref<128xf32>
  %16 = bufferization.to_tensor %alloc restrict writable : memref<128xf32>
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
    // CHECK: %[[VAL_5:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[VAL_3]], {{.*}}, %[[VAL_4]], %[[VAL_2]] : tensor<16x512xf16>, tensor<512x128xf16>, i1, index, index, index, tensor<128xf32>) outs({{.*}} : tensor<16x128xf32>) -> tensor<16x128xf32>
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
  %expanded = tensor.expand_shape %16 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
  // CHECK-NOT: hivm.hir.vbrc
  %19 = hivm.hir.vbrc ins(%expanded : tensor<1x128xf32>) outs(%5 : tensor<16x128xf32>) broadcast_dims = [0] -> tensor<16x128xf32>
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
}

// -----
// CHECK-LABEL:   func.func @fused_matmul_fwd_kernel_MMInitPerChannelAddWithSplitK(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @fused_matmul_fwd_kernel_MMInitPerChannelAddWithSplitK(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %false = arith.constant false
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c64 = arith.constant 64 : index
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %c128_i32 = arith.constant 128 : i32
  %c63_i32 = arith.constant 63 : i32
  %c127_i32 = arith.constant 127 : i32
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg9 : i32
  %7 = arith.addi %arg6, %c63_i32 : i32
  %8 = arith.divsi %7, %c64_i32 : i32
  %9 = arith.addi %arg7, %c63_i32 : i32
  %10 = arith.divsi %9, %c64_i32 : i32
  %11 = arith.muli %10, %c8_i32 : i32
  %12 = arith.divsi %6, %11 : i32
  %13 = arith.muli %12, %c8_i32 : i32
  %14 = arith.subi %8, %13 : i32
  %15 = arith.minsi %14, %c8_i32 : i32
  %16 = arith.remsi %6, %11 : i32
  %17 = arith.remsi %16, %15 : i32
  %18 = arith.addi %13, %17 : i32
  %19 = arith.divsi %16, %15 : i32
  %20 = arith.muli %18, %c64_i32 : i32
  %21 = arith.cmpi sge, %20, %arg6 : i32
  %22 = arith.muli %19, %c64_i32 : i32
  %23 = arith.cmpi sge, %22, %arg7 : i32
  %24 = arith.ori %21, %23 : i1
  scf.if %24 {
  } else {
    %25 = arith.index_cast %22 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [64], strides: [1] : memref<?xf16> to memref<64xf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<64xf16>
    %26 = arith.addi %25, %c64 : index
    %27 = arith.index_cast %arg7 : i32 to index
    %28 = arith.maxsi %25, %27 : index
    %29 = arith.minsi %26, %28 : index
    %30 = arith.subi %29, %25 : index
    %31 = arith.cmpi slt, %30, %c64 : index
    %subview = memref.subview %reinterpret_cast[0] [%30] [1] : memref<64xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%30] [1] : memref<64xf16> to memref<?xf16, strided<[1]>>
    hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xf16, strided<[1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %31 : i1 eviction_policy = <EvictFirst>
    %32 = bufferization.to_tensor %alloc restrict writable : memref<64xf16>
    // CHECK: %[[VAL_32:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<64xf16>
    %33 = tensor.empty() : tensor<64xf32>
    %34 = hivm.hir.vcast ins(%32 : tensor<64xf16>) outs(%33 : tensor<64xf32>) -> tensor<64xf32>
    %35 = tensor.empty() : tensor<64x64xf32>
    %expanded = tensor.expand_shape %34 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
    %36 = hivm.hir.vbrc ins(%expanded : tensor<1x64xf32>) outs(%35 : tensor<64x64xf32>) broadcast_dims = [0] -> tensor<64x64xf32>
    // CHECK-NOT: hivm.hir.vbrc
    %37 = arith.addi %arg8, %c127_i32 : i32
    %38 = arith.divsi %37, %c128_i32 : i32
    // CHECK: %[[VAL_35:.*]] = tensor.empty() : tensor<64x64xf32>
    // CHECK-NOT: %36 = scf.for %[[VAL_arg12:.*]] = %[[VAL_c0_i32:.*]] to {{.*}} step {{.*}} iter_args(%[[VAL_arg13:.*]] = %[[VAL_35]]) -> (tensor<64x64xf32>)  : i32 {
    %39 = scf.for %arg12 = %c0_i32 to %38 step %c1_i32 iter_args(%arg13 = %36) -> (tensor<64x64xf32>)  : i32 {
      %52 = arith.muli %arg12, %c128_i32 : i32
      %53 = arith.index_cast %20 : i32 to index
      %54 = arith.index_cast %arg8 : i32 to index
      %55 = arith.muli %53, %54 : index
      %56 = arith.index_cast %52 : i32 to index
      %57 = arith.addi %55, %56 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%57], sizes: [64, 128], strides: [%54, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %alloc_4 = memref.alloc() : memref<64x128xf16>
      %58 = arith.addi %53, %c64 : index
      %59 = arith.index_cast %arg6 : i32 to index
      %60 = arith.maxsi %53, %59 : index
      %61 = arith.minsi %58, %60 : index
      %62 = arith.subi %61, %53 : index
      %63 = arith.addi %56, %c128 : index
      %64 = arith.maxsi %56, %54 : index
      %65 = arith.minsi %63, %64 : index
      %66 = arith.subi %65, %56 : index
      %67 = arith.minsi %62, %c64 : index
      %68 = arith.minsi %66, %c128 : index
      %69 = arith.cmpi slt, %67, %c64 : index
      %70 = arith.cmpi slt, %68, %c128 : index
      %71 = arith.ori %69, %70 : i1
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%67, %68] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %alloc_4[0, 0] [%67, %68] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      hivm.hir.load ins(%subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_6 : memref<?x?xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %71 : i1 eviction_policy = <EvictFirst>
      %72 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x128xf16>
      %73 = arith.muli %56, %27 : index
      %74 = arith.addi %73, %25 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%74], sizes: [128, 64], strides: [%27, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
      %alloc_8 = memref.alloc() : memref<128x64xf16>
      %75 = arith.minsi %30, %c64 : index
      %76 = arith.cmpi slt, %75, %c64 : index
      %77 = arith.ori %70, %76 : i1
      %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%68, %75] [1, 1] : memref<128x64xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_10 = memref.subview %alloc_8[0, 0] [%68, %75] [1, 1] : memref<128x64xf16> to memref<?x?xf16, strided<[64, 1]>>
      hivm.hir.load ins(%subview_9 : memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_10 : memref<?x?xf16, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %77 : i1 eviction_policy = <EvictFirst>
      %78 = bufferization.to_tensor %alloc_8 restrict writable : memref<128x64xf16>
      %79 = hivm.hir.mmadL1 ins(%72, %78, %false, %c0, %c0, %c0 : tensor<64x128xf16>, tensor<128x64xf16>, i1, index, index, index) outs(%arg13 : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %79 : tensor<64x64xf32>
    }
    %40 = tensor.empty() : tensor<64x64xf16>
    %41 = hivm.hir.vcast ins(%39 : tensor<64x64xf32>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
    %42 = arith.index_cast %20 : i32 to index
    %43 = arith.muli %42, %27 : index
    %44 = arith.addi %43, %25 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [64, 64], strides: [%27, 1] : memref<?xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
    %45 = arith.addi %42, %c64 : index
    %46 = arith.index_cast %arg6 : i32 to index
    %47 = arith.maxsi %42, %46 : index
    %48 = arith.minsi %45, %47 : index
    %49 = arith.subi %48, %42 : index
    %50 = arith.minsi %49, %c64 : index
    %51 = arith.minsi %30, %c64 : index
    %extracted_slice = tensor.extract_slice %41[0, 0] [%50, %51] [1, 1] : tensor<64x64xf16> to tensor<?x?xf16>
    %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%50, %51] [1, 1] : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<?x?xf16>) outs(%subview_2 : memref<?x?xf16, strided<[?, 1], offset: ?>>)
  }
  return
}
}

// -----
// CHECK-LABEL: func.func @dot_with_gm_bias(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
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
// CHECK-LABEL: func.func @dot_with_no_bias(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dot_with_no_bias(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
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
    %alloc_3 = memref.alloc() : memref<16x16xi32, #hivm.address_space<cc>>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<16x16xi32, strided<[16, 1]>>) outs(%alloc_3 : memref<16x16xi32, #hivm.address_space<cc>>)
    %memspacecast_22 = memref.memory_space_cast %alloc_3 : memref<16x16xi32, #hivm.address_space<cc>> to memref<16x16xi32>
    %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xi32, #hivm.address_space<cc>>
    %c16 = arith.constant 16 : index
    %c16_4 = arith.constant 16 : index
    %c16_5 = arith.constant 16 : index
    // CHECK: hivm.hir.vadd
    %3 = hivm.hir.mmadL1 ins(%0, %1, %false, %c16, %c16_4, %c16_5 : tensor<16x16xi8>, tensor<16x16xi8>, i1, index, index, index) outs(%2 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi32> to memref<16x16xi32, strided<[16, 1]>>
    hivm.hir.store ins(%3 : tensor<16x16xi32>) outs(%reinterpret_cast_6 : memref<16x16xi32, strided<[16, 1]>>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
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
      // %5 = hivm.hir.mmadL1 {{.*}} ins({{.*}} %cst : {{.*}} tensor<1x64xf32>) outs(%[[EMPTY]] : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mad = hivm.hir.mmadL1 ins(%aTensor, %bTensor, %false, %c0, %c0, %c0 : tensor<128x256xf16>, tensor<256x64xf16>, i1, index, index, index) outs(%bias_brc : tensor<128x64xf32>) -> tensor<128x64xf32>
      %4 = arith.muli %3, %c64 : index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%4], sizes: [128, 64], strides: [64, 1] : memref<?xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
      hivm.hir.store ins(%mad : tensor<128x64xf32>) outs(%reinterpret_cast : memref<128x64xf32, strided<[64, 1], offset: ?>>)
    }
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @only_pad_k(
  // CHECK-SAME:                        {{.*}}: memref<?x?xi8>, {{.*}}: memref<?x?xi8>,
  // CHECK-SAME:                        {{.*}}: index, {{.*}}: index, {{.*}}: index, %[[REAL_M:.*]]: index, %[[REAL_K:.*]]: index, %[[REAL_N:.*]]: index)
  func.func @only_pad_k(%gm_a : memref<?x?xi8>, %gm_b : memref<?x?xi8>,
                        %tile_m : index, %tile_k : index, %tile_n : index,
                        %real_m : index, %real_k : index, %real_n : index) {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %init_cond = arith.constant 1 : i1

    %gm_subview_a = memref.subview %gm_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    %alloc_a = memref.alloc(%tile_m, %tile_k) : memref<?x?xi8>
    %subview_a = memref.subview %alloc_a[0, 0] [%real_m, %real_k] [1, 1] : memref<?x?xi8> to memref<?x?xi8, strided<[?, 1]>>
    // CHECK-NOT: init_out_buffer
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
    // CHECK-NOT: init_out_buffer
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
    // CHECK: %[[REAL_M]], %[[REAL_K]], %[[REAL_N]]
    %tensor_c = hivm.hir.mmadL1 ins(%tensor_a, %tensor_b, %init_cond, %c0, %c0, %c0 : tensor<?x?xi8>, tensor<?x?xi8>, i1, index, index, index) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
    "some_use"(%tensor_c) : (tensor<?x?xi32>) -> ()
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
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
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
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
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_madL1_with_PerChannelAdd_in_cond(
  func.func @test_madL1_with_PerChannelAdd_in_cond(%arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xf32>, %cond: i1) {
    %false = arith.constant false
    %c29 = arith.constant 29 : index
    %c128 = arith.constant 128 : index
    %c768 = arith.constant 768 : index
    %alloc = memref.alloc() : memref<29x128xf16>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<29x128xf16>
    %alloc_3 = memref.alloc() : memref<128x768xf16>
    %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x768xf16>
    %alloc_4 = memref.alloc() : memref<1x768xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 768], strides: [768, 1] : memref<?xf32> to memref<1x768xf32, strided<[768, 1]>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [29, 768], strides: [768, 1] : memref<?xf16> to memref<29x768xf16, strided<[768, 1]>>
    hivm.hir.load ins(%reinterpret_cast_1 : memref<1x768xf32, strided<[768, 1]>>) outs(%alloc_4 : memref<1x768xf32>)
    %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x768xf32>
    %12 = tensor.empty() : tensor<29x768xf32>
    %outC = hivm.hir.vbrc ins(%11 : tensor<1x768xf32>) outs(%12 : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
    // CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[BRC_1D:.*]] : tensor<1x768xf32>) outs(%[[BRC_INIT:.*]] : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
    // CHECK: memref.store %[[C0:.*]], %[[CNT:.*]][] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<29x768xf32>
    // CHECK: %[[IFRES:.*]] = scf.if
    // CHECK: %[[LOAD:.*]] = memref.load %[[CNT]][] : memref<i32>
    // CHECK: %[[INIT:.*]] = arith.cmpi eq, %[[LOAD]], %[[C0]] : i32
    // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
    // CHECK: scf.yield %[[MMAD]] : tensor<29x768xf32>
    // CHECK: } else {
    // CHECK: scf.yield %[[BRC]] : tensor<29x768xf32>
    // CHECK: } {may_not_exec
    %res = scf.if %cond -> tensor<29x768xf32> {
      %14 = hivm.hir.mmadL1 ins(%9, %10, %false, %c29, %c128, %c768 : tensor<29x128xf16>, tensor<128x768xf16>, i1, index, index, index) outs(%outC : tensor<29x768xf32>) -> tensor<29x768xf32>
      scf.yield %14 : tensor<29x768xf32>
    } else {
      scf.yield %outC : tensor<29x768xf32>
    }
    %15 = tensor.empty() : tensor<29x768xf16>
    %16 = hivm.hir.vcast ins(%res : tensor<29x768xf32>) outs(%15 : tensor<29x768xf16>) round_mode = <rint> -> tensor<29x768xf16>
    hivm.hir.store ins(%16 : tensor<29x768xf16>) outs(%reinterpret_cast_2 : memref<29x768xf16, strided<[768, 1]>>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dot_used_chain_access_if() -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<16x16xf16>
    %alloc_b = memref.alloc() : memref<16x16xf16>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<16x16xf16>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<16x16xf16>
    %c = tensor.empty() : tensor<16x16xf32>
    %0 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %c) -> (tensor<16x16xf32>) : i32 {
      // CHECK: scf.for
      // CHECK-NEXT: tensor.empty
      // CHECK-NEXT: hivm.hir.mmadL1
      // CHECK-NEXT: tensor.empty
      // CHECK-NEXT: hivm.hir.vadd
      // CHECK-NEXT: arith.andi
      %mmadL1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%arg1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %and = arith.andi %arg0, %c1_i32 : i32
      %cond = arith.cmpi ne, %and, %c0_i32 : i32
      %1 = scf.if %cond -> (tensor<16x16xf32>) {
        scf.yield %mmadL1 : tensor<16x16xf32>
      } else {
        %empty = tensor.empty() : tensor<16x16xf32>
        %2 = hivm.hir.vadd ins(%mmadL1, %mmadL1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %2 : tensor<16x16xf32>
      }
      scf.yield %1 : tensor<16x16xf32>
    }
    return %0 : tensor<16x16xf32>
  }
}

// -----
// CHECK-LABEL:   func.func @dot_used_chain_access_if_with_non_const_init(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: i32) -> tensor<16x16xf32> {
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dot_used_chain_access_if_with_non_const_init(%input : i32) -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<16x16xf16>
    %alloc_b = memref.alloc() : memref<16x16xf16>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<16x16xf16>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<16x16xf16>
    %c = tensor.empty() : tensor<16x16xf32>
    %0 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %c) -> (tensor<16x16xf32>) : i32 {
    // CHECK:             %[[VAL_14:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_3]] : i32
    // CHECK:             %[[VAL_15:.*]] = tensor.empty() : tensor<16x16xf32>
    // CHECK:             %[[VAL_16:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}) outs(%[[VAL_15]] : tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK:             %[[VAL_17:.*]] = scf.if %[[VAL_14]] -> (tensor<16x16xf32>) {
    // CHECK:               scf.yield %[[VAL_16]] : tensor<16x16xf32>
    // CHECK:             } else {
    // CHECK:               %[[VAL_18:.*]] = tensor.empty() : tensor<16x16xf32>
    // CHECK:               %[[VAL_19:.*]] = hivm.hir.vadd ins(%[[VAL_16]], {{.*}} : tensor<16x16xf32>, tensor<16x16xf32>) outs(%[[VAL_18]] : tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK:               scf.yield %[[VAL_19]] : tensor<16x16xf32>
    // CHECK:             }
      %init = arith.cmpi eq, %input, %c0_i32 : i32
      %mmadL1 = hivm.hir.mmadL1 ins(%a, %b, %init, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%arg1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %and = arith.andi %arg0, %c1_i32 : i32
      %cond = arith.cmpi ne, %and, %c0_i32 : i32
      %1 = scf.if %cond -> (tensor<16x16xf32>) {
        // CHECK:               scf.yield %[[VAL_17]] : tensor<16x16xf32>
        scf.yield %mmadL1 : tensor<16x16xf32>
      } else {
        // CHECK:               %[[VAL_23:.*]] = tensor.empty() : tensor<16x16xf32>
        // CHECK:               %[[VAL_24:.*]] = hivm.hir.vadd ins(%[[VAL_17]], %[[VAL_17]] : tensor<16x16xf32>, tensor<16x16xf32>) outs(%[[VAL_23]] : tensor<16x16xf32>) -> tensor<16x16xf32>
        %empty = tensor.empty() : tensor<16x16xf32>
        %2 = hivm.hir.vadd ins(%mmadL1, %mmadL1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %2 : tensor<16x16xf32>
      }
      scf.yield %1 : tensor<16x16xf32>
    }
    return %0 : tensor<16x16xf32>
  }
}

// -----
  // CHECK-LABEL:   func.func @dot_bias_init_with_vec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dot_bias_init_with_vec() -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<64x32xf32>
    %c = hivm.hir.vmul ins(%a, %b : tensor<64x32xf32>, tensor<64x32xf32>) outs(%a : tensor<64x32xf32>) -> tensor<64x32xf32>
    %alloc_d = memref.alloc() : memref<32x32xf32>
    %d = bufferization.to_tensor %alloc_d restrict writable : memref<32x32xf32>
    // CHECK: %[[C:.*]] = hivm.hir.vmul
    // CHECK: %[[E:.*]] = tensor.empty()
    // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[E]])
    %0 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %c) -> (tensor<64x32xf32>) : i32 {
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
      %mmadL1 = hivm.hir.mmadL1 ins(%a, %d, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%arg1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmadL1 : tensor<64x32xf32>
    }
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64x32xf32>
    // CHECK: hivm.hir.vadd ins(%[[FOR]], %[[C]] : tensor<64x32xf32>, tensor<64x32xf32>) outs(%[[EMPTY]] : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
}

// -----
  // CHECK-LABEL:   func.func @dot_reuse_l0c
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dot_reuse_l0c() -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %true = arith.constant true
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %alloc_c = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %a_c = bufferization.to_tensor %alloc_c restrict writable : memref<64x32xf32>
    %c = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%a_c : tensor<64x32xf32>) -> tensor<64x32xf32>
    %alloc_d = memref.alloc() : memref<32x32xf32>
    %d = bufferization.to_tensor %alloc_d restrict writable : memref<32x32xf32>
    // CHECK: %[[C:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c
    // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[C]])
    %0 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %c) -> (tensor<64x32xf32>) : i32 {
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
      %mmadL1 = hivm.hir.mmadL1 ins(%a, %d, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%arg1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmadL1 : tensor<64x32xf32>
    }
    // CHECK-NOT: hivm.hir.vadd 
    return %0 : tensor<64x32xf32>
  }
}

// -----
  // CHECK-LABEL:   func.func @dot_reuse_for_l0c
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dot_reuse_for_l0c() -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %true = arith.constant true
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %empty_c = tensor.empty() : tensor<64x32xf32>
    %a_c = hivm.hir.vbrc ins(%cst : f32) outs(%empty_c : tensor<64x32xf32>) -> tensor<64x32xf32>
    %alloc_d = memref.alloc() : memref<32x32xf32>
    %d = bufferization.to_tensor %alloc_d restrict writable : memref<32x32xf32>
    // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[C:.*]])
    %0 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %a_c) -> (tensor<64x32xf32>) : i32 {
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
      %mmadL1 = hivm.hir.mmadL1 ins(%a, %d, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%arg1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmadL1 : tensor<64x32xf32>
    }
    // CHECK: %[[c:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
    %c = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    // CHECK-NOT: hivm.hir.vadd
    return %c : tensor<64x32xf32>
  }
}

// -----
// CHECK-LABEL:   func.func @simplicial_bwd_kv1_kernel
// CHECK-NOT: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @simplicial_bwd_kv1_kernel(%arg0: tensor<1xi16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i16 = arith.constant 0 : i16
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = scf.for %arg5 = %arg1 to %arg2 step %c1_i32 iter_args(%arg6 = %0) -> (tensor<64x64xf32>)  : i32 {
      %2 = scf.for %arg7 = %arg5 to %arg3 step %c32_i32 iter_args(%arg8 = %arg6) -> (tensor<64x64xf32>)  : i32 {
        %alloc = memref.alloc() : memref<32x64xbf16>
        %alloc_0 = memref.alloc() : memref<64x32xbf16>
        %3 = bufferization.to_tensor %alloc restrict writable : memref<32x64xbf16>
        %4 = bufferization.to_tensor %alloc_0 restrict writable : memref<64x32xbf16>
        %5 = bufferization.alloc_tensor() : tensor<i1>
        %6 = bufferization.alloc_tensor() : tensor<i16>
        %7 = tensor.empty() : tensor<1xi1>
        %8 = hivm.hir.vcmp ins(%arg0, %c0_i16 : tensor<1xi16>, i16) outs(%7 : tensor<1xi1>) compare_mode = <ne> -> tensor<1xi1>
        %extracted = tensor.extract %8[%c0] : tensor<1xi1>
        %9 = arith.extui %extracted : i1 to i32
        %10 = arith.cmpi sgt, %9, %c0_i32 : i32
        %11 = scf.if %10 -> (tensor<64x64xf32>) {
          %12 = arith.cmpi eq, %9, %arg7 : i32
          %13 = arith.cmpi eq, %arg4, %arg5 : i32
          %14 = arith.andi %12, %13 : i1
          %15 = hivm.hir.mmadL1 ins(%4, %3, %14, %c0, %c0, %c0 : tensor<64x32xbf16>, tensor<32x64xbf16>, i1, index, index, index) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
          scf.yield %15 : tensor<64x64xf32>
        } else {
          scf.yield %arg8 : tensor<64x64xf32>
        }
        scf.yield %11 : tensor<64x64xf32>
      } {tt.num_stages = 1 : i32}
      scf.yield %2 : tensor<64x64xf32>
    }
    return
  }
}

// -----
// CHECK-LABEL:   func.func @simplicial_chunk_kda_bwd_kernel_wy_dqkg_fused_kernel
// CHECK-NOT: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @simplicial_chunk_kda_bwd_kernel_wy_dqkg_fused_kernel(%arg0: tensor<1xi16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i16 = arith.constant 0 : i16
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = scf.for %arg5 = %arg1 to %arg2 step %c1_i32 iter_args(%arg6 = %0) -> (tensor<64x64xf32>)  : i32 {
      %2 = scf.for %arg7 = %arg5 to %arg3 step %c32_i32 iter_args(%arg8 = %arg6) -> (tensor<64x64xf32>)  : i32 {
        %alloc = memref.alloc() : memref<32x64xbf16>
        %alloc_0 = memref.alloc() : memref<64x32xbf16>
        %3 = bufferization.to_tensor %alloc restrict writable : memref<32x64xbf16>
        %4 = bufferization.to_tensor %alloc_0 restrict writable : memref<64x32xbf16>
        %5 = bufferization.alloc_tensor() : tensor<i1>
        %6 = bufferization.alloc_tensor() : tensor<i16>
        %7 = tensor.empty() : tensor<1xi1>
        %8 = hivm.hir.vcmp ins(%arg0, %c0_i16 : tensor<1xi16>, i16) outs(%7 : tensor<1xi1>) compare_mode = <ne> -> tensor<1xi1>
        %extracted = tensor.extract %8[%c0] : tensor<1xi1>
        %9 = arith.extui %extracted : i1 to i32
        %10 = arith.cmpi sgt, %9, %c0_i32 : i32
        %11 = scf.if %10 -> (tensor<64x64xf32>) {
          %12 = arith.cmpi eq, %9, %arg7 : i32
          %13 = arith.cmpi eq, %arg4, %arg5 : i32
          %14 = arith.andi %12, %13 : i1
          %15 = hivm.hir.mmadL1 ins(%4, %3, %14, %c0, %c0, %c0 : tensor<64x32xbf16>, tensor<32x64xbf16>, i1, index, index, index) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
          scf.yield %15 : tensor<64x64xf32>
        } else {
          scf.yield %arg8 : tensor<64x64xf32>
        }
        scf.yield %11 : tensor<64x64xf32>
      } {tt.num_stages = 1 : i32}
      %3 = tensor.empty() : tensor<64x64xf32>
      %alloc = memref.alloc() : memref<64x64xf32>
      %alloc_0 = memref.alloc() : memref<64x64xf32>
      %4 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
      %5 = bufferization.to_tensor %alloc_0 restrict writable : memref<64x64xf32>
      %6 = hivm.hir.mmadL1 ins(%4, %5, %true, %c0, %c0, %c0 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%3 : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %6 : tensor<64x64xf32>
    }
    return
  }
}

// -----
// CHECK-LABEL: func.func @matmul_operands_from_scope_result
// CHECK: scope.scope
// CHECK: hivm.hir.mmadL1
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @matmul_operands_from_scope_result() -> tensor<16x16xf32> {
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<16x16xf16>
    %alloc_b = memref.alloc() : memref<16x16xf16>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<16x16xf16>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<16x16xf16>
    %scope_a = scope.scope : () -> tensor<16x16xf16> {
      scope.return %a : tensor<16x16xf16>
    }
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = scope.scope : () -> tensor<16x16xf32> {
      scope.return %empty : tensor<16x16xf32>
    }
    %mmad = hivm.hir.mmadL1 ins(%scope_a, %b, %false, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mmad : tensor<16x16xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_matmul_with_scope_matmul_limited_in_cube
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_matmul_with_scope_matmul_limited_in_cube(%arg1: memref<16x16xf16>) {

  // CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index

  // CHECK: scope.scope : () -> () {
  scope.scope : () -> () {
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.0 : f32

    %ma = tensor.empty() : tensor<64x64xf16>
    %mb = tensor.empty() : tensor<64x64xf16>

    %empty = tensor.empty() : tensor<64x64xf32>
    %init_acc = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>

    %alloc_15 = memref.alloc() : memref<64x64xf32>

    // CHECK: %[[ALLOCA:.*]] = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.store %[[C0_I32]], %[[ALLOCA]][] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>

    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg22 = %init_acc) -> tensor<64x64xf32> {

      %187 = arith.constant 1 : i1

      %189 = scf.if %187 -> tensor<64x64xf32> {
        // CHECK: %[[LOADED:.*]] = memref.load %[[ALLOCA]][] : memref<i32>
        // CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[LOADED]], %[[C0_I32]] : i32
        // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[CMP]], %[[C64]], %[[C64]], %[[C64]]
        // CHECK: %[[INCREMENTED:.*]] = arith.addi %[[LOADED]], %[[C1_I32]] : i32
        // CHECK: memref.store %[[INCREMENTED]], %[[ALLOCA]][] : memref<i32>
        %190 = hivm.hir.mmadL1 ins(%ma, %mb, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%arg22 : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %190 : tensor<64x64xf32>
      } else {
        scf.yield %arg22 : tensor<64x64xf32>
      } {hivm.matmul_limited_in_cube, ssbuffer.if = 6 : i32}

      scf.yield %189 : tensor<64x64xf32>
    // CHECK: } {normalized_in_L0C = [0 : i32]}
    }

    bufferization.materialize_in_destination %result in writable %alloc_15 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()

    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>, hivm.matmul_limited_in_cube}

  return
}
}

// -----
// Test: with dot_pad_only_k hint, real_m should use l1M (aligned size from C)
// instead of actualM when actualM is not aligned.
// A has shape 128x128 but actualM=29 (non-aligned), C has shape 128x128 (l1M=128).
// CHECK-LABEL: func.func @test_dot_pad_only_k_l1M_for_real_m
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C128]], {{.*}}, {{.*}}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_dot_pad_only_k_l1M_for_real_m() -> tensor<128x128xf32> {
    %c128 = arith.constant 128 : index
    %c29 = arith.constant 29 : index
    %true = arith.constant true

    %allocA = memref.alloc() : memref<128x128xf16>
    %tensorA = bufferization.to_tensor %allocA restrict writable : memref<128x128xf16>
    annotation.mark %tensorA {dot_pad_only_k} : tensor<128x128xf16>

    %allocB = memref.alloc() : memref<128x128xf16>
    %tensorB = bufferization.to_tensor %allocB restrict writable : memref<128x128xf16>

    %empty = tensor.empty() : tensor<128x128xf32>
    %result = hivm.hir.mmadL1 ins(%tensorA, %tensorB, %true, %c29, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
}
}

// -----
// Test: annotation.mark {matmul_at_least_once} on scf.for result overrides
// mayNotExec to false, suppressing tail fallback and kMayNotExec attr.
// CHECK-LABEL: func.func @test_matmul_at_least_once_annotation
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK-NOT: may_not_exec
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_matmul_at_least_once_annotation(%arg0: i32, %arg1: i32) -> tensor<64x64xf32> {
  %c64 = arith.constant 64 : index
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %init_brc = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %alloc_a = memref.alloc() : memref<64x64xf16>
  %tensor_a = bufferization.to_tensor %alloc_a restrict writable : memref<64x64xf16>
  %alloc_b = memref.alloc() : memref<64x64xf16>
  %tensor_b = bufferization.to_tensor %alloc_b restrict writable : memref<64x64xf16>

  %for_res = scf.for %i = %arg0 to %arg1 step %c1_i32 iter_args(%acc = %init_brc) -> (tensor<64x64xf32>) : i32 {
    %mmad = hivm.hir.mmadL1 ins(%tensor_a, %tensor_b, %false, %c0, %c0, %c0 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %mmad : tensor<64x64xf32>
  }
  annotation.mark %for_res {matmul_at_least_once} : tensor<64x64xf32>

  return %for_res : tensor<64x64xf32>
}
}

// -----
// Test mmadL1 normalization in nested scf.for and scf.if (no counter should be generated)
// CHECK-LABEL: func.func @test_mmadl1_normalize_in_nested_ccf
// CHECK-NOT: normalize_matmul_counter
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadl1_normalize_in_nested_ccf(%arg0: i1, %arg1: i1, %arg2: i32, %arg3: memref<112x64xf32>) -> tensor<112x64xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<112x64xf32>
  %init_brc = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
  %alloc_a = memref.alloc() : memref<112x1xf32>
  %tensor_a = bufferization.to_tensor %alloc_a restrict writable : memref<112x1xf32>
  %alloc_b = memref.alloc() : memref<1x64xf32>
  %tensor_b = bufferization.to_tensor %alloc_b restrict writable : memref<1x64xf32>
  %for_res:2 = scf.for %i = %c0_i32 to %arg2 step %c1_i32 
      iter_args(%acc = %init_brc, %sum = %init_brc) -> (tensor<112x64xf32>, tensor<112x64xf32>) : i32 {
    %if_res:2 = scf.if %arg0 -> (tensor<112x64xf32>, tensor<112x64xf32>) {
      %mmad1 = hivm.hir.mmadL1 ins(%tensor_a, %tensor_b, %false, %c0, %c0, %c0 : tensor<112x1xf32>, tensor<1x64xf32>, i1, index, index, index) 
          outs(%init_brc : tensor<112x64xf32>) -> tensor<112x64xf32>
      %abs = hivm.hir.vabs ins(%mmad1 : tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
      %add1 = hivm.hir.vadd ins(%sum, %abs : tensor<112x64xf32>, tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
      scf.yield %mmad1, %add1 : tensor<112x64xf32>, tensor<112x64xf32>
    } else {
      %inner_if:2 = scf.if %arg1 -> (tensor<112x64xf32>, tensor<112x64xf32>) {
        %bias_tensor = bufferization.to_tensor %arg3 restrict writable : memref<112x64xf32>
        %mmad2 = hivm.hir.mmadL1 ins(%tensor_a, %tensor_b, %false, %c0, %c0, %c0 : tensor<112x1xf32>, tensor<1x64xf32>, i1, index, index, index) 
            outs(%acc : tensor<112x64xf32>) -> tensor<112x64xf32>
        %add2 = hivm.hir.vadd ins(%mmad2, %bias_tensor : tensor<112x64xf32>, tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
        scf.yield %mmad2, %add2 : tensor<112x64xf32>, tensor<112x64xf32>
      } else {
        scf.yield %acc, %init_brc : tensor<112x64xf32>, tensor<112x64xf32>
      }
      %add3 = hivm.hir.vadd ins(%sum, %inner_if#1 : tensor<112x64xf32>, tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
      scf.yield %inner_if#0, %add3 : tensor<112x64xf32>, tensor<112x64xf32>
    }
    scf.yield %if_res#0, %if_res#1 : tensor<112x64xf32>, tensor<112x64xf32>
  }
  
  return %for_res#1 : tensor<112x64xf32>
}
}

// -----
// CHECK-LABEL: func.func @test_mmadmx_elemwise_bias_decompose(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mmadmx_elemwise_bias_decompose(%bias: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %a = tensor.empty() : tensor<4x8xf8E5M2>
    %b = tensor.empty() : tensor<8x16xf8E5M2>
    %scaleA = tensor.empty() : tensor<1xui8>
    %scaleB = tensor.empty() : tensor<1xui8>
    // CHECK: tensor.empty
    // CHECK: hivm.hir.mmadmxL1
    // CHECK: hivm.hir.vadd
    %mad = hivm.hir.mmadmxL1 ins(%a, %b, %scaleA, %scaleB, %false, %c4, %c8, %c16 : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>, tensor<1xui8>, tensor<1xui8>, i1, index, index, index) outs(%bias : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %mad : tensor<4x16xf32>
  }
}

// Test counterPrevious chain across mixed CCF types:
// CCF1: for+if, variable bounds (mayNotExec=true) → NoBias, addTailFallback, counterPrevious=true
// CCF2: for, variable bounds (mayNotExec=true) → ReuseL0C from CCF1's IfOp, counterPrevious=runtime → andi
// CCF3: bare mmad (no for) → ReuseL0C from CCF2's IfOp, no counter/addTailFallback
// CCF4: for, constant bounds (mayNotExec=false) → ReuseL0C from CCF3's MmadL1Op, counterPrevious=false (folded)

// CHECK-LABEL: func.func @test_counter_previous_mixed_ccf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_counter_previous_mixed_ccf(%lb: i32, %ub: i32) -> tensor<64x32xf32> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %empty = tensor.empty() : tensor<64x32xf32>

    // CCF1: for+if with variable bounds → addTailFallback, counterPrevious=true
    // CHECK: %[[ALLOCA1:.*]] = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xf32>)
    // CCF1 mmad inside if: initCondition = firstIter (counter == 0)
    // CHECK: %[[FIRST1:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 : i32
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[FIRST1]], %c64, %c32, %c32
    // counterPrevious1: counterPrevIn=true → m_One → CmpIOp{counter_previous}
    //   compare: load(counter1) == 0  (did CCF1's mmad ever execute?)
    // CHECK: %[[CNT1_ZERO:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 {counter_previous} : i32
    %for1 = scf.for %i1 = %lb to %ub step %c1_i32 iter_args(%acc1 = %empty) -> (tensor<64x32xf32>) : i32 {
      %cond = arith.cmpi eq, %i1, %c0_i32 : i32
      %if1 = scf.if %cond -> (tensor<64x32xf32>) {
        %mmad1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc1 : tensor<64x32xf32>) -> tensor<64x32xf32>
        scf.yield %mmad1 : tensor<64x32xf32>
      } else {
        scf.yield %acc1 : tensor<64x32xf32>
      }
      scf.yield %if1 : tensor<64x32xf32>
    }

    // CCF2: for with variable bounds → ReuseL0C, no IfOp (init not replaced)
    // CHECK: %[[ALLOCA2:.*]] = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CCF2 mmad inside for: initCondition = andi(counterPrevious1, firstIter2), remain_in_l0c
    // CHECK: %[[INIT2:.*]] = arith.andi %[[CNT1_ZERO]], {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT2]], %c64, %c32, %c32
    // counterPrevious2: counterPrevIn=counterPrevious1(runtime) → andi(counter1==0, counter2==0)
    //   compare: andi(counter1==0, counter2==0)  (did CCF1 and CCF2 both never execute?)
    // CHECK: %[[CNT_PREV2:.*]] = arith.andi %[[CNT1_ZERO]], {{.*}} {counter_previous} : i1
    // CHECK-NOT: scf.if
    %for2 = scf.for %i2 = %lb to %ub step %c1_i32 iter_args(%acc2 = %for1) -> (tensor<64x32xf32>) : i32 {
      %mmad2 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc2 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad2 : tensor<64x32xf32>
    }

    // CCF3: bare mmad (no for) → ReuseL0C from CCF2's IfOp, no counter/addTailFallback
    // CCF3 mmad: initCondition = counterPrevious2 (AndIOp{counter_previous}), remain_in_l0c
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[CNT_PREV2]], %c64, %c32, %c32
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    %mmad3 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%for2 : tensor<64x32xf32>) -> tensor<64x32xf32>

    // CCF4: for with constant bounds (mayNotExec=false) → ReuseL0C, counterPrevious=false (folded)
    // CCF4 mmad: initCondition = false (always accumulate), remain_in_l0c, no may_not_exec
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %false, %c64, %c32, %c32
    // CHECK-NOT: may_not_exec
    // CHECK-NOT: scf.if {{.*}} -> (tensor<64x32xf32>)
    // CHECK: return
    %for4 = scf.for %i4 = %c0 to %c8 step %c1 iter_args(%acc4 = %mmad3) -> (tensor<64x32xf32>) {
      %mmad4 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc4 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad4 : tensor<64x32xf32>
    }

    return %for4 : tensor<64x32xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_mmadmx_PerChannelAdd(
// CHECK-SAME:                                      %[[BIAS:.*]]: tensor<1x16xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_PerChannelAdd(%bias: tensor<1x16xf32>) -> tensor<4x16xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %a = tensor.empty() : tensor<4x8xf8E5M2>
  %b = tensor.empty() : tensor<8x16xf8E5M2>
  %scaleA = tensor.empty() : tensor<1xui8>
  %scaleB = tensor.empty() : tensor<1xui8>
  %brc_out = tensor.empty() : tensor<4x16xf32>
  %brc = hivm.hir.vbrc ins(%bias : tensor<1x16xf32>) outs(%brc_out : tensor<4x16xf32>) broadcast_dims = [0] -> tensor<4x16xf32>
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[OUT:.*]] = tensor.empty() : tensor<4x16xf32>
  // CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn, normalized_in_L0C, normalized_init_or_bias}
  // CHECK-SAME: ins(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[TRUE]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[BIAS]] :{{.*}}tensor<1x16xf32>)
  // CHECK-SAME: outs(%[[OUT]] : tensor<4x16xf32>)
  // CHECK-NOT: hivm.hir.vbrc
  %mad = hivm.hir.mmadmxL1 ins(%a, %b, %scaleA, %scaleB, %false, %c4, %c8, %c16 : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>, tensor<1xui8>, tensor<1xui8>, i1, index, index, index) outs(%brc : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %mad : tensor<4x16xf32>
}
}

// -----
// CHECK-LABEL: func.func @test_mmadmx_normalize_decompose_matmul(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<16x16xf32>) -> tensor<16x16xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_3:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<16x16xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<16x16xf8E4M3FN>
// CHECK:           %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] restrict writable : memref<16x16xf8E4M3FN>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<16x16xf8E4M3FN>
// CHECK:           %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<16x16xf8E4M3FN>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.load ins(%[[VAL_3]] : tensor<16x16xf32>) outs(%[[VAL_8]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           hivm.hir.mmadmxL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK:           hivm.hir.vadd
// CHECK:           return
// CHECK:         }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_normalize_decompose_matmul(%arg0: memref<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf8E4M3FN>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf8E4M3FN>
    %alloc_0 = memref.alloc() : memref<16x16xf8E4M3FN>
    %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x16xf8E4M3FN>
    %alloc_scaleA = memref.alloc() : memref<1xi8>
    %scaleA = bufferization.to_tensor %alloc_scaleA restrict writable : memref<1xi8>
    %alloc_scaleB = memref.alloc() : memref<1xi8>
    %scaleB = bufferization.to_tensor %alloc_scaleB restrict writable : memref<1xi8>
    %false = arith.constant false
    %3 = tensor.empty() : tensor<16x16xf32>
    %c0 = arith.constant 0 : index
    %5 = hivm.hir.load ins(%0 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = hivm.hir.mmadmxL1 ins(%1, %2, %scaleA, %scaleB, %false, %c0, %c0, %c0 : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %4 : tensor<16x16xf32>
}
}

// -----
// CHECK-LABEL: func.func @test_mmadmx_normalize_in_nested_ccf
// CHECK-NOT: normalize_matmul_counter
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_normalize_in_nested_ccf(%arg0: i1, %arg1: i1, %arg2: i32, %arg3: memref<112x64xf32>) -> tensor<112x64xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<112x64xf32>
  %init_brc = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
  %alloc_a = memref.alloc() : memref<112x1xf8E4M3FN>
  %tensor_a = bufferization.to_tensor %alloc_a restrict writable : memref<112x1xf8E4M3FN>
  %alloc_b = memref.alloc() : memref<1x64xf8E4M3FN>
  %tensor_b = bufferization.to_tensor %alloc_b restrict writable : memref<1x64xf8E4M3FN>
  %alloc_scaleA = memref.alloc() : memref<1xi8>
  %scaleA = bufferization.to_tensor %alloc_scaleA restrict writable : memref<1xi8>
  %alloc_scaleB = memref.alloc() : memref<1xi8>
  %scaleB = bufferization.to_tensor %alloc_scaleB restrict writable : memref<1xi8>
  %for_res:2 = scf.for %i = %c0_i32 to %arg2 step %c1_i32 
      iter_args(%acc = %init_brc, %sum = %init_brc) -> (tensor<112x64xf32>, tensor<112x64xf32>) : i32 {
    %if_res:2 = scf.if %arg0 -> (tensor<112x64xf32>, tensor<112x64xf32>) {
      %mmad1 = hivm.hir.mmadmxL1 ins(%tensor_a, %tensor_b, %scaleA, %scaleB, %false, %c0, %c0, %c0 : tensor<112x1xf8E4M3FN>, tensor<1x64xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) 
          outs(%init_brc : tensor<112x64xf32>) -> tensor<112x64xf32>
      %abs = hivm.hir.vabs ins(%mmad1 : tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
      %add1 = hivm.hir.vadd ins(%sum, %abs : tensor<112x64xf32>, tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
      scf.yield %mmad1, %add1 : tensor<112x64xf32>, tensor<112x64xf32>
    } else {
      %inner_if:2 = scf.if %arg1 -> (tensor<112x64xf32>, tensor<112x64xf32>) {
        %bias_tensor = bufferization.to_tensor %arg3 restrict writable : memref<112x64xf32>
        %mmad2 = hivm.hir.mmadmxL1 ins(%tensor_a, %tensor_b, %scaleA, %scaleB, %false, %c0, %c0, %c0 : tensor<112x1xf8E4M3FN>, tensor<1x64xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) 
            outs(%acc : tensor<112x64xf32>) -> tensor<112x64xf32>
        %add2 = hivm.hir.vadd ins(%mmad2, %bias_tensor : tensor<112x64xf32>, tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
        scf.yield %mmad2, %add2 : tensor<112x64xf32>, tensor<112x64xf32>
      } else {
        scf.yield %acc, %init_brc : tensor<112x64xf32>, tensor<112x64xf32>
      }
      %add3 = hivm.hir.vadd ins(%sum, %inner_if#1 : tensor<112x64xf32>, tensor<112x64xf32>) outs(%empty : tensor<112x64xf32>) -> tensor<112x64xf32>
      scf.yield %inner_if#0, %add3 : tensor<112x64xf32>, tensor<112x64xf32>
    }
    scf.yield %if_res#0, %if_res#1 : tensor<112x64xf32>, tensor<112x64xf32>
  }
  
  return %for_res#1 : tensor<112x64xf32>
}
}

// -----
// CHECK-LABEL:   func.func @dotscale_reuse_l0c
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dotscale_reuse_l0c() -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %true = arith.constant true
    %alloc_a = memref.alloc() : memref<64x32xf8E4M3FN>
    %alloc_b = memref.alloc() : memref<32x32xf8E4M3FN>
    %alloc_c = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf8E4M3FN>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf8E4M3FN>
    %a_c = bufferization.to_tensor %alloc_c restrict writable : memref<64x32xf32>
    %alloc_scaleA = memref.alloc() : memref<1xi8>
    %scaleA = bufferization.to_tensor %alloc_scaleA restrict writable : memref<1xi8>
    %alloc_scaleB = memref.alloc() : memref<1xi8>
    %scaleB = bufferization.to_tensor %alloc_scaleB restrict writable : memref<1xi8>
    %c = hivm.hir.mmadmxL1 ins(%a, %b, %scaleA, %scaleB, %true, %c16, %c16, %c16 : tensor<64x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%a_c : tensor<64x32xf32>) -> tensor<64x32xf32>
    %alloc_d = memref.alloc() : memref<32x32xf8E4M3FN>
    %d = bufferization.to_tensor %alloc_d restrict writable : memref<32x32xf8E4M3FN>
    // CHECK: %[[C:.*]] = hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c
    // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[C]])
    %0 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %c) -> (tensor<64x32xf32>) : i32 {
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
      %mmadMx = hivm.hir.mmadmxL1 ins(%a, %d, %scaleA, %scaleB, %false, %c16, %c16, %c16 : tensor<64x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%arg1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmadMx : tensor<64x32xf32>
    }
    // CHECK-NOT: hivm.hir.vadd 
    return %0 : tensor<64x32xf32>
  }
}

// -----
// CHECK-LABEL:   func.func @dotscale_reuse_for_l0c
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dotscale_reuse_for_l0c() -> tensor<64x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<64x32xf8E4M3FN>
    %alloc_b = memref.alloc() : memref<32x32xf8E4M3FN>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf8E4M3FN>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf8E4M3FN>
    %cst = arith.constant 0.000000e+00 : f32
    %empty_c = tensor.empty() : tensor<64x32xf32>
    %a_c = hivm.hir.vbrc ins(%cst : f32) outs(%empty_c : tensor<64x32xf32>) -> tensor<64x32xf32>
    %alloc_d = memref.alloc() : memref<32x32xf8E4M3FN>
    %d = bufferization.to_tensor %alloc_d restrict writable : memref<32x32xf8E4M3FN>
    %alloc_scaleA = memref.alloc() : memref<1xi8>
    %scaleA = bufferization.to_tensor %alloc_scaleA restrict writable : memref<1xi8>
    %alloc_scaleB = memref.alloc() : memref<1xi8>
    %scaleB = bufferization.to_tensor %alloc_scaleB restrict writable : memref<1xi8>
    // CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[C:.*]])
    %0 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %a_c) -> (tensor<64x32xf32>) : i32 {
      // CHECK: %[[MMAD:.*]] = hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
      %mmadMx = hivm.hir.mmadmxL1 ins(%a, %d, %scaleA, %scaleB, %false, %c16, %c16, %c16 : tensor<64x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%arg1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmadMx : tensor<64x32xf32>
    }
    // CHECK: %[[c:.*]] = hivm.hir.mmadmxL1 {already_set_real_mkn, normalized_in_L0C}
    %c = hivm.hir.mmadmxL1 ins(%a, %b, %scaleA, %scaleB, %false, %c16, %c16, %c16 : tensor<64x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
    // CHECK-NOT: hivm.hir.vadd
    return %c : tensor<64x32xf32>
  }
}

// Test: when counterPrevious is compile-time false (previous CCF definitely
// executed), ReuseL0C skips addTailFallback and kMayNotExec.
// CCF1: bare mmadL1 (definitely executes) → result in L0C
// CCF2: for with variable bounds, ReuseL0C from CCF1 → counterPrevious=false
//        → no addTailFallback, no kMayNotExec, initCondition=false (always accumulate)

// CHECK-LABEL: func.func @test_reuse_l0c_skip_tail_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_reuse_l0c_skip_tail_fallback(%lb: i32, %ub: i32) -> tensor<64x32xf32> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %empty = tensor.empty() : tensor<64x32xf32>

    // CCF1: bare mmadL1 (definitely executes) → kNormalizedInL0C
    // CHECK: hivm.hir.mmadL1 {{.*}} normalized_in_L0C
    %mmad1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>

    // CCF2: for with variable bounds, ReuseL0C from CCF1
    // counterPrevious = false (from MmadL1Op) → skip addTailFallback
    // CHECK: scf.for
    // CHECK-NOT: may_not_exec
    // CHECK-NOT: scf.if {{.*}} -> (tensor<64x32xf32>, i1)
    // CHECK: return
    %for2 = scf.for %i = %lb to %ub step %c1_i32 iter_args(%acc = %mmad1) -> (tensor<64x32xf32>) : i32 {
      %mmad2 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad2 : tensor<64x32xf32>
    }

    return %for2 : tensor<64x32xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_mmadmx_with_scope_matmul_limited_in_cube
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_with_scope_matmul_limited_in_cube() {

  // CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index

  // CHECK: scope.scope : () -> () {
  scope.scope : () -> () {
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.0 : f32

    %ma = tensor.empty() : tensor<64x64xf8E4M3FN>
    %mb = tensor.empty() : tensor<64x64xf8E4M3FN>
    %alloc_scaleA = memref.alloc() : memref<1xi8>
    %scaleA = bufferization.to_tensor %alloc_scaleA restrict writable : memref<1xi8>
    %alloc_scaleB = memref.alloc() : memref<1xi8>
    %scaleB = bufferization.to_tensor %alloc_scaleB restrict writable : memref<1xi8>

    %empty = tensor.empty() : tensor<64x64xf32>
    %init_acc = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>

    %alloc_15 = memref.alloc() : memref<64x64xf32>

    // CHECK: %[[ALLOCA:.*]] = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.store %[[C0_I32]], %[[ALLOCA]][] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>

    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg22 = %init_acc) -> tensor<64x64xf32> {

      %187 = arith.constant 1 : i1

      %189 = scf.if %187 -> tensor<64x64xf32> {
        // CHECK: %[[LOADED:.*]] = memref.load %[[ALLOCA]][] : memref<i32>
        // CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[LOADED]], %[[C0_I32]] : i32
        // CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn, normalized_in_L0C} ins({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[CMP]], %[[C64]], %[[C64]], %[[C64]]
        // CHECK: %[[INCREMENTED:.*]] = arith.addi %[[LOADED]], %[[C1_I32]] : i32
        // CHECK: memref.store %[[INCREMENTED]], %[[ALLOCA]][] : memref<i32>
        %190 = hivm.hir.mmadmxL1 ins(%ma, %mb, %scaleA, %scaleB, %false, %c0, %c0, %c0 : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%arg22 : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %190 : tensor<64x64xf32>
      } else {
        scf.yield %arg22 : tensor<64x64xf32>
      } {hivm.matmul_limited_in_cube, ssbuffer.if = 6 : i32}

      scf.yield %189 : tensor<64x64xf32>
    // CHECK: } {normalized_in_L0C = [0 : i32]}
    }

    bufferization.materialize_in_destination %result in writable %alloc_15 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()

    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>, hivm.matmul_limited_in_cube}

  return
}
}

// -----
// A5: batchMmadL1 real MKN extraction from static memref shapes.
// CHECK-LABEL: func.func @test_batchMmadL1_Normalize_Mkn
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C16]], %[[C16]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_Normalize_Mkn(%arg0: memref<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x16x16xf32>
  %alloc = memref.alloc() : memref<2x16x16xf16>
  %1 = bufferization.to_tensor %alloc restrict writable : memref<2x16x16xf16>
  %alloc_0 = memref.alloc() : memref<2x16x16xf16>
  %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<2x16x16xf16>
  %true = arith.constant true
  %3 = tensor.empty() : tensor<2x16x16xf32>
  %c0 = arith.constant 0 : index
  %4 = hivm.hir.batchMmadL1 ins(%1, %2, %true, %c0, %c0, %c0, %0 : tensor<2x16x16xf16>, tensor<2x16x16xf16>, i1, index, index, index, tensor<2x16x16xf32>) outs(%3 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  return %4 : tensor<2x16x16xf32>
}
}

// -----
// A5: batchMmadL1 elementwise bias decompose into mmad + vadd.
// CHECK-LABEL: func.func @test_batchMmadL1_decompose_elemwise_bias
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_decompose_elemwise_bias(%arg0: memref<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x16x16xf32>
  %alloc = memref.alloc() : memref<2x16x16xf16>
  %1 = bufferization.to_tensor %alloc restrict writable : memref<2x16x16xf16>
  %alloc_0 = memref.alloc() : memref<2x16x16xf16>
  %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<2x16x16xf16>
  %false = arith.constant false
  %3 = tensor.empty() : tensor<2x16x16xf32>
  %c0 = arith.constant 0 : index
  %5 = hivm.hir.load ins(%0 : tensor<2x16x16xf32>) outs(%3 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  %4 = hivm.hir.batchMmadL1 ins(%1, %2, %false, %c0, %c0, %c0 : tensor<2x16x16xf16>, tensor<2x16x16xf16>, i1, index, index, index) outs(%5 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  return %4 : tensor<2x16x16xf32>
}
}

// -----
// A5: a_transpose real MKN is (A[1], A[0], B[1]) = (16, 64, 32).
// CHECK-LABEL: func.func @test_mmadL1_a_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK: hivm.hir.mmadL1 {a_transpose, already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_a_transpose_Normalize_Mkn() -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %alloc_a = memref.alloc() : memref<64x16xf16>
  %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x16xf16>
  %alloc_b = memref.alloc() : memref<64x32xf16>
  %b = bufferization.to_tensor %alloc_b restrict writable : memref<64x32xf16>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadL1 {a_transpose} ins(%a, %b, %true, %c0, %c0, %c0 : tensor<64x16xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}

// -----
// A5: b_transpose real MKN is (A[0], A[1], B[0]) = (16, 64, 32).
// CHECK-LABEL: func.func @test_mmadL1_b_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, b_transpose} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_b_transpose_Normalize_Mkn() -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %alloc_a = memref.alloc() : memref<16x64xf16>
  %a = bufferization.to_tensor %alloc_a restrict writable : memref<16x64xf16>
  %alloc_b = memref.alloc() : memref<32x64xf16>
  %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x64xf16>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadL1 {b_transpose} ins(%a, %b, %true, %c0, %c0, %c0 : tensor<16x64xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}

// -----
// A5: a_transpose + b_transpose real MKN is (A[1], A[0], B[0]) = (16, 64, 32).
// CHECK-LABEL: func.func @test_mmadL1_ab_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK: hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, b_transpose} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_ab_transpose_Normalize_Mkn() -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %alloc_a = memref.alloc() : memref<64x16xf16>
  %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x16xf16>
  %alloc_b = memref.alloc() : memref<32x64xf16>
  %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x64xf16>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadL1 {a_transpose, b_transpose} ins(%a, %b, %true, %c0, %c0, %c0 : tensor<64x16xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}

// -----
// A5: already_set_real_mkn must not overwrite pre-set MKN (c8 stays as K).
// CHECK-LABEL: func.func @test_mmadL1_already_set_real_mkn_noop
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C8]], %[[C16]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_already_set_real_mkn_noop() -> tensor<16x16xf32> {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %alloc_a = memref.alloc() : memref<16x16xf16>
  %a = bufferization.to_tensor %alloc_a restrict writable : memref<16x16xf16>
  %alloc_b = memref.alloc() : memref<16x16xf16>
  %b = bufferization.to_tensor %alloc_b restrict writable : memref<16x16xf16>
  %empty = tensor.empty() : tensor<16x16xf32>
  %r = hivm.hir.mmadL1 {already_set_real_mkn} ins(%a, %b, %true, %c16, %c8, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}
}

// -----
// A5: batchMmadL1 split-K CCF keeps L0C with counter-based init.
// CHECK-LABEL: func.func @test_batchMmadL1_splitk_reuse_l0c
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {normalized_in_L0C = [0 : i32]}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_splitk_reuse_l0c() -> tensor<2x16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  %alloc_a = memref.alloc() : memref<2x16x32xf16>
  %a = bufferization.to_tensor %alloc_a restrict writable : memref<2x16x32xf16>
  %alloc_b = memref.alloc() : memref<2x32x16xf16>
  %b = bufferization.to_tensor %alloc_b restrict writable : memref<2x32x16xf16>
  %0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<2x16x16xf32>) : i32 {
    %mmad = hivm.hir.batchMmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<2x16x32xf16>, tensor<2x32x16xf16>, i1, index, index, index) outs(%acc : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    scf.yield %mmad : tensor<2x16x16xf32>
  }
  return %0 : tensor<2x16x16xf32>
}
}

// -----
// A5: bf16 mmadL1 real MKN extraction.
// CHECK-LABEL: func.func @test_mmadL1_bf16_Normalize_Mkn
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C32]], %[[C128]], %[[C64]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_bf16_Normalize_Mkn() -> tensor<32x64xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %alloc_a = memref.alloc() : memref<32x128xbf16>
  %a = bufferization.to_tensor %alloc_a restrict writable : memref<32x128xbf16>
  %alloc_b = memref.alloc() : memref<128x64xbf16>
  %b = bufferization.to_tensor %alloc_b restrict writable : memref<128x64xbf16>
  %empty = tensor.empty() : tensor<32x64xf32>
  %r = hivm.hir.mmadL1 ins(%a, %b, %true, %c0, %c0, %c0 : tensor<32x128xbf16>, tensor<128x64xbf16>, i1, index, index, index) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}
}

// -----
// A5: mmadmx a_transpose real MKN extraction.
// CHECK-LABEL: func.func @test_mmadmx_a_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK: hivm.hir.mmadmxL1 {{.*}}a_transpose{{.*}} ins({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_a_transpose_Normalize_Mkn() -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<64x16xf8E4M3FN>
  %b = tensor.empty() : tensor<64x32xf8E4M3FN>
  %scaleA = tensor.empty() : tensor<1xi8>
  %scaleB = tensor.empty() : tensor<1xi8>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadmxL1 {a_transpose} ins(%a, %b, %scaleA, %scaleB, %true, %c0, %c0, %c0 : tensor<64x16xf8E4M3FN>, tensor<64x32xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}

// -----
// A5: mmadmx b_transpose real MKN extraction.
// CHECK-LABEL: func.func @test_mmadmx_b_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: hivm.hir.mmadmxL1 {{.*}}b_transpose{{.*}} ins({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_b_transpose_Normalize_Mkn() -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<16x64xf8E4M3FN>
  %b = tensor.empty() : tensor<32x64xf8E4M3FN>
  %scaleA = tensor.empty() : tensor<1xi8>
  %scaleB = tensor.empty() : tensor<1xi8>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadmxL1 {b_transpose} ins(%a, %b, %scaleA, %scaleB, %true, %c0, %c0, %c0 : tensor<16x64xf8E4M3FN>, tensor<32x64xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}

// -----
// A5: pure tensor operands (no memref.alloc) still yield static real MKN.
// CHECK-LABEL: func.func @test_mmadL1_tensor_operands_Normalize_Mkn
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C128]], %[[C64]], %[[C256]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_tensor_operands_Normalize_Mkn() -> tensor<128x256xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<128x64xf16>
  %b = tensor.empty() : tensor<64x256xf16>
  %empty = tensor.empty() : tensor<128x256xf32>
  %r = hivm.hir.mmadL1 ins(%a, %b, %true, %c0, %c0, %c0 : tensor<128x64xf16>, tensor<64x256xf16>, i1, index, index, index) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}
}

// -----
// A5: dynamic for-bounds => may_not_exec, and no zero-fill guard after the loop.
// CHECK-LABEL: func.func @test_mmadL1_may_not_exec_dynamic_bounds
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {may_not_exec, normalized_in_L0C = [0 : i32]}
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_may_not_exec_dynamic_bounds(%lb: i32, %ub: i32) -> tensor<16x16xf32> {
  %c1 = arith.constant 1 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf16>
  %b = tensor.empty() : tensor<16x16xf16>
  %0 = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: hfusion.disableHfusionVectorize skips CCF; falls back to mmad+vadd.
// CHECK-LABEL: func.func @test_mmadL1_skip_ccf_when_disable_hfusion_vectorize
// CHECK-NOT: normalize_matmul_counter
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn}
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
func.func @test_mmadL1_skip_ccf_when_disable_hfusion_vectorize() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf16>
  %b = tensor.empty() : tensor<16x16xf16>
  %0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: plain scope (no matmul_limited_in_cube) skips CCF; mmad+vadd.
// CHECK-LABEL: func.func @test_mmadL1_skip_ccf_in_plain_scope
// CHECK-NOT: normalize_matmul_counter
// CHECK: scope.scope
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn}
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_skip_ccf_in_plain_scope() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %res = scope.scope : () -> tensor<16x16xf32> {
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %a = tensor.empty() : tensor<16x16xf16>
    %b = tensor.empty() : tensor<16x16xf16>
    %0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mmad : tensor<16x16xf32>
    }
    scope.return %0 : tensor<16x16xf32>
  }
  return %res : tensor<16x16xf32>
}
}

// -----
// A5: i8->i32 mmadL1 real MKN.
// CHECK-LABEL: func.func @test_mmadL1_i8_Normalize_Mkn
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C32]], %[[C128]], %[[C64]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_i8_Normalize_Mkn() -> tensor<32x64xi32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<32x128xi8>
  %b = tensor.empty() : tensor<128x64xi8>
  %empty = tensor.empty() : tensor<32x64xi32>
  %r = hivm.hir.mmadL1 ins(%a, %b, %true, %c0, %c0, %c0 : tensor<32x128xi8>, tensor<128x64xi8>, i1, index, index, index) outs(%empty : tensor<32x64xi32>) -> tensor<32x64xi32>
  return %r : tensor<32x64xi32>
}
}

// -----
// A5: f32 cube mmadL1 real MKN.
// CHECK-LABEL: func.func @test_mmadL1_f32_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C32]], %[[C16]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_f32_Normalize_Mkn() -> tensor<16x16xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<16x32xf32>
  %b = tensor.empty() : tensor<32x16xf32>
  %empty = tensor.empty() : tensor<16x16xf32>
  %r = hivm.hir.mmadL1 ins(%a, %b, %true, %c0, %c0, %c0 : tensor<16x32xf32>, tensor<32x16xf32>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}
}

// -----
// A5: batch a_transpose real MKN = (16, 64, 32).
// CHECK-LABEL: func.func @test_batchMmadL1_a_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK: hivm.hir.batchMmadL1 {a_transpose, already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_a_transpose_Normalize_Mkn() -> tensor<2x16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<2x64x16xf16>
  %b = tensor.empty() : tensor<2x64x32xf16>
  %empty = tensor.empty() : tensor<2x16x32xf32>
  %r = hivm.hir.batchMmadL1 {a_transpose} ins(%a, %b, %true, %c0, %c0, %c0 : tensor<2x64x16xf16>, tensor<2x64x32xf16>, i1, index, index, index) outs(%empty : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %r : tensor<2x16x32xf32>
}
}

// -----
// A5: batch b_transpose real MKN = (16, 64, 32).
// CHECK-LABEL: func.func @test_batchMmadL1_b_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn, b_transpose} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_b_transpose_Normalize_Mkn() -> tensor<2x16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<2x16x64xf16>
  %b = tensor.empty() : tensor<2x32x64xf16>
  %empty = tensor.empty() : tensor<2x16x32xf32>
  %r = hivm.hir.batchMmadL1 {b_transpose} ins(%a, %b, %true, %c0, %c0, %c0 : tensor<2x16x64xf16>, tensor<2x32x64xf16>, i1, index, index, index) outs(%empty : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %r : tensor<2x16x32xf32>
}
}

// -----
// A5: mmadmx already_set_real_mkn keeps pre-set K=8.
// CHECK-LABEL: func.func @test_mmadmx_already_set_real_mkn_noop
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C8]], %[[C16]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_already_set_real_mkn_noop() -> tensor<16x16xf32> {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %a = tensor.empty() : tensor<16x16xf8E4M3FN>
  %b = tensor.empty() : tensor<16x16xf8E4M3FN>
  %sa = tensor.empty() : tensor<1xi8>
  %sb = tensor.empty() : tensor<1xi8>
  %empty = tensor.empty() : tensor<16x16xf32>
  %r = hivm.hir.mmadmxL1 {already_set_real_mkn} ins(%a, %b, %sa, %sb, %true, %c16, %c8, %c16 : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}
}

// -----
// A5: empty init (NoBias) split-K still gets counter + remain_in_l0c.
// CHECK-LABEL: func.func @test_mmadL1_empty_init_splitk
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {normalized_in_L0C = [0 : i32]}
// CHECK-NOT: may_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_empty_init_splitk() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf16>
  %b = tensor.empty() : tensor<16x16xf16>
  %0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %empty) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: mmadmx dynamic bounds => may_not_exec, and no zero-fill guard after the loop.
// CHECK-LABEL: func.func @test_mmadmx_may_not_exec_dynamic_bounds
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {may_not_exec, normalized_in_L0C = [0 : i32]}
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_may_not_exec_dynamic_bounds(%lb: i32, %ub: i32) -> tensor<16x16xf32> {
  %c1 = arith.constant 1 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf8E4M3FN>
  %b = tensor.empty() : tensor<16x16xf8E4M3FN>
  %sa = tensor.empty() : tensor<1xi8>
  %sb = tensor.empty() : tensor<1xi8>
  %0 = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadmxL1 ins(%a, %b, %sa, %sb, %false, %c0, %c0, %c0 : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: mmadmx matmul_at_least_once suppresses may_not_exec.
// CHECK-LABEL: func.func @test_mmadmx_at_least_once_annotation
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK-NOT: may_not_exec
// CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_at_least_once_annotation(%lb: i32, %ub: i32) -> tensor<16x16xf32> {
  %c1 = arith.constant 1 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf8E4M3FN>
  %b = tensor.empty() : tensor<16x16xf8E4M3FN>
  %sa = tensor.empty() : tensor<1xi8>
  %sb = tensor.empty() : tensor<1xi8>
  %0 = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadmxL1 ins(%a, %b, %sa, %sb, %false, %c0, %c0, %c0 : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  annotation.mark %0 {matmul_at_least_once} : tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: dynamic memref.alloc sizes become real MKN SSA values.
// CHECK-LABEL: func.func @test_mmadL1_dynamic_alloc_Normalize_Mkn
// CHECK-SAME: %[[M:.*]]: index, %[[K:.*]]: index, %[[N:.*]]: index
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, %[[M]], %[[K]], %[[N]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_dynamic_alloc_Normalize_Mkn(%m: index, %k: index, %n: index) -> tensor<?x?xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %alloc_a = memref.alloc(%m, %k) : memref<?x?xf16>
  %a = bufferization.to_tensor %alloc_a restrict writable : memref<?x?xf16>
  %alloc_b = memref.alloc(%k, %n) : memref<?x?xf16>
  %b = bufferization.to_tensor %alloc_b restrict writable : memref<?x?xf16>
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %r = hivm.hir.mmadL1 ins(%a, %b, %true, %c0, %c0, %c0 : tensor<?x?xf16>, tensor<?x?xf16>, i1, index, index, index) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
}

// -----
// A5: batch pre-mmad + for currently cannot ReuseL0C (no Batch in couldReuse); falls to vadd.
// CHECK-LABEL: func.func @test_batchMmadL1_reuse_l0c_pre
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn}
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_reuse_l0c_pre() -> tensor<2x16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %false = arith.constant false
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<2x16x32xf16>
  %b = tensor.empty() : tensor<2x32x16xf16>
  %empty = tensor.empty() : tensor<2x16x16xf32>
  %pre = hivm.hir.batchMmadL1 ins(%a, %b, %true, %c0, %c0, %c0 : tensor<2x16x32xf16>, tensor<2x32x16xf16>, i1, index, index, index) outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  %d = tensor.empty() : tensor<2x32x16xf16>
  %0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %pre) -> (tensor<2x16x16xf32>) : i32 {
    %mmad = hivm.hir.batchMmadL1 ins(%a, %d, %false, %c0, %c0, %c0 : tensor<2x16x32xf16>, tensor<2x32x16xf16>, i1, index, index, index) outs(%acc : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    scf.yield %mmad : tensor<2x16x16xf32>
  }
  return %0 : tensor<2x16x16xf32>
}
}

// -----
// A5: mmadmx f8E5M2 real MKN.
// CHECK-LABEL: func.func @test_mmadmx_f8e5m2_Normalize_Mkn
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[C32]], %[[C128]], %[[C64]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_f8e5m2_Normalize_Mkn() -> tensor<32x64xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<32x128xf8E5M2>
  %b = tensor.empty() : tensor<128x64xf8E5M2>
  %sa = tensor.empty() : tensor<1xui8>
  %sb = tensor.empty() : tensor<1xui8>
  %empty = tensor.empty() : tensor<32x64xf32>
  %r = hivm.hir.mmadmxL1 ins(%a, %b, %sa, %sb, %true, %c0, %c0, %c0 : tensor<32x128xf8E5M2>, tensor<128x64xf8E5M2>, tensor<1xui8>, tensor<1xui8>, i1, index, index, index) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}
}

// -----
// A5: constant empty loop (ub==lb) sets may_not_exec.
// CHECK-LABEL: func.func @test_mmadL1_empty_loop_bounds_may_not_exec
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: } {may_not_exec, normalized_in_L0C = [0 : i32]}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_empty_loop_bounds_may_not_exec() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf16>
  %b = tensor.empty() : tensor<16x16xf16>
  // ub == lb => mayNotExec
  %0 = scf.for %i = %c0_i32 to %c0_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: batch dynamic bounds => may_not_exec.
// CHECK-LABEL: func.func @test_batchMmadL1_may_not_exec_dynamic
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {may_not_exec, normalized_in_L0C = [0 : i32]}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_may_not_exec_dynamic(%lb: i32, %ub: i32) -> tensor<2x16x16xf32> {
  %c1 = arith.constant 1 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  %a = tensor.empty() : tensor<2x16x32xf16>
  %b = tensor.empty() : tensor<2x32x16xf16>
  %0 = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %init) -> (tensor<2x16x16xf32>) : i32 {
    %mmad = hivm.hir.batchMmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<2x16x32xf16>, tensor<2x32x16xf16>, i1, index, index, index) outs(%acc : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    scf.yield %mmad : tensor<2x16x16xf32>
  }
  return %0 : tensor<2x16x16xf32>
}
}

// -----
// A5: mmadmx empty-init split-K CCF.
// CHECK-LABEL: func.func @test_mmadmx_empty_init_splitk
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C}
// CHECK: } {normalized_in_L0C = [0 : i32]}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_empty_init_splitk() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf8E4M3FN>
  %b = tensor.empty() : tensor<16x16xf8E4M3FN>
  %sa = tensor.empty() : tensor<1xi8>
  %sb = tensor.empty() : tensor<1xi8>
  %0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %empty) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadmxL1 ins(%a, %b, %sa, %sb, %false, %c0, %c0, %c0 : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: scf.if-only CCF sets may_not_exec (no remain_in_l0c on if parent).
// CHECK-LABEL: func.func @test_mmadL1_if_only_may_not_exec
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32}
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK: } {may_not_exec, normalized_in_L0C = [0 : i32]}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_if_only_may_not_exec(%cond: i1) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %a = tensor.empty() : tensor<16x16xf16>
  %b = tensor.empty() : tensor<16x16xf16>
  %0 = scf.if %cond -> (tensor<16x16xf32>) {
    %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  } else {
    scf.yield %init : tensor<16x16xf32>
  }
  return %0 : tensor<16x16xf32>
}
}

// -----
// A5: bf16 a+b transpose MKN = (32, 128, 64).
// CHECK-LABEL: func.func @test_mmadL1_ab_transpose_bf16
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, b_transpose} ins({{.*}}, {{.*}}, {{.*}}, %[[C32]], %[[C128]], %[[C64]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_ab_transpose_bf16() -> tensor<32x64xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<128x32xbf16>
  %b = tensor.empty() : tensor<64x128xbf16>
  %empty = tensor.empty() : tensor<32x64xf32>
  %r = hivm.hir.mmadL1 {a_transpose, b_transpose} ins(%a, %b, %true, %c0, %c0, %c0 : tensor<128x32xbf16>, tensor<64x128xbf16>, i1, index, index, index) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}
}

// -----
// A5: batch a+b transpose MKN = (16, 64, 32).
// CHECK-LABEL: func.func @test_batchMmadL1_ab_transpose_Normalize_Mkn
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK: hivm.hir.batchMmadL1 {a_transpose, already_set_real_mkn, b_transpose} ins({{.*}}, {{.*}}, {{.*}}, %[[C16]], %[[C64]], %[[C32]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_ab_transpose_Normalize_Mkn() -> tensor<2x16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<2x64x16xf16>
  %b = tensor.empty() : tensor<2x32x64xf16>
  %empty = tensor.empty() : tensor<2x16x32xf32>
  %r = hivm.hir.batchMmadL1 {a_transpose, b_transpose} ins(%a, %b, %true, %c0, %c0, %c0 : tensor<2x64x16xf16>, tensor<2x32x64xf16>, i1, index, index, index) outs(%empty : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %r : tensor<2x16x32xf32>
}
}

// -----
// A5: block-arg init condition must not crash isInitConstant; decompose via scf.if+vadd.
// CHECK-LABEL: func.func @test_mmadL1_conditional_init_elemwise
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn}
// CHECK: scf.if %[[COND:.*]] -> (tensor<16x16xf32>) {
// CHECK: scf.yield
// CHECK: } else {
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_conditional_init_elemwise(%cond: i1, %bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<16x16xf16>
  %b = tensor.empty() : tensor<16x16xf16>
  %r = hivm.hir.mmadL1 ins(%a, %b, %cond, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}
}

// -----
// A5: batch block-arg init condition -> scf.if + vadd.
// CHECK-LABEL: func.func @test_batchMmadL1_conditional_init_elemwise
// CHECK: hivm.hir.batchMmadL1 {already_set_real_mkn}
// CHECK: scf.if
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_conditional_init_elemwise(%cond: i1, %bias: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<2x16x16xf16>
  %b = tensor.empty() : tensor<2x16x16xf16>
  %r = hivm.hir.batchMmadL1 ins(%a, %b, %cond, %c0, %c0, %c0 : tensor<2x16x16xf16>, tensor<2x16x16xf16>, i1, index, index, index) outs(%bias : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  return %r : tensor<2x16x16xf32>
}
}

// -----
// A5: mmadmx block-arg init condition -> scf.if + vadd.
// CHECK-LABEL: func.func @test_mmadmx_conditional_init_elemwise
// CHECK: hivm.hir.mmadmxL1 {already_set_real_mkn}
// CHECK: scf.if
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadmx_conditional_init_elemwise(%cond: i1, %bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %a = tensor.empty() : tensor<16x16xf8E4M3FN>
  %b = tensor.empty() : tensor<16x16xf8E4M3FN>
  %sa = tensor.empty() : tensor<1xi8>
  %sb = tensor.empty() : tensor<1xi8>
  %r = hivm.hir.mmadmxL1 ins(%a, %b, %sa, %sb, %cond, %c0, %c0, %c0 : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>, tensor<1xi8>, tensor<1xi8>, i1, index, index, index) outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}
}


// -----
// A5: a vtranspose feeding A is absorbed into a_transpose, and real MKN is then
// read from the pre-transpose operand: (A[1], A[0], B[1]) = (16, 64, 32).
// CHECK-LABEL: func.func @test_mmadL1_fold_vtranspose_a
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: hivm.hir.mmadL1 {a_transpose, already_set_real_mkn} ins(%{{.*}}, %{{.*}}, %{{.*}}, %[[C16]], %[[C64]], %[[C32]] : tensor<64x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_fold_vtranspose_a(%a: tensor<64x16xf16>, %b: tensor<64x32xf16>) -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %at_init = tensor.empty() : tensor<16x64xf16>
  %at = hivm.hir.vtranspose ins(%a : tensor<64x16xf16>) outs(%at_init : tensor<16x64xf16>) permutation = [1, 0] -> tensor<16x64xf16>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadL1 ins(%at, %b, %true, %c0, %c0, %c0 : tensor<16x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}

// -----
// A5: a vtranspose feeding B is absorbed into b_transpose.
// CHECK-LABEL: func.func @test_mmadL1_fold_vtranspose_b
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, b_transpose} ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<16x64xf16>, tensor<32x64xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_fold_vtranspose_b(%a: tensor<16x64xf16>, %b: tensor<32x64xf16>) -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %bt_init = tensor.empty() : tensor<64x32xf16>
  %bt = hivm.hir.vtranspose ins(%b : tensor<32x64xf16>) outs(%bt_init : tensor<64x32xf16>) permutation = [1, 0] -> tensor<64x32xf16>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadL1 ins(%a, %bt, %true, %c0, %c0, %c0 : tensor<16x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}

// -----
// A5: a batch vtranspose swapping the two innermost axes is absorbed.
// CHECK-LABEL: func.func @test_batchMmadL1_fold_vtranspose_innermost
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: hivm.hir.batchMmadL1 {a_transpose, already_set_real_mkn} ins(%{{.*}} : tensor<2x64x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_fold_vtranspose_innermost(%a: tensor<2x64x16xf16>, %b: tensor<2x64x32xf16>) -> tensor<2x16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %at_init = tensor.empty() : tensor<2x16x64xf16>
  %at = hivm.hir.vtranspose ins(%a : tensor<2x64x16xf16>) outs(%at_init : tensor<2x16x64xf16>) permutation = [0, 2, 1] -> tensor<2x16x64xf16>
  %empty = tensor.empty() : tensor<2x16x32xf32>
  %r = hivm.hir.batchMmadL1 ins(%at, %b, %true, %c0, %c0, %c0 : tensor<2x16x64xf16>, tensor<2x64x32xf16>, i1, index, index, index) outs(%empty : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %r : tensor<2x16x32xf32>
}
}

// -----
// A5: a_transpose/b_transpose can only express a swap of the two innermost
// axes. This vtranspose swaps the outermost and innermost axis, which is legal
// for vtranspose but not representable by the flag, so it must be left alone.
// CHECK-LABEL: func.func @test_batchMmadL1_no_fold_outer_axis_vtranspose
// CHECK: hivm.hir.vtranspose
// CHECK-NOT: batchMmadL1 {a_transpose
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_no_fold_outer_axis_vtranspose(%a: tensor<64x16x2xf16>, %b: tensor<2x64x32xf16>) -> tensor<2x16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %at_init = tensor.empty() : tensor<2x16x64xf16>
  %at = hivm.hir.vtranspose ins(%a : tensor<64x16x2xf16>) outs(%at_init : tensor<2x16x64xf16>) permutation = [2, 1, 0] -> tensor<2x16x64xf16>
  %empty = tensor.empty() : tensor<2x16x32xf32>
  %r = hivm.hir.batchMmadL1 ins(%at, %b, %true, %c0, %c0, %c0 : tensor<2x16x64xf16>, tensor<2x64x32xf16>, i1, index, index, index) outs(%empty : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %r : tensor<2x16x32xf32>
}
}

// -----
// A5: the operand is a slice of the transposed tensor, so the vtranspose source
// cannot be handed to the mmad directly and the fold must be skipped.
// CHECK-LABEL: func.func @test_mmadL1_no_fold_vtranspose_through_slice
// CHECK: hivm.hir.vtranspose
// CHECK-NOT: mmadL1 {a_transpose
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_mmadL1_no_fold_vtranspose_through_slice(%a: tensor<64x64xf16>, %b: tensor<32x32xf16>) -> tensor<32x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %at_init = tensor.empty() : tensor<64x64xf16>
  %at = hivm.hir.vtranspose ins(%a : tensor<64x64xf16>) outs(%at_init : tensor<64x64xf16>) permutation = [1, 0] -> tensor<64x64xf16>
  %slice = tensor.extract_slice %at[0, 0] [32, 32] [1, 1] : tensor<64x64xf16> to tensor<32x32xf16>
  %empty = tensor.empty() : tensor<32x32xf32>
  %r = hivm.hir.mmadL1 ins(%slice, %b, %true, %c0, %c0, %c0 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}
}

// -----
// A3 / mem-based: NormalizeMatmul does not run the fold there, so the
// vtranspose survives.
// CHECK-LABEL: func.func @test_mmadL1_membase_keeps_vtranspose
// CHECK: hivm.hir.vtranspose
// CHECK-NOT: mmadL1 {a_transpose
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @test_mmadL1_membase_keeps_vtranspose(%a: tensor<64x16xf16>, %b: tensor<64x32xf16>) -> tensor<16x32xf32> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %at_init = tensor.empty() : tensor<16x64xf16>
  %at = hivm.hir.vtranspose ins(%a : tensor<64x16xf16>) outs(%at_init : tensor<16x64xf16>) permutation = [1, 0] -> tensor<16x64xf16>
  %empty = tensor.empty() : tensor<16x32xf32>
  %r = hivm.hir.mmadL1 ins(%at, %b, %true, %c0, %c0, %c0 : tensor<16x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %r : tensor<16x32xf32>
}
}


//===----------------------------------------------------------------------===//
// A3 / mem-based (Ascend910B4): per-channel bias MUST NOT be folded into
// MmadL1Op when A or B is transposed. The mem-based backend only registers
// the non-transposed mma_tile BIAS symbol variants, so folding bias into a
// transposed mmad would later lower to an undefined library call. The bias
// is kept as a separate hivm.hir.vadd instead.
//===----------------------------------------------------------------------===//

// -----
// CHECK-LABEL: func.func @test_madL1_perChannelAdd_b_transpose_keeps_vadd(
// On mem-based arch with b_transpose, per-channel bias must NOT fold into
// mmadL1 (no BIAS_TB symbol exists on 910B); the bias stays materialised as
// a separate hivm.hir.vadd taking the mmad result and the bias broadcast.
// CHECK: %[[BIAS:.*]] = hivm.hir.vbrc ins({{.*}} : tensor<1x768xf32>) outs({{.*}} : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {b_transpose} ins({{.*}} : tensor<29x128xf16>, tensor<768x128xf16>, i1, index, index, index) outs({{.*}} : tensor<29x768xf32>) -> tensor<29x768xf32>
// CHECK: %[[VADD:.*]] = hivm.hir.vadd ins(%[[MMAD]], %[[BIAS]] : tensor<29x768xf32>, tensor<29x768xf32>) outs({{.*}} : tensor<29x768xf32>) -> tensor<29x768xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @test_madL1_perChannelAdd_b_transpose_keeps_vadd(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32>) {
  %false = arith.constant false
  %c128 = arith.constant 128 : index
  %c768 = arith.constant 768 : index
  %c29 = arith.constant 29 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [29, 128], strides: [128, 1] : memref<?xf16> to memref<29x128xf16, strided<[128, 1]>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [768, 128], strides: [128, 1] : memref<?xf16> to memref<768x128xf16, strided<[128, 1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 768], strides: [768, 1] : memref<?xf32> to memref<1x768xf32, strided<[768, 1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [29, 768], strides: [768, 1] : memref<?xf16> to memref<29x768xf16, strided<[768, 1]>>
  %alloc = memref.alloc() : memref<29x128xf16>
  hivm.hir.load ins(%reinterpret_cast : memref<29x128xf16, strided<[128, 1]>>) outs(%alloc : memref<29x128xf16>)
  %9 = bufferization.to_tensor %alloc restrict writable : memref<29x128xf16>
  %alloc_3 = memref.alloc() : memref<768x128xf16>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<768x128xf16, strided<[128, 1]>>) outs(%alloc_3 : memref<768x128xf16>)
  %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<768x128xf16>
  %alloc_4 = memref.alloc() : memref<1x768xf32>
  hivm.hir.load ins(%reinterpret_cast_1 : memref<1x768xf32, strided<[768, 1]>>) outs(%alloc_4 : memref<1x768xf32>)
  %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x768xf32>
  %12 = tensor.empty() : tensor<29x768xf32>
  %13 = hivm.hir.vbrc ins(%11 : tensor<1x768xf32>) outs(%12 : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
  %14 = hivm.hir.mmadL1 {b_transpose} ins(%9, %10, %false, %c29, %c128, %c768 : tensor<29x128xf16>, tensor<768x128xf16>, i1, index, index, index)
        outs(%13 : tensor<29x768xf32>) -> tensor<29x768xf32>
  %15 = tensor.empty() : tensor<29x768xf16>
  %16 = hivm.hir.vcast ins(%14 : tensor<29x768xf32>) outs(%15 : tensor<29x768xf16>) round_mode = <rint> -> tensor<29x768xf16>
  hivm.hir.store ins(%16 : tensor<29x768xf16>) outs(%reinterpret_cast_2 : memref<29x768xf16, strided<[768, 1]>>)
  return
}
}
// -----
// CHECK-LABEL: func.func @test_madL1_perChannelAdd_a_transpose_keeps_vadd(
// On mem-based arch with a_transpose, per-channel bias must NOT fold into
// mmadL1 (no BIAS_TA symbol exists on 910B); the bias stays materialised as
// a separate hivm.hir.vadd taking the mmad result and the bias broadcast.
// CHECK: %[[BIAS:.*]] = hivm.hir.vbrc ins({{.*}} : tensor<1x768xf32>) outs({{.*}} : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {a_transpose} ins({{.*}} : tensor<128x29xf16>, tensor<128x768xf16>, i1, index, index, index) outs({{.*}} : tensor<29x768xf32>) -> tensor<29x768xf32>
// CHECK: %[[VADD:.*]] = hivm.hir.vadd ins(%[[MMAD]], %[[BIAS]] : tensor<29x768xf32>, tensor<29x768xf32>) outs({{.*}} : tensor<29x768xf32>) -> tensor<29x768xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @test_madL1_perChannelAdd_a_transpose_keeps_vadd(%arg2: memref<?xf16> , %arg3: memref<?xf16>, %arg4: memref<?xf16> , %arg5: memref<?xf32>) {
  %false = arith.constant false
  %c128 = arith.constant 128 : index
  %c768 = arith.constant 768 : index
  %c29 = arith.constant 29 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128, 29], strides: [29, 1] : memref<?xf16> to memref<128x29xf16, strided<[29, 1]>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<?xf16> to memref<128x768xf16, strided<[768, 1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 768], strides: [768, 1] : memref<?xf32> to memref<1x768xf32, strided<[768, 1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [29, 768], strides: [768, 1] : memref<?xf16> to memref<29x768xf16, strided<[768, 1]>>
  %alloc = memref.alloc() : memref<128x29xf16>
  hivm.hir.load ins(%reinterpret_cast : memref<128x29xf16, strided<[29, 1]>>) outs(%alloc : memref<128x29xf16>)
  %9 = bufferization.to_tensor %alloc restrict writable : memref<128x29xf16>
  %alloc_3 = memref.alloc() : memref<128x768xf16>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<128x768xf16, strided<[768, 1]>>) outs(%alloc_3 : memref<128x768xf16>)
  %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x768xf16>
  %alloc_4 = memref.alloc() : memref<1x768xf32>
  hivm.hir.load ins(%reinterpret_cast_1 : memref<1x768xf32, strided<[768, 1]>>) outs(%alloc_4 : memref<1x768xf32>)
  %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<1x768xf32>
  %12 = tensor.empty() : tensor<29x768xf32>
  %13 = hivm.hir.vbrc ins(%11 : tensor<1x768xf32>) outs(%12 : tensor<29x768xf32>) broadcast_dims = [0] -> tensor<29x768xf32>
  %14 = hivm.hir.mmadL1 {a_transpose} ins(%9, %10, %false, %c29, %c128, %c768 : tensor<128x29xf16>, tensor<128x768xf16>, i1, index, index, index)
        outs(%13 : tensor<29x768xf32>) -> tensor<29x768xf32>
  %15 = tensor.empty() : tensor<29x768xf16>
  %16 = hivm.hir.vcast ins(%14 : tensor<29x768xf32>) outs(%15 : tensor<29x768xf16>) round_mode = <rint> -> tensor<29x768xf16>
  hivm.hir.store ins(%16 : tensor<29x768xf16>) outs(%reinterpret_cast_2 : memref<29x768xf16, strided<[768, 1]>>)
  return
}
}


// -----
// 4D Fractal A (zN, !a_transpose): shape [K1,M1,16,16] = [20,10,16,16]
// M = dim1*dim2 = 160, K = dim0*dim3 = 320
// CHECK-LABEL: func.func @test_fractal_zN_A_normalize
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-SAME: tensor<20x10x16x16xf16>, tensor<320x80xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fractal_zN_A_normalize(%arg0: memref<20x10x16x16xf16>, %arg1: memref<320x80xf16>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %a_mem = memref.alloc() : memref<20x10x16x16xf16>
    memref.copy %arg0, %a_mem : memref<20x10x16x16xf16> to memref<20x10x16x16xf16>
    %a = bufferization.to_tensor %a_mem restrict writable : memref<20x10x16x16xf16>
    %b_mem = memref.alloc() : memref<320x80xf16>
    memref.copy %arg1, %b_mem : memref<320x80xf16> to memref<320x80xf16>
    %b = bufferization.to_tensor %b_mem restrict writable : memref<320x80xf16>
    %empty = tensor.empty() : tensor<160x80xf32>
    %0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 : tensor<20x10x16x16xf16>, tensor<320x80xf16>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %0 : tensor<160x80xf32>
}
}

// -----
// 4D Fractal A (nZ, a_transpose): shape [M1,K1,16,16] = [10,20,16,16]
// M = dim0*dim3 = 160, K = dim1*dim2 = 320
// CHECK-LABEL: func.func @test_fractal_nZ_A_normalize
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-SAME: tensor<10x20x16x16xf16>, tensor<320x80xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fractal_nZ_A_normalize(%arg0: memref<10x20x16x16xf16>, %arg1: memref<320x80xf16>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %a_mem = memref.alloc() : memref<10x20x16x16xf16>
    memref.copy %arg0, %a_mem : memref<10x20x16x16xf16> to memref<10x20x16x16xf16>
    %a = bufferization.to_tensor %a_mem restrict writable : memref<10x20x16x16xf16>
    %b_mem = memref.alloc() : memref<320x80xf16>
    memref.copy %arg1, %b_mem : memref<320x80xf16> to memref<320x80xf16>
    %b = bufferization.to_tensor %b_mem restrict writable : memref<320x80xf16>
    %empty = tensor.empty() : tensor<160x80xf32>
    %0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 : tensor<10x20x16x16xf16>, tensor<320x80xf16>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %0 : tensor<160x80xf32>
}
}

// -----
// 4D Fractal B (zN, !b_transpose): shape [N1,K1,16,16] = [5,20,16,16]
// K = dim1*dim2 = 320, N = dim0*dim3 = 80
// CHECK-LABEL: func.func @test_fractal_zN_B_normalize
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-SAME: tensor<160x320xf16>, tensor<5x20x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fractal_zN_B_normalize(%arg0: memref<160x320xf16>, %arg1: memref<5x20x16x16xf16>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %a_mem = memref.alloc() : memref<160x320xf16>
    memref.copy %arg0, %a_mem : memref<160x320xf16> to memref<160x320xf16>
    %a = bufferization.to_tensor %a_mem restrict writable : memref<160x320xf16>
    %b_mem = memref.alloc() : memref<5x20x16x16xf16>
    memref.copy %arg1, %b_mem : memref<5x20x16x16xf16> to memref<5x20x16x16xf16>
    %b = bufferization.to_tensor %b_mem restrict writable : memref<5x20x16x16xf16>
    %empty = tensor.empty() : tensor<160x80xf32>
    %0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 : tensor<160x320xf16>, tensor<5x20x16x16xf16>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %0 : tensor<160x80xf32>
}
}

// -----
// 4D Fractal B (nZ, b_transpose): shape [K1,N1,16,16] = [20,5,16,16]
// K = dim0*dim3 = 320, N = dim1*dim2 = 80
// CHECK-LABEL: func.func @test_fractal_nZ_B_normalize
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-SAME: tensor<160x320xf16>, tensor<20x5x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fractal_nZ_B_normalize(%arg0: memref<160x320xf16>, %arg1: memref<20x5x16x16xf16>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %a_mem = memref.alloc() : memref<160x320xf16>
    memref.copy %arg0, %a_mem : memref<160x320xf16> to memref<160x320xf16>
    %a = bufferization.to_tensor %a_mem restrict writable : memref<160x320xf16>
    %b_mem = memref.alloc() : memref<20x5x16x16xf16>
    memref.copy %arg1, %b_mem : memref<20x5x16x16xf16> to memref<20x5x16x16xf16>
    %b = bufferization.to_tensor %b_mem restrict writable : memref<20x5x16x16xf16>
    %empty = tensor.empty() : tensor<160x80xf32>
    %0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 : tensor<160x320xf16>, tensor<20x5x16x16xf16>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %0 : tensor<160x80xf32>
}
}

// -----
// Both 4D Fractal: A zN [K1,M1,16,16] + B zN [N1,K1,16,16]
// M = 160, K = 320, N = 80
// CHECK-LABEL: func.func @test_fractal_both_zN_normalize
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%{{.*}}, %{{.*}}, %true, %c160, %c320, %c80 : tensor<20x10x16x16xf16>, tensor<5x20x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fractal_both_zN_normalize(%arg0: memref<20x10x16x16xf16>, %arg1: memref<5x20x16x16xf16>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %a_mem = memref.alloc() : memref<20x10x16x16xf16>
    memref.copy %arg0, %a_mem : memref<20x10x16x16xf16> to memref<20x10x16x16xf16>
    %a = bufferization.to_tensor %a_mem restrict writable : memref<20x10x16x16xf16>
    %b_mem = memref.alloc() : memref<5x20x16x16xf16>
    memref.copy %arg1, %b_mem : memref<5x20x16x16xf16> to memref<5x20x16x16xf16>
    %b = bufferization.to_tensor %b_mem restrict writable : memref<5x20x16x16xf16>
    %empty = tensor.empty() : tensor<160x80xf32>
    %0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 : tensor<20x10x16x16xf16>, tensor<5x20x16x16xf16>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %0 : tensor<160x80xf32>
}
}

// -----
// f32 Fractal A zN [40,10,16,8] f32: M=160, K=320 (f32 uses block [16,8])
// CHECK-LABEL: func.func @test_fractal_f32_A_normalize
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
// CHECK-SAME: tensor<40x10x16x8xf32>, tensor<320x80xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fractal_f32_A_normalize(%arg0: memref<40x10x16x8xf32>, %arg1: memref<320x80xf32>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %a_mem = memref.alloc() : memref<40x10x16x8xf32>
    memref.copy %arg0, %a_mem : memref<40x10x16x8xf32> to memref<40x10x16x8xf32>
    %a = bufferization.to_tensor %a_mem restrict writable : memref<40x10x16x8xf32>
    %b_mem = memref.alloc() : memref<320x80xf32>
    memref.copy %arg1, %b_mem : memref<320x80xf32> to memref<320x80xf32>
    %b = bufferization.to_tensor %b_mem restrict writable : memref<320x80xf32>
    %empty = tensor.empty() : tensor<160x80xf32>
    %0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 : tensor<40x10x16x8xf32>, tensor<320x80xf32>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %0 : tensor<160x80xf32>
}
}

// -----
// K-padding: Fractal A zN [K1,M1,16,16] with dot_pad_only_k, K_real=310
// real M = dim1*dim2 = 160, real K = dim0*dim3 = 320 (padded from 310)
// CHECK-LABEL: func.func @test_fractal_Kpad_dot_pad_only_k
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, dot_pad_only_k, normalized_in_L0C}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_fractal_Kpad_dot_pad_only_k(%arg0: memref<20x10x16x16xf16>, %arg1: memref<320x80xf16>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %a_mem = memref.alloc() : memref<20x10x16x16xf16>
    memref.copy %arg0, %a_mem : memref<20x10x16x16xf16> to memref<20x10x16x16xf16>
    %a = bufferization.to_tensor %a_mem restrict writable : memref<20x10x16x16xf16>
    %b_mem = memref.alloc() : memref<320x80xf16>
    memref.copy %arg1, %b_mem : memref<320x80xf16> to memref<320x80xf16>
    %b = bufferization.to_tensor %b_mem restrict writable : memref<320x80xf16>
    %empty = tensor.empty() : tensor<160x80xf32>
    %0 = hivm.hir.mmadL1 {dot_pad_only_k} ins(%a, %b, %false, %c160, %c320, %c80 : tensor<20x10x16x16xf16>, tensor<320x80xf16>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %0 : tensor<160x80xf32>
  }
}

// -----
// CHECK-LABEL: func.func @test_mmadmx_chain_no_elemwise_decompose(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mmadmx_chain_no_elemwise_decompose() -> tensor<4x16xf32> {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %true = arith.constant true
    %a = tensor.empty() : tensor<4x8xf8E5M2>
    %b = tensor.empty() : tensor<8x16xf8E5M2>
    %scaleA = tensor.empty() : tensor<1xui8>
    %scaleB = tensor.empty() : tensor<1xui8>
    %initC = tensor.empty() : tensor<4x16xf32>
    %first = hivm.hir.mmadmxL1 ins(%a, %b, %scaleA, %scaleB, %true, %c4, %c8, %c16 : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>, tensor<1xui8>, tensor<1xui8>, i1, index, index, index) outs(%initC : tensor<4x16xf32>) -> tensor<4x16xf32>
    // CHECK-NOT: hivm.hir.vadd
    %second = hivm.hir.mmadmxL1 ins(%a, %b, %scaleA, %scaleB, %false, %c4, %c8, %c16 : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>, tensor<1xui8>, tensor<1xui8>, i1, index, index, index) outs(%first : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %second : tensor<4x16xf32>
  }
}

// -----
// vtranspose feeding into mmadmxL1 A side — folds to a_transpose
// CHECK-LABEL: func.func @test_fold_vtranspose_mmadmx_a
// CHECK: hivm.hir.mmadmxL1
// CHECK-SAME: lhsFormat = 1 : i32
// CHECK-SAME: rhsFormat = 1 : i32
// CHECK-SAME: a_transpose
// CHECK-SAME: tensor<8x4xi8>, tensor<8x16xi8>
// CHECK-NOT: hivm.hir.vtranspose
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fold_vtranspose_mmadmx_a() -> tensor<4x16xf32> {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %alloc_a = memref.alloc() : memref<8x4xi8>
    %a_src = bufferization.to_tensor %alloc_a restrict writable : memref<8x4xi8>
    %empty_t = tensor.empty() : tensor<4x8xi8>
    %a_vtrans = hivm.hir.vtranspose ins(%a_src : tensor<8x4xi8>) outs(%empty_t : tensor<4x8xi8>) permutation = [1, 0] -> tensor<4x8xi8>
    %alloc_b = memref.alloc() : memref<8x16xi8>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<8x16xi8>
    %alloc_sa = memref.alloc() : memref<4x1xi8>
    %scaleA = bufferization.to_tensor %alloc_sa restrict writable : memref<4x1xi8>
    %alloc_sb = memref.alloc() : memref<16x1xi8>
    %scaleB = bufferization.to_tensor %alloc_sb restrict writable : memref<16x1xi8>
    %empty = tensor.empty() : tensor<4x16xf32>
    %result = hivm.hir.mmadmxL1 {lhsFormat = 1 : i32, rhsFormat = 1 : i32}
        ins(%a_vtrans, %b, %scaleA, %scaleB, %true, %c0, %c0, %c0
            : tensor<4x8xi8>, tensor<8x16xi8>, tensor<4x1xi8>, tensor<16x1xi8>, i1, index, index, index)
        outs(%empty : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %result : tensor<4x16xf32>
}
}

// -----
// vtranspose feeding into mmadmxL1 B side — folds to b_transpose
// CHECK-LABEL: func.func @test_fold_vtranspose_mmadmx_b
// CHECK: hivm.hir.mmadmxL1
// CHECK-SAME: b_transpose
// CHECK-SAME: tensor<4x8xf8E5M2>, tensor<16x8xf8E5M2>
// CHECK-NOT: hivm.hir.vtranspose
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_fold_vtranspose_mmadmx_b() -> tensor<4x16xf32> {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %alloc_a = memref.alloc() : memref<4x8xf8E5M2>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<4x8xf8E5M2>
    %alloc_b = memref.alloc() : memref<16x8xf8E5M2>
    %b_src = bufferization.to_tensor %alloc_b restrict writable : memref<16x8xf8E5M2>
    %empty_t = tensor.empty() : tensor<8x16xf8E5M2>
    %b_vtrans = hivm.hir.vtranspose ins(%b_src : tensor<16x8xf8E5M2>) outs(%empty_t : tensor<8x16xf8E5M2>) permutation = [1, 0] -> tensor<8x16xf8E5M2>
    %alloc_sa = memref.alloc() : memref<1xui8>
    %scaleA = bufferization.to_tensor %alloc_sa restrict writable : memref<1xui8>
    %alloc_sb = memref.alloc() : memref<1xui8>
    %scaleB = bufferization.to_tensor %alloc_sb restrict writable : memref<1xui8>
    %empty = tensor.empty() : tensor<4x16xf32>
    %result = hivm.hir.mmadmxL1
        ins(%a, %b_vtrans, %scaleA, %scaleB, %true, %c0, %c0, %c0
            : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>, tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
        outs(%empty : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %result : tensor<4x16xf32>
}
}

// Test: all CCFs are mayNotExec (variable bounds), 2 mmads per for.
// CCF1 is ZeroInit, CCF2/CCF3 are ReuseL0C. AddIf should create IfOp only at
// the last CCF (CCF3) for each mmad, providing vbrc(0) fallback when all
// didn't execute. Also verifies counterBuf result index matching across the
// chain.

// CHECK-LABEL: func.func @test_all_may_not_exec_chain
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_all_may_not_exec_chain(%lb: i32, %ub: i32) -> (tensor<64x32xf32>, tensor<64x32xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %cst_zero = arith.constant 0.000000e+00 : f32
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %empty1 = tensor.empty() : tensor<64x32xf32>
    %empty2 = tensor.empty() : tensor<64x32xf32>
    %vbrc_zero1 = hivm.hir.vbrc ins(%cst_zero : f32) outs(%empty1 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %vbrc_zero2 = hivm.hir.vbrc ins(%cst_zero : f32) outs(%empty2 : tensor<64x32xf32>) -> tensor<64x32xf32>

    // CCF1: 2 ZeroInit mmads, variable bounds
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.alloca() {normalize_matmul_counter = 1 : i32} : memref<i32>
    // CCF1 mmad: initCondition = firstIter (CmpIOp, counter == 0), remain_in_l0c
    // CHECK: %[[FIRST1_A:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 : i32
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[FIRST1_A]], %c64, %c32, %c32
    // CHECK: %[[FIRST1_B:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 : i32
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[FIRST1_B]], %c64, %c32, %c32
    %for1:2 = scf.for %i1 = %lb to %ub step %c1_i32 iter_args(%acc1_0 = %vbrc_zero1, %acc1_1 = %vbrc_zero2) -> (tensor<64x32xf32>, tensor<64x32xf32>) : i32 {
      %mmad1_0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc1_0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %mmad1_1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc1_1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad1_0, %mmad1_1 : tensor<64x32xf32>, tensor<64x32xf32>
    }

    // CCF2: 2 ReuseL0C mmads (init from CCF1), variable bounds
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.alloca() {normalize_matmul_counter = 1 : i32} : memref<i32>
    // CCF2 mmad: initCondition = andi(counterPrevious1, firstIter2) (AndIOp), remain_in_l0c
    // CHECK: %[[INIT2_A:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT2_A]], %c64, %c32, %c32
    // CHECK: %[[INIT2_B:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT2_B]], %c64, %c32, %c32
    %for2:2 = scf.for %i2 = %lb to %ub step %c1_i32 iter_args(%acc2_0 = %for1#0, %acc2_1 = %for1#1) -> (tensor<64x32xf32>, tensor<64x32xf32>) : i32 {
      %mmad2_0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc2_0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %mmad2_1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc2_1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad2_0, %mmad2_1 : tensor<64x32xf32>, tensor<64x32xf32>
    }

    // CCF3: 2 ReuseL0C mmads (init from CCF2), variable bounds — last mayNotExec
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.alloca() {normalize_matmul_counter = 1 : i32} : memref<i32>
    // CCF3 mmad: initCondition = andi(counterPrevious2, firstIter3) (AndIOp), remain_in_l0c
    // CHECK: %[[INIT3_A:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT3_A]], %c64, %c32, %c32
    // CHECK: %[[INIT3_B:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT3_B]], %c64, %c32, %c32
    // CHECK-COUNT-2: scf.if {{.*}} -> (tensor<64x32xf32>)
    // CHECK: hivm.hir.vbrc
    // CHECK: return
    %for3:2 = scf.for %i3 = %lb to %ub step %c1_i32 iter_args(%acc3_0 = %for2#0, %acc3_1 = %for2#1) -> (tensor<64x32xf32>, tensor<64x32xf32>) : i32 {
      %mmad3_0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc3_0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %mmad3_1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc3_1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad3_0, %mmad3_1 : tensor<64x32xf32>, tensor<64x32xf32>
    }

    return %for3#0, %for3#1 : tensor<64x32xf32>, tensor<64x32xf32>
  }
}

// -----

// Test: NoBias for (var bounds), result goes to return → AddIfPattern creates fallback IfOp.
// The deferred_tail_fallback tag must survive (set on tmpNewMmad) and AddIfPattern
// must create scf.if with vbrc(0) when the chain ends at return.
// CHECK-LABEL: func.func @test_nobias_for_var_fallback
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: may_not_exec
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_nobias_for_var_fallback(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %lb: i32, %ub: i32) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %empty = tensor.empty() : tensor<16x16xf32>
  %for = scf.for %i = %lb to %ub step %c1_i32 iter_args(%acc = %empty) -> tensor<16x16xf32> : i32 {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  }
  return %for : tensor<16x16xf32>
}
}

// -----

// Test: ZeroInit for (var bounds), result goes to return → AddIfPattern creates fallback IfOp.
// CHECK-LABEL: func.func @test_zeroinit_for_var_fallback
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: may_not_exec
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_zeroinit_for_var_fallback(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %lb: i32, %ub: i32) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %zero = hivm.hir.vbrc ins(%cst_0 : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %for = scf.for %i = %lb to %ub step %c1_i32 iter_args(%acc = %zero) -> tensor<16x16xf32> : i32 {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  }
  return %for : tensor<16x16xf32>
}
}

// -----

// Test: ZeroInit for (var) with i32 element type (int8 matmul). AddIfPattern's
// fallback then-block must create vbrc(0) with IntegerAttr, not FloatAttr.
// CHECK-LABEL: func.func @test_zeroinit_for_var_i32_fallback
// CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for
// CHECK: may_not_exec
// CHECK: scf.if
// CHECK: hivm.hir.vbrc ins(%{{.*}} : i32) outs
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_zeroinit_for_var_i32_fallback(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %lb: i32, %ub: i32) -> tensor<16x16xi32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16x16xi32>
  %zero = hivm.hir.vbrc ins(%c0_i32 : i32) outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
  %for = scf.for %i = %lb to %ub step %c1_i32 iter_args(%acc = %zero) -> tensor<16x16xi32> : i32 {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xi32>) -> tensor<16x16xi32>
    scf.yield %m : tensor<16x16xi32>
  }
  return %for : tensor<16x16xi32>
}
}

// -----
// vbrc(0) + vadd with perChannel bias: PostPerChannelAddWithSplitK (non-split-K)
// The mmadL1 init is vbrc(0), and the mmadL1 result is added with a perChannel
// vbrc bias via vadd. Bias alloc/load is before mmadL1 to ensure dominance.
// After normalization, vbrc(0) and vadd should be fused into mmadL1 with perChannelBias.
// CHECK-LABEL: func.func @test_vbrc_zero_vadd_postPerChannel
// CHECK-NOT: hivm.hir.vbrc
// CHECK-NOT: hivm.hir.vadd
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C, normalized_init_or_bias}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_vbrc_zero_vadd_postPerChannel(%arg0: memref<1x16xf32>) -> tensor<16x16xf32> {
    %cst_zero = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index

    %alloc_a = memref.alloc() : memref<16x16xf16>
    %a_tensor = bufferization.to_tensor %alloc_a restrict writable : memref<16x16xf16>
    %alloc_b = memref.alloc() : memref<16x16xf16>
    %b_tensor = bufferization.to_tensor %alloc_b restrict writable : memref<16x16xf16>

    %bias_alloc = memref.alloc() : memref<1x16xf32>
    hivm.hir.load ins(%arg0 : memref<1x16xf32>) outs(%bias_alloc : memref<1x16xf32>)
    %bias_tensor = bufferization.to_tensor %bias_alloc restrict writable : memref<1x16xf32>

    %empty_init = tensor.empty() : tensor<16x16xf32>
    %vbrc_zero = hivm.hir.vbrc ins(%cst_zero : f32) outs(%empty_init : tensor<16x16xf32>) -> tensor<16x16xf32>

    %4 = hivm.hir.mmadL1 ins(%a_tensor, %b_tensor, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%vbrc_zero : tensor<16x16xf32>) -> tensor<16x16xf32>

    %empty_vbrc = tensor.empty() : tensor<16x16xf32>
    %vbrc_bias = hivm.hir.vbrc ins(%bias_tensor : tensor<1x16xf32>) outs(%empty_vbrc : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>

    %empty_add = tensor.empty() : tensor<16x16xf32>
    %add_result = hivm.hir.vadd ins(%4, %vbrc_bias : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty_add : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %add_result : tensor<16x16xf32>
}
}

// -----
// vbrc(0) + vadd with perChannel bias whose defining op is AFTER mmadL1.
// isSatisfiedBrcForPerChannel is given the mmadL1 as hookOp, so a bias not
// defined before the mmad is rejected (mirroring isPostPerChannelSplitKPattern);
// the vadd/vbrc are kept and no perChannelBias is fused -> no dominance issue.
// CHECK-LABEL: func.func @test_vbrc_zero_vadd_postPerChannel_bias_after_mmad
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.vadd
func.func @test_vbrc_zero_vadd_postPerChannel_bias_after_mmad(%arg0: memref<1x16xf32>) -> tensor<16x16xf32> {
    %cst_zero = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %c0 = arith.constant 0 : index

    %alloc_a = memref.alloc() : memref<16x16xf16>
    %a_tensor = bufferization.to_tensor %alloc_a restrict writable : memref<16x16xf16>
    %alloc_b = memref.alloc() : memref<16x16xf16>
    %b_tensor = bufferization.to_tensor %alloc_b restrict writable : memref<16x16xf16>

    %empty_init = tensor.empty() : tensor<16x16xf32>
    %vbrc_zero = hivm.hir.vbrc ins(%cst_zero : f32) outs(%empty_init : tensor<16x16xf32>) -> tensor<16x16xf32>

    %4 = hivm.hir.mmadL1 ins(%a_tensor, %b_tensor, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%vbrc_zero : tensor<16x16xf32>) -> tensor<16x16xf32>

    %bias_alloc = memref.alloc() : memref<1x16xf32>
    hivm.hir.load ins(%arg0 : memref<1x16xf32>) outs(%bias_alloc : memref<1x16xf32>)
    %bias_tensor = bufferization.to_tensor %bias_alloc restrict writable : memref<1x16xf32>

    %empty_vbrc = tensor.empty() : tensor<16x16xf32>
    %vbrc_bias = hivm.hir.vbrc ins(%bias_tensor : tensor<1x16xf32>) outs(%empty_vbrc : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>

    %empty_add = tensor.empty() : tensor<16x16xf32>
    %add_result = hivm.hir.vadd ins(%4, %vbrc_bias : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty_add : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %add_result : tensor<16x16xf32>
}

// -----
// vbrc(0) + vadd with perChannel bias in split-K (scf.for): PostPerChannelAddWithSplitK
// The for-loop init is vbrc(0), mmadL1 runs inside the for loop, and after the
// for loop the result is added with a perChannel vbrc bias via vadd.
// In the split-K vbrc(0) case, PostPerChannel does not match because ccfOutVal
// (for result) is not directly consumed by vadd after normalize adds if nesting.
// The mmadL1 init is replaced with tensor.empty, vbrc(0) and vadd remain outside
// the for loop.
// CHECK-LABEL: func.func @test_vbrc_zero_vadd_postPerChannel_splitK
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn}
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func @test_vbrc_zero_vadd_postPerChannel_splitK(%arg0: memref<1x16xf32>, %arg1: memref<16x16xf16>) -> tensor<16x16xf16> {
    %cst_zero = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false

    %alloc_a = memref.alloc() : memref<16x16xf16>
    %a_tensor = bufferization.to_tensor %alloc_a restrict writable : memref<16x16xf16>
    %alloc_b = memref.alloc() : memref<16x16xf16>
    %b_tensor = bufferization.to_tensor %alloc_b restrict writable : memref<16x16xf16>

    %bias_alloc = memref.alloc() : memref<1x16xf32>
    hivm.hir.load ins(%arg0 : memref<1x16xf32>) outs(%bias_alloc : memref<1x16xf32>)
    %bias_tensor = bufferization.to_tensor %bias_alloc restrict writable : memref<1x16xf32>

    %empty_init = tensor.empty() : tensor<16x16xf32>
    %vbrc_zero = hivm.hir.vbrc ins(%cst_zero : f32) outs(%empty_init : tensor<16x16xf32>) -> tensor<16x16xf32>

    %for_result = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %vbrc_zero) -> (tensor<16x16xf32>) {
      %idx_i32 = arith.index_cast %i : index to i32
      %cmp = arith.cmpi eq, %idx_i32, %c0_i32 : i32
      %mmad = hivm.hir.mmadL1 ins(%a_tensor, %b_tensor, %cmp, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mmad : tensor<16x16xf32>
    }

    %empty_vbrc = tensor.empty() : tensor<16x16xf32>
    %vbrc_bias = hivm.hir.vbrc ins(%bias_tensor : tensor<1x16xf32>) outs(%empty_vbrc : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>

    %empty_add = tensor.empty() : tensor<16x16xf32>
    %add_result = hivm.hir.vadd ins(%for_result, %vbrc_bias : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty_add : tensor<16x16xf32>) -> tensor<16x16xf32>

    %empty_cast = tensor.empty() : tensor<16x16xf16>
    %cast_result = hivm.hir.vcast ins(%add_result : tensor<16x16xf32>) outs(%empty_cast : tensor<16x16xf16>) round_mode = <rint> -> tensor<16x16xf16>
    return %cast_result : tensor<16x16xf16>
}
}
// CHECK-LABEL: func.func @if_only_nobias
// CHECK: memref.alloca() {normalize_matmul_counter
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: deferred_tail_fallback
// CHECK-NOT: scf.if
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func @if_only_nobias(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %cond: i1) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %r = scf.if %cond -> tensor<16x16xf32> {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  } else {
    scf.yield %empty : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @if_only_zeroinit
// CHECK-DAG: memref.alloca() {normalize_matmul_counter
// CHECK-DAG: hivm.hir.vbrc
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: deferred_tail_fallback
// CHECK-NOT: scf.if
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func @if_only_zeroinit(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %cond: i1) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %zero = hivm.hir.vbrc ins(%cst_0 : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %r = scf.if %cond -> tensor<16x16xf32> {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%zero : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  } else {
    scf.yield %zero : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @if_only_perchannel
// CHECK-DAG: memref.alloca() {normalize_matmul_counter
// CHECK-DAG: hivm.hir.vbrc
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK-SAME: normalized_init_or_bias
// CHECK-NOT: deferred_tail_fallback
// CHECK-NOT: scf.if
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func @if_only_perchannel(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %bias_1d: tensor<1x16xf32>, %cond: i1) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %bias = hivm.hir.vbrc ins(%bias_1d : tensor<1x16xf32>) outs(%empty : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
  %r = scf.if %cond -> tensor<16x16xf32> {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  } else {
    scf.yield %bias : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @if_only_postperchannel
// CHECK: memref.alloca() {normalize_matmul_counter
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK-SAME: normalized_init_or_bias
// CHECK-NOT: deferred_tail_fallback
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK-NOT: hivm.hir.vadd
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func @if_only_postperchannel(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %cond: i1) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %cst_1d = arith.constant dense<1.000000e+00> : tensor<1x16xf32>
  %init = tensor.empty() : tensor<16x16xf32>
  %r = scf.if %cond -> tensor<16x16xf32> {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  } else {
    scf.yield %init : tensor<16x16xf32>
  }
  %empty = tensor.empty() : tensor<16x16xf32>
  %bias = hivm.hir.vbrc ins(%cst_1d : tensor<1x16xf32>) outs(%empty : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
  %add = hivm.hir.vadd ins(%r, %bias : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %add : tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @if_only_elemadd
// CHECK: memref.alloca() {normalize_matmul_counter
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: } else {
// CHECK: scf.yield %{{.*}} : tensor<16x16xf32>
// CHECK: } {may_not_exec
// CHECK: scf.if
// CHECK: scf.yield %{{.*}} : tensor<16x16xf32>
// CHECK: } else {
// CHECK: hivm.hir.vadd
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func @if_only_elemadd(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %bias: tensor<16x16xf32>, %cond: i1) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<16x16xf32>
  %r = scf.if %cond -> tensor<16x16xf32> {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  } else {
    scf.yield %bias : tensor<16x16xf32>
  }
  return %r : tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @test_postperchannel_vcast_after_for
// CHECK: memref.alloca() {normalize_matmul_counter
// CHECK: scf.for
// CHECK: may_not_exec
// CHECK: hivm.hir.vcast
// CHECK: scf.if
// CHECK: hivm.hir.vbrc ins(%{{.*}} : tensor<1x16xf32>) outs
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_postperchannel_vcast_after_for(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %lb: i32, %ub: i32) -> tensor<16x16xf32> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %cst_f16 = arith.constant dense<1.000000e+00> : tensor<1x16xf16>
  %init = tensor.empty() : tensor<16x16xf32>
  %mat = scf.for %i = %lb to %ub step %c1_i32 iter_args(%acc = %init) -> tensor<16x16xf32> : i32 {
    %m = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %m : tensor<16x16xf32>
  }
  %empty1 = tensor.empty() : tensor<1x16xf32>
  %cast = hivm.hir.vcast ins(%cst_f16 : tensor<1x16xf16>) outs(%empty1 : tensor<1x16xf32>) -> tensor<1x16xf32>
  %empty2 = tensor.empty() : tensor<16x16xf32>
  %bias = hivm.hir.vbrc ins(%cast : tensor<1x16xf32>) outs(%empty2 : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
  %empty3 = tensor.empty() : tensor<16x16xf32>
  %add = hivm.hir.vadd ins(%mat, %bias : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %add : tensor<16x16xf32>
}
}
