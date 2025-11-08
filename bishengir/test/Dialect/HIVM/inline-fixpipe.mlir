// RUN: bishengir-opt -hivm-inline-fixpipe %s -split-input-file -verify-diagnostics | FileCheck %s


// -----
// CHECK: func.func @matmul
func.func @matmul(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {global_kernel = "local"} {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x128xf32>
    %1 = hivm.hir.vbrc ins(%cst : f32) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
    scf.for %arg9 = %arg6 to %c1_i32 step %arg3  : i32 {
      %2 = arith.divsi %arg9, %c2_i32 : i32
      %3 = arith.remsi %arg9, %c2_i32 : i32
      %4 = arith.cmpi eq, %2, %c0_i32 : i32
      %5 = scf.if %4 -> (i32) {
        %47 = arith.muli %2, %c2_i32 : i32
        %48 = arith.subi %c1_i32, %47 : i32
        scf.yield %48 : i32
      } else {
        scf.yield %c2_i32 : i32
      }
      %6 = arith.divsi %3, %5 : i32
      %7 = arith.muli %2, %c2_i32 : i32
      %8 = arith.remsi %3, %5 : i32
      %9 = arith.addi %7, %8 : i32
      %10 = arith.remsi %2, %c2_i32 : i32
      %11 = arith.cmpi ne, %10, %c0_i32 : i32
      %12 = scf.if %11 -> (i32) {
        %47 = arith.subi %c0_i32, %6 : i32
        scf.yield %47 : i32
      } else {
        scf.yield %6 : i32
      }
      %13 = arith.muli %12, %c256_i32 : i32
      %14 = arith.muli %9, %c128_i32 : i32
      %15 = arith.index_cast %13 : i32 to index
      %16 = arith.muli %15, %c256 : index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [256, 256], strides: [256, 1] : memref<?xf16> to memref<256x256xf16, strided<[256, 1], offset: ?>>
      %cast = memref.cast %reinterpret_cast : memref<256x256xf16, strided<[256, 1], offset: ?>> to memref<256x256xf16, strided<[?, ?], offset: ?>>
      %17 = arith.index_cast %14 : i32 to index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%17], sizes: [256, 128], strides: [128, 1] : memref<?xf16> to memref<256x128xf16, strided<[128, 1], offset: ?>>
      %cast_1 = memref.cast %reinterpret_cast_0 : memref<256x128xf16, strided<[128, 1], offset: ?>> to memref<256x128xf16, strided<[?, ?], offset: ?>>
      %alloc = memref.alloc() : memref<256x256xf16>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %cast : memref<256x256xf16, strided<[?, ?], offset: ?>> -> memref<f16>, index, index, index, index, index
      %18 = arith.remsi %offset, %c256 : index
      %19 = arith.subi %c256, %18 : index
      %20 = arith.minsi %19, %c256 : index
      %21 = arith.divsi %offset, %c256 : index
      %22 = arith.remsi %21, %c256 : index
      %23 = arith.subi %c256, %22 : index
      %24 = arith.minsi %23, %c256 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%24, %20] [1, 1] : memref<256x256xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[0, 0] [%24, %20] [1, 1] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1]>>
      memref.copy %subview, %subview_2 : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1]>>
      %25 = bufferization.to_tensor %alloc restrict writable : memref<256x256xf16>
      %alloc_3 = memref.alloc() : memref<256x128xf16>
      %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %cast_1 : memref<256x128xf16, strided<[?, ?], offset: ?>> -> memref<f16>, index, index, index, index, index
      %26 = arith.remsi %offset_5, %c128 : index
      %27 = arith.subi %c128, %26 : index
      %28 = arith.minsi %27, %c128 : index
      %29 = arith.divsi %offset_5, %c128 : index
      %30 = arith.remsi %29, %c256 : index
      %31 = arith.subi %c256, %30 : index
      %32 = arith.minsi %31, %c256 : index
      %subview_8 = memref.subview %reinterpret_cast_0[0, 0] [%32, %28] [1, 1] : memref<256x128xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
      %subview_9 = memref.subview %alloc_3[0, 0] [%32, %28] [1, 1] : memref<256x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview_8, %subview_9 : memref<?x?xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %33 = bufferization.to_tensor %alloc_3 restrict writable : memref<256x128xf16>
      %true = arith.constant true
      %34 = tensor.empty() : tensor<256x128xf32>
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %35 = hivm.hir.mmadL1 ins(%25, %33, %true, %24, %20, %28 : tensor<256x256xf16>, tensor<256x128xf16>, i1, index, index, index) outs(%34 : tensor<256x128xf32>) -> tensor<256x128xf32>
      %36 = arith.muli %15, %c128 : index
      %37 = arith.addi %36, %17 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg2 to offset: [%37], sizes: [256, 128], strides: [128, 1] : memref<?xf16> to memref<256x128xf16, strided<[128, 1], offset: ?>>
      %cast_11 = memref.cast %reinterpret_cast_10 : memref<256x128xf16, strided<[128, 1], offset: ?>> to memref<256x128xf16, strided<[?, ?], offset: ?>>
      %38 = tensor.empty() : tensor<256x128xf16>
      %39 = hivm.hir.vcast ins(%35 : tensor<256x128xf32>) outs(%38 : tensor<256x128xf16>) round_mode = <rint> -> tensor<256x128xf16>
      %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %cast_11 : memref<256x128xf16, strided<[?, ?], offset: ?>> -> memref<f16>, index, index, index, index, index
      %40 = arith.remsi %offset_13, %c128 : index
      %41 = arith.subi %c128, %40 : index
      %42 = arith.minsi %41, %c128 : index
      %43 = arith.divsi %offset_13, %c128 : index
      %44 = arith.remsi %43, %c256 : index
      %45 = arith.subi %c256, %44 : index
      %46 = arith.minsi %45, %c256 : index
      %extracted_slice = tensor.extract_slice %39[0, 0] [%46, %42] [1, 1] : tensor<256x128xf16> to tensor<?x?xf16>
      // CHECK: hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
      %subview_16 = memref.subview %reinterpret_cast_10[0, 0] [%46, %42] [1, 1] : memref<256x128xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?x?xf16>) outs(%subview_16 : memref<?x?xf16, strided<[128, 1], offset: ?>>)
    }
    return
  }

// -----
// CHECK-LABEL: func.func @test_batchMmadL1_fixpipe
func.func @test_batchMmadL1_fixpipe(%ma : tensor<2x256x128xf16>, %mb : tensor<2x128x256xf16>, %dst : memref<2x256x256xf16>){

  %mc = tensor.empty() : tensor<2x256x256xf32>
  %true = arith.constant true
  %M = arith.constant 256 : index
  %K = arith.constant 128 : index
  %N = arith.constant 256 : index
  // CHECK: %[[RET:.*]] = hivm.hir.batchMmadL1 {fixpipe_already_inserted = true}
  %ret = hivm.hir.batchMmadL1 ins(%ma, %mb, %true, %M, %K, %N: tensor<2x256x128xf16>, tensor<2x128x256xf16>, i1, index, index, index)
                              outs(%mc: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  %mc_cast = tensor.empty() : tensor<2x256x256xf16>
  %casted = hivm.hir.vcast ins(%ret : tensor<2x256x256xf32>) outs(%mc_cast : tensor<2x256x256xf16>) round_mode = <rint> -> tensor<2x256x256xf16>
  // CHECK: hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
  // CHECK-SAME: ins(%[[RET]] : tensor<2x256x256xf32>) outs({{.*}} : memref<2x256x256xf16>)
  hivm.hir.store ins(%casted : tensor<2x256x256xf16>) outs(%dst : memref<2x256x256xf16>)
  return
}

// -----
// CHECK-LABEL: func.func @matmul_kernel
func.func @matmul_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %cst = arith.constant 0.000000e+00 : f16
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c15_i32 = arith.constant 15 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %2 = arith.addi %arg4, %c15_i32 : i32
  %3 = arith.divsi %2, %c16_i32 : i32
  %4 = arith.addi %arg5, %c15_i32 : i32
  %5 = arith.divsi %4, %c16_i32 : i32
  %6 = arith.divsi %arg13, %5 : i32
  %7 = arith.subi %3, %6 : i32
  %8 = arith.minsi %7, %c1_i32 : i32
  %9 = arith.remsi %arg13, %5 : i32
  %10 = arith.remsi %9, %8 : i32
  %11 = arith.addi %6, %10 : i32
  %12 = arith.divsi %9, %8 : i32
  %13 = arith.muli %11, %c16_i32 : i32
  %14 = arith.muli %12, %c16_i32 : i32
  %15 = arith.index_cast %13 : i32 to index
  %16 = arith.index_cast %arg7 : i32 to index
  %17 = arith.muli %15, %16 : index
  %18 = arith.index_cast %arg8 : i32 to index
  %19 = arith.index_cast %14 : i32 to index
  %20 = arith.addi %arg6, %c15_i32 : i32
  %21 = arith.divsi %20, %c16_i32 : i32
  %22 = arith.muli %arg8, %c16_i32 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%17], sizes: [16, 16], strides: [%16, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
  %cast = memref.cast %reinterpret_cast : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [16, 16], strides: [%18, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
  %cast_2 = memref.cast %reinterpret_cast_1 : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
  %23 = tensor.empty() : tensor<16x16xf32>
  %24:7 = scf.for %arg16 = %c0_i32 to %21 step %c1_i32 iter_args(%arg17 = %23, %arg18 = %cast, %arg19 = %cast_2, %arg20 = %17, %arg21 = %c0, %arg22 = %19, %arg23 = %c0) -> (tensor<16x16xf32>, memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x16xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
    %42 = arith.muli %arg16, %c16_i32 : i32
    %43 = arith.subi %arg6, %42 : i32
    %alloc = memref.alloc() : memref<16x16xf16>
    %44 = arith.index_cast %43 : i32 to index
    %45 = arith.maxsi %44, %c0 : index
    %46 = arith.minsi %45, %c16 : index
    %47 = arith.cmpi slt, %46, %c16 : index
    scf.if %47 {
    }
    %subview_4 = memref.subview %arg18[0, 0] [16, %46] [1, 1] : memref<16x16xf16, strided<[?, ?], offset: ?>> to memref<16x?xf16, strided<[?, ?], offset: ?>>
    %subview_5 = memref.subview %alloc[0, 0] [16, %46] [1, 1] : memref<16x16xf16> to memref<16x?xf16, strided<[16, 1]>>
    memref.copy %subview_4, %subview_5 : memref<16x?xf16, strided<[?, ?], offset: ?>> to memref<16x?xf16, strided<[16, 1]>>
    %48 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %alloc_6 = memref.alloc() : memref<16x16xf16>
    scf.if %47 {
    }
    %subview_7 = memref.subview %arg19[0, 0] [%46, 16] [1, 1] : memref<16x16xf16, strided<[?, ?], offset: ?>> to memref<?x16xf16, strided<[?, ?], offset: ?>>
    %subview_8 = memref.subview %alloc_6[0, 0] [%46, 16] [1, 1] : memref<16x16xf16> to memref<?x16xf16, strided<[16, 1]>>
    memref.copy %subview_7, %subview_8 : memref<?x16xf16, strided<[?, ?], offset: ?>> to memref<?x16xf16, strided<[16, 1]>>
    %49 = bufferization.to_tensor %alloc_6 restrict writable : memref<16x16xf16>
    %true = arith.constant true
    %50 = arith.cmpi eq, %c0_i32, %arg16 : i32
    %51 = arith.andi %true, %50 : i1
    %c16_9 = arith.constant 16 : index
    %c16_10 = arith.constant 16 : index
    %52 = hivm.hir.mmadL1 ins(%48, %49, %51, %c16_9, %46, %c16_10 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%arg17 : tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK: hivm.hir.fixpipe {enable_nz2nd}
    // CHECK: hivm.hir.debug {debugtype = "print", hex = false, prefix = " partial_sum: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} {{.*}} : tensor<16x16xf32>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " partial_sum: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %52 : tensor<16x16xf32>
    %53 = arith.addi %arg20, %c16 : index
    %54 = arith.addi %53, %arg21 : index
    %reinterpret_cast_11 = memref.reinterpret_cast %arg1 to offset: [%54], sizes: [16, 16], strides: [%16, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
    %cast_12 = memref.cast %reinterpret_cast_11 : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
    %55 = arith.index_cast %22 : i32 to index
    %56 = arith.addi %arg22, %55 : index
    %57 = arith.addi %56, %arg23 : index
    %reinterpret_cast_13 = memref.reinterpret_cast %arg2 to offset: [%57], sizes: [16, 16], strides: [%18, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
    %cast_14 = memref.cast %reinterpret_cast_13 : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<16x16xf16, strided<[?, ?], offset: ?>>
    scf.yield %52, %cast_12, %cast_14, %54, %c0, %57, %c0 : tensor<16x16xf32>, memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x16xf16, strided<[?, ?], offset: ?>>, index, index, index, index
  }
  %25 = tensor.empty() : tensor<16x16xf16>
  %26 = hivm.hir.vcast ins(%24#0 : tensor<16x16xf32>) outs(%25 : tensor<16x16xf16>) round_mode = <rint> -> tensor<16x16xf16>
  %27 = arith.index_cast %arg9 : i32 to index
  %28 = arith.muli %15, %27 : index
  %29 = arith.addi %28, %19 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [16, 16], strides: [%27, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
  %30 = arith.addi %15, %c16 : index
  %31 = arith.index_cast %arg4 : i32 to index
  %32 = arith.maxsi %15, %31 : index
  %33 = arith.minsi %30, %32 : index
  %34 = arith.subi %33, %15 : index
  %35 = arith.addi %19, %c16 : index
  %36 = arith.index_cast %arg5 : i32 to index
  %37 = arith.maxsi %19, %36 : index
  %38 = arith.minsi %35, %37 : index
  %39 = arith.subi %38, %19 : index
  %40 = arith.minsi %34, %c16 : index
  %41 = arith.minsi %39, %c16 : index
  %extracted_slice = tensor.extract_slice %26[0, 0] [%40, %41] [1, 1] : tensor<16x16xf16> to tensor<?x?xf16>
  // CHECK: hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
  // CHECK: ins({{.*}} : tensor<?x?xf32>) outs({{.*}} : memref<?x?xf16, strided<[?, 1], offset: ?>>)
  %subview = memref.subview %reinterpret_cast_3[0, 0] [%40, %41] [1, 1] : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
  hivm.hir.store ins(%extracted_slice : tensor<?x?xf16>) outs(%subview : memref<?x?xf16, strided<[?, 1], offset: ?>>)
  return
}

// -----
// CHECK-LABEL: func.func @mm_01
module {
  func.func @mm_01(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %true = arith.constant true
    %c16_i32 = arith.constant 16 : i32
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg8, %arg7 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg6 : i32
    hivm.hir.set_mask_norm
    %5 = arith.muli %4, %c16_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.muli %6, %c64 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%7], sizes: [16, 64], strides: [64, 1] : memref<?xf16> to memref<16x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 32], strides: [32, 1] : memref<?xf16> to memref<64x32xf16, strided<[32, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
    %8 = arith.muli %6, %c32 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [16, 32], strides: [32, 1] : memref<?xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x64xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<16x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<16x64xf16>)
    %9 = bufferization.to_tensor %alloc restrict writable : memref<16x64xf16>
    %alloc_3 = memref.alloc() : memref<64x32xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<64x32xf16, strided<[32, 1]>>) outs(%alloc_3 : memref<64x32xf16>)
    %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<64x32xf16>
    %alloc_4 = memref.alloc() : memref<32xf32>
    hivm.hir.load ins(%reinterpret_cast_1 : memref<32xf32, strided<[1]>>) outs(%alloc_4 : memref<32xf32>)
    %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<32xf32>
    %expanded = tensor.expand_shape %11 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
    %12 = tensor.empty() : tensor<16x32xf32>
    %c16_5 = arith.constant 16 : index
    %c64_6 = arith.constant 64 : index
    %c32_7 = arith.constant 32 : index
    // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
    %13 = hivm.hir.mmadL1 ins(%9, %10, %true, %c16_5, %c64_6, %c32_7, %expanded : tensor<16x64xf16>, tensor<64x32xf16>, i1, index, index, index, tensor<1x32xf32>) outs(%12 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %14 = hivm.hir.vrelu ins(%13 : tensor<16x32xf32>) outs(%13 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %15 = tensor.empty() : tensor<16x32xf16>
    %16 = hivm.hir.vcast ins(%14 : tensor<16x32xf32>) outs(%15 : tensor<16x32xf16>) round_mode = <rint> -> tensor<16x32xf16>
    %17 = arith.addi %6, %c16 : index
    %18 = arith.maxsi %6, %c32 : index
    %19 = arith.minsi %17, %18 : index
    %20 = arith.subi %19, %6 : index
    %21 = arith.minsi %20, %c16 : index
    %extracted_slice = tensor.extract_slice %16[0, 0] [%21, 32] [1, 1] : tensor<16x32xf16> to tensor<?x32xf16>
    %subview = memref.subview %reinterpret_cast_2[0, 0] [%21, 32] [1, 1] : memref<16x32xf16, strided<[32, 1], offset: ?>> to memref<?x32xf16, strided<[32, 1], offset: ?>>
    // CHECK: hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>, pre_relu = #hivm.fixpipe_pre_relu_mode<NORMAL_RELU>}
    hivm.hir.store ins(%extracted_slice : tensor<?x32xf16>) outs(%subview : memref<?x32xf16, strided<[32, 1], offset: ?>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @_attn_fwd
func.func @_attn_fwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg_cube : memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: f32, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %cst = arith.constant 2.000000e+00 : f32
  %cst_0 = arith.constant 0.693147182 : f32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst_1 = arith.constant 1.44269502 : f32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %cst_3 = arith.constant -1.000000e+06 : f32
  %c1_i32 = arith.constant 1 : i32
  %cst_4 = arith.constant 0xFF800000 : f32
  %cst_5 = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = hivm.hir.vbrc ins(%cst_5 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = hivm.hir.vbrc ins(%cst_4 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %3 = tensor.empty() : tensor<64x32xf32>
  %4 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
  %5 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
  %6 = tensor.empty() : tensor<64x64xf32>
  %7 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %8 = arith.muli %arg25, %c64_i32 : i32
  %9 = arith.index_cast %8 : i32 to index
  %10 = arith.index_cast %arg9 : i32 to index
  %11 = arith.muli %9, %10 : index
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [64, 64], strides: [%10, 1] : memref<?xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
  %12 = arith.index_cast %arg15 : i32 to index
  %13 = arith.index_cast %arg12 : i32 to index
  %14 = arith.index_cast %arg18 : i32 to index
  %15 = arith.muli %9, %14 : index
  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%15], sizes: [64, 64], strides: [%14, 1] : memref<?xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
  %cast = memref.cast %reinterpret_cast_6 : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16, strided<[?, ?], offset: ?>>
  %16 = tensor.empty() : tensor<64xi32>
  %c1 = arith.constant 1 : index
  %17 = hivm.hir.varange strides[%c1] outs(%16 : tensor<64xi32>) -> tensor<64xi32>
  %18 = hivm.hir.vadd ins(%17, %8 : tensor<64xi32>, i32) outs(%16 : tensor<64xi32>) -> tensor<64xi32>
  %19 = tensor.empty() : tensor<32xi32>
  %20 = hivm.hir.varange strides[%c1] outs(%19 : tensor<32xi32>) -> tensor<32xi32>
  %21 = arith.mulf %arg4, %cst_1 : f32
  %alloc = memref.alloc() : memref<64x64xf16>
  memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16>
  %22 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 64], strides: [%12, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1]>>
  %cast_8 = memref.cast %reinterpret_cast_7 : memref<32x64xf16, strided<[?, 1]>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
  %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 32], strides: [1, %13] : memref<?xf16> to memref<64x32xf16, strided<[1, ?]>>
  %cast_10 = memref.cast %reinterpret_cast_9 : memref<64x32xf16, strided<[1, ?]>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
  %23:9 = scf.for %arg28 = %c0_i32 to %8 step %c32_i32 iter_args(%arg29 = %1, %arg30 = %7, %arg31 = %2, %arg32 = %cast_8, %arg33 = %cast_10, %arg34 = %c0, %arg35 = %c0, %arg36 = %c0, %arg37 = %c0) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<32x64xf16, strided<[?, ?], offset: ?>>, memref<64x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
    %alloc_17 = memref.alloc() : memref<64x32xf16>
    memref.copy %arg33, %alloc_17 : memref<64x32xf16, strided<[?, ?], offset: ?>> to memref<64x32xf16>
    %43 = bufferization.to_tensor %alloc_17 restrict writable : memref<64x32xf16>
    %true = arith.constant true
    %44 = tensor.empty() : tensor<64x32xf32>
    %c64 = arith.constant 64 : index
    %c64_18 = arith.constant 64 : index
    %c64_19 = arith.constant 64 : index
    %c32_20 = arith.constant 32 : index
    // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
    %45 = hivm.hir.mmadL1 ins(%22, %43, %true, %c64, %c64_18, %c32_20 : tensor<64x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%44 : tensor<64x32xf32>) -> tensor<64x32xf32>
    // CHECK: tensor.empty
    // CHECK: hivm.hir.fixpipe
    %expanded_21 = tensor.expand_shape %0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %46 = hivm.hir.vreduce <max> ins(%45 : tensor<64x32xf32>) outs(%expanded_21 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
    %collapsed = tensor.collapse_shape %46 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %47 = hivm.hir.vmul ins(%collapsed, %21 : tensor<64xf32>, f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %48 = hivm.hir.vmax ins(%arg31, %47 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %49 = hivm.hir.vmul ins(%45, %21 : tensor<64x32xf32>, f32) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %expanded_22 = tensor.expand_shape %48 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %50 = hivm.hir.vbrc ins(%expanded_22 : tensor<64x1xf32>) outs(%3 : tensor<64x32xf32>) broadcast_dims = [1] -> tensor<64x32xf32>
    %51 = hivm.hir.vsub ins(%49, %50 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %52 = hivm.hir.vmul ins(%51, %cst_0 : tensor<64x32xf32>, f32) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %53 = hivm.hir.vexp ins(%52 : tensor<64x32xf32>) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %expanded_23 = tensor.expand_shape %0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %54 = hivm.hir.vreduce <sum> ins(%53 : tensor<64x32xf32>) outs(%expanded_23 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
    %collapsed_24 = tensor.collapse_shape %54 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %55 = hivm.hir.vsub ins(%arg31, %48 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %56 = hivm.hir.vmul ins(%55, %cst_0 : tensor<64xf32>, f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %57 = hivm.hir.vexp ins(%56 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %58 = hivm.hir.vmul ins(%arg29, %57 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %59 = hivm.hir.vadd ins(%58, %collapsed_24 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %expanded_25 = tensor.expand_shape %57 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %60 = hivm.hir.vbrc ins(%expanded_25 : tensor<64x1xf32>) outs(%6 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
    %61 = hivm.hir.vmul ins(%arg30, %60 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %alloc_26 = memref.alloc() : memref<32x64xf16>
    memref.copy %arg32, %alloc_26 : memref<32x64xf16, strided<[?, ?], offset: ?>> to memref<32x64xf16>
    %62 = bufferization.to_tensor %alloc_26 restrict writable : memref<32x64xf16>
    %63 = tensor.empty() : tensor<64x32xf16>
    %64 = hivm.hir.vcast ins(%53 : tensor<64x32xf32>) outs(%63 : tensor<64x32xf16>) round_mode = <rint> -> tensor<64x32xf16>
    %true_27 = arith.constant true
    %alloc_28 = memref.alloc() : memref<64x32xf16>
    %c64_29 = arith.constant 64 : index
    %c32_30 = arith.constant 32 : index
    %c32_31 = arith.constant 32 : index
    %c64_32 = arith.constant 64 : index
    // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
    %65 = hivm.hir.mmadL1 ins(%64, %62, %true_27, %c64_29, %c32_30, %c64_32 : tensor<64x32xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%61 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %66 = arith.muli %12, %c32 : index
    %67 = arith.addi %66, %arg34 : index
    %68 = arith.addi %67, %arg35 : index
    %reinterpret_cast_33 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [32, 64], strides: [%12, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1], offset: ?>>
    %cast_34 = memref.cast %reinterpret_cast_33 : memref<32x64xf16, strided<[?, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %69 = arith.muli %13, %c32 : index
    %70 = arith.addi %69, %arg37 : index
    %71 = arith.addi %arg36, %70 : index
    %reinterpret_cast_35 = memref.reinterpret_cast %arg2 to offset: [%71], sizes: [64, 32], strides: [1, %13] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
    %cast_36 = memref.cast %reinterpret_cast_35 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
    scf.yield %59, %65, %48, %cast_34, %cast_36, %68, %c0, %71, %c0 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<32x64xf16, strided<[?, ?], offset: ?>>, memref<64x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index
  }
  // CHECK: tensor.empty
  // CHECK: hivm.hir.fixpipe
  %24 = arith.muli %arg25, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32
  %25 = arith.addi %arg25, %c1_i32 : i32
  %26 = arith.muli %25, %c64_i32 : i32
  %27 = arith.index_cast %24 : i32 to index
  %28 = arith.muli %27, %13 : index
  %29 = arith.muli %27, %12 : index
  %expanded = tensor.expand_shape %20 [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
  %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [32, 64], strides: [%12, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1], offset: ?>>
  %cast_12 = memref.cast %reinterpret_cast_11 : memref<32x64xf16, strided<[?, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
  %reinterpret_cast_13 = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [64, 32], strides: [1, %13] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
  %cast_14 = memref.cast %reinterpret_cast_13 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
  %30:9 = scf.for %arg28 = %24 to %26 step %c32_i32 iter_args(%arg29 = %23#0, %arg30 = %23#1, %arg31 = %23#2, %arg32 = %cast_12, %arg33 = %cast_14, %arg34 = %29, %arg35 = %c0, %arg36 = %28, %arg37 = %c0) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<32x64xf16, strided<[?, ?], offset: ?>>, memref<64x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
    %alloc_17 = memref.alloc() : memref<64x32xf16>
    memref.copy %arg33, %alloc_17 : memref<64x32xf16, strided<[?, ?], offset: ?>> to memref<64x32xf16>
    %43 = bufferization.to_tensor %alloc_17 restrict writable : memref<64x32xf16>
    %true = arith.constant true
    %44 = tensor.empty() : tensor<64x32xf32>
    %c64 = arith.constant 64 : index
    %c64_18 = arith.constant 64 : index
    %c64_19 = arith.constant 64 : index
    %c32_20 = arith.constant 32 : index
    // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
    %45 = hivm.hir.mmadL1 ins(%22, %43, %true, %c64, %c64_18, %c32_20 : tensor<64x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%44 : tensor<64x32xf32>) -> tensor<64x32xf32>
    // CHECK: tensor.empty
    // CHECK: hivm.hir.fixpipe
    %46 = tensor.empty() : tensor<1x32xi32>
    %47 = hivm.hir.vadd ins(%expanded, %arg28 : tensor<1x32xi32>, i32) outs(%46 : tensor<1x32xi32>) -> tensor<1x32xi32>
    %collapsed = tensor.collapse_shape %47 [[0, 1]] : tensor<1x32xi32> into tensor<32xi32>
    %48 = tensor.empty() : tensor<64x32xi1>
    %49 = tensor.empty() : tensor<64xi64>
    %50 = hivm.hir.vcast ins(%18 : tensor<64xi32>) outs(%49 : tensor<64xi64>) -> tensor<64xi64>
    %51 = tensor.empty() : tensor<64x32xi64>
    %expanded_21 = tensor.expand_shape %50 [[0, 1]] output_shape [64, 1] : tensor<64xi64> into tensor<64x1xi64>
    %52 = hivm.hir.vbrc ins(%expanded_21 : tensor<64x1xi64>) outs(%51 : tensor<64x32xi64>) broadcast_dims = [1] -> tensor<64x32xi64>
    %53 = tensor.empty() : tensor<32xi64>
    %54 = hivm.hir.vcast ins(%collapsed : tensor<32xi32>) outs(%53 : tensor<32xi64>) -> tensor<32xi64>
    %expanded_22 = tensor.expand_shape %54 [[0, 1]] output_shape [1, 32] : tensor<32xi64> into tensor<1x32xi64>
    %55 = hivm.hir.vbrc ins(%expanded_22 : tensor<1x32xi64>) outs(%51 : tensor<64x32xi64>) broadcast_dims = [0] -> tensor<64x32xi64>
    %56 = hivm.hir.vcmp ins(%52, %55 : tensor<64x32xi64>, tensor<64x32xi64>) outs(%48 : tensor<64x32xi1>) compare_mode = <ge> -> tensor<64x32xi1>
    %57 = hivm.hir.vmul ins(%45, %21 : tensor<64x32xf32>, f32) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %58 = hivm.hir.vsel ins(%56, %5, %4 : tensor<64x32xi1>, tensor<64x32xf32>, tensor<64x32xf32>) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %59 = hivm.hir.vadd ins(%57, %58 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %expanded_23 = tensor.expand_shape %0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %60 = hivm.hir.vreduce <max> ins(%59 : tensor<64x32xf32>) outs(%expanded_23 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
    %collapsed_24 = tensor.collapse_shape %60 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %61 = hivm.hir.vmax ins(%arg31, %collapsed_24 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %expanded_25 = tensor.expand_shape %61 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %62 = hivm.hir.vbrc ins(%expanded_25 : tensor<64x1xf32>) outs(%3 : tensor<64x32xf32>) broadcast_dims = [1] -> tensor<64x32xf32>
    %63 = hivm.hir.vsub ins(%59, %62 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %64 = hivm.hir.vmul ins(%63, %cst_0 : tensor<64x32xf32>, f32) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %65 = hivm.hir.vexp ins(%64 : tensor<64x32xf32>) outs(%3 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %expanded_26 = tensor.expand_shape %0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %66 = hivm.hir.vreduce <sum> ins(%65 : tensor<64x32xf32>) outs(%expanded_26 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
    %collapsed_27 = tensor.collapse_shape %66 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %67 = hivm.hir.vsub ins(%arg31, %61 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %68 = hivm.hir.vmul ins(%67, %cst_0 : tensor<64xf32>, f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %69 = hivm.hir.vexp ins(%68 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %70 = hivm.hir.vmul ins(%arg29, %69 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %71 = hivm.hir.vadd ins(%70, %collapsed_27 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %expanded_28 = tensor.expand_shape %69 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %72 = hivm.hir.vbrc ins(%expanded_28 : tensor<64x1xf32>) outs(%6 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
    %73 = hivm.hir.vmul ins(%arg30, %72 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %alloc_29 = memref.alloc() : memref<32x64xf16>
    memref.copy %arg32, %alloc_29 : memref<32x64xf16, strided<[?, ?], offset: ?>> to memref<32x64xf16>
    %74 = bufferization.to_tensor %alloc_29 restrict writable : memref<32x64xf16>
    %75 = tensor.empty() : tensor<64x32xf16>
    %76 = hivm.hir.vcast ins(%65 : tensor<64x32xf32>) outs(%75 : tensor<64x32xf16>) round_mode = <rint> -> tensor<64x32xf16>
    %true_30 = arith.constant true
    %alloc_31 = memref.alloc() : memref<64x32xf16>
    %c64_32 = arith.constant 64 : index
    %c32_33 = arith.constant 32 : index
    %c32_34 = arith.constant 32 : index
    %c64_35 = arith.constant 64 : index
    // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
    %77 = hivm.hir.mmadL1 ins(%76, %74, %true_30, %c64_32, %c32_33, %c64_35 : tensor<64x32xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%73 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %78 = arith.muli %12, %c32 : index
    %79 = arith.addi %78, %arg34 : index
    %80 = arith.addi %79, %arg35 : index
    %reinterpret_cast_36 = memref.reinterpret_cast %arg3 to offset: [%80], sizes: [32, 64], strides: [%12, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1], offset: ?>>
    %cast_37 = memref.cast %reinterpret_cast_36 : memref<32x64xf16, strided<[?, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    %81 = arith.muli %13, %c32 : index
    %82 = arith.addi %81, %arg37 : index
    %83 = arith.addi %arg36, %82 : index
    %reinterpret_cast_38 = memref.reinterpret_cast %arg2 to offset: [%83], sizes: [64, 32], strides: [1, %13] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
    %cast_39 = memref.cast %reinterpret_cast_38 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
    scf.yield %71, %77, %61, %cast_37, %cast_39, %80, %c0, %83, %c0 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<32x64xf16, strided<[?, ?], offset: ?>>, memref<64x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index
  }
  %31 = hivm.hir.vln ins(%30#0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %32 = hivm.hir.vbrc ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %33 = hivm.hir.vln ins(%32 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %34 = hivm.hir.vdiv ins(%31, %33 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %35 = hivm.hir.vadd ins(%30#2, %34 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %expanded_15 = tensor.expand_shape %30#0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %36 = hivm.hir.vbrc ins(%expanded_15 : tensor<64x1xf32>) outs(%6 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
  // CHECK: tensor.empty
  // CHECK: hivm.hir.fixpipe
  // CHECK: hivm.hir.vdiv
  %37 = hivm.hir.vdiv ins(%30#1, %36 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %38 = arith.muli %arg26, %arg21 : i32
  %39 = arith.index_cast %38 : i32 to index
  %40 = arith.addi %39, %9 : index
  %reinterpret_cast_16 = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %35 in writable %reinterpret_cast_16 : (tensor<64xf32>, memref<64xf32, strided<[1], offset: ?>>) -> ()
  %41 = tensor.empty() : tensor<64x64xf16>
  %42 = hivm.hir.vcast ins(%37 : tensor<64x64xf32>) outs(%41 : tensor<64x64xf16>) round_mode = <rint> -> tensor<64x64xf16>
  hivm.hir.store ins(%42 : tensor<64x64xf16>) outs(%cast : memref<64x64xf16, strided<[?, ?], offset: ?>>)
  return
}

// -----
// CHECK-LABEL: func.func @_attn_bwd(
// CHECK-SAME: %[[ARG0:.*]]: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %[[ARG_cube:.*]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %[[ARG1:.*]]: memref<?xf16> {tt.divisibility = 16 : i32}, %[[ARG2:.*]]: memref<?xf16> {tt.divisibility = 16 : i32}, %[[ARG3:.*]]: memref<?xf16> {tt.divisibility = 16 : i32}, %[[ARG4:.*]]: f32, %[[ARG5:.*]]: memref<?xf16> {tt.divisibility = 16 : i32}, %[[ARG6:.*]]: memref<?xf16> {tt.divisibility = 16 : i32}, %[[ARG7:.*]]: memref<?xf16> {tt.divisibility = 16 : i32}, %[[ARG8:.*]]: memref<?xf16> {tt.divisibility = 16 : i32}, %[[ARG9:.*]]: memref<?xf32> {tt.divisibility = 16 : i32}, %[[ARG10:.*]]: memref<?xf32> {tt.divisibility = 16 : i32}, %[[ARG11:.*]]: i32 {tt.divisibility = 16 : i32}, %[[ARG12:.*]]: i32 {tt.divisibility = 16 : i32}, %[[ARG13:.*]]: i32 {tt.divisibility = 16 : i32}, %[[ARG14:.*]]: i32, %[[ARG15:.*]]: i32 {tt.divisibility = 16 : i32}, %[[ARG16:.*]]: i32, %[[ARG17:.*]]: i32, %[[ARG18:.*]]: i32)
module {
  func.func @_attn_bwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg_cube : memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: f32, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32, %arg18: i32) attributes {func_dyn_memref_args = dense<[false, true, true, true, false, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<19xi1>, global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.693147182 : f32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.remsi %1, %arg18 : i32
    %3 = arith.muli %arg18, %arg17 : i32
    %4 = arith.divsi %1, %3 : i32
    %5 = arith.remsi %4, %arg16 : i32
    hivm.hir.set_mask_norm
    %6 = tensor.empty() : tensor<128x32xf32>
    %7 = tensor.empty() : tensor<128x16xf32>
    %8 = hivm.hir.vbrc ins(%cst : f32) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
    %9 = tensor.empty() : tensor<128x64xf32>
    %10 = arith.muli %2, %arg15 : i32
    %11 = arith.remsi %2, %arg14 : i32
    %12 = arith.muli %arg12, %11 : i32
    %13 = arith.divsi %2, %arg14 : i32
    %14 = arith.muli %arg11, %13 : i32
    %15 = arith.addi %12, %14 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.index_cast %10 : i32 to index
    %18 = arith.muli %5, %c128_i32 : i32
    %19 = tensor.empty() : tensor<128xi32>
    %expanded = tensor.expand_shape %19 [[0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
    %c1 = arith.constant 1 : index
    %20 = hivm.hir.varange strides[%c1] outs(%19 : tensor<128xi32>) -> tensor<128xi32>
    %expanded_1 = tensor.expand_shape %20 [[0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
    %21 = hivm.hir.vadd ins(%expanded_1, %18 : tensor<128x1xi32>, i32) outs(%expanded : tensor<128x1xi32>) -> tensor<128x1xi32>
    %22 = arith.index_cast %18 : i32 to index
    %23 = arith.index_cast %arg13 : i32 to index
    %24 = arith.muli %22, %23 : index
    %25 = arith.addi %16, %24 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [128, 64], strides: [%23, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x64xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<128x64xf16, strided<[?, 1], offset: ?>>) outs(%alloc : memref<128x64xf16>)
    %26 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [128, 64], strides: [%23, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<128x64xf16>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<128x64xf16, strided<[?, 1], offset: ?>>) outs(%alloc_3 : memref<128x64xf16>)
    %27 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x64xf16>
    %28 = tensor.empty() : tensor<16xi32>
    %expanded_4 = tensor.expand_shape %28 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %expanded_5 = tensor.expand_shape %28 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %29 = hivm.hir.varange strides[%c1] outs(%28 : tensor<16xi32>) -> tensor<16xi32>
    %expanded_6 = tensor.expand_shape %29 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %expanded_7 = tensor.expand_shape %29 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %30 = arith.muli %arg13, %c16_i32 : i32
    %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [64, 16], strides: [1, %23] : memref<?xf16> to memref<64x16xf16, strided<[1, ?], offset: ?>>
    %cast = memref.cast %reinterpret_cast_8 : memref<64x16xf16, strided<[1, ?], offset: ?>> to memref<64x16xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [16, 64], strides: [%23, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
    %cast_10 = memref.cast %reinterpret_cast_9 : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<16x64xf16, strided<[?, ?], offset: ?>>
    %31 = tensor.empty() : tensor<128x64xf32>
    %32:9 = scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg20 = %31, %arg21 = %31, %arg22 = %18, %arg23 = %cast, %arg24 = %cast_10, %arg25 = %25, %arg26 = %c0, %arg27 = %25, %arg28 = %c0) -> (tensor<128x64xf32>, tensor<128x64xf32>, i32, memref<64x16xf16, strided<[?, ?], offset: ?>>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_39 = memref.alloc() : memref<64x16xf16>
      hivm.hir.load ins(%arg23 : memref<64x16xf16, strided<[?, ?], offset: ?>>) outs(%alloc_39 : memref<64x16xf16>)
      %65 = bufferization.to_tensor %alloc_39 restrict writable : memref<64x16xf16>
      %66 = hivm.hir.vadd ins(%expanded_6, %arg22 : tensor<1x16xi32>, i32) outs(%expanded_4 : tensor<1x16xi32>) -> tensor<1x16xi32>
      %67 = arith.index_cast %arg22 : i32 to index
      %68 = arith.addi %17, %67 : index
      %reinterpret_cast_40 = memref.reinterpret_cast %arg9 to offset: [%68], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %alloc_41 = memref.alloc() : memref<16xf32>
      hivm.hir.load ins(%reinterpret_cast_40 : memref<16xf32, strided<[1], offset: ?>>) outs(%alloc_41 : memref<16xf32>)
      %69 = bufferization.to_tensor %alloc_41 restrict writable : memref<16xf32>
      %70 = tensor.empty() : tensor<128x16xf32>
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %71 = hivm.hir.mmadL1 ins(%26, %65, %true, %c128, %c64, %c16 : tensor<128x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%70 : tensor<128x16xf32>) -> tensor<128x16xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %expanded_42 = tensor.expand_shape %69 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
      %72 = hivm.hir.vbrc ins(%expanded_42 : tensor<1x16xf32>) outs(%7 : tensor<128x16xf32>) broadcast_dims = [0] -> tensor<128x16xf32>
      %73 = hivm.hir.vsub ins(%71, %72 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %74 = hivm.hir.vmul ins(%73, %cst_0 : tensor<128x16xf32>, f32) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %75 = hivm.hir.vexp ins(%74 : tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %76 = tensor.empty() : tensor<128x16xi1>
      %77 = tensor.empty() : tensor<16xi64>
      %expanded_43 = tensor.expand_shape %77 [[0, 1]] output_shape [1, 16] : tensor<16xi64> into tensor<1x16xi64>
      %78 = hivm.hir.vcast ins(%66 : tensor<1x16xi32>) outs(%expanded_43 : tensor<1x16xi64>) -> tensor<1x16xi64>
      %79 = tensor.empty() : tensor<128x16xi64>
      %80 = hivm.hir.vbrc ins(%78 : tensor<1x16xi64>) outs(%79 : tensor<128x16xi64>) broadcast_dims = [0] -> tensor<128x16xi64>
      %81 = tensor.empty() : tensor<128xi64>
      %expanded_44 = tensor.expand_shape %81 [[0, 1]] output_shape [128, 1] : tensor<128xi64> into tensor<128x1xi64>
      %82 = hivm.hir.vcast ins(%21 : tensor<128x1xi32>) outs(%expanded_44 : tensor<128x1xi64>) -> tensor<128x1xi64>
      %83 = hivm.hir.vbrc ins(%82 : tensor<128x1xi64>) outs(%79 : tensor<128x16xi64>) broadcast_dims = [1] -> tensor<128x16xi64>
      %84 = hivm.hir.vcmp ins(%80, %83 : tensor<128x16xi64>, tensor<128x16xi64>) outs(%76 : tensor<128x16xi1>) compare_mode = <ge> -> tensor<128x16xi1>
      %85 = hivm.hir.vsel ins(%84, %75, %8 : tensor<128x16xi1>, tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %alloc_45 = memref.alloc() : memref<16x64xf16>
      hivm.hir.load ins(%arg24 : memref<16x64xf16, strided<[?, ?], offset: ?>>) outs(%alloc_45 : memref<16x64xf16>)
      %86 = bufferization.to_tensor %alloc_45 restrict writable : memref<16x64xf16>
      %87 = tensor.empty() : tensor<128x16xf16>
      %88 = hivm.hir.vcast ins(%85 : tensor<128x16xf32>) outs(%87 : tensor<128x16xf16>) round_mode = <rint> -> tensor<128x16xf16>
      %89 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %c128_46 = arith.constant 128 : index
      %c16_47 = arith.constant 16 : index
      %c64_48 = arith.constant 64 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %90 = hivm.hir.mmadL1 ins(%88, %86, %89, %c128_46, %c16_47, %c64_48 : tensor<128x16xf16>, tensor<16x64xf16>, i1, index, index, index) outs(%arg20 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %reinterpret_cast_49 = memref.reinterpret_cast %arg10 to offset: [%68], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %alloc_50 = memref.alloc() : memref<16xf32>
      hivm.hir.load ins(%reinterpret_cast_49 : memref<16xf32, strided<[1], offset: ?>>) outs(%alloc_50 : memref<16xf32>)
      %91 = bufferization.to_tensor %alloc_50 restrict writable : memref<16xf32>
      %92 = tensor.empty() : tensor<128x16xf32>
      %c128_51 = arith.constant 128 : index
      %c64_52 = arith.constant 64 : index
      %c16_53 = arith.constant 16 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %93 = hivm.hir.mmadL1 {b_transpose} ins(%27, %86, %true, %c128_51, %c64_52, %c16_53 : tensor<128x64xf16>, tensor<16x64xf16>, i1, index, index, index) outs(%92 : tensor<128x16xf32>) -> tensor<128x16xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %expanded_54 = tensor.expand_shape %91 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
      %94 = hivm.hir.vbrc ins(%expanded_54 : tensor<1x16xf32>) outs(%7 : tensor<128x16xf32>) broadcast_dims = [0] -> tensor<128x16xf32>
      %95 = hivm.hir.vsub ins(%93, %94 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %96 = hivm.hir.vmul ins(%85, %95 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %97 = hivm.hir.vcast ins(%96 : tensor<128x16xf32>) outs(%87 : tensor<128x16xf16>) round_mode = <rint> -> tensor<128x16xf16>
      %98 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %c128_55 = arith.constant 128 : index
      %c16_56 = arith.constant 16 : index
      %c64_57 = arith.constant 64 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %99 = hivm.hir.mmadL1 {b_transpose} ins(%97, %65, %98, %c128_55, %c16_56, %c64_57 : tensor<128x16xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%arg21 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %100 = arith.addi %arg22, %c16_i32 : i32
      %101 = arith.index_cast %30 : i32 to index
      %102 = arith.addi %arg25, %101 : index
      %103 = arith.addi %102, %arg26 : index
      %reinterpret_cast_58 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64, 16], strides: [1, %23] : memref<?xf16> to memref<64x16xf16, strided<[1, ?], offset: ?>>
      %cast_59 = memref.cast %reinterpret_cast_58 : memref<64x16xf16, strided<[1, ?], offset: ?>> to memref<64x16xf16, strided<[?, ?], offset: ?>>
      %104 = arith.addi %arg27, %101 : index
      %105 = arith.addi %104, %arg28 : index
      %reinterpret_cast_60 = memref.reinterpret_cast %arg5 to offset: [%105], sizes: [16, 64], strides: [%23, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
      %cast_61 = memref.cast %reinterpret_cast_60 : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<16x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %90, %99, %100, %cast_59, %cast_61, %103, %c0, %105, %c0 : tensor<128x64xf32>, tensor<128x64xf32>, i32, memref<64x16xf16, strided<[?, ?], offset: ?>>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %33 = arith.addi %18, %c128_i32 : i32
    %34 = arith.subi %arg15, %33 : i32
    %35 = arith.divsi %34, %c32_i32 : i32
    %36 = arith.index_cast %33 : i32 to index
    %37 = arith.muli %36, %23 : index
    %38 = arith.addi %16, %37 : index
    %39 = arith.muli %arg13, %c32_i32 : i32
    %reinterpret_cast_11 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [64, 32], strides: [1, %23] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
    %cast_12 = memref.cast %reinterpret_cast_11 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg5 to offset: [%38], sizes: [32, 64], strides: [%23, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1], offset: ?>>
    %cast_14 = memref.cast %reinterpret_cast_13 : memref<32x64xf16, strided<[?, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
    // CHECK: tensor.empty
    // CHECK: hivm.hir.fixpipe
    // CHECK: tensor.empty
    // CHECK: hivm.hir.fixpipe
    %40:9 = scf.for %arg19 = %c0_i32 to %35 step %c1_i32 iter_args(%arg20 = %32#0, %arg21 = %32#1, %arg22 = %33, %arg23 = %cast_12, %arg24 = %cast_14, %arg25 = %38, %arg26 = %c0, %arg27 = %38, %arg28 = %c0) -> (tensor<128x64xf32>, tensor<128x64xf32>, i32, memref<64x32xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_39 = memref.alloc() : memref<64x32xf16>
      hivm.hir.load ins(%arg23 : memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%alloc_39 : memref<64x32xf16>)
      %65 = bufferization.to_tensor %alloc_39 restrict writable : memref<64x32xf16>
      %66 = arith.index_cast %arg22 : i32 to index
      %67 = arith.addi %17, %66 : index
      %reinterpret_cast_40 = memref.reinterpret_cast %arg9 to offset: [%67], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      %alloc_41 = memref.alloc() : memref<32xf32>
      hivm.hir.load ins(%reinterpret_cast_40 : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc_41 : memref<32xf32>)
      %68 = bufferization.to_tensor %alloc_41 restrict writable : memref<32xf32>
      %69 = tensor.empty() : tensor<128x32xf32>
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %70 = hivm.hir.mmadL1 ins(%26, %65, %true, %c128, %c64, %c32 : tensor<128x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%69 : tensor<128x32xf32>) -> tensor<128x32xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %expanded_42 = tensor.expand_shape %68 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
      %71 = hivm.hir.vbrc ins(%expanded_42 : tensor<1x32xf32>) outs(%6 : tensor<128x32xf32>) broadcast_dims = [0] -> tensor<128x32xf32>
      %72 = hivm.hir.vsub ins(%70, %71 : tensor<128x32xf32>, tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %73 = hivm.hir.vmul ins(%72, %cst_0 : tensor<128x32xf32>, f32) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %74 = hivm.hir.vexp ins(%73 : tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %alloc_43 = memref.alloc() : memref<32x64xf16>
      hivm.hir.load ins(%arg24 : memref<32x64xf16, strided<[?, ?], offset: ?>>) outs(%alloc_43 : memref<32x64xf16>)
      %75 = bufferization.to_tensor %alloc_43 restrict writable : memref<32x64xf16>
      %76 = tensor.empty() : tensor<128x32xf16>
      %77 = hivm.hir.vcast ins(%74 : tensor<128x32xf32>) outs(%76 : tensor<128x32xf16>) round_mode = <rint> -> tensor<128x32xf16>
      %78 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %79 = tensor.empty() : tensor<128x64xf32>
      %c128_44 = arith.constant 128 : index
      %c32_45 = arith.constant 32 : index
      %c64_46 = arith.constant 64 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %80 = hivm.hir.mmadL1 ins(%77, %75, %78, %c128_44, %c32_45, %c64_46 : tensor<128x32xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%79 : tensor<128x64xf32>) -> tensor<128x64xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %81 = tensor.empty() : tensor<128x64xf32>
      %82 = hivm.hir.vadd ins(%80, %arg20 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%81 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %reinterpret_cast_47 = memref.reinterpret_cast %arg10 to offset: [%67], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      %alloc_48 = memref.alloc() : memref<32xf32>
      hivm.hir.load ins(%reinterpret_cast_47 : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc_48 : memref<32xf32>)
      %83 = bufferization.to_tensor %alloc_48 restrict writable : memref<32xf32>
      %84 = tensor.empty() : tensor<128x32xf32>
      %c128_49 = arith.constant 128 : index
      %c64_50 = arith.constant 64 : index
      %c32_51 = arith.constant 32 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %85 = hivm.hir.mmadL1 {b_transpose} ins(%27, %75, %true, %c128_49, %c64_50, %c32_51 : tensor<128x64xf16>, tensor<32x64xf16>, i1, index, index, index) outs(%84 : tensor<128x32xf32>) -> tensor<128x32xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %expanded_52 = tensor.expand_shape %83 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
      %86 = hivm.hir.vbrc ins(%expanded_52 : tensor<1x32xf32>) outs(%6 : tensor<128x32xf32>) broadcast_dims = [0] -> tensor<128x32xf32>
      %87 = hivm.hir.vsub ins(%85, %86 : tensor<128x32xf32>, tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %88 = hivm.hir.vmul ins(%74, %87 : tensor<128x32xf32>, tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %89 = hivm.hir.vcast ins(%88 : tensor<128x32xf32>) outs(%76 : tensor<128x32xf16>) round_mode = <rint> -> tensor<128x32xf16>
      %90 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %91 = tensor.empty() : tensor<128x64xf32>
      %c128_53 = arith.constant 128 : index
      %c32_54 = arith.constant 32 : index
      %c64_55 = arith.constant 64 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %92 = hivm.hir.mmadL1 {b_transpose} ins(%89, %65, %90, %c128_53, %c32_54, %c64_55 : tensor<128x32xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%91 : tensor<128x64xf32>) -> tensor<128x64xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %93 = tensor.empty() : tensor<128x64xf32>
      %94 = hivm.hir.vadd ins(%92, %arg21 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%93 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %95 = arith.addi %arg22, %c32_i32 : i32
      %96 = arith.index_cast %39 : i32 to index
      %97 = arith.addi %arg25, %96 : index
      %98 = arith.addi %97, %arg26 : index
      %reinterpret_cast_56 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [64, 32], strides: [1, %23] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
      %cast_57 = memref.cast %reinterpret_cast_56 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
      %99 = arith.addi %arg27, %96 : index
      %100 = arith.addi %99, %arg28 : index
      %reinterpret_cast_58 = memref.reinterpret_cast %arg5 to offset: [%100], sizes: [32, 64], strides: [%23, 1] : memref<?xf16> to memref<32x64xf16, strided<[?, 1], offset: ?>>
      %cast_59 = memref.cast %reinterpret_cast_58 : memref<32x64xf16, strided<[?, 1], offset: ?>> to memref<32x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %82, %94, %95, %cast_57, %cast_59, %98, %c0, %100, %c0 : tensor<128x64xf32>, tensor<128x64xf32>, i32, memref<64x32xf16, strided<[?, ?], offset: ?>>, memref<32x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [%25], sizes: [128, 64], strides: [%23, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
    %41 = tensor.empty() : tensor<128x64xf16>
    %42 = hivm.hir.vcast ins(%40#0 : tensor<128x64xf32>) outs(%41 : tensor<128x64xf16>) round_mode = <rint> -> tensor<128x64xf16>
    hivm.hir.store ins(%42 : tensor<128x64xf16>) outs(%reinterpret_cast_15 : memref<128x64xf16, strided<[?, 1], offset: ?>>)
    %43 = hivm.hir.vmul ins(%40#1, %arg4 : tensor<128x64xf32>, f32) outs(%9 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg7 to offset: [%25], sizes: [128, 64], strides: [%23, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
    %44 = hivm.hir.vcast ins(%43 : tensor<128x64xf32>) outs(%41 : tensor<128x64xf16>) round_mode = <rint> -> tensor<128x64xf16>
    hivm.hir.store ins(%44 : tensor<128x64xf16>) outs(%reinterpret_cast_16 : memref<128x64xf16, strided<[?, 1], offset: ?>>)
    %reinterpret_cast_17 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [128, 64], strides: [%23, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
    %alloc_18 = memref.alloc() : memref<128x64xf16>
    hivm.hir.load ins(%reinterpret_cast_17 : memref<128x64xf16, strided<[?, 1], offset: ?>>) outs(%alloc_18 : memref<128x64xf16>)
    %45 = bufferization.to_tensor %alloc_18 restrict writable : memref<128x64xf16>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [128, 64], strides: [%23, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
    %alloc_20 = memref.alloc() : memref<128x64xf16>
    hivm.hir.load ins(%reinterpret_cast_19 : memref<128x64xf16, strided<[?, 1], offset: ?>>) outs(%alloc_20 : memref<128x64xf16>)
    %46 = bufferization.to_tensor %alloc_20 restrict writable : memref<128x64xf16>
    %47 = arith.addi %17, %22 : index
    %reinterpret_cast_21 = memref.reinterpret_cast %arg9 to offset: [%47], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc_22 = memref.alloc() : memref<128xf32>
    hivm.hir.load ins(%reinterpret_cast_21 : memref<128xf32, strided<[1], offset: ?>>) outs(%alloc_22 : memref<128xf32>)
    %48 = bufferization.to_tensor %alloc_22 restrict writable : memref<128xf32>
    %reinterpret_cast_23 = memref.reinterpret_cast %arg10 to offset: [%47], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc_24 = memref.alloc() : memref<128xf32>
    hivm.hir.load ins(%reinterpret_cast_23 : memref<128xf32, strided<[1], offset: ?>>) outs(%alloc_24 : memref<128xf32>)
    %49 = bufferization.to_tensor %alloc_24 restrict writable : memref<128xf32>
    %expanded_25 = tensor.expand_shape %48 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %50 = hivm.hir.vbrc ins(%expanded_25 : tensor<128x1xf32>) outs(%7 : tensor<128x16xf32>) broadcast_dims = [1] -> tensor<128x16xf32>
    %expanded_26 = tensor.expand_shape %49 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %51 = hivm.hir.vbrc ins(%expanded_26 : tensor<128x1xf32>) outs(%7 : tensor<128x16xf32>) broadcast_dims = [1] -> tensor<128x16xf32>
    %reinterpret_cast_27 = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 16], strides: [1, %23] : memref<?xf16> to memref<64x16xf16, strided<[1, ?], offset: ?>>
    %cast_28 = memref.cast %reinterpret_cast_27 : memref<64x16xf16, strided<[1, ?], offset: ?>> to memref<64x16xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_29 = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [64, 16], strides: [1, %23] : memref<?xf16> to memref<64x16xf16, strided<[1, ?], offset: ?>>
    %cast_30 = memref.cast %reinterpret_cast_29 : memref<64x16xf16, strided<[1, ?], offset: ?>> to memref<64x16xf16, strided<[?, ?], offset: ?>>
    %52:8 = scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg20 = %31, %arg21 = %18, %arg22 = %cast_28, %arg23 = %cast_30, %arg24 = %25, %arg25 = %c0, %arg26 = %25, %arg27 = %c0) -> (tensor<128x64xf32>, i32, memref<64x16xf16, strided<[?, ?], offset: ?>>, memref<64x16xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_39 = memref.alloc() : memref<64x16xf16>
      hivm.hir.load ins(%arg22 : memref<64x16xf16, strided<[?, ?], offset: ?>>) outs(%alloc_39 : memref<64x16xf16>)
      %65 = bufferization.to_tensor %alloc_39 restrict writable : memref<64x16xf16>
      %alloc_40 = memref.alloc() : memref<64x16xf16>
      hivm.hir.load ins(%arg23 : memref<64x16xf16, strided<[?, ?], offset: ?>>) outs(%alloc_40 : memref<64x16xf16>)
      %66 = bufferization.to_tensor %alloc_40 restrict writable : memref<64x16xf16>
      %67 = tensor.empty() : tensor<128x16xf32>
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %68 = hivm.hir.mmadL1 ins(%45, %65, %true, %c128, %c64, %c16 : tensor<128x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%67 : tensor<128x16xf32>) -> tensor<128x16xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %69 = hivm.hir.vsub ins(%68, %50 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %70 = hivm.hir.vmul ins(%69, %cst_0 : tensor<128x16xf32>, f32) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %71 = hivm.hir.vexp ins(%70 : tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %72 = hivm.hir.vadd ins(%expanded_7, %arg21 : tensor<1x16xi32>, i32) outs(%expanded_5 : tensor<1x16xi32>) -> tensor<1x16xi32>
      %73 = tensor.empty() : tensor<128x16xi1>
      %74 = tensor.empty() : tensor<128xi64>
      %expanded_41 = tensor.expand_shape %74 [[0, 1]] output_shape [128, 1] : tensor<128xi64> into tensor<128x1xi64>
      %75 = hivm.hir.vcast ins(%21 : tensor<128x1xi32>) outs(%expanded_41 : tensor<128x1xi64>) -> tensor<128x1xi64>
      %76 = tensor.empty() : tensor<128x16xi64>
      %77 = hivm.hir.vbrc ins(%75 : tensor<128x1xi64>) outs(%76 : tensor<128x16xi64>) broadcast_dims = [1] -> tensor<128x16xi64>
      %78 = tensor.empty() : tensor<16xi64>
      %expanded_42 = tensor.expand_shape %78 [[0, 1]] output_shape [1, 16] : tensor<16xi64> into tensor<1x16xi64>
      %79 = hivm.hir.vcast ins(%72 : tensor<1x16xi32>) outs(%expanded_42 : tensor<1x16xi64>) -> tensor<1x16xi64>
      %80 = hivm.hir.vbrc ins(%79 : tensor<1x16xi64>) outs(%76 : tensor<128x16xi64>) broadcast_dims = [0] -> tensor<128x16xi64>
      %81 = hivm.hir.vcmp ins(%77, %80 : tensor<128x16xi64>, tensor<128x16xi64>) outs(%73 : tensor<128x16xi1>) compare_mode = <ge> -> tensor<128x16xi1>
      %82 = hivm.hir.vsel ins(%81, %71, %8 : tensor<128x16xi1>, tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %83 = tensor.empty() : tensor<128x16xf32>
      %c128_43 = arith.constant 128 : index
      %c64_44 = arith.constant 64 : index
      %c16_45 = arith.constant 16 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %84 = hivm.hir.mmadL1 ins(%46, %66, %true, %c128_43, %c64_44, %c16_45 : tensor<128x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%83 : tensor<128x16xf32>) -> tensor<128x16xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %85 = hivm.hir.vsub ins(%84, %51 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %86 = hivm.hir.vmul ins(%82, %85 : tensor<128x16xf32>, tensor<128x16xf32>) outs(%7 : tensor<128x16xf32>) -> tensor<128x16xf32>
      %87 = tensor.empty() : tensor<128x16xf16>
      %88 = hivm.hir.vcast ins(%86 : tensor<128x16xf32>) outs(%87 : tensor<128x16xf16>) round_mode = <rint> -> tensor<128x16xf16>
      %89 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %c128_46 = arith.constant 128 : index
      %c16_47 = arith.constant 16 : index
      %c64_48 = arith.constant 64 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %90 = hivm.hir.mmadL1 {b_transpose} ins(%88, %65, %89, %c128_46, %c16_47, %c64_48 : tensor<128x16xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%arg20 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %91 = arith.addi %arg21, %c16_i32 : i32
      %92 = arith.index_cast %30 : i32 to index
      %93 = arith.addi %arg24, %92 : index
      %94 = arith.addi %93, %arg25 : index
      %reinterpret_cast_49 = memref.reinterpret_cast %arg2 to offset: [%94], sizes: [64, 16], strides: [1, %23] : memref<?xf16> to memref<64x16xf16, strided<[1, ?], offset: ?>>
      %cast_50 = memref.cast %reinterpret_cast_49 : memref<64x16xf16, strided<[1, ?], offset: ?>> to memref<64x16xf16, strided<[?, ?], offset: ?>>
      %95 = arith.addi %arg26, %92 : index
      %96 = arith.addi %95, %arg27 : index
      %reinterpret_cast_51 = memref.reinterpret_cast %arg3 to offset: [%96], sizes: [64, 16], strides: [1, %23] : memref<?xf16> to memref<64x16xf16, strided<[1, ?], offset: ?>>
      %cast_52 = memref.cast %reinterpret_cast_51 : memref<64x16xf16, strided<[1, ?], offset: ?>> to memref<64x16xf16, strided<[?, ?], offset: ?>>
      scf.yield %90, %91, %cast_50, %cast_52, %94, %c0, %96, %c0 : tensor<128x64xf32>, i32, memref<64x16xf16, strided<[?, ?], offset: ?>>, memref<64x16xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %53 = arith.divsi %18, %c32_i32 : i32
    %54 = arith.muli %53, %c32_i32 : i32
    %55 = arith.subi %18, %54 : i32
    %56 = arith.index_cast %55 : i32 to index
    %57 = arith.muli %56, %23 : index
    %58 = arith.addi %16, %57 : index
    %alloc_31 = memref.alloc() : memref<128xf32>
    hivm.hir.load ins(%reinterpret_cast_23 : memref<128xf32, strided<[1], offset: ?>>) outs(%alloc_31 : memref<128xf32>)
    %59 = bufferization.to_tensor %alloc_31 restrict writable : memref<128xf32>
    %expanded_32 = tensor.expand_shape %48 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %60 = hivm.hir.vbrc ins(%expanded_32 : tensor<128x1xf32>) outs(%6 : tensor<128x32xf32>) broadcast_dims = [1] -> tensor<128x32xf32>
    %expanded_33 = tensor.expand_shape %59 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %61 = hivm.hir.vbrc ins(%expanded_33 : tensor<128x1xf32>) outs(%6 : tensor<128x32xf32>) broadcast_dims = [1] -> tensor<128x32xf32>
    %reinterpret_cast_34 = memref.reinterpret_cast %arg2 to offset: [%58], sizes: [64, 32], strides: [1, %23] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
    %cast_35 = memref.cast %reinterpret_cast_34 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_36 = memref.reinterpret_cast %arg3 to offset: [%58], sizes: [64, 32], strides: [1, %23] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
    %cast_37 = memref.cast %reinterpret_cast_36 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
    // CHECK: tensor.empty
    // CHECK: hivm.hir.fixpipe
    %62:8 = scf.for %arg19 = %c0_i32 to %53 step %c1_i32 iter_args(%arg20 = %52#0, %arg21 = %55, %arg22 = %cast_35, %arg23 = %cast_37, %arg24 = %58, %arg25 = %c0, %arg26 = %58, %arg27 = %c0) -> (tensor<128x64xf32>, i32, memref<64x32xf16, strided<[?, ?], offset: ?>>, memref<64x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_39 = memref.alloc() : memref<64x32xf16>
      hivm.hir.load ins(%arg22 : memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%alloc_39 : memref<64x32xf16>)
      %65 = bufferization.to_tensor %alloc_39 restrict writable : memref<64x32xf16>
      %alloc_40 = memref.alloc() : memref<64x32xf16>
      hivm.hir.load ins(%arg23 : memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%alloc_40 : memref<64x32xf16>)
      %66 = bufferization.to_tensor %alloc_40 restrict writable : memref<64x32xf16>
      %67 = tensor.empty() : tensor<128x32xf32>
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %68 = hivm.hir.mmadL1 ins(%45, %65, %true, %c128, %c64, %c32 : tensor<128x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%67 : tensor<128x32xf32>) -> tensor<128x32xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %69 = hivm.hir.vsub ins(%68, %60 : tensor<128x32xf32>, tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %70 = hivm.hir.vmul ins(%69, %cst_0 : tensor<128x32xf32>, f32) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %71 = hivm.hir.vexp ins(%70 : tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %72 = tensor.empty() : tensor<128x32xf32>
      %c128_41 = arith.constant 128 : index
      %c64_42 = arith.constant 64 : index
      %c32_43 = arith.constant 32 : index
      // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
      %73 = hivm.hir.mmadL1 ins(%46, %66, %true, %c128_41, %c64_42, %c32_43 : tensor<128x64xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%72 : tensor<128x32xf32>) -> tensor<128x32xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %74 = hivm.hir.vsub ins(%73, %61 : tensor<128x32xf32>, tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %75 = hivm.hir.vmul ins(%71, %74 : tensor<128x32xf32>, tensor<128x32xf32>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %76 = tensor.empty() : tensor<128x32xf16>
      %77 = hivm.hir.vcast ins(%75 : tensor<128x32xf32>) outs(%76 : tensor<128x32xf16>) round_mode = <rint> -> tensor<128x32xf16>
      %78 = arith.cmpi eq, %arg19, %c0_i32 : i32
      %79 = tensor.empty() : tensor<128x64xf32>
      %c128_44 = arith.constant 128 : index
      %c32_45 = arith.constant 32 : index
      %c64_46 = arith.constant 64 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %80 = hivm.hir.mmadL1 {b_transpose} ins(%77, %65, %78, %c128_44, %c32_45, %c64_46 : tensor<128x32xf16>, tensor<64x32xf16>, i1, index, index, index) outs(%79 : tensor<128x64xf32>) -> tensor<128x64xf32>
      // CHECK: tensor.empty
      // CHECK: hivm.hir.fixpipe
      %81 = tensor.empty() : tensor<128x64xf32>
      %82 = hivm.hir.vadd ins(%80, %arg20 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%81 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %83 = arith.addi %arg21, %c32_i32 : i32
      %84 = arith.index_cast %39 : i32 to index
      %85 = arith.addi %arg24, %84 : index
      %86 = arith.addi %85, %arg25 : index
      %reinterpret_cast_47 = memref.reinterpret_cast %arg2 to offset: [%86], sizes: [64, 32], strides: [1, %23] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
      %cast_48 = memref.cast %reinterpret_cast_47 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
      %87 = arith.addi %arg26, %84 : index
      %88 = arith.addi %87, %arg27 : index
      %reinterpret_cast_49 = memref.reinterpret_cast %arg3 to offset: [%88], sizes: [64, 32], strides: [1, %23] : memref<?xf16> to memref<64x32xf16, strided<[1, ?], offset: ?>>
      %cast_50 = memref.cast %reinterpret_cast_49 : memref<64x32xf16, strided<[1, ?], offset: ?>> to memref<64x32xf16, strided<[?, ?], offset: ?>>
      scf.yield %82, %83, %cast_48, %cast_50, %86, %c0, %88, %c0 : tensor<128x64xf32>, i32, memref<64x32xf16, strided<[?, ?], offset: ?>>, memref<64x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %reinterpret_cast_38 = memref.reinterpret_cast %arg6 to offset: [%25], sizes: [128, 64], strides: [%23, 1] : memref<?xf16> to memref<128x64xf16, strided<[?, 1], offset: ?>>
    %63 = hivm.hir.vmul ins(%62#0, %cst_0 : tensor<128x64xf32>, f32) outs(%9 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %64 = hivm.hir.vcast ins(%63 : tensor<128x64xf32>) outs(%41 : tensor<128x64xf16>) round_mode = <rint> -> tensor<128x64xf16>
    hivm.hir.store ins(%64 : tensor<128x64xf16>) outs(%reinterpret_cast_38 : memref<128x64xf16, strided<[?, 1], offset: ?>>)
    return
  }
}

// -----
func.func @mm_with_add(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %true = arith.constant true
  %cst = arith.constant 0.000000e+00 : f32
  %c256_i32 = arith.constant 256 : i32
  %c768_i32 = arith.constant 768 : i32
  %c0_i32 = arith.constant 0 : i32
  %c29_i32 = arith.constant 29 : i32
  %c128 = arith.constant 128 : index
  %c768 = arith.constant 768 : index
  %c29 = arith.constant 29 : index
  %c86 = arith.constant 86 : index
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %arg8, %arg7 : i32
  %3 = arith.divsi %1, %2 : i32
  %4 = arith.remsi %3, %arg6 : i32
  hivm.hir.set_mask_norm
  %5 = tensor.empty() : tensor<29x768xf32>
  %6 = hivm.hir.vbrc ins(%cst : f32) outs(%5 : tensor<29x768xf32>) -> tensor<29x768xf32>
  %7 = arith.muli %4, %c29_i32 : i32
  %8 = arith.index_cast %7 : i32 to index
  %9 = arith.muli %8, %c128 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [29, 128], strides: [128, 1] : memref<?xf16> to memref<29x128xf16, strided<[128, 1], offset: ?>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<?xf16> to memref<128x768xf16, strided<[768, 1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [768], strides: [1] : memref<?xf32> to memref<768xf32, strided<[1]>>
  %10 = arith.muli %8, %c768 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%10], sizes: [29, 768], strides: [768, 1] : memref<?xf32> to memref<29x768xf32, strided<[768, 1], offset: ?>>
  %alloc = memref.alloc() : memref<29x128xf16>
  hivm.hir.load ins(%reinterpret_cast : memref<29x128xf16, strided<[128, 1], offset: ?>>) outs(%alloc : memref<29x128xf16>)
  %11 = bufferization.to_tensor %alloc restrict writable : memref<29x128xf16>
  %alloc_3 = memref.alloc() : memref<128x768xf16>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<128x768xf16, strided<[768, 1]>>) outs(%alloc_3 : memref<128x768xf16>)
  %12 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x768xf16>
  %alloc_4 = memref.alloc() : memref<768xf32>
  hivm.hir.load ins(%reinterpret_cast_1 : memref<768xf32, strided<[1]>>) outs(%alloc_4 : memref<768xf32>)
  %13 = bufferization.to_tensor %alloc_4 restrict writable : memref<768xf32>
  %14 = scf.for %arg9 = %c0_i32 to %c768_i32 step %c256_i32 iter_args(%arg10 = %6) -> (tensor<29x768xf32>)  : i32 {
    %20 = arith.index_cast %arg9 : i32 to index
    %extracted_slice_5 = tensor.extract_slice %12[0, %20] [128, 256] [1, 1] : tensor<128x768xf16> to tensor<128x256xf16>
    %expanded = tensor.expand_shape %13 [[0, 1]] output_shape [1, 768] : tensor<768xf32> into tensor<1x768xf32>
    %extracted_slice_6 = tensor.extract_slice %expanded[0, %20] [1, 256] [1, 1] : tensor<1x768xf32> to tensor<1x256xf32>
    %21 = tensor.empty() : tensor<29x256xf32>
    %c29_7 = arith.constant 29 : index
    %c128_8 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
    %22 = hivm.hir.mmadL1 ins(%11, %extracted_slice_5, %true, %c29_7, %c128_8, %c256, %extracted_slice_6 : tensor<29x128xf16>, tensor<128x256xf16>, i1, index, index, index, tensor<1x256xf32>) outs(%21 : tensor<29x256xf32>) -> tensor<29x256xf32>
    // CHECK-NOT: hivm.hir.fixpipe
    %inserted_slice = tensor.insert_slice %22 into %arg10[0, %20] [29, 256] [1, 1] : tensor<29x256xf32> into tensor<29x768xf32>
    scf.yield %inserted_slice : tensor<29x768xf32>
  }
  %15 = arith.addi %8, %c29 : index
  %16 = arith.maxsi %8, %c86 : index
  %17 = arith.minsi %15, %16 : index
  %18 = arith.subi %17, %8 : index
  %19 = arith.minsi %18, %c29 : index
  %extracted_slice = tensor.extract_slice %14[0, 0] [%19, 768] [1, 1] : tensor<29x768xf32> to tensor<?x768xf32>
  %subview = memref.subview %reinterpret_cast_2[0, 0] [%19, 768] [1, 1] : memref<29x768xf32, strided<[768, 1], offset: ?>> to memref<?x768xf32, strided<[768, 1], offset: ?>>
  // CHECK: hivm.hir.fixpipe
  hivm.hir.store ins(%extracted_slice : tensor<?x768xf32>) outs(%subview : memref<?x768xf32, strided<[768, 1], offset: ?>>)
  return
}


// -----
func.func @mmad_mmad_fixpipe_case(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32}, %arg11: f32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %cst = arith.constant 0.693147182 : f32
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128_i32 = arith.constant 128 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c96_i32 = arith.constant 96 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<32x32xf16>
    %1 = hivm.hir.vbrc ins(%cst_1 : f16) outs(%0 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %2 = tensor.empty() : tensor<32x16xf32>
    %3 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = arith.muli %arg17, %c128_i32 : i32
    %7 = arith.muli %arg17, %c4096_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.index_cast %6 : i32 to index
    %10 = tensor.empty() : tensor<32xi32>
    %11 = hivm.hir.varange offset[%c0] strides[%c1] outs(%10 : tensor<32xi32>) -> tensor<32xi32>
    %12 = arith.muli %arg15, %c32_i32 : i32
    %13 = hivm.hir.vadd ins(%11, %12 : tensor<32xi32>, i32) outs(%10 : tensor<32xi32>) -> tensor<32xi32>
    %14 = arith.index_cast %12 : i32 to index
    %15 = arith.muli %14, %c32 : index
    %16 = arith.addi %8, %15 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %alloc = memref.alloc() : memref<32x32xf16>
    memref.copy %reinterpret_cast, %alloc : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
    %17 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<32x32xf16>
    memref.copy %reinterpret_cast_2, %alloc_3 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
    %18 = bufferization.to_tensor %alloc_3 restrict writable : memref<32x32xf16>
    %19 = tensor.empty() : tensor<16xi32>
    %20 = hivm.hir.varange offset[%c0] strides[%c1] outs(%19 : tensor<16xi32>) -> tensor<16xi32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [16, 32], strides: [32, 1] : memref<?xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast_4 : memref<16x32xf16, strided<[32, 1], offset: ?>> to memref<16x32xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%16], sizes: [16, 32], strides: [32, 1] : memref<?xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
    %cast_6 = memref.cast %reinterpret_cast_5 : memref<16x32xf16, strided<[32, 1], offset: ?>> to memref<16x32xf16, strided<[?, ?], offset: ?>>
    %21 = tensor.empty() : tensor<32x32xf32>
    %22 = tensor.empty() : tensor<32x32xf32>
    %23:9 = scf.for %arg18 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg19 = %21, %arg20 = %22, %arg21 = %12, %arg22 = %cast, %arg23 = %cast_6, %arg24 = %16, %arg25 = %c0, %arg26 = %16, %arg27 = %c0) -> (tensor<32x32xf32>, tensor<32x32xf32>, i32, memref<16x32xf16, strided<[?, ?], offset: ?>>, memref<16x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_14 = memref.alloc() : memref<16x32xf16>
      memref.copy %arg22, %alloc_14 : memref<16x32xf16, strided<[?, ?], offset: ?>> to memref<16x32xf16>
      %34 = bufferization.to_tensor %alloc_14 restrict writable : memref<16x32xf16>
      %35 = tensor.empty() : tensor<32x16xf16>
      %36 = hivm.hir.vtranspose ins(%34 : tensor<16x32xf16>) outs(%35 : tensor<32x16xf16>) permutation = [1, 0] -> tensor<32x16xf16>
      %37 = hivm.hir.vadd ins(%20, %arg21 : tensor<16xi32>, i32) outs(%19 : tensor<16xi32>) -> tensor<16xi32>
      %38 = arith.index_cast %arg21 : i32 to index
      %39 = arith.addi %9, %38 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg9 to offset: [%39], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %alloc_16 = memref.alloc() : memref<16xf32>
      memref.copy %reinterpret_cast_15, %alloc_16 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32>
      %40 = bufferization.to_tensor %alloc_16 restrict writable : memref<16xf32>
      %true = arith.constant true
      %41 = tensor.empty() : tensor<32x16xf32>
      %c32_17 = arith.constant 32 : index
      %c32_18 = arith.constant 32 : index
      %c0_19 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32_20 = arith.constant 32 : index
      %42 = hivm.hir.mmadL1 {b_transpose} ins(%17, %34, %true, %c32_17, %c32_18, %c16 : tensor<32x32xf16>, tensor<16x32xf16>, i1, index, index, index) outs(%41 : tensor<32x16xf32>) -> tensor<32x16xf32>
      // CHECK: hivm.hir.fixpipe
      %expanded = tensor.expand_shape %40 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
      %43 = hivm.hir.vbrc ins(%expanded : tensor<1x16xf32>) outs(%2 : tensor<32x16xf32>) broadcast_dims = [0] -> tensor<32x16xf32>
      %44 = hivm.hir.vsub ins(%42, %43 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %45 = hivm.hir.vmul ins(%44, %cst : tensor<32x16xf32>, f32) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %46 = hivm.hir.vexp ins(%45 : tensor<32x16xf32>) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %47 = tensor.empty() : tensor<32x16xi1>
      %48 = tensor.empty() : tensor<16xi64>
      %49 = hivm.hir.vcast ins(%37 : tensor<16xi32>) outs(%48 : tensor<16xi64>) -> tensor<16xi64>
      %50 = tensor.empty() : tensor<32x16xi64>
      %expanded_21 = tensor.expand_shape %49 [[0, 1]] output_shape [1, 16] : tensor<16xi64> into tensor<1x16xi64>
      %51 = hivm.hir.vbrc ins(%expanded_21 : tensor<1x16xi64>) outs(%50 : tensor<32x16xi64>) broadcast_dims = [0] -> tensor<32x16xi64>
      %52 = tensor.empty() : tensor<32xi64>
      %53 = hivm.hir.vcast ins(%13 : tensor<32xi32>) outs(%52 : tensor<32xi64>) -> tensor<32xi64>
      %expanded_22 = tensor.expand_shape %53 [[0, 1]] output_shape [32, 1] : tensor<32xi64> into tensor<32x1xi64>
      %54 = hivm.hir.vbrc ins(%expanded_22 : tensor<32x1xi64>) outs(%50 : tensor<32x16xi64>) broadcast_dims = [1] -> tensor<32x16xi64>
      %55 = hivm.hir.vcmp ins(%51, %54 : tensor<32x16xi64>, tensor<32x16xi64>) outs(%47 : tensor<32x16xi1>) compare_mode = <ge> -> tensor<32x16xi1>
      %56 = hivm.hir.vsel ins(%55, %46, %3 : tensor<32x16xi1>, tensor<32x16xf32>, tensor<32x16xf32>) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %alloc_23 = memref.alloc() : memref<16x32xf16>
      memref.copy %arg23, %alloc_23 : memref<16x32xf16, strided<[?, ?], offset: ?>> to memref<16x32xf16>
      %57 = bufferization.to_tensor %alloc_23 restrict writable : memref<16x32xf16>
      %58 = hivm.hir.vcast ins(%56 : tensor<32x16xf32>) outs(%35 : tensor<32x16xf16>) round_mode = <rint> -> tensor<32x16xf16>
      %true_24 = arith.constant true
      %59 = arith.cmpi eq, %c0_i32, %arg18 : i32
      %60 = arith.andi %true_24, %59 : i1
      %c0_25 = arith.constant 0 : index
      %61 = hivm.hir.mmadL1 ins(%58, %57, %60, %c0_25, %c0_25, %c0_25 : tensor<32x16xf16>, tensor<16x32xf16>, i1, index, index, index) outs(%arg19 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %reinterpret_cast_26 = memref.reinterpret_cast %arg10 to offset: [%39], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %alloc_27 = memref.alloc() : memref<16xf32>
      memref.copy %reinterpret_cast_26, %alloc_27 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32>
      %62 = bufferization.to_tensor %alloc_27 restrict writable : memref<16xf32>
      %63 = hivm.hir.vtranspose ins(%57 : tensor<16x32xf16>) outs(%35 : tensor<32x16xf16>) permutation = [1, 0] -> tensor<32x16xf16>
      %true_28 = arith.constant true
      %64 = tensor.empty() : tensor<32x16xf32>
      %c32_29 = arith.constant 32 : index
      %c32_30 = arith.constant 32 : index
      %c0_31 = arith.constant 0 : index
      %c16_32 = arith.constant 16 : index
      %c32_33 = arith.constant 32 : index
      %65 = hivm.hir.mmadL1 {b_transpose} ins(%18, %57, %true_28, %c32_29, %c32_30, %c16_32 : tensor<32x32xf16>, tensor<16x32xf16>, i1, index, index, index) outs(%64 : tensor<32x16xf32>) -> tensor<32x16xf32>
      // CHECK: hivm.hir.fixpipe
      %expanded_34 = tensor.expand_shape %62 [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
      %66 = hivm.hir.vbrc ins(%expanded_34 : tensor<1x16xf32>) outs(%2 : tensor<32x16xf32>) broadcast_dims = [0] -> tensor<32x16xf32>
      %67 = hivm.hir.vsub ins(%65, %66 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %68 = hivm.hir.vmul ins(%56, %67 : tensor<32x16xf32>, tensor<32x16xf32>) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
      %69 = hivm.hir.vcast ins(%68 : tensor<32x16xf32>) outs(%35 : tensor<32x16xf16>) round_mode = <rint> -> tensor<32x16xf16>
      %true_35 = arith.constant true
      %70 = arith.cmpi eq, %c0_i32, %arg18 : i32
      %71 = arith.andi %true_35, %70 : i1
      %c0_36 = arith.constant 0 : index
      %72 = hivm.hir.mmadL1 ins(%69, %34, %71, %c0_36, %c0_36, %c0_36 : tensor<32x16xf16>, tensor<16x32xf16>, i1, index, index, index) outs(%arg20 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %73 = arith.addi %arg21, %c16_i32 : i32
      %74 = arith.addi %arg24, %c512 : index
      %75 = arith.addi %74, %arg25 : index
      %reinterpret_cast_37 = memref.reinterpret_cast %arg2 to offset: [%75], sizes: [16, 32], strides: [32, 1] : memref<?xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
      %cast_38 = memref.cast %reinterpret_cast_37 : memref<16x32xf16, strided<[32, 1], offset: ?>> to memref<16x32xf16, strided<[?, ?], offset: ?>>
      %76 = arith.addi %arg26, %c512 : index
      %77 = arith.addi %76, %arg27 : index
      %reinterpret_cast_39 = memref.reinterpret_cast %arg5 to offset: [%77], sizes: [16, 32], strides: [32, 1] : memref<?xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
      %cast_40 = memref.cast %reinterpret_cast_39 : memref<16x32xf16, strided<[32, 1], offset: ?>> to memref<16x32xf16, strided<[?, ?], offset: ?>>
      scf.yield %61, %72, %73, %cast_38, %cast_40, %75, %c0, %77, %c0 : tensor<32x32xf32>, tensor<32x32xf32>, i32, memref<16x32xf16, strided<[?, ?], offset: ?>>, memref<16x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    // CHECK-NOT: hivm.hir.fixpipe
    // CHECK-NOT: hivm.hir.fixpipe
    %24 = arith.addi %12, %c32_i32 : i32
    %25 = arith.subi %c96_i32, %12 : i32
    %26 = arith.divsi %25, %c32_i32 : i32
    %27 = arith.index_cast %24 : i32 to index
    %28 = arith.muli %27, %c32 : index
    %29 = arith.addi %8, %28 : index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %cast_8 = memref.cast %reinterpret_cast_7 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %cast_10 = memref.cast %reinterpret_cast_9 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
    %30:9 = scf.for %arg18 = %c0_i32 to %26 step %c1_i32 iter_args(%arg19 = %23#0, %arg20 = %23#1, %arg21 = %24, %arg22 = %cast_8, %arg23 = %cast_10, %arg24 = %29, %arg25 = %c0, %arg26 = %29, %arg27 = %c0) -> (tensor<32x32xf32>, tensor<32x32xf32>, i32, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_14 = memref.alloc() : memref<32x32xf16>
      memref.copy %arg22, %alloc_14 : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<32x32xf16>
      %34 = bufferization.to_tensor %alloc_14 restrict writable : memref<32x32xf16>
      %35 = hivm.hir.vtranspose ins(%34 : tensor<32x32xf16>) outs(%0 : tensor<32x32xf16>) permutation = [1, 0] -> tensor<32x32xf16>
      %36 = arith.index_cast %arg21 : i32 to index
      %37 = arith.addi %9, %36 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %arg9 to offset: [%37], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      %alloc_16 = memref.alloc() : memref<32xf32>
      memref.copy %reinterpret_cast_15, %alloc_16 : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
      %38 = bufferization.to_tensor %alloc_16 restrict writable : memref<32xf32>
      %true = arith.constant true
      %39 = tensor.empty() : tensor<32x32xf32>
      %c32_17 = arith.constant 32 : index
      %c32_18 = arith.constant 32 : index
      %c0_19 = arith.constant 0 : index
      %c32_20 = arith.constant 32 : index
      %c32_21 = arith.constant 32 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %40 = hivm.hir.mmadL1 {b_transpose} ins(%17, %34, %true, %c32_17, %c32_18, %c32_20 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%39 : tensor<32x32xf32>) -> tensor<32x32xf32>
      // CHECK: hivm.hir.fixpipe
      %expanded = tensor.expand_shape %38 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
      %41 = hivm.hir.vbrc ins(%expanded : tensor<1x32xf32>) outs(%4 : tensor<32x32xf32>) broadcast_dims = [0] -> tensor<32x32xf32>
      %42 = hivm.hir.vsub ins(%40, %41 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %43 = hivm.hir.vmul ins(%42, %cst : tensor<32x32xf32>, f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %44 = hivm.hir.vexp ins(%43 : tensor<32x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %alloc_22 = memref.alloc() : memref<32x32xf16>
      memref.copy %arg23, %alloc_22 : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<32x32xf16>
      %45 = bufferization.to_tensor %alloc_22 restrict writable : memref<32x32xf16>
      %46 = hivm.hir.vcast ins(%44 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf16>) round_mode = <rint> -> tensor<32x32xf16>
      %true_23 = arith.constant true
      %47 = arith.cmpi eq, %c0_i32, %arg18 : i32
      %48 = arith.andi %true_23, %47 : i1
      %c0_24 = arith.constant 0 : index
      %49 = hivm.hir.mmadL1 ins(%46, %45, %48, %c0_24, %c0_24, %c0_24 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%arg19 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %reinterpret_cast_25 = memref.reinterpret_cast %arg10 to offset: [%37], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      %alloc_26 = memref.alloc() : memref<32xf32>
      memref.copy %reinterpret_cast_25, %alloc_26 : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
      %50 = bufferization.to_tensor %alloc_26 restrict writable : memref<32xf32>
      %51 = hivm.hir.vtranspose ins(%45 : tensor<32x32xf16>) outs(%0 : tensor<32x32xf16>) permutation = [1, 0] -> tensor<32x32xf16>
      %true_27 = arith.constant true
      %52 = tensor.empty() : tensor<32x32xf32>
      %c32_28 = arith.constant 32 : index
      %c32_29 = arith.constant 32 : index
      %c0_30 = arith.constant 0 : index
      %c32_31 = arith.constant 32 : index
      %c32_32 = arith.constant 32 : index
      // CHECK: hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
      %53 = hivm.hir.mmadL1 {b_transpose} ins(%18, %45, %true_27, %c32_28, %c32_29, %c32_31 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%52 : tensor<32x32xf32>) -> tensor<32x32xf32>
      // CHECK: hivm.hir.fixpipe
      %expanded_33 = tensor.expand_shape %50 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
      %54 = hivm.hir.vbrc ins(%expanded_33 : tensor<1x32xf32>) outs(%4 : tensor<32x32xf32>) broadcast_dims = [0] -> tensor<32x32xf32>
      %55 = hivm.hir.vsub ins(%53, %54 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %56 = hivm.hir.vmul ins(%44, %55 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %57 = hivm.hir.vcast ins(%56 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf16>) round_mode = <rint> -> tensor<32x32xf16>
      %true_34 = arith.constant true
      %58 = arith.cmpi eq, %c0_i32, %arg18 : i32
      %59 = arith.andi %true_34, %58 : i1
      %c0_35 = arith.constant 0 : index
      %60 = hivm.hir.mmadL1 ins(%57, %34, %59, %c0_35, %c0_35, %c0_35 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%arg20 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %61 = arith.addi %arg21, %c32_i32 : i32
      %62 = arith.addi %arg24, %c1024 : index
      %63 = arith.addi %62, %arg25 : index
      %reinterpret_cast_36 = memref.reinterpret_cast %arg2 to offset: [%63], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      %cast_37 = memref.cast %reinterpret_cast_36 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      %64 = arith.addi %arg26, %c1024 : index
      %65 = arith.addi %64, %arg27 : index
      %reinterpret_cast_38 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      %cast_39 = memref.cast %reinterpret_cast_38 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      scf.yield %49, %60, %61, %cast_37, %cast_39, %63, %c0, %65, %c0 : tensor<32x32xf32>, tensor<32x32xf32>, i32, memref<32x32xf16, strided<[?, ?], offset: ?>>, memref<32x32xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    // CHECK: hivm.hir.fixpipe
    // CHECK: hivm.hir.fixpipe
    %reinterpret_cast_11 = memref.reinterpret_cast %arg8 to offset: [%16], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %31 = hivm.hir.vcast ins(%30#0 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf16>) round_mode = <rint> -> tensor<32x32xf16>
    bufferization.materialize_in_destination %31 in writable %reinterpret_cast_11 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> ()
    %32 = hivm.hir.vmul ins(%30#1, %arg11 : tensor<32x32xf32>, f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg7 to offset: [%16], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %33 = hivm.hir.vcast ins(%32 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf16>) round_mode = <rint> -> tensor<32x32xf16>
    bufferization.materialize_in_destination %33 in writable %reinterpret_cast_12 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> ()
    %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%16], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast_13 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> ()
    return
  }


// -----
// CHECK-LABEL: func.func @test_mmadL1_fixpipe_no_quant
func.func @test_mmadL1_fixpipe_no_quant(%ma : tensor<256x128xi8>, %mb : tensor<128x256xi8>, %dst : memref<256x256xf32>){

  %mc = tensor.empty() : tensor<256x256xi32>
  %true = arith.constant true
  %M = arith.constant 256 : index
  %K = arith.constant 128 : index
  %N = arith.constant 256 : index
  // CHECK: hivm.hir.mmadL1 {fixpipe_already_inserted = true}
  %ret = hivm.hir.mmadL1 ins(%ma, %mb, %true, %M, %K, %N: tensor<256x128xi8>, tensor<128x256xi8>, i1, index, index, index)
                              outs(%mc: tensor<256x256xi32>) -> tensor<256x256xi32>
  %mc_cast = tensor.empty() : tensor<256x256xf32>
  // CHECK: hivm.hir.fixpipe {enable_nz2nd}
  // CHECK: %[[RET:.*]] = hivm.hir.vcast
  %casted = hivm.hir.vcast ins(%ret : tensor<256x256xi32>) outs(%mc_cast : tensor<256x256xf32>) -> tensor<256x256xf32>
  hivm.hir.store ins(%casted : tensor<256x256xf32>) outs(%dst : memref<256x256xf32>)
  return
}