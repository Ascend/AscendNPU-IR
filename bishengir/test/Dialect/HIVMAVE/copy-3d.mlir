// RUN: bishengir-compile -enable-lir-compile=true -enable-hfusion-compile=true \
// RUN: -block-dim=1 -target=Ascend310B4 -enable-triton-kernel-compile=true \
// RUN: %s -save-linked-ir -o %t1.ll
// RUN: cat %t1_linked.ll | FileCheck %s


// CHECK-LABEL: @load_gm_to_ubuf_3d_float
// CHECK: @_mlir_ciface_load_gm_to_ubuf_3d_float
// CHECK: @_Z23load_gm_to_ubuf_3d_coreIfEvP8memref_tIU3AS1T_Lm3EEPS0_IU3AS6S1_Lm3EE7PadModeN12PadValueTypeIS1_E4typeEl14EvictionPolicy10AtomicKind
// CEHCK: @_Z27load_gm_to_ubuf_intrin_coreIfEvPU3AS1T_lPU3AS6S0_ltjlmj
// CHECK: @_Z48load_gm_to_ubuf_3d_core_with_contiguous_last_dimIfEvP8memref_tIU3AS1T_Lm3EEPS0_IU3AS6S1_Lm3EEl
// CHECK: @_Z48load_gm_to_ubuf_2d_core_with_contiguous_last_dimIfEvP8memref_tIU3AS1T_Lm2EEPS0_IU3AS6S1_Lm2EEl
// CHECK: @_Z48load_gm_to_ubuf_1d_core_with_contiguous_last_dimIfEvP8memref_tIU3AS1T_Lm1EEPS0_IU3AS6S1_Lm1EEl
// CHECK: @_Z27load_gm_to_ubuf_intrin_coreIfEvPU3AS1T_lPU3AS6S0_ltjlmj

module {
  func.func @triton_kernel(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.muli %arg12, %c2_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %arg5 : i32 to index
    %3 = arith.muli %1, %2 : index
    %4 = arith.index_cast %arg6 : i32 to index
    %5 = arith.muli %3, %4 : index
    %6 = arith.muli %2, %4 : index
    %7 = arith.index_cast %arg7 : i32 to index
    %8 = arith.muli %5, %7 : index
    %9 = arith.muli %6, %7 : index
    %10 = arith.index_cast %arg8 : i32 to index
    %11 = arith.muli %8, %10 : index
    %12 = arith.muli %9, %10 : index
    %13 = arith.muli %4, %7 : index
    %14 = arith.muli %13, %10 : index
    %15 = arith.muli %7, %10 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [2, 2, 2, 2, 8], strides: [%12, %14, %15, %10, 1] : memref<?xf32> to memref<2x2x2x2x8xf32, strided<[?, ?, ?, ?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<2x2x2x2x8xf32>
    %16 = arith.addi %1, %c2 : index
    %17 = arith.index_cast %arg4 : i32 to index
    %18 = arith.maxsi %1, %17 : index
    %19 = arith.minsi %16, %18 : index
    %20 = arith.subi %19, %1 : index
    %21 = arith.maxsi %2, %c0 : index
    %22 = arith.minsi %21, %c2 : index
    %23 = arith.minsi %20, %c2 : index
    %24 = arith.minsi %22, %c2 : index
    %25 = arith.maxsi %4, %c0 : index
    %26 = arith.minsi %25, %c2 : index
    %27 = arith.minsi %23, %c2 : index
    %28 = arith.minsi %24, %c2 : index
    %29 = arith.minsi %26, %c2 : index
    %30 = arith.maxsi %7, %c0 : index
    %31 = arith.minsi %30, %c2 : index
    %32 = arith.minsi %27, %c2 : index
    %33 = arith.minsi %28, %c2 : index
    %34 = arith.minsi %29, %c2 : index
    %35 = arith.minsi %31, %c2 : index
    %36 = arith.maxsi %10, %c0 : index
    %37 = arith.minsi %36, %c8 : index
    %38 = arith.minsi %32, %c2 : index
    %39 = arith.minsi %33, %c2 : index
    %40 = arith.minsi %34, %c2 : index
    %41 = arith.minsi %35, %c2 : index
    %42 = arith.minsi %37, %c8 : index
    %subview = memref.subview %reinterpret_cast[0, 0, 0, 0, 0] [%38, %39, %40, %41, %42] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xf32, strided<[?, ?, ?, ?, 1], offset: ?>> to memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, 0, 0, 0, 0] [%38, %39, %40, %41, %42] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xf32> to memref<?x?x?x?x?xf32, strided<[64, 32, 16, 8, 1]>>
    memref.copy %subview, %subview_0 : memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1], offset: ?>> to memref<?x?x?x?x?xf32, strided<[64, 32, 16, 8, 1]>>
    %43 = bufferization.to_tensor %alloc restrict writable : memref<2x2x2x2x8xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [2, 2, 2, 2, 8], strides: [%12, %14, %15, %10, 1] : memref<?xf32> to memref<2x2x2x2x8xf32, strided<[?, ?, ?, ?, 1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<2x2x2x2x8xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0, 0, 0, 0, 0] [%38, %39, %40, %41, %42] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xf32, strided<[?, ?, ?, ?, 1], offset: ?>> to memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0, 0, 0, 0, 0] [%38, %39, %40, %41, %42] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xf32> to memref<?x?x?x?x?xf32, strided<[64, 32, 16, 8, 1]>>
    memref.copy %subview_3, %subview_4 : memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1], offset: ?>> to memref<?x?x?x?x?xf32, strided<[64, 32, 16, 8, 1]>>
    %44 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x2x2x2x8xf32>
    %45 = arith.addf %43, %44 : tensor<2x2x2x2x8xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [2, 2, 2, 2, 8], strides: [%12, %14, %15, %10, 1] : memref<?xf32> to memref<2x2x2x2x8xf32, strided<[?, ?, ?, ?, 1], offset: ?>>
    %extracted_slice = tensor.extract_slice %45[0, 0, 0, 0, 0] [%38, %39, %40, %41, %42] [1, 1, 1, 1, 1] : tensor<2x2x2x2x8xf32> to tensor<?x?x?x?x?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0, 0, 0, 0, 0] [%38, %39, %40, %41, %42] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xf32, strided<[?, ?, ?, ?, 1], offset: ?>> to memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1], offset: ?>>) -> ()
    return
  }
}