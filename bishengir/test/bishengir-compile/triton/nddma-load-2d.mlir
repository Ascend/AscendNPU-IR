// RUN: bishengir-compile -enable-lir-compile=true -enable-hfusion-compile=true \
// RUN: -block-dim=1 -target=Ascend950PR_9589 -enable-triton-kernel-compile=true \
// RUN: %s -save-linked-ir -o %t1.ll
// RUN: cat %t1_linked.ll | FileCheck %s

module {
  func.func @triton_unk_fused_clone_mul_15_new(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c384 = arith.constant 384 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c1_i32 = arith.constant 1 : i32
    %c110_i32 = arith.constant 110 : i32
    %c8_i32 = arith.constant 8 : i32
    %c294912_i32 = arith.constant 294912 : i32
    %c14_i32 = arith.constant 14 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.353553385 : f32
    %0 = tensor.empty() : tensor<8x384xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<8x384xf32>) -> tensor<8x384xf32>
    %2 = arith.muli %arg12, %c110_i32 : i32
    %3 = arith.addi %arg11, %c1_i32 : i32
    %4 = arith.minsi %3, %arg5 : i32
    %5 = arith.addi %2, %c110_i32 : i32
    %6 = arith.minsi %5, %arg6 : i32
    scf.for %arg14 = %arg11 to %4 step %c1_i32  : i32 {
      %7 = arith.muli %arg14, %c294912_i32 : i32
      scf.for %arg15 = %c0_i32 to %c14_i32 step %c1_i32  : i32 {
        %8 = arith.muli %arg15, %c8_i32 : i32
        %9 = arith.addi %2, %8 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = arith.index_cast %7 : i32 to index
        %12 = arith.addi %10, %11 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%12], sizes: [8, 384], strides: [1, 768] : memref<?xf32> to memref<8x384xf32, strided<[1, 768], offset: ?>>
        %alloc = memref.alloc() : memref<8x384xf32>
        %13 = arith.addi %10, %c8 : index
        %14 = arith.index_cast %6 : i32 to index
        %15 = arith.maxsi %10, %14 : index
        %16 = arith.minsi %13, %15 : index
        %17 = arith.subi %16, %10 : index
        %18 = arith.cmpi slt, %17, %c8 : index
        scf.if %18 {
          linalg.fill ins(%cst : f32) outs(%alloc : memref<8x384xf32>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%17, 384] [1, 1] : memref<8x384xf32, strided<[1, 768], offset: ?>> to memref<?x384xf32, strided<[1, 768], offset: ?>>
        %subview_1 = memref.subview %alloc[0, 0] [%17, 384] [1, 1] : memref<8x384xf32> to memref<?x384xf32, strided<[384, 1]>>
        memref.copy %subview, %subview_1 : memref<?x384xf32, strided<[1, 768], offset: ?>> to memref<?x384xf32, strided<[384, 1]>>
        // CHECK: @_Z23load_gm_to_ubuf_2d_coreIfEvP8memref_tIU3AS1T_Lm2EEPS0_IU3AS6S1_Lm2EE7PadModeN12PadValueTypeIS1_E4typeEl14EvictionPolicy10AtomicKind
        %19 = bufferization.to_tensor %alloc restrict writable : memref<8x384xf32>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%10], sizes: [8, 1], strides: [1, 1] : memref<?xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
        %alloc_3 = memref.alloc() : memref<8x1xf32>
        scf.if %18 {
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<8x1xf32>)
        } {hivm.unlikely_condition}
        %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%17, 1] [1, 1] : memref<8x1xf32, strided<[1, 1], offset: ?>> to memref<?x1xf32, strided<[1, 1], offset: ?>>
        %subview_5 = memref.subview %alloc_3[0, 0] [%17, 1] [1, 1] : memref<8x1xf32> to memref<?x1xf32, strided<[1, 1]>>
        memref.copy %subview_4, %subview_5 : memref<?x1xf32, strided<[1, 1], offset: ?>> to memref<?x1xf32, strided<[1, 1]>>
        %20 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x1xf32>
        %collapsed = tensor.collapse_shape %20 [[0, 1]] : tensor<8x1xf32> into tensor<8xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<8xf32>) outs(%0 : tensor<8x384xf32>) dimensions = [1] 
        %21 = arith.addf %19, %broadcasted : tensor<8x384xf32>
        %22 = arith.mulf %21, %1 : tensor<8x384xf32>
        %23 = arith.muli %10, %c384 : index
        %24 = arith.addi %23, %11 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [8, 384], strides: [384, 1] : memref<?xf32> to memref<8x384xf32, strided<[384, 1], offset: ?>>
        %extracted_slice = tensor.extract_slice %22[0, 0] [%17, 384] [1, 1] : tensor<8x384xf32> to tensor<?x384xf32>
        %subview_7 = memref.subview %reinterpret_cast_6[0, 0] [%17, 384] [1, 1] : memref<8x384xf32, strided<[384, 1], offset: ?>> to memref<?x384xf32, strided<[384, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?x384xf32>, memref<?x384xf32, strided<[384, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
