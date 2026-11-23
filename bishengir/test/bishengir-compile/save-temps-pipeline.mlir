// REQUIRES: hivmc
// Test that --save-temps=<dir> stores the optimized HIVM module
// (module.hivm.opt.mlir) of a Triton kernel compiled with the regbase pipeline
// and LLVM IR(*.ll)
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: bishengir-compile %s \
// RUN:   --target=Ascend950PR_9579 \
// RUN:   --save-temps=%t.dir \
// RUN:   --enable-auto-multi-buffer=True \
// RUN:   --enable-auto-bind-sub-block=True \
// RUN:   --disable-ffts \
// RUN:   --limit-auto-multi-buffer-of-local-buffer=no-limit \
// RUN:   --enable-auto-blockify-loop \
// RUN:   --enable-hfusion-compile=true \
// RUN:   --enable-triton-kernel-compile=true \
// RUN:   --enable-vf-merge-level=1 \
// RUN:   -o %t.o
// RUN: FileCheck --input-file %t.dir/module.hivm.opt.mlir %s
// RUN: FileCheck --input-file %t.dir/*.ll %s

// CHECK: {{[Mm]odule}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @add_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg9, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    %2 = arith.addi %1, %c1024 : index
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.maxsi %1, %3 : index
    %5 = arith.minsi %2, %4 : index
    %6 = arith.subi %5, %1 : index
    %7 = arith.cmpi slt, %6, %c1024 : index
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<1024xf32>)
    } {hivm.unlikely_condition}
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32>
    %10 = arith.addf %8, %9 : tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %10[0] [%6] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
