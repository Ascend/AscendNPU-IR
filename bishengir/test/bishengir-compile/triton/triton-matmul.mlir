 // REQUIRES: issue893
 // RUN: bishengir-compile -enable-triton-kernel-compile -enable-lir-compile=false -enable-hfusion-compile -block-dim=1 -target=Ascend310B4  %s | FileCheck %s

module {
  func.func @matmul_with_act_kernel(%arg0: memref<?xi8>, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "mix", hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = arith.muli %arg7, %c2_i32 : i32
    %3 = arith.addi %2, %c2_i32 : i32
    %4 = arith.minsi %3, %c32_i32 : i32
    scf.for %arg10 = %2 to %4 step %c1_i32  : i32 {
      %5 = arith.divsi %arg10, %c4_i32 : i32
      %6 = arith.remsi %arg10, %c4_i32 : i32
      %7 = arith.muli %5, %c128_i32 : i32
      %8 = arith.muli %6, %c256_i32 : i32
      %9 = scf.for %arg11 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg12 = %1) -> (tensor<128x256xf32>)  : i32 {
        %25 = arith.index_cast %7 : i32 to index
        %26 = arith.muli %25, %c1024 : index
        %27 = arith.index_cast %arg11 : i32 to index
        %28 = arith.addi %26, %27 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [128, 128], strides: [1024, 1] : memref<?xf16> to memref<128x128xf16, strided<[1024, 1], offset: ?>>
        %alloc = memref.alloc() : memref<128x128xf16>
        %29 = arith.addi %25, %c128 : index
        %30 = arith.maxsi %25, %c1024 : index
        %31 = arith.minsi %29, %30 : index
        %32 = arith.subi %31, %25 : index
        %33 = arith.addi %27, %c128 : index
        %34 = arith.maxsi %27, %c1024 : index
        %35 = arith.minsi %33, %34 : index
        %36 = arith.subi %35, %27 : index
        %37 = arith.minsi %32, %c128 : index
        %38 = arith.minsi %36, %c128 : index
        %39 = arith.cmpi slt, %37, %c128 : index
        %40 = arith.cmpi slt, %38, %c128 : index
        %41 = arith.ori %39, %40 : i1
        scf.if %41 {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<128x128xf16>)
        }
        %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%37, %38] [1, 1] : memref<128x128xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %subview_3 = memref.subview %alloc[0, 0] [%37, %38] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        // CHECK: nd2nz_half
        memref.copy %subview_2, %subview_3 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %42 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
        %43 = arith.muli %27, %c1024 : index
        %44 = arith.index_cast %8 : i32 to index
        %45 = arith.addi %43, %44 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [128, 256], strides: [1024, 1] : memref<?xf16> to memref<128x256xf16, strided<[1024, 1], offset: ?>>
        %alloc_5 = memref.alloc() : memref<128x256xf16>
        %46 = arith.addi %44, %c256 : index
        %47 = arith.maxsi %44, %c1024 : index
        %48 = arith.minsi %46, %47 : index
        %49 = arith.subi %48, %44 : index
        %50 = arith.minsi %49, %c256 : index
        %51 = arith.cmpi slt, %50, %c256 : index
        %52 = arith.ori %40, %51 : i1
        scf.if %52 {
          linalg.fill ins(%cst : f16) outs(%alloc_5 : memref<128x256xf16>)
        }
        %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%38, %50] [1, 1] : memref<128x256xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_5[0, 0] [%38, %50] [1, 1] : memref<128x256xf16> to memref<?x?xf16, strided<[256, 1]>>
        // CHECK: nd2nz_half
        memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1]>>
        %53 = bufferization.to_tensor %alloc_5 restrict writable : memref<128x256xf16>
        // CHECK: mma_tile_half_to_float
        %54 = linalg.matmul {input_precison = "ieee"} ins(%42, %53 : tensor<128x128xf16>, tensor<128x256xf16>) outs(%arg12 : tensor<128x256xf32>) -> tensor<128x256xf32>
        scf.yield %54 : tensor<128x256xf32>
      }
      %10 = arith.index_cast %7 : i32 to index
      %11 = arith.muli %10, %c1024 : index
      %12 = arith.index_cast %8 : i32 to index
      %13 = arith.addi %11, %12 : index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [128, 256], strides: [1024, 1] : memref<?xf16> to memref<128x256xf16, strided<[1024, 1], offset: ?>>
      %14 = arith.truncf %9 : tensor<128x256xf32> to tensor<128x256xf16>
      %15 = arith.addi %10, %c128 : index
      %16 = arith.maxsi %10, %c1024 : index
      %17 = arith.minsi %15, %16 : index
      %18 = arith.subi %17, %10 : index
      %19 = arith.addi %12, %c256 : index
      %20 = arith.maxsi %12, %c1024 : index
      %21 = arith.minsi %19, %20 : index
      %22 = arith.subi %21, %12 : index
      %23 = arith.minsi %18, %c128 : index
      %24 = arith.minsi %22, %c256 : index
      %extracted_slice = tensor.extract_slice %14[0, 0] [%23, %24] [1, 1] : tensor<128x256xf16> to tensor<?x?xf16>
      %subview = memref.subview %reinterpret_cast[0, 0] [%23, %24] [1, 1] : memref<128x256xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      // CHECK: fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf
      // CHECK: kernel_outlined_vf
      // CHECK: store_ubuf_to_gm_2d_half
      bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf16>, memref<?x?xf16, strided<[1024, 1], offset: ?>>) -> ()
    }
    return
  }
}

