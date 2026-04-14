// REQUIRES: hivmc
// RUN: bishengir-compile %s --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true -o %t.o

module attributes {hacc.target = #hacc.target<"Ascend910B4">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @kernel_allgather_gemm(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0_i64 = arith.constant 0 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.addi %arg8, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg9, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.subi %4, %c1_i32 : i32
    %8 = arith.addi %arg10, %c31_i32 : i32
    %9 = arith.divsi %8, %c32_i32 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
    %10 = arith.cmpi eq, %2, %c0_i64 : i64
    %11 = arith.muli %6, %arg7 : i32
    %12 = arith.muli %arg17, %c8_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%13], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %14 = arith.remsi %c2_i32, %arg7 : i32
    scf.for %arg20 = %c0_i32 to %4 step %c1_i32  : i32 {
      %15 = arith.cmpi eq, %arg20, %7 : i32
      %16 = scf.if %15 -> (i32) {
        %21 = arith.muli %arg20, %c128_i32 : i32
        %22 = arith.subi %arg8, %21 : i32
        scf.yield %22 : i32
      } else {
        scf.yield %c128_i32 : i32
      }
      %17 = arith.muli %arg20, %c128_i32 : i32
      %18 = arith.addi %16, %c128_i32 : i32
      %19 = hivm.hir.custom {hivm.is_distributed, no_side_effect, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int64"} "dist.aclshmem_wait_int64" ins(%reinterpret_cast, %arg14, %c0_i64 : memref<1xi64, strided<[1]>>, i32, i64) -> i32
      hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_barrier_all_vec"} "dist.aclshmemx_barrier_all_vec"
      scf.if %10 {
        %21 = arith.muli %9, %arg7 : i32
        scf.for %arg21 = %arg17 to %21 step %arg14  : i32 {
          %22 = arith.divsi %arg21, %9 : i32
          %23 = arith.remsi %arg21, %9 : i32
          %24 = hivm.hir.custom {hivm.is_distributed, no_side_effect, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_half"} "dist.aclshmem_ptr_half" ins(%arg5, %22 : memref<?xf16>, i32) -> memref<?xf16>
          %25 = arith.muli %23, %c32_i32 : i32
          %26 = arith.index_cast %17 : i32 to index
          %27 = arith.index_cast %arg11 : i32 to index
          %28 = arith.muli %26, %27 : index
          %29 = arith.index_cast %25 : i32 to index
          %30 = arith.addi %28, %29 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [128, 32], strides: [%27, 1] : memref<?xf16> to memref<128x32xf16, strided<[?, 1], offset: ?>>
          %31 = arith.muli %27, %c128 : index
          %32 = arith.addi %31, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %24 to offset: [%32], sizes: [128, 32], strides: [%27, 1] : memref<?xf16> to memref<128x32xf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<128x32xf16>
          %33 = arith.addi %29, %c32 : index
          %34 = arith.index_cast %arg10 : i32 to index
          %35 = arith.maxsi %29, %34 : index
          %36 = arith.minsi %33, %35 : index
          %37 = arith.subi %36, %29 : index
          %38 = arith.addi %26, %c128 : index
          %39 = arith.index_cast %arg8 : i32 to index
          %40 = arith.maxsi %26, %39 : index
          %41 = arith.minsi %38, %40 : index
          %42 = arith.subi %41, %26 : index
          %43 = arith.minsi %42, %c128 : index
          %44 = arith.minsi %37, %c32 : index
          %45 = arith.cmpi slt, %43, %c128 : index
          %46 = arith.cmpi slt, %44, %c32 : index
          %47 = arith.ori %45, %46 : i1
          scf.if %47 {
            linalg.fill ins(%cst : f16) outs(%alloc : memref<128x32xf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_2[0, 0] [%43, %44] [1, 1] : memref<128x32xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_4 = memref.subview %alloc[0, 0] [%43, %44] [1, 1] : memref<128x32xf16> to memref<?x?xf16, strided<[32, 1]>>
          memref.copy %subview, %subview_4 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
          %48 = bufferization.to_tensor %alloc restrict writable : memref<128x32xf16>
          %49 = hivm.hir.custom {SrcPtrIndex = array<i32: 0>, hivm.is_distributed, no_side_effect, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_consume_token_half_ptr_2d"} "dist.aclshmem_consume_token_half_ptr_2d" ins(%reinterpret_cast_3, %19 : memref<128x32xf16, strided<[?, 1], offset: ?>>, i32) -> memref<128x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %49 to offset: [%32], sizes: [128, 32], strides: [%27, 1] : memref<128x32xf16> to memref<128x32xf16, strided<[?, 1], offset: ?>>
          %50 = arith.index_cast %18 : i32 to index
          %51 = arith.maxsi %50, %c128 : index
          %52 = arith.minsi %51, %c256 : index
          %53 = arith.subi %52, %c128 : index
          %54 = arith.minsi %53, %c128 : index
          %extracted_slice = tensor.extract_slice %48[0, 0] [%54, %44] [1, 1] : tensor<128x32xf16> to tensor<?x?xf16>
          %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%54, %44] [1, 1] : memref<128x32xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
        }
        hivm.hir.custom {commScope = 2 : i32, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%reinterpret_cast_1, %c1_i64, %14 : memref<1xi64, strided<[1], offset: ?>>, i64, i32)
      }
      %20 = hivm.hir.custom {hivm.is_distributed, no_side_effect, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int64"} "dist.aclshmem_wait_int64" ins(%reinterpret_cast, %arg14, %c1_i64 : memref<1xi64, strided<[1]>>, i32, i64) -> i32
      hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
      scf.for %arg21 = %arg17 to %11 step %arg14  : i32 {
        %21 = arith.divsi %arg21, %6 : i32
        %22 = arith.remsi %arg21, %6 : i32
        %23 = arith.muli %21, %c128_i32 : i32
        %24 = arith.addi %23, %16 : i32
        %25 = arith.muli %22, %c128_i32 : i32
        %26 = scf.for %arg22 = %c0_i32 to %9 step %c1_i32 iter_args(%arg23 = %1) -> (tensor<128x128xf32>)  : i32 {
          %49 = arith.muli %arg22, %c32_i32 : i32
          %50 = arith.index_cast %23 : i32 to index
          %51 = arith.index_cast %arg11 : i32 to index
          %52 = arith.muli %50, %51 : index
          %53 = arith.index_cast %49 : i32 to index
          %54 = arith.addi %52, %53 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [128, 32], strides: [%51, 1] : memref<?xf16> to memref<128x32xf16, strided<[?, 1], offset: ?>>
          %55 = arith.index_cast %arg12 : i32 to index
          %56 = arith.muli %53, %55 : index
          %57 = arith.index_cast %25 : i32 to index
          %58 = arith.addi %56, %57 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%58], sizes: [32, 128], strides: [%55, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
          %59 = hivm.hir.custom {SrcPtrIndex = array<i32: 0>, hivm.is_distributed, no_side_effect, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_consume_token_half_ptr_2d"} "dist.aclshmem_consume_token_half_ptr_2d" ins(%reinterpret_cast_3, %20 : memref<128x32xf16, strided<[?, 1], offset: ?>>, i32) -> memref<128x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %59 to offset: [%54], sizes: [128, 32], strides: [%51, 1] : memref<128x32xf16> to memref<128x32xf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<128x32xf16>
          %60 = arith.addi %53, %c32 : index
          %61 = arith.index_cast %arg10 : i32 to index
          %62 = arith.maxsi %53, %61 : index
          %63 = arith.minsi %60, %62 : index
          %64 = arith.subi %63, %53 : index
          %65 = arith.addi %50, %c128 : index
          %66 = arith.index_cast %24 : i32 to index
          %67 = arith.maxsi %50, %66 : index
          %68 = arith.minsi %65, %67 : index
          %69 = arith.subi %68, %50 : index
          %70 = arith.minsi %69, %c128 : index
          %71 = arith.minsi %64, %c32 : index
          %72 = arith.cmpi slt, %70, %c128 : index
          %73 = arith.cmpi slt, %71, %c32 : index
          %74 = arith.ori %72, %73 : i1
          scf.if %74 {
            linalg.fill ins(%cst : f16) outs(%alloc : memref<128x32xf16>)
          } {hivm.unlikely_condition}
          %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%70, %71] [1, 1] : memref<128x32xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_7 = memref.subview %alloc[0, 0] [%70, %71] [1, 1] : memref<128x32xf16> to memref<?x?xf16, strided<[32, 1]>>
          memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
          %75 = bufferization.to_tensor %alloc restrict writable : memref<128x32xf16>
          %alloc_8 = memref.alloc() : memref<32x128xf16>
          %76 = arith.addi %57, %c128 : index
          %77 = arith.index_cast %arg9 : i32 to index
          %78 = arith.maxsi %57, %77 : index
          %79 = arith.minsi %76, %78 : index
          %80 = arith.subi %79, %57 : index
          %81 = arith.minsi %80, %c128 : index
          %82 = arith.cmpi slt, %81, %c128 : index
          %83 = arith.ori %73, %82 : i1
          scf.if %83 {
            linalg.fill ins(%cst : f16) outs(%alloc_8 : memref<32x128xf16>)
          } {hivm.unlikely_condition}
          %subview_9 = memref.subview %reinterpret_cast_4[0, 0] [%71, %81] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_10 = memref.subview %alloc_8[0, 0] [%71, %81] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
          memref.copy %subview_9, %subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
          %84 = bufferization.to_tensor %alloc_8 restrict writable : memref<32x128xf16>
          %85 = linalg.matmul {input_precision = "ieee"} ins(%75, %84 : tensor<128x32xf16>, tensor<32x128xf16>) outs(%arg23 : tensor<128x128xf32>) -> tensor<128x128xf32>
          scf.yield %85 : tensor<128x128xf32>
        }
        %27 = arith.truncf %26 : tensor<128x128xf32> to tensor<128x128xf16>
        %28 = arith.muli %21, %arg8 : i32
        %29 = arith.addi %28, %17 : i32
        %30 = arith.index_cast %arg13 : i32 to index
        %31 = arith.index_cast %29 : i32 to index
        %32 = arith.muli %31, %30 : index
        %33 = arith.index_cast %25 : i32 to index
        %34 = arith.addi %32, %33 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [128, 128], strides: [%30, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
        %35 = arith.addi %21, %c1_i32 : i32
        %36 = arith.muli %arg8, %35 : i32
        %37 = arith.addi %31, %c128 : index
        %38 = arith.index_cast %36 : i32 to index
        %39 = arith.maxsi %31, %38 : index
        %40 = arith.minsi %37, %39 : index
        %41 = arith.subi %40, %31 : index
        %42 = arith.addi %33, %c128 : index
        %43 = arith.index_cast %arg9 : i32 to index
        %44 = arith.maxsi %33, %43 : index
        %45 = arith.minsi %42, %44 : index
        %46 = arith.subi %45, %33 : index
        %47 = arith.minsi %41, %c128 : index
        %48 = arith.minsi %46, %c128 : index
        %extracted_slice = tensor.extract_slice %27[0, 0] [%47, %48] [1, 1] : tensor<128x128xf16> to tensor<?x?xf16>
        %subview = memref.subview %reinterpret_cast_2[0, 0] [%47, %48] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      }
      hivm.hir.custom {commScope = 2 : i32, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%reinterpret_cast_1, %c0_i64, %14 : memref<1xi64, strided<[1], offset: ?>>, i64, i32)
    }
    return
  }
}