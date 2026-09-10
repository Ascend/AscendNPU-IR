// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @kernel_gemm_allreduce_notify_wait_pull(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c7_i32 = arith.constant 7 : i32
    %c2_i32 = arith.constant 2 : i32
    %c6_i32 = arith.constant 6 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i64 = arith.constant 0 : i64
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.muli %arg16, %c4_i32 : i32
    %4 = arith.addi %arg10, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg11, %c255_i32 : i32
    %7 = arith.divsi %6, %c256_i32 : i32
    %8 = arith.muli %5, %7 : i32
    %9 = arith.addi %8, %3 : i32
    %10 = arith.subi %9, %c1_i32 : i32
    %11 = arith.divsi %10, %3 : i32
    %12 = arith.subi %11, %c1_i32 : i32
    %13 = arith.addi %arg12, %c255_i32 : i32
    %14 = arith.divsi %13, %c256_i32 : i32
    %15 = arith.addi %7, %c6_i32 : i32
    %16 = arith.divsi %15, %c7_i32 : i32
    %17 = arith.muli %5, %c7_i32 : i32
    %18 = arith.subi %16, %c1_i32 : i32
    %19 = arith.muli %arg7, %c16_i32 : i32
    %20 = arith.cmpi eq, %2, %c0_i64 : i64
    scf.for %arg22 = %c0_i32 to %11 step %c1_i32  : i32 {
      %21 = arith.muli %arg22, %3 : i32
      %22 = arith.cmpi eq, %arg22, %12 : i32
      %23 = scf.if %22 -> (i32) {
        %27 = arith.subi %8, %21 : i32
        scf.yield %27 : i32
      } else {
        scf.yield %3 : i32
      }
      scf.for %arg23 = %arg19 to %23 step %arg16  : i32 {
        %27 = arith.addi %21, %arg23 : i32
        %28 = arith.divsi %27, %17 : i32
        %29 = arith.remsi %27, %17 : i32
        %30 = arith.cmpi eq, %28, %18 : i32
        %31 = scf.if %30 -> (i32) {
          %66 = arith.muli %28, %c7_i32 : i32
          %67 = arith.subi %7, %66 : i32
          scf.yield %67 : i32
        } else {
          scf.yield %c7_i32 : i32
        }
        %32 = arith.divsi %29, %31 : i32
        %33 = arith.muli %28, %c7_i32 : i32
        %34 = arith.remsi %29, %31 : i32
        %35 = arith.addi %33, %34 : i32
        %36 = arith.remsi %28, %c2_i32 : i32
        %37 = arith.cmpi eq, %36, %c1_i32 : i32
        %38 = scf.if %37 -> (i32) {
          %66 = arith.subi %5, %32 : i32
          %67 = arith.subi %66, %c1_i32 : i32
          scf.yield %67 : i32
        } else {
          scf.yield %32 : i32
        }
        %39 = arith.muli %38, %c128_i32 : i32
        %40 = arith.muli %35, %c256_i32 : i32
        %41 = scf.for %arg24 = %c0_i32 to %14 step %c1_i32 iter_args(%arg25 = %1) -> (tensor<128x256xf32>)  : i32 {
          %66 = arith.muli %arg24, %c256_i32 : i32
          %67 = arith.index_cast %39 : i32 to index
          %68 = arith.index_cast %arg13 : i32 to index
          %69 = arith.muli %67, %68 : index
          %70 = arith.index_cast %66 : i32 to index
          %71 = arith.addi %69, %70 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%71], sizes: [128, 256], strides: [%68, 1] : memref<?xf16> to memref<128x256xf16, strided<[?, 1], offset: ?>>
          %72 = arith.index_cast %arg14 : i32 to index
          %73 = arith.muli %70, %72 : index
          %74 = arith.index_cast %40 : i32 to index
          %75 = arith.addi %73, %74 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%75], sizes: [256, 256], strides: [%72, 1] : memref<?xf16> to memref<256x256xf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<128x256xf16>
          %76 = arith.addi %70, %c256 : index
          %77 = arith.index_cast %arg12 : i32 to index
          %78 = arith.maxsi %70, %77 : index
          %79 = arith.minsi %76, %78 : index
          %80 = arith.subi %79, %70 : index
          %81 = arith.addi %67, %c128 : index
          %82 = arith.index_cast %arg10 : i32 to index
          %83 = arith.maxsi %67, %82 : index
          %84 = arith.minsi %81, %83 : index
          %85 = arith.subi %84, %67 : index
          %86 = arith.minsi %85, %c128 : index
          %87 = arith.maxsi %86, %c0 : index
          %88 = arith.minsi %80, %c256 : index
          %89 = arith.maxsi %88, %c0 : index
          %90 = arith.cmpi slt, %87, %c128 : index
          %91 = arith.cmpi slt, %89, %c256 : index
          %92 = arith.ori %90, %91 : i1
          scf.if %92 {
            linalg.fill ins(%cst : f16) outs(%alloc : memref<128x256xf16>)
          } {hivm.unlikely_condition}
          %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%87, %89] [1, 1] : memref<128x256xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_5 = memref.subview %alloc[0, 0] [%87, %89] [1, 1] : memref<128x256xf16> to memref<?x?xf16, strided<[256, 1]>>
          memref.copy %subview_4, %subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1]>>
          %93 = bufferization.to_tensor %alloc restrict writable : memref<128x256xf16>
          %alloc_6 = memref.alloc() : memref<256x256xf16>
          %94 = arith.addi %74, %c256 : index
          %95 = arith.index_cast %arg11 : i32 to index
          %96 = arith.maxsi %74, %95 : index
          %97 = arith.minsi %94, %96 : index
          %98 = arith.subi %97, %74 : index
          %99 = arith.minsi %98, %c256 : index
          %100 = arith.maxsi %99, %c0 : index
          %101 = arith.cmpi slt, %100, %c256 : index
          %102 = arith.ori %91, %101 : i1
          scf.if %102 {
            linalg.fill ins(%cst : f16) outs(%alloc_6 : memref<256x256xf16>)
          } {hivm.unlikely_condition}
          %subview_7 = memref.subview %reinterpret_cast_3[0, 0] [%89, %100] [1, 1] : memref<256x256xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_6[0, 0] [%89, %100] [1, 1] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1]>>
          memref.copy %subview_7, %subview_8 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1]>>
          %103 = bufferization.to_tensor %alloc_6 restrict writable : memref<256x256xf16>
          %104 = linalg.matmul {input_precision = "ieee"} ins(%93, %103 : tensor<128x256xf16>, tensor<256x256xf16>) outs(%arg25 : tensor<128x256xf32>) -> tensor<128x256xf32>
          scf.yield %104 : tensor<128x256xf32>
        }
        %42 = arith.index_cast %arg15 : i32 to index
        %43 = arith.index_cast %39 : i32 to index
        %44 = arith.muli %43, %42 : index
        %45 = arith.index_cast %40 : i32 to index
        %46 = arith.addi %44, %45 : index
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [128, 256], strides: [%42, 1] : memref<?xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
        %47 = arith.addi %43, %c128 : index
        %48 = arith.index_cast %arg10 : i32 to index
        %49 = arith.maxsi %43, %48 : index
        %50 = arith.minsi %47, %49 : index
        %51 = arith.subi %50, %43 : index
        %52 = arith.addi %45, %c256 : index
        %53 = arith.index_cast %arg11 : i32 to index
        %54 = arith.maxsi %45, %53 : index
        %55 = arith.minsi %52, %54 : index
        %56 = arith.subi %55, %45 : index
        %57 = arith.minsi %51, %c128 : index
        %58 = arith.maxsi %57, %c0 : index
        %59 = arith.minsi %56, %c256 : index
        %60 = arith.maxsi %59, %c0 : index
        %extracted_slice = tensor.extract_slice %41[0, 0] [%58, %60] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
        %subview = memref.subview %reinterpret_cast[0, 0] [%58, %60] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
        hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_fence"} "dist.aclshmem_fence"
        %61 = arith.muli %27, %arg8 : i32
        %62 = arith.muli %61, %c16_i32 : i32
        %63 = arith.index_cast %62 : i32 to index
        %64 = arith.index_cast %19 : i32 to index
        %65 = arith.addi %63, %64 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        scf.for %arg24 = %c0_i32 to %arg8 step %c1_i32  : i32 {
          hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmemx_signal_op"} "dist.aclshmemx_signal_op" ins(%reinterpret_cast_1, %c1_i32, %c0_i32, %arg24 : memref<1xi32, strided<[1], offset: ?>>, i32, i32, i32)
        }
      }
      %24 = arith.muli %23, %c128_i32 : i32
      %25 = arith.addi %24, %c7_i32 : i32
      %26 = arith.divsi %25, %c8_i32 : i32
      scf.if %20 {
        %27 = arith.muli %26, %arg8 : i32
        scf.for %arg23 = %arg19 to %27 step %arg16  : i32 {
          %28 = arith.divsi %arg23, %26 : i32
          %29 = arith.remsi %arg23, %26 : i32
          %30 = arith.addi %28, %29 : i32
          %31 = arith.remsi %30, %arg8 : i32
          %32 = arith.muli %29, %c8_i32 : i32
          %33 = arith.divsi %32, %c128_i32 : i32
          %34 = arith.addi %21, %33 : i32
          %35 = arith.divsi %34, %17 : i32
          %36 = arith.remsi %34, %17 : i32
          %37 = arith.cmpi eq, %35, %18 : i32
          %38 = scf.if %37 -> (i32) {
            %80 = arith.muli %35, %c7_i32 : i32
            %81 = arith.subi %7, %80 : i32
            scf.yield %81 : i32
          } else {
            scf.yield %c7_i32 : i32
          }
          %39 = arith.divsi %36, %38 : i32
          %40 = arith.muli %35, %c7_i32 : i32
          %41 = arith.remsi %36, %38 : i32
          %42 = arith.addi %40, %41 : i32
          %43 = arith.remsi %35, %c2_i32 : i32
          %44 = arith.cmpi eq, %43, %c1_i32 : i32
          %45 = scf.if %44 -> (i32) {
            %80 = arith.subi %5, %39 : i32
            %81 = arith.subi %80, %c1_i32 : i32
            scf.yield %81 : i32
          } else {
            scf.yield %39 : i32
          }
          %46 = arith.remsi %32, %c128_i32 : i32
          %47 = arith.muli %45, %c128_i32 : i32
          %48 = arith.addi %47, %46 : i32
          %49 = arith.muli %42, %c256_i32 : i32
          %50 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_int32"} "dist.aclshmem_ptr_int32" ins(%arg6, %31 : memref<?xi32>, i32) -> memref<?xi32>
          %51 = arith.muli %34, %arg8 : i32
          %52 = arith.muli %51, %c16_i32 : i32
          %53 = arith.index_cast %52 : i32 to index
          %54 = arith.muli %31, %c16_i32 : i32
          %55 = arith.index_cast %54 : i32 to index
          %56 = arith.addi %53, %55 : index
          %reinterpret_cast = memref.reinterpret_cast %50 to offset: [%56], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %57 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 3 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int32"} "dist.aclshmem_wait_int32" ins(%reinterpret_cast, %c1_i32, %c1_i32 : memref<1xi32, strided<[1], offset: ?>>, i32, i32) -> i32
          %58 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_float"} "dist.aclshmem_ptr_float" ins(%arg5, %31 : memref<?xf32>, i32) -> memref<?xf32>
          %59 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_float_ptr_1d"} "dist.aclshmem_consume_token_float_ptr_1d" ins(%58, %57 : memref<?xf32>, i32) -> memref<?xf32>
          %60 = arith.index_cast %arg15 : i32 to index
          %61 = arith.index_cast %48 : i32 to index
          %62 = arith.muli %61, %60 : index
          %63 = arith.index_cast %49 : i32 to index
          %64 = arith.addi %62, %63 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %59 to offset: [%64], sizes: [8, 256], strides: [%60, 1] : memref<?xf32> to memref<8x256xf32, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<8x256xf32>
          memref.copy %reinterpret_cast_1, %alloc : memref<8x256xf32, strided<[?, 1], offset: ?>> to memref<8x256xf32>
          %65 = bufferization.to_tensor %alloc restrict writable : memref<8x256xf32>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [8, 256], strides: [%60, 1] : memref<?xf32> to memref<8x256xf32, strided<[?, 1], offset: ?>>
          %66 = arith.addi %61, %c8 : index
          %67 = arith.index_cast %arg10 : i32 to index
          %68 = arith.maxsi %61, %67 : index
          %69 = arith.minsi %66, %68 : index
          %70 = arith.subi %69, %61 : index
          %71 = arith.addi %63, %c256 : index
          %72 = arith.index_cast %arg11 : i32 to index
          %73 = arith.maxsi %63, %72 : index
          %74 = arith.minsi %71, %73 : index
          %75 = arith.subi %74, %63 : index
          %76 = arith.minsi %70, %c8 : index
          %77 = arith.maxsi %76, %c0 : index
          %78 = arith.minsi %75, %c256 : index
          %79 = arith.maxsi %78, %c0 : index
          %subview = memref.subview %reinterpret_cast_2[0, 0] [%77, %79] [1, 1] : memref<8x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %65[0, 0] [%77, %79] [1, 1] : tensor<8x256xf32> to tensor<?x?xf32>
          hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>>) atomic = <add>
        }
      }
    }
    return
  }
}
