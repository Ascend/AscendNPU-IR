// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">} {
  func.func @kernel_hccl_reverse_a2a_pipelined(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = arith.remsi %arg19, %c2_i32 : i32
    %1 = arith.divsi %arg19, %c2_i32 : i32
    %2 = arith.divsi %arg16, %c2_i32 : i32
    %3 = arith.addi %arg9, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.addi %arg11, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %arg10, %arg7 : i32
    %8 = arith.muli %7, %arg11 : i32
    %9 = arith.cmpi eq, %0, %c0_i32 : i32
    %10 = arith.cmpi eq, %0, %c1_i32 : i32
    scf.for %arg22 = %c0_i32 to %4 step %c1_i32  : i32 {
      %11 = arith.remsi %arg22, %arg8 : i32
      scf.if %9 {
        %12 = arith.muli %6, %arg10 : i32
        %13 = arith.muli %12, %arg7 : i32
        %14 = arith.muli %arg22, %c64_i32 : i32
        %15 = arith.muli %arg6, %arg10 : i32
        %16 = arith.muli %8, %c64_i32 : i32
        %17 = arith.muli %11, %16 : i32
        %18 = arith.muli %11, %6 : i32
        %19 = arith.muli %18, %arg10 : i32
        %20 = arith.muli %19, %arg7 : i32
        %21 = arith.muli %20, %c16_i32 : i32
        %22 = arith.muli %12, %arg6 : i32
        %23 = arith.muli %22, %c16_i32 : i32
        scf.for %arg23 = %1 to %13 step %2  : i32 {
          %24 = arith.remsi %arg23, %arg7 : i32
          %25 = arith.divsi %arg23, %arg7 : i32
          %26 = arith.remsi %25, %arg10 : i32
          %27 = arith.divsi %25, %arg10 : i32
          %28 = arith.addi %arg6, %24 : i32
          %29 = arith.remsi %28, %arg7 : i32
          %30 = arith.muli %27, %c128_i32 : i32
          %31 = arith.muli %29, %arg9 : i32
          %32 = arith.muli %26, %arg13 : i32
          %33 = arith.index_cast %31 : i32 to index
          %34 = arith.index_cast %14 : i32 to index
          %35 = arith.addi %33, %34 : index
          %36 = arith.index_cast %arg12 : i32 to index
          %37 = arith.muli %35, %36 : index
          %38 = arith.index_cast %32 : i32 to index
          %39 = arith.addi %37, %38 : index
          %40 = arith.index_cast %30 : i32 to index
          %41 = arith.addi %39, %40 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [64, 128], strides: [%36, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<64x128xbf16>
          %42 = arith.addi %34, %c64 : index
          %43 = arith.index_cast %arg9 : i32 to index
          %44 = arith.maxsi %34, %43 : index
          %45 = arith.minsi %42, %44 : index
          %46 = arith.subi %45, %34 : index
          %47 = arith.addi %40, %c128 : index
          %48 = arith.index_cast %arg11 : i32 to index
          %49 = arith.maxsi %40, %48 : index
          %50 = arith.minsi %47, %49 : index
          %51 = arith.subi %50, %40 : index
          %52 = arith.minsi %46, %c64 : index
          %53 = arith.maxsi %52, %c0 : index
          %54 = arith.minsi %51, %c128 : index
          %55 = arith.maxsi %54, %c0 : index
          %56 = arith.cmpi slt, %53, %c64 : index
          %57 = arith.cmpi slt, %55, %c128 : index
          %58 = arith.ori %56, %57 : i1
          scf.if %58 {
            linalg.fill ins(%cst : bf16) outs(%alloc : memref<64x128xbf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast[0, 0] [%53, %55] [1, 1] : memref<64x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_0 = memref.subview %alloc[0, 0] [%53, %55] [1, 1] : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1]>>
          memref.copy %subview, %subview_0 : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1]>>
          %59 = bufferization.to_tensor %alloc restrict writable : memref<64x128xbf16>
          %60 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_bfloat16"} "dist.aclshmem_ptr_bfloat16" ins(%arg4, %29 : memref<?xbf16>, i32) -> memref<?xbf16>
          %61 = arith.addi %15, %26 : i32
          %62 = arith.muli %61, %arg11 : i32
          %63 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_int32"} "dist.aclshmem_ptr_int32" ins(%arg5, %29 : memref<?xi32>, i32) -> memref<?xi32>
          %64 = arith.index_cast %21 : i32 to index
          %65 = arith.index_cast %23 : i32 to index
          %66 = arith.addi %64, %65 : index
          %67 = arith.muli %25, %c16_i32 : i32
          %68 = arith.index_cast %67 : i32 to index
          %69 = arith.addi %66, %68 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %63 to offset: [%69], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %70 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int32"} "dist.aclshmem_wait_int32" ins(%reinterpret_cast_1, %c1_i32, %c0_i32 : memref<1xi32, strided<[1], offset: ?>>, i32, i32) -> i32
          %71 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_bfloat16_ptr_1d"} "dist.aclshmem_consume_token_bfloat16_ptr_1d" ins(%60, %70 : memref<?xbf16>, i32) -> memref<?xbf16>
          %72 = arith.index_cast %17 : i32 to index
          %73 = arith.index_cast %8 : i32 to index
          %74 = arith.index_cast %62 : i32 to index
          %75 = arith.addi %72, %74 : index
          %76 = arith.addi %75, %40 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %71 to offset: [%76], sizes: [64, 128], strides: [%73, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %59[0, 0] [%53, %55] [1, 1] : tensor<64x128xbf16> to tensor<?x?xbf16>
          %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%53, %55] [1, 1] : memref<64x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
        }
        hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_fence"} "dist.aclshmem_fence"
        scf.for %arg23 = %1 to %13 step %2  : i32 {
          %24 = arith.remsi %arg23, %arg7 : i32
          %25 = arith.divsi %arg23, %arg7 : i32
          %26 = arith.addi %arg6, %24 : i32
          %27 = arith.remsi %26, %arg7 : i32
          %28 = arith.muli %25, %c16_i32 : i32
          %29 = arith.index_cast %21 : i32 to index
          %30 = arith.index_cast %23 : i32 to index
          %31 = arith.addi %29, %30 : index
          %32 = arith.index_cast %28 : i32 to index
          %33 = arith.addi %31, %32 : index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmemx_signal_op"} "dist.aclshmemx_signal_op" ins(%reinterpret_cast, %c1_i32, %c0_i32, %27 : memref<1xi32, strided<[1], offset: ?>>, i32, i32, i32)
        }
      }
      scf.if %10 {
        %12 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_bfloat16"} "dist.aclshmem_ptr_bfloat16" ins(%arg4, %arg6 : memref<?xbf16>, i32) -> memref<?xbf16>
        %13 = arith.muli %6, %arg10 : i32
        %14 = arith.muli %13, %arg7 : i32
        %15 = arith.muli %arg22, %c64_i32 : i32
        %16 = arith.muli %8, %c64_i32 : i32
        %17 = arith.muli %11, %16 : i32
        %18 = arith.muli %11, %6 : i32
        %19 = arith.muli %18, %arg10 : i32
        %20 = arith.muli %19, %arg7 : i32
        %21 = arith.muli %20, %c16_i32 : i32
        %22 = scf.for %arg23 = %1 to %14 step %2 iter_args(%arg24 = %c0_i64) -> (i64)  : i32 {
          %23 = arith.index_cast %arg24 : i64 to index
          %reinterpret_cast = memref.reinterpret_cast %12 to offset: [%23], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
          %24 = arith.remsi %arg23, %arg7 : i32
          %25 = arith.divsi %arg23, %arg7 : i32
          %26 = arith.remsi %25, %arg10 : i32
          %27 = arith.divsi %25, %arg10 : i32
          %28 = arith.muli %27, %c128_i32 : i32
          %29 = arith.muli %24, %arg10 : i32
          %30 = arith.addi %29, %26 : i32
          %31 = arith.muli %30, %arg11 : i32
          %32 = arith.muli %13, %24 : i32
          %33 = arith.muli %32, %c16_i32 : i32
          %34 = arith.index_cast %21 : i32 to index
          %35 = arith.index_cast %33 : i32 to index
          %36 = arith.addi %34, %35 : index
          %37 = arith.muli %25, %c16_i32 : i32
          %38 = arith.index_cast %37 : i32 to index
          %39 = arith.addi %36, %38 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %40 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int32"} "dist.aclshmem_wait_int32" ins(%reinterpret_cast_0, %c1_i32, %c1_i32 : memref<1xi32, strided<[1], offset: ?>>, i32, i32) -> i32
          %41 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_bfloat16_ptr_1d"} "dist.aclshmem_consume_token_bfloat16_ptr_1d" ins(%reinterpret_cast, %40 : memref<1xbf16, strided<[1], offset: ?>>, i32) -> memref<?xbf16>
          %42 = arith.index_cast %17 : i32 to index
          %43 = arith.index_cast %8 : i32 to index
          %44 = arith.index_cast %31 : i32 to index
          %45 = arith.addi %42, %44 : index
          %46 = arith.index_cast %28 : i32 to index
          %47 = arith.addi %45, %46 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %41 to offset: [%47], sizes: [64, 128], strides: [%43, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<64x128xbf16>
          %48 = arith.index_cast %15 : i32 to index
          %49 = arith.addi %48, %c64 : index
          %50 = arith.index_cast %arg9 : i32 to index
          %51 = arith.maxsi %48, %50 : index
          %52 = arith.minsi %49, %51 : index
          %53 = arith.subi %52, %48 : index
          %54 = arith.addi %46, %c128 : index
          %55 = arith.index_cast %arg11 : i32 to index
          %56 = arith.maxsi %46, %55 : index
          %57 = arith.minsi %54, %56 : index
          %58 = arith.subi %57, %46 : index
          %59 = arith.minsi %53, %c64 : index
          %60 = arith.maxsi %59, %c0 : index
          %61 = arith.minsi %58, %c128 : index
          %62 = arith.maxsi %61, %c0 : index
          %63 = arith.cmpi slt, %60, %c64 : index
          %64 = arith.cmpi slt, %62, %c128 : index
          %65 = arith.ori %63, %64 : i1
          scf.if %65 {
            linalg.fill ins(%cst : bf16) outs(%alloc : memref<64x128xbf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_1[0, 0] [%60, %62] [1, 1] : memref<64x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_2 = memref.subview %alloc[0, 0] [%60, %62] [1, 1] : memref<64x128xbf16> to memref<?x?xbf16, strided<[128, 1]>>
          memref.copy %subview, %subview_2 : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1]>>
          %66 = bufferization.to_tensor %alloc restrict writable : memref<64x128xbf16>
          %67 = arith.muli %30, %arg15 : i32
          %68 = arith.index_cast %arg14 : i32 to index
          %69 = arith.muli %48, %68 : index
          %70 = arith.index_cast %67 : i32 to index
          %71 = arith.addi %69, %70 : index
          %72 = arith.addi %71, %46 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [64, 128], strides: [%68, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %66[0, 0] [%60, %62] [1, 1] : tensor<64x128xbf16> to tensor<?x?xbf16>
          %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%60, %62] [1, 1] : memref<64x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice in writable %subview_4 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
          scf.yield %c0_i64 : i64
        }
        hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_fence"} "dist.aclshmem_fence"
        scf.for %arg23 = %1 to %14 step %2  : i32 {
          %23 = arith.remsi %arg23, %arg7 : i32
          %24 = arith.divsi %arg23, %arg7 : i32
          %25 = arith.muli %13, %23 : i32
          %26 = arith.muli %25, %c16_i32 : i32
          %27 = arith.index_cast %21 : i32 to index
          %28 = arith.index_cast %26 : i32 to index
          %29 = arith.addi %27, %28 : index
          %30 = arith.muli %24, %c16_i32 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.addi %29, %31 : index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%32], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmemx_signal_op"} "dist.aclshmemx_signal_op" ins(%reinterpret_cast, %c0_i32, %c0_i32, %arg6 : memref<1xi32, strided<[1], offset: ?>>, i32, i32, i32)
        }
      }
    }
    return
  }
}
