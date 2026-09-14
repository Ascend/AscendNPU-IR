// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">} {
  func.func @kernel_hccl_transpose_a2a(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = arith.remsi %arg17, %c2_i32 : i32
    %1 = arith.divsi %arg17, %c2_i32 : i32
    %2 = arith.divsi %arg14, %c2_i32 : i32
    %3 = arith.addi %arg8, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %arg6, %c5_i32 : i32
    %6 = arith.muli %5, %arg9 : i32
    %7 = arith.muli %4, %c16_i32 : i32
    %8 = arith.cmpi eq, %0, %c0_i32 : i32
    scf.if %8 {
      %10 = arith.muli %arg6, %4 : i32
      scf.for %arg20 = %1 to %10 step %2  : i32 {
        %11 = arith.divsi %arg20, %4 : i32
        %12 = arith.remsi %arg20, %4 : i32
        %13 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_bfloat16"} "dist.aclshmem_ptr_bfloat16" ins(%arg4, %11 : memref<?xbf16>, i32) -> memref<?xbf16>
        %14 = arith.cmpi slt, %12, %4 : i32
        scf.for %arg21 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
          scf.if %14 {
            %19 = arith.remsi %12, %arg7 : i32
            %20 = arith.muli %12, %c128_i32 : i32
            %21 = arith.muli %11, %c5_i32 : i32
            %22 = arith.addi %21, %arg21 : i32
            %23 = arith.muli %22, %arg11 : i32
            %24 = arith.index_cast %20 : i32 to index
            %25 = arith.index_cast %arg10 : i32 to index
            %26 = arith.muli %24, %25 : index
            %27 = arith.index_cast %23 : i32 to index
            %28 = arith.addi %26, %27 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [128, 128], strides: [%25, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
            %alloc = memref.alloc() : memref<128x128xbf16>
            %29 = arith.addi %24, %c128 : index
            %30 = arith.index_cast %arg8 : i32 to index
            %31 = arith.maxsi %24, %30 : index
            %32 = arith.minsi %29, %31 : index
            %33 = arith.subi %32, %24 : index
            %34 = arith.index_cast %arg9 : i32 to index
            %35 = arith.maxsi %34, %c0 : index
            %36 = arith.minsi %35, %c128 : index
            %37 = arith.minsi %33, %c128 : index
            %38 = arith.maxsi %37, %c0 : index
            %39 = arith.minsi %36, %c128 : index
            %40 = arith.maxsi %39, %c0 : index
            %41 = arith.cmpi slt, %38, %c128 : index
            %42 = arith.cmpi slt, %40, %c128 : index
            %43 = arith.ori %41, %42 : i1
            scf.if %43 {
              linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x128xbf16>)
            } {hivm.unlikely_condition}
            %subview = memref.subview %reinterpret_cast_0[0, 0] [%38, %40] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
            %subview_1 = memref.subview %alloc[0, 0] [%38, %40] [1, 1] : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1]>>
            memref.copy %subview, %subview_1 : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1]>>
            %44 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
            %45 = arith.muli %6, %c128_i32 : i32
            %46 = arith.muli %19, %45 : i32
            %47 = arith.addi %arg21, %c5_i32 : i32
            %48 = arith.muli %47, %arg9 : i32
            %49 = arith.index_cast %46 : i32 to index
            %50 = arith.index_cast %6 : i32 to index
            %51 = arith.index_cast %48 : i32 to index
            %52 = arith.addi %49, %51 : index
            %reinterpret_cast_2 = memref.reinterpret_cast %13 to offset: [%52], sizes: [128, 128], strides: [%50, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
            %extracted_slice = tensor.extract_slice %44[0, 0] [%38, %40] [1, 1] : tensor<128x128xbf16> to tensor<?x?xbf16>
            %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%38, %40] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
            bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
          }
        }
        hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_fence"} "dist.aclshmem_fence"
        %15 = arith.muli %12, %c16_i32 : i32
        %16 = arith.index_cast %7 : i32 to index
        %17 = arith.index_cast %15 : i32 to index
        %18 = arith.addi %16, %17 : index
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmemx_signal_op"} "dist.aclshmemx_signal_op" ins(%reinterpret_cast, %c1_i32, %c0_i32, %11 : memref<1xi32, strided<[1], offset: ?>>, i32, i32, i32)
      }
    }
    %9 = arith.cmpi eq, %0, %c1_i32 : i32
    scf.if %9 {
      %10 = arith.muli %arg6, %4 : i32
      scf.for %arg20 = %1 to %10 step %2  : i32 {
        %11 = arith.divsi %arg20, %4 : i32
        %12 = arith.remsi %arg20, %4 : i32
        %13 = arith.muli %11, %7 : i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = arith.muli %12, %c16_i32 : i32
        %16 = arith.index_cast %15 : i32 to index
        %17 = arith.addi %14, %16 : index
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %18 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int32"} "dist.aclshmem_wait_int32" ins(%reinterpret_cast, %c1_i32, %c1_i32 : memref<1xi32, strided<[1], offset: ?>>, i32, i32) -> i32
        %19 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_bfloat16_ptr_1d"} "dist.aclshmem_consume_token_bfloat16_ptr_1d" ins(%arg4, %18 : memref<?xbf16>, i32) -> memref<?xbf16>
        %20 = arith.muli %11, %c5_i32 : i32
        %21 = arith.cmpi slt, %12, %4 : i32
        scf.for %arg21 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
          scf.if %21 {
            %22 = arith.remsi %12, %arg7 : i32
            %23 = arith.muli %12, %c128_i32 : i32
            %24 = arith.muli %6, %c128_i32 : i32
            %25 = arith.muli %22, %24 : i32
            %26 = arith.addi %20, %arg21 : i32
            %27 = arith.muli %26, %arg9 : i32
            %28 = arith.index_cast %25 : i32 to index
            %29 = arith.index_cast %6 : i32 to index
            %30 = arith.index_cast %27 : i32 to index
            %31 = arith.addi %28, %30 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %19 to offset: [%31], sizes: [128, 128], strides: [%29, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
            %alloc = memref.alloc() : memref<128x128xbf16>
            %32 = arith.index_cast %23 : i32 to index
            %33 = arith.addi %32, %c128 : index
            %34 = arith.index_cast %arg8 : i32 to index
            %35 = arith.maxsi %32, %34 : index
            %36 = arith.minsi %33, %35 : index
            %37 = arith.subi %36, %32 : index
            %38 = arith.index_cast %arg9 : i32 to index
            %39 = arith.maxsi %38, %c0 : index
            %40 = arith.minsi %39, %c128 : index
            %41 = arith.minsi %37, %c128 : index
            %42 = arith.maxsi %41, %c0 : index
            %43 = arith.minsi %40, %c128 : index
            %44 = arith.maxsi %43, %c0 : index
            %45 = arith.cmpi slt, %42, %c128 : index
            %46 = arith.cmpi slt, %44, %c128 : index
            %47 = arith.ori %45, %46 : i1
            scf.if %47 {
              linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x128xbf16>)
            } {hivm.unlikely_condition}
            %subview = memref.subview %reinterpret_cast_0[0, 0] [%42, %44] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
            %subview_1 = memref.subview %alloc[0, 0] [%42, %44] [1, 1] : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1]>>
            memref.copy %subview, %subview_1 : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1]>>
            %48 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
            %49 = arith.muli %11, %arg8 : i32
            %50 = arith.muli %arg21, %arg13 : i32
            %51 = arith.index_cast %49 : i32 to index
            %52 = arith.addi %51, %32 : index
            %53 = arith.index_cast %arg12 : i32 to index
            %54 = arith.muli %52, %53 : index
            %55 = arith.index_cast %50 : i32 to index
            %56 = arith.addi %54, %55 : index
            %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [128, 128], strides: [%53, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
            %extracted_slice = tensor.extract_slice %48[0, 0] [%42, %44] [1, 1] : tensor<128x128xbf16> to tensor<?x?xbf16>
            %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%42, %44] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
            bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
          }
        }
      }
    }
    return
  }
}
