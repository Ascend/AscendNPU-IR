// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @kernel_hccl_reverse_a2a_pipelined(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c127_i64 = arith.constant 127 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2_i64 = arith.constant 2 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = hivm.hir.get_sub_block_idx -> i64
    %1 = arith.extsi %arg18 : i32 to i64
    %2 = arith.addi %1, %0 : i64
    %3 = arith.divsi %2, %c2_i64 : i64
    %4 = arith.divsi %arg15, %c2_i32 : i32
    %5 = arith.extsi %arg8 : i32 to i64
    %6 = arith.extsi %arg9 : i32 to i64
    %7 = arith.extsi %arg10 : i32 to i64
    %8 = arith.addi %5, %c127_i64 : i64
    %9 = arith.divsi %8, %c128_i64 : i64
    %10 = arith.addi %7, %c127_i64 : i64
    %11 = arith.divsi %10, %c128_i64 : i64
    %12 = arith.muli %11, %6 : i64
    %13 = arith.muli %6, %7 : i64
    %14 = arith.extsi %arg6 : i32 to i64
    %15 = arith.muli %9, %14 : i64
    %16 = arith.muli %15, %c8_i64 : i64
    %17 = arith.extsi %arg7 : i32 to i64
    %18 = arith.remsi %2, %c2_i64 : i64
    %19 = arith.cmpi eq, %18, %c0_i64 : i64
    %20 = arith.cmpi slt, %3, %14 : i64
    %21 = arith.andi %19, %20 : i1
    %22 = arith.cmpi sge, %3, %14 : i64
    %23 = arith.andi %19, %22 : i1
    %24 = arith.cmpi eq, %18, %c1_i64 : i64
    scf.for %arg21 = %c0_i64 to %9 step %c1_i64  : i64 {
      %25 = arith.remsi %arg21, %17 : i64
      %26 = arith.muli %arg21, %c128_i64 : i64
      %27 = arith.subi %5, %26 : i64
      %28 = arith.minsi %27, %c128_i64 : i64
      scf.if %21 {
        %29 = arith.extsi %4 : i32 to i64
        scf.for %arg22 = %3 to %14 step %29  : i64 {
          %30 = arith.cmpi ne, %arg22, %c1_i64 : i64
          scf.if %30 {
            %31 = arith.muli %5, %6 : i64
            %32 = arith.muli %31, %7 : i64
            %33 = arith.muli %26, %13 : i64
            %34 = arith.addi %32, %33 : i64
            %35 = arith.index_cast %34 : i64 to index
            %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
            %36 = arith.muli %arg22, %5 : i64
            %37 = arith.muli %36, %6 : i64
            %38 = arith.muli %37, %7 : i64
            %39 = arith.extsi %arg11 : i32 to i64
            %40 = arith.muli %26, %39 : i64
            %41 = arith.addi %38, %40 : i64
            %42 = arith.index_cast %41 : i64 to index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
            %43 = arith.muli %28, %6 : i64
            %44 = arith.muli %43, %7 : i64
            %45 = arith.muli %44, %c2_i64 : i64
            %46 = arith.trunci %45 : i64 to i32
            %47 = arith.trunci %arg22 : i64 to i32
            hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_putmem_bfloat16"} "dist.aclshmem_putmem_bfloat16" ins(%reinterpret_cast, %reinterpret_cast_0, %46, %47 : memref<1xbf16, strided<[1], offset: ?>>, memref<1xbf16, strided<[1], offset: ?>>, i32, i32)
          }
        }
        scf.for %arg22 = %3 to %14 step %29  : i64 {
          %30 = arith.cmpi ne, %arg22, %c1_i64 : i64
          scf.if %30 {
            %31 = arith.muli %arg21, %14 : i64
            %32 = arith.addi %31, %c1_i64 : i64
            %33 = arith.muli %32, %c8_i64 : i64
            %34 = arith.index_cast %33 : i64 to index
            %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%34], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %35 = arith.trunci %arg22 : i64 to i32
            hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%reinterpret_cast, %c1_i64, %35 : memref<1xi64, strided<[1], offset: ?>>, i64, i32)
          }
        }
      }
      scf.if %23 {
        %29 = arith.subi %3, %14 : i64
        %30 = arith.subi %4, %arg6 : i32
        %31 = arith.extsi %30 : i32 to i64
        %32 = arith.extsi %arg12 : i32 to i64
        %33 = arith.muli %25, %12 : i64
        %34 = arith.muli %5, %6 : i64
        %35 = arith.muli %34, %7 : i64
        %36 = arith.muli %13, %c128_i64 : i64
        %37 = arith.muli %25, %36 : i64
        %38 = arith.addi %35, %37 : i64
        scf.for %arg22 = %29 to %12 step %31  : i64 {
          %39 = arith.remsi %arg22, %6 : i64
          %40 = arith.divsi %arg22, %6 : i64
          %41 = arith.muli %40, %c128_i64 : i64
          %42 = arith.muli %39, %32 : i64
          %43 = arith.index_cast %arg8 : i32 to index
          %44 = arith.index_cast %26 : i64 to index
          %45 = arith.addi %43, %44 : index
          %46 = arith.index_cast %arg11 : i32 to index
          %47 = arith.muli %45, %46 : index
          %48 = arith.index_cast %42 : i64 to index
          %49 = arith.addi %47, %48 : index
          %50 = arith.index_cast %41 : i64 to index
          %51 = arith.addi %49, %50 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%51], sizes: [128, 128], strides: [%46, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<128x128xbf16>
          %52 = arith.addi %44, %c128 : index
          %53 = arith.maxsi %44, %43 : index
          %54 = arith.minsi %52, %53 : index
          %55 = arith.subi %54, %44 : index
          %56 = arith.addi %50, %c128 : index
          %57 = arith.index_cast %arg10 : i32 to index
          %58 = arith.maxsi %50, %57 : index
          %59 = arith.minsi %56, %58 : index
          %60 = arith.subi %59, %50 : index
          %61 = arith.minsi %55, %c128 : index
          %62 = arith.maxsi %61, %c0 : index
          %63 = arith.minsi %60, %c128 : index
          %64 = arith.maxsi %63, %c0 : index
          %65 = arith.cmpi slt, %62, %c128 : index
          %66 = arith.cmpi slt, %64, %c128 : index
          %67 = arith.ori %65, %66 : i1
          scf.if %67 {
            linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x128xbf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast[0, 0] [%62, %64] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          %subview_0 = memref.subview %alloc[0, 0] [%62, %64] [1, 1] : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1]>>
          memref.copy %subview, %subview_0 : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1]>>
          %68 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
          %69 = arith.addi %33, %arg22 : i64
          %70 = arith.muli %69, %c8_i64 : i64
          %71 = arith.addi %16, %70 : i64
          %72 = arith.index_cast %71 : i64 to index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%72], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %73 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int64"} "dist.aclshmem_wait_int64" ins(%reinterpret_cast_1, %c1_i32, %c0_i64 : memref<1xi64, strided<[1], offset: ?>>, i32, i64) -> i32
          %74 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_bfloat16_ptr_1d"} "dist.aclshmem_consume_token_bfloat16_ptr_1d" ins(%arg4, %73 : memref<?xbf16>, i32) -> memref<?xbf16>
          %75 = arith.muli %39, %7 : i64
          %76 = arith.index_cast %38 : i64 to index
          %77 = arith.index_cast %13 : i64 to index
          %78 = arith.index_cast %75 : i64 to index
          %79 = arith.addi %76, %78 : index
          %80 = arith.addi %79, %50 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %74 to offset: [%80], sizes: [128, 128], strides: [%77, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %68[0, 0] [%62, %64] [1, 1] : tensor<128x128xbf16> to tensor<?x?xbf16>
          %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%62, %64] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
        }
        hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_fence"} "dist.aclshmem_fence"
        scf.for %arg22 = %29 to %12 step %31  : i64 {
          %39 = arith.addi %33, %arg22 : i64
          %40 = arith.muli %39, %c8_i64 : i64
          %41 = arith.addi %16, %40 : i64
          %42 = arith.index_cast %41 : i64 to index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%42], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%reinterpret_cast, %c1_i64, %c1_i32 : memref<1xi64, strided<[1], offset: ?>>, i64, i32)
        }
      }
      scf.if %24 {
        %29 = arith.muli %12, %14 : i64
        %30 = arith.extsi %4 : i32 to i64
        %31 = arith.extsi %arg14 : i32 to i64
        scf.for %arg22 = %3 to %29 step %30  : i64 {
          %32 = arith.remsi %arg22, %14 : i64
          %33 = arith.divsi %arg22, %14 : i64
          %34 = arith.remsi %33, %6 : i64
          %35 = arith.divsi %33, %6 : i64
          %36 = arith.muli %35, %c128_i64 : i64
          %37 = arith.cmpi eq, %32, %c1_i64 : i64
          %38 = scf.if %37 -> (tensor<128x128xbf16>) {
            %63 = arith.muli %25, %12 : i64
            %64 = arith.addi %63, %33 : i64
            %65 = arith.muli %64, %c8_i64 : i64
            %66 = arith.addi %16, %65 : i64
            %67 = arith.index_cast %66 : i64 to index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%67], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %68 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int64"} "dist.aclshmem_wait_int64" ins(%reinterpret_cast_0, %c1_i32, %c1_i64 : memref<1xi64, strided<[1], offset: ?>>, i32, i64) -> i32
            %69 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_bfloat16_ptr_1d"} "dist.aclshmem_consume_token_bfloat16_ptr_1d" ins(%arg4, %68 : memref<?xbf16>, i32) -> memref<?xbf16>
            %70 = arith.muli %5, %6 : i64
            %71 = arith.muli %70, %7 : i64
            %72 = arith.muli %13, %c128_i64 : i64
            %73 = arith.muli %25, %72 : i64
            %74 = arith.addi %71, %73 : i64
            %75 = arith.muli %34, %7 : i64
            %76 = arith.index_cast %74 : i64 to index
            %77 = arith.index_cast %13 : i64 to index
            %78 = arith.index_cast %75 : i64 to index
            %79 = arith.addi %76, %78 : index
            %80 = arith.index_cast %36 : i64 to index
            %81 = arith.addi %79, %80 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %69 to offset: [%81], sizes: [128, 128], strides: [%77, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
            %alloc = memref.alloc() : memref<128x128xbf16>
            %82 = arith.index_cast %26 : i64 to index
            %83 = arith.addi %82, %c128 : index
            %84 = arith.index_cast %arg8 : i32 to index
            %85 = arith.maxsi %82, %84 : index
            %86 = arith.minsi %83, %85 : index
            %87 = arith.subi %86, %82 : index
            %88 = arith.addi %80, %c128 : index
            %89 = arith.index_cast %arg10 : i32 to index
            %90 = arith.maxsi %80, %89 : index
            %91 = arith.minsi %88, %90 : index
            %92 = arith.subi %91, %80 : index
            %93 = arith.minsi %87, %c128 : index
            %94 = arith.maxsi %93, %c0 : index
            %95 = arith.minsi %92, %c128 : index
            %96 = arith.maxsi %95, %c0 : index
            %97 = arith.cmpi slt, %94, %c128 : index
            %98 = arith.cmpi slt, %96, %c128 : index
            %99 = arith.ori %97, %98 : i1
            scf.if %99 {
              linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x128xbf16>)
            } {hivm.unlikely_condition}
            %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%94, %96] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
            %subview_3 = memref.subview %alloc[0, 0] [%94, %96] [1, 1] : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1]>>
            memref.copy %subview_2, %subview_3 : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1]>>
            %100 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
            scf.yield %100 : tensor<128x128xbf16>
          } else {
            %63 = arith.muli %arg21, %14 : i64
            %64 = arith.addi %63, %32 : i64
            %65 = arith.muli %64, %c8_i64 : i64
            %66 = arith.index_cast %65 : i64 to index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%66], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %67 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int64"} "dist.aclshmem_wait_int64" ins(%reinterpret_cast_0, %c1_i32, %c1_i64 : memref<1xi64, strided<[1], offset: ?>>, i32, i64) -> i32
            %68 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_bfloat16_ptr_1d"} "dist.aclshmem_consume_token_bfloat16_ptr_1d" ins(%arg4, %67 : memref<?xbf16>, i32) -> memref<?xbf16>
            %69 = arith.muli %5, %32 : i64
            %70 = arith.muli %69, %6 : i64
            %71 = arith.muli %70, %7 : i64
            %72 = arith.muli %26, %13 : i64
            %73 = arith.addi %71, %72 : i64
            %74 = arith.muli %34, %7 : i64
            %75 = arith.index_cast %73 : i64 to index
            %76 = arith.index_cast %13 : i64 to index
            %77 = arith.index_cast %74 : i64 to index
            %78 = arith.addi %75, %77 : index
            %79 = arith.index_cast %36 : i64 to index
            %80 = arith.addi %78, %79 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %68 to offset: [%80], sizes: [128, 128], strides: [%76, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
            %alloc = memref.alloc() : memref<128x128xbf16>
            %81 = arith.index_cast %26 : i64 to index
            %82 = arith.addi %81, %c128 : index
            %83 = arith.index_cast %arg8 : i32 to index
            %84 = arith.maxsi %81, %83 : index
            %85 = arith.minsi %82, %84 : index
            %86 = arith.subi %85, %81 : index
            %87 = arith.addi %79, %c128 : index
            %88 = arith.index_cast %arg10 : i32 to index
            %89 = arith.maxsi %79, %88 : index
            %90 = arith.minsi %87, %89 : index
            %91 = arith.subi %90, %79 : index
            %92 = arith.minsi %86, %c128 : index
            %93 = arith.maxsi %92, %c0 : index
            %94 = arith.minsi %91, %c128 : index
            %95 = arith.maxsi %94, %c0 : index
            %96 = arith.cmpi slt, %93, %c128 : index
            %97 = arith.cmpi slt, %95, %c128 : index
            %98 = arith.ori %96, %97 : i1
            scf.if %98 {
              linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x128xbf16>)
            } {hivm.unlikely_condition}
            %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%93, %95] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
            %subview_3 = memref.subview %alloc[0, 0] [%93, %95] [1, 1] : memref<128x128xbf16> to memref<?x?xbf16, strided<[128, 1]>>
            memref.copy %subview_2, %subview_3 : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1]>>
            %99 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
            scf.yield %99 : tensor<128x128xbf16>
          }
          %39 = arith.muli %32, %6 : i64
          %40 = arith.addi %39, %34 : i64
          %41 = arith.muli %40, %31 : i64
          %42 = arith.index_cast %26 : i64 to index
          %43 = arith.index_cast %arg13 : i32 to index
          %44 = arith.muli %42, %43 : index
          %45 = arith.index_cast %41 : i64 to index
          %46 = arith.addi %44, %45 : index
          %47 = arith.index_cast %36 : i64 to index
          %48 = arith.addi %46, %47 : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%48], sizes: [128, 128], strides: [%43, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
          %49 = arith.addi %42, %c128 : index
          %50 = arith.index_cast %arg8 : i32 to index
          %51 = arith.maxsi %42, %50 : index
          %52 = arith.minsi %49, %51 : index
          %53 = arith.subi %52, %42 : index
          %54 = arith.addi %47, %c128 : index
          %55 = arith.index_cast %arg10 : i32 to index
          %56 = arith.maxsi %47, %55 : index
          %57 = arith.minsi %54, %56 : index
          %58 = arith.subi %57, %47 : index
          %59 = arith.minsi %53, %c128 : index
          %60 = arith.maxsi %59, %c0 : index
          %61 = arith.minsi %58, %c128 : index
          %62 = arith.maxsi %61, %c0 : index
          %extracted_slice = tensor.extract_slice %38[0, 0] [%60, %62] [1, 1] : tensor<128x128xbf16> to tensor<?x?xbf16>
          %subview = memref.subview %reinterpret_cast[0, 0] [%60, %62] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
        }
        hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_fence"} "dist.aclshmem_fence"
        scf.for %arg22 = %3 to %29 step %30  : i64 {
          %32 = arith.remsi %arg22, %14 : i64
          %33 = arith.divsi %arg22, %14 : i64
          %34 = arith.cmpi eq, %32, %c1_i64 : i64
          scf.if %34 {
            %35 = arith.muli %25, %12 : i64
            %36 = arith.addi %35, %33 : i64
            %37 = arith.muli %36, %c8_i64 : i64
            %38 = arith.addi %16, %37 : i64
            %39 = arith.index_cast %38 : i64 to index
            %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%reinterpret_cast, %c0_i64, %c1_i32 : memref<1xi64, strided<[1], offset: ?>>, i64, i32)
          }
        }
      }
    }
    return
  }
}
