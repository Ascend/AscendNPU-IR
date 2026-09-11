// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @kernel_gemm_reduce_scatter(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c268435456_i32 = arith.constant 268435456 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.divsi %arg8, %arg7 : i32
    %4 = arith.addi %arg9, %c255_i32 : i32
    %5 = arith.divsi %4, %c256_i32 : i32
    %6 = arith.addi %arg10, %c255_i32 : i32
    %7 = arith.divsi %6, %c256_i32 : i32
    %8 = arith.addi %3, %c127_i32 : i32
    %9 = arith.divsi %8, %c128_i32 : i32
    %10 = arith.muli %9, %5 : i32
    %11 = arith.cmpi eq, %arg17, %c0_i32 : i32
    %12 = arith.cmpi eq, %2, %c1_i64 : i64
    %13 = arith.andi %11, %12 : i1
    scf.for %arg20 = %c0_i32 to %arg7 step %c1_i32  : i32 {
      %16 = arith.muli %arg20, %3 : i32
      %17 = arith.addi %arg20, %c1_i32 : i32
      %18 = arith.muli %17, %3 : i32
      scf.for %arg21 = %arg17 to %10 step %arg14  : i32 {
        %19 = arith.divsi %arg21, %5 : i32
        %20 = arith.remsi %arg21, %5 : i32
        %21 = arith.muli %19, %c128_i32 : i32
        %22 = arith.addi %16, %21 : i32
        %23 = arith.muli %20, %c256_i32 : i32
        %24 = scf.for %arg22 = %c0_i32 to %7 step %c1_i32 iter_args(%arg23 = %1) -> (tensor<128x256xf32>)  : i32 {
          %65 = arith.muli %arg22, %c256_i32 : i32
          %66 = arith.index_cast %22 : i32 to index
          %67 = arith.index_cast %arg11 : i32 to index
          %68 = arith.muli %66, %67 : index
          %69 = arith.index_cast %65 : i32 to index
          %70 = arith.addi %68, %69 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%70], sizes: [128, 256], strides: [%67, 1] : memref<?xf16> to memref<128x256xf16, strided<[?, 1], offset: ?>>
          %71 = arith.index_cast %arg12 : i32 to index
          %72 = arith.muli %69, %71 : index
          %73 = arith.index_cast %23 : i32 to index
          %74 = arith.addi %72, %73 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%74], sizes: [256, 256], strides: [%71, 1] : memref<?xf16> to memref<256x256xf16, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<128x256xf16>
          %75 = arith.addi %69, %c256 : index
          %76 = arith.index_cast %arg10 : i32 to index
          %77 = arith.maxsi %69, %76 : index
          %78 = arith.minsi %75, %77 : index
          %79 = arith.subi %78, %69 : index
          %80 = arith.addi %66, %c128 : index
          %81 = arith.index_cast %18 : i32 to index
          %82 = arith.maxsi %66, %81 : index
          %83 = arith.minsi %80, %82 : index
          %84 = arith.subi %83, %66 : index
          %85 = arith.index_cast %16 : i32 to index
          %86 = arith.maxsi %66, %85 : index
          %87 = arith.minsi %80, %86 : index
          %88 = arith.subi %87, %66 : index
          %89 = arith.subi %80, %87 : index
          %90 = arith.maxsi %88, %c0 : index
          %91 = arith.addi %88, %89 : index
          %92 = arith.minsi %84, %91 : index
          %93 = arith.subi %92, %90 : index
          %94 = arith.maxsi %93, %c0 : index
          %95 = arith.index_cast %arg8 : i32 to index
          %96 = arith.maxsi %66, %95 : index
          %97 = arith.minsi %80, %96 : index
          %98 = arith.subi %97, %66 : index
          %99 = arith.maxsi %90, %c0 : index
          %100 = arith.addi %90, %94 : index
          %101 = arith.minsi %100, %98 : index
          %102 = arith.subi %101, %99 : index
          %103 = arith.maxsi %102, %c0 : index
          %104 = arith.maxsi %99, %c0 : index
          %105 = arith.addi %99, %103 : index
          %106 = arith.minsi %105, %c128 : index
          %107 = arith.subi %106, %104 : index
          %108 = arith.maxsi %107, %c0 : index
          %109 = arith.minsi %79, %c256 : index
          %110 = arith.maxsi %109, %c0 : index
          %111 = arith.cmpi slt, %108, %c128 : index
          %112 = arith.cmpi slt, %110, %c256 : index
          %113 = arith.ori %111, %112 : i1
          scf.if %113 {
            linalg.fill ins(%cst : f16) outs(%alloc : memref<128x256xf16>)
          } {hivm.unlikely_condition}
          %subview_3 = memref.subview %reinterpret_cast_1[%104, 0] [%108, %110] [1, 1] : memref<128x256xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_4 = memref.subview %alloc[%104, 0] [%108, %110] [1, 1] : memref<128x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          memref.copy %subview_3, %subview_4 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          %114 = bufferization.to_tensor %alloc restrict writable : memref<128x256xf16>
          %alloc_5 = memref.alloc() : memref<256x256xf16>
          %115 = arith.addi %73, %c256 : index
          %116 = arith.index_cast %arg9 : i32 to index
          %117 = arith.maxsi %73, %116 : index
          %118 = arith.minsi %115, %117 : index
          %119 = arith.subi %118, %73 : index
          %120 = arith.minsi %119, %c256 : index
          %121 = arith.maxsi %120, %c0 : index
          %122 = arith.cmpi slt, %121, %c256 : index
          %123 = arith.ori %112, %122 : i1
          scf.if %123 {
            linalg.fill ins(%cst : f16) outs(%alloc_5 : memref<256x256xf16>)
          } {hivm.unlikely_condition}
          %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%110, %121] [1, 1] : memref<256x256xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_7 = memref.subview %alloc_5[0, 0] [%110, %121] [1, 1] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1]>>
          memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1]>>
          %124 = bufferization.to_tensor %alloc_5 restrict writable : memref<256x256xf16>
          %125 = linalg.matmul {input_precision = "ieee"} ins(%114, %124 : tensor<128x256xf16>, tensor<256x256xf16>) outs(%arg23 : tensor<128x256xf32>) -> tensor<128x256xf32>
          scf.yield %125 : tensor<128x256xf32>
        }
        annotation.mark %24 {matmul_at_least_once} : tensor<128x256xf32>
        %25 = arith.index_cast %22 : i32 to index
        %26 = arith.index_cast %arg9 : i32 to index
        %27 = arith.muli %25, %26 : index
        %28 = arith.index_cast %23 : i32 to index
        %29 = arith.addi %27, %28 : index
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%29], sizes: [128, 256], strides: [%26, 1] : memref<?xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
        %30 = arith.addi %25, %c128 : index
        %31 = arith.index_cast %18 : i32 to index
        %32 = arith.maxsi %25, %31 : index
        %33 = arith.minsi %30, %32 : index
        %34 = arith.subi %33, %25 : index
        %35 = arith.index_cast %16 : i32 to index
        %36 = arith.maxsi %25, %35 : index
        %37 = arith.minsi %30, %36 : index
        %38 = arith.subi %37, %25 : index
        %39 = arith.subi %30, %37 : index
        %40 = arith.maxsi %38, %c0 : index
        %41 = arith.addi %38, %39 : index
        %42 = arith.minsi %34, %41 : index
        %43 = arith.subi %42, %40 : index
        %44 = arith.maxsi %43, %c0 : index
        %45 = arith.index_cast %arg8 : i32 to index
        %46 = arith.maxsi %25, %45 : index
        %47 = arith.minsi %30, %46 : index
        %48 = arith.subi %47, %25 : index
        %49 = arith.maxsi %40, %c0 : index
        %50 = arith.addi %40, %44 : index
        %51 = arith.minsi %50, %48 : index
        %52 = arith.subi %51, %49 : index
        %53 = arith.maxsi %52, %c0 : index
        %54 = arith.addi %28, %c256 : index
        %55 = arith.maxsi %28, %26 : index
        %56 = arith.minsi %54, %55 : index
        %57 = arith.subi %56, %28 : index
        %58 = arith.maxsi %49, %c0 : index
        %59 = arith.addi %49, %53 : index
        %60 = arith.minsi %59, %c128 : index
        %61 = arith.subi %60, %58 : index
        %62 = arith.maxsi %61, %c0 : index
        %63 = arith.minsi %57, %c256 : index
        %64 = arith.maxsi %63, %c0 : index
        %extracted_slice = tensor.extract_slice %24[%58, 0] [%62, %64] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
        %subview = memref.subview %reinterpret_cast[%58, 0] [%62, %64] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
      }
      scf.if %13 {
        %19 = arith.cmpi ne, %arg20, %c1_i32 : i32
        scf.if %19 {
          %20 = arith.muli %3, %arg9 : i32
          %21 = arith.muli %20, %c4_i32 : i32
          %22 = arith.cmpi sgt, %21, %c268435456_i32 : i32
          scf.if %22 {
            %23 = arith.muli %arg9, %c4_i32 : i32
            %24 = arith.divsi %c268435456_i32, %23 : i32
            %25 = arith.addi %3, %24 : i32
            %26 = arith.subi %25, %c1_i32 : i32
            %27 = arith.divsi %26, %24 : i32
            %28 = arith.muli %16, %arg9 : i32
            scf.for %arg21 = %c0_i32 to %27 step %c1_i32  : i32 {
              %29 = arith.muli %arg21, %24 : i32
              %30 = arith.subi %3, %29 : i32
              %31 = arith.minsi %24, %30 : i32
              %32 = arith.muli %31, %arg9 : i32
              %33 = arith.muli %32, %c4_i32 : i32
              %34 = arith.muli %29, %arg9 : i32
              %35 = arith.index_cast %28 : i32 to index
              %36 = arith.index_cast %34 : i32 to index
              %37 = arith.addi %35, %36 : index
              %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
              %38 = arith.index_cast %20 : i32 to index
              %39 = arith.addi %38, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%39], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
              hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_putmem_float"} "dist.aclshmem_putmem_float" ins(%reinterpret_cast_1, %reinterpret_cast, %33, %arg20 : memref<1xf32, strided<[1], offset: ?>>, memref<1xf32, strided<[1], offset: ?>>, i32, i32)
            }
          } else {
            %23 = arith.index_cast %20 : i32 to index
            %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%23], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %24 = arith.muli %16, %arg9 : i32
            %25 = arith.index_cast %24 : i32 to index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_putmem_float"} "dist.aclshmem_putmem_float" ins(%reinterpret_cast, %reinterpret_cast_1, %21, %arg20 : memref<1xf32, strided<[1], offset: ?>>, memref<1xf32, strided<[1], offset: ?>>, i32, i32)
          }
        }
      }
    }
    hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    %14 = arith.muli %10, %arg7 : i32
    %15 = arith.cmpi eq, %2, %c0_i64 : i64
    scf.if %15 {
      scf.for %arg20 = %arg17 to %14 step %arg14  : i32 {
        %16 = arith.divsi %arg20, %10 : i32
        %17 = arith.remsi %arg20, %10 : i32
        %18 = arith.divsi %17, %5 : i32
        %19 = arith.remsi %17, %5 : i32
        %20 = arith.muli %18, %c128_i32 : i32
        %21 = arith.muli %19, %c256_i32 : i32
        %22 = arith.cmpi eq, %16, %c1_i32 : i32
        scf.if %22 {
          %23 = arith.muli %16, %3 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = arith.index_cast %20 : i32 to index
          %26 = arith.addi %24, %25 : index
          %27 = arith.index_cast %arg9 : i32 to index
          %28 = arith.muli %26, %27 : index
          %29 = arith.index_cast %21 : i32 to index
          %30 = arith.addi %28, %29 : index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%30], sizes: [128, 256], strides: [%27, 1] : memref<?xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<128x256xf32>
          %31 = arith.addi %25, %c128 : index
          %32 = arith.index_cast %3 : i32 to index
          %33 = arith.maxsi %25, %32 : index
          %34 = arith.minsi %31, %33 : index
          %35 = arith.subi %34, %25 : index
          %36 = arith.addi %29, %c256 : index
          %37 = arith.maxsi %29, %27 : index
          %38 = arith.minsi %36, %37 : index
          %39 = arith.subi %38, %29 : index
          %40 = arith.minsi %35, %c128 : index
          %41 = arith.maxsi %40, %c0 : index
          %42 = arith.minsi %39, %c256 : index
          %43 = arith.maxsi %42, %c0 : index
          %44 = arith.cmpi slt, %41, %c128 : index
          %45 = arith.cmpi slt, %43, %c256 : index
          %46 = arith.ori %44, %45 : i1
          scf.if %46 {
            linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<128x256xf32>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast[0, 0] [%41, %43] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_1 = memref.subview %alloc[0, 0] [%41, %43] [1, 1] : memref<128x256xf32> to memref<?x?xf32, strided<[256, 1]>>
          memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
          %47 = bufferization.to_tensor %alloc restrict writable : memref<128x256xf32>
          %48 = arith.index_cast %arg13 : i32 to index
          %49 = arith.muli %25, %48 : index
          %50 = arith.addi %49, %29 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [128, 256], strides: [%48, 1] : memref<?xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
          %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%41, %43] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %47[0, 0] [%41, %43] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
          hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>>) atomic = <add>
        } else {
          %23 = arith.muli %16, %3 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = arith.index_cast %20 : i32 to index
          %26 = arith.addi %24, %25 : index
          %27 = arith.index_cast %arg9 : i32 to index
          %28 = arith.muli %26, %27 : index
          %29 = arith.index_cast %21 : i32 to index
          %30 = arith.addi %28, %29 : index
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%30], sizes: [128, 256], strides: [%27, 1] : memref<?xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<128x256xf32>
          %31 = arith.addi %25, %c128 : index
          %32 = arith.index_cast %3 : i32 to index
          %33 = arith.maxsi %25, %32 : index
          %34 = arith.minsi %31, %33 : index
          %35 = arith.subi %34, %25 : index
          %36 = arith.addi %29, %c256 : index
          %37 = arith.maxsi %29, %27 : index
          %38 = arith.minsi %36, %37 : index
          %39 = arith.subi %38, %29 : index
          %40 = arith.minsi %35, %c128 : index
          %41 = arith.maxsi %40, %c0 : index
          %42 = arith.minsi %39, %c256 : index
          %43 = arith.maxsi %42, %c0 : index
          %44 = arith.cmpi slt, %41, %c128 : index
          %45 = arith.cmpi slt, %43, %c256 : index
          %46 = arith.ori %44, %45 : i1
          scf.if %46 {
            linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<128x256xf32>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast[0, 0] [%41, %43] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_1 = memref.subview %alloc[0, 0] [%41, %43] [1, 1] : memref<128x256xf32> to memref<?x?xf32, strided<[256, 1]>>
          memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1]>>
          %47 = bufferization.to_tensor %alloc restrict writable : memref<128x256xf32>
          %48 = arith.index_cast %arg13 : i32 to index
          %49 = arith.muli %25, %48 : index
          %50 = arith.addi %49, %29 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [128, 256], strides: [%48, 1] : memref<?xf32> to memref<128x256xf32, strided<[?, 1], offset: ?>>
          %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%41, %43] [1, 1] : memref<128x256xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %47[0, 0] [%41, %43] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
          hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>>) atomic = <add>
        }
      }
    }
    return
  }
}
