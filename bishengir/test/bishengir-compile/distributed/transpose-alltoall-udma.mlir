// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @kernel_hccl_transpose_a2a_put(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c128 = arith.constant 128 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c128_i64 = arith.constant 128 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i32 = arith.constant 1 : i32
    %c784128_i32 = arith.constant 784128 : i32
    %c2_i64 = arith.constant 2 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c784127_i32 = arith.constant 784127 : i32
    %0 = arith.muli %arg13, %c2_i32 : i32
    %1 = arith.divsi %0, %c2_i32 : i32
    %2 = arith.muli %arg16, %c2_i32 : i32
    %3 = arith.divsi %2, %c2_i32 : i32
    %4 = arith.muli %3, %c2_i32 : i32
    %5 = hivm.hir.get_sub_block_idx -> i64
    %6 = arith.extsi %4 : i32 to i64
    %7 = arith.addi %6, %5 : i64
    %8 = arith.remsi %7, %c2_i64 : i64
    %9 = arith.addi %arg7, %c127_i32 : i32
    %10 = arith.divsi %9, %c128_i32 : i32
    %11 = arith.muli %arg8, %c3_i32 : i32
    %12 = arith.muli %arg7, %c3_i32 : i32
    %13 = arith.muli %12, %arg8 : i32
    %14 = arith.addi %13, %c784127_i32 : i32
    %15 = arith.divsi %14, %c784128_i32 : i32
    %16 = arith.muli %arg6, %10 : i32
    %17 = arith.muli %16, %c3_i32 : i32
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.extsi %0 : i32 to i64
    %20 = arith.extsi %10 : i32 to i64
    %21 = arith.extsi %arg10 : i32 to i64
    %22 = arith.extsi %arg5 : i32 to i64
    scf.for %arg19 = %7 to %18 step %19  : i64 {
      %24 = arith.remsi %arg19, %c3_i64 : i64
      %25 = arith.divsi %arg19, %c3_i64 : i64
      %26 = arith.remsi %25, %20 : i64
      %27 = arith.divsi %25, %20 : i64
      %28 = arith.muli %26, %c128_i64 : i64
      %29 = arith.muli %27, %c3_i64 : i64
      %30 = arith.addi %29, %24 : i64
      %31 = arith.muli %30, %21 : i64
      %32 = arith.index_cast %28 : i64 to index
      %33 = arith.index_cast %arg9 : i32 to index
      %34 = arith.muli %32, %33 : index
      %35 = arith.index_cast %31 : i64 to index
      %36 = arith.addi %34, %35 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [128, 128], strides: [%33, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xbf16>
      %37 = arith.addi %32, %c128 : index
      %38 = arith.index_cast %arg7 : i32 to index
      %39 = arith.maxsi %32, %38 : index
      %40 = arith.minsi %37, %39 : index
      %41 = arith.subi %40, %32 : index
      %42 = arith.cmpi slt, %41, %c128 : index
      scf.if %42 {
        linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x128xbf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0, 0] [%41, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      %subview_0 = memref.subview %alloc[0, 0] [%41, 128] [1, 1] : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %subview, %subview_0 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %43 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
      %44 = arith.cmpi eq, %27, %22 : i64
      scf.if %44 {
        %45 = arith.extsi %arg12 : i32 to i64
        %46 = arith.muli %24, %45 : i64
        %47 = arith.muli %arg5, %arg7 : i32
        %48 = arith.index_cast %47 : i32 to index
        %49 = arith.addi %48, %32 : index
        %50 = arith.index_cast %arg11 : i32 to index
        %51 = arith.muli %49, %50 : index
        %52 = arith.index_cast %46 : i64 to index
        %53 = arith.addi %51, %52 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%53], sizes: [128, 128], strides: [%50, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %extracted_slice = tensor.extract_slice %43[0, 0] [%41, 128] [1, 1] : tensor<128x128xbf16> to tensor<?x128xbf16>
        %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%41, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
      } else {
        %45 = arith.extsi %arg8 : i32 to i64
        %46 = arith.muli %24, %45 : i64
        %47 = arith.extsi %13 : i32 to i64
        %48 = arith.muli %27, %47 : i64
        %49 = arith.index_cast %48 : i64 to index
        %50 = arith.index_cast %11 : i32 to index
        %51 = arith.muli %32, %50 : index
        %52 = arith.addi %49, %51 : index
        %53 = arith.index_cast %46 : i64 to index
        %54 = arith.addi %52, %53 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [128, 128], strides: [%50, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %extracted_slice = tensor.extract_slice %43[0, 0] [%41, 128] [1, 1] : tensor<128x128xbf16> to tensor<?x128xbf16>
        %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%41, 128] [1, 1] : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
      }
    }
    %23 = arith.cmpi sgt, %arg6, %c1_i32 : i32
    scf.if %23 {
      hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
      %24 = arith.cmpi eq, %8, %c0_i64 : i64
      %25 = arith.cmpi slt, %3, %arg6 : i32
      %26 = arith.andi %24, %25 : i1
      scf.if %26 {
        scf.for %arg19 = %3 to %arg6 step %1  : i32 {
          %27 = arith.cmpi ne, %arg19, %arg5 : i32
          scf.if %27 {
            scf.for %arg20 = %c0_i32 to %15 step %c1_i32  : i32 {
              %28 = arith.muli %arg20, %c784128_i32 : i32
              %29 = arith.cmpi slt, %28, %13 : i32
              scf.if %29 {
                %30 = arith.subi %13, %28 : i32
                %31 = arith.minsi %30, %c784128_i32 : i32
                %32 = arith.muli %arg5, %13 : i32
                %33 = arith.index_cast %32 : i32 to index
                %34 = arith.index_cast %28 : i32 to index
                %35 = arith.addi %33, %34 : index
                %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
                %36 = arith.muli %arg19, %13 : i32
                %37 = arith.index_cast %36 : i32 to index
                %38 = arith.addi %37, %34 : index
                %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
                %39 = arith.muli %31, %c2_i32 : i32
                hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_putmem_bfloat16"} "dist.aclshmem_putmem_bfloat16" ins(%reinterpret_cast, %reinterpret_cast_0, %39, %arg19 : memref<1xbf16, strided<[1], offset: ?>>, memref<1xbf16, strided<[1], offset: ?>>, i32, i32)
              }
            }
          }
        }
      }
    }
    return
  }
}
