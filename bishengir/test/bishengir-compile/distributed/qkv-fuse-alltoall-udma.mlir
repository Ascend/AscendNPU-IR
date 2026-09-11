// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @kernel_qkv_fuse_a2a_udma_get(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c768 = arith.constant 768 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c112 = arith.constant 112 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c112_i32 = arith.constant 112 : i32
    %c112_i64 = arith.constant 112 : i64
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c2_i64 = arith.constant 2 : i64
    %c780_i32 = arith.constant 780 : i32
    %c0_i32 = arith.constant 0 : i32
    %c86016_i32 = arith.constant 86016 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %c111_i32 = arith.constant 111 : i32
    %0 = arith.extsi %arg12 : i32 to i64
    %1 = arith.muli %arg13, %c2_i32 : i32
    %2 = arith.divsi %1, %c2_i32 : i32
    %3 = arith.muli %arg16, %c2_i32 : i32
    %4 = arith.divsi %3, %c2_i32 : i32
    %5 = arith.muli %4, %c2_i32 : i32
    %6 = hivm.hir.get_sub_block_idx -> i64
    %7 = arith.extsi %5 : i32 to i64
    %8 = arith.addi %7, %6 : i64
    %9 = arith.addi %arg9, %c111_i32 : i32
    %10 = arith.divsi %9, %c112_i32 : i32
    %11 = arith.muli %arg8, %arg7 : i32
    %12 = arith.extsi %11 : i32 to i64
    %13 = arith.muli %12, %0 : i64
    %14 = arith.subi %arg7, %c1_i32 : i32
    %15 = arith.muli %14, %10 : i32
    %16 = arith.muli %15, %c2_i32 : i32
    %17 = arith.extsi %16 : i32 to i64
    %18 = arith.extsi %1 : i32 to i64
    %19 = arith.extsi %10 : i32 to i64
    %20 = arith.extsi %arg11 : i32 to i64
    scf.for %arg19 = %8 to %17 step %18  : i64 {
      %25 = arith.remsi %arg19, %c2_i64 : i64
      %26 = arith.divsi %arg19, %c2_i64 : i64
      %27 = arith.remsi %26, %19 : i64
      %28 = arith.divsi %26, %19 : i64
      %29 = arith.cmpi sge, %28, %c1_i64 : i64
      %30 = scf.if %29 -> (i64) {
        %59 = arith.addi %28, %c1_i64 : i64
        scf.yield %59 : i64
      } else {
        scf.yield %28 : i64
      }
      %31 = arith.muli %27, %c112_i64 : i64
      %32 = arith.muli %25, %c128_i64 : i64
      %33 = arith.muli %30, %c2_i64 : i64
      %34 = arith.addi %33, %25 : i64
      %35 = arith.muli %34, %20 : i64
      %36 = arith.muli %30, %0 : i64
      %37 = arith.addi %13, %36 : i64
      %38 = arith.index_cast %31 : i64 to index
      %39 = arith.index_cast %arg10 : i32 to index
      %40 = arith.muli %38, %39 : index
      %41 = arith.index_cast %35 : i64 to index
      %42 = arith.addi %40, %41 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [112, 128], strides: [%39, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<112x128xbf16>
      %43 = arith.addi %38, %c112 : index
      %44 = arith.index_cast %arg9 : i32 to index
      %45 = arith.maxsi %38, %44 : index
      %46 = arith.minsi %43, %45 : index
      %47 = arith.subi %46, %38 : index
      %48 = arith.cmpi slt, %47, %c112 : index
      scf.if %48 {
        linalg.fill ins(%cst : bf16) outs(%alloc : memref<112x128xbf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      %subview_0 = memref.subview %alloc[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %subview, %subview_0 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %49 = bufferization.to_tensor %alloc restrict writable : memref<112x128xbf16>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [112, 128], strides: [%39, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<112x128xbf16>
      scf.if %48 {
        linalg.fill ins(%cst : bf16) outs(%alloc_2 : memref<112x128xbf16>)
      } {hivm.unlikely_condition}
      %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %alloc_2[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %subview_3, %subview_4 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %50 = bufferization.to_tensor %alloc_2 restrict writable : memref<112x128xbf16>
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [112, 128], strides: [%39, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
      %alloc_6 = memref.alloc() : memref<112x128xbf16>
      scf.if %48 {
        linalg.fill ins(%cst : bf16) outs(%alloc_6 : memref<112x128xbf16>)
      } {hivm.unlikely_condition}
      %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
      %subview_8 = memref.subview %alloc_6[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %subview_7, %subview_8 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %51 = bufferization.to_tensor %alloc_6 restrict writable : memref<112x128xbf16>
      %52 = arith.index_cast %37 : i64 to index
      %53 = arith.muli %38, %c768 : index
      %54 = arith.addi %52, %53 : index
      %55 = arith.index_cast %32 : i64 to index
      %56 = arith.addi %54, %55 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%56], sizes: [112, 128], strides: [768, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[768, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %49[0, 0] [%47, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
      %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[768, 1], offset: ?>> to memref<?x128xbf16, strided<[768, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[768, 1], offset: ?>>) -> ()
      %57 = arith.addi %56, %c256 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%57], sizes: [112, 128], strides: [768, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[768, 1], offset: ?>>
      %extracted_slice_12 = tensor.extract_slice %50[0, 0] [%47, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
      %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[768, 1], offset: ?>> to memref<?x128xbf16, strided<[768, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_12 in writable %subview_13 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[768, 1], offset: ?>>) -> ()
      %58 = arith.addi %56, %c512 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [%58], sizes: [112, 128], strides: [768, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[768, 1], offset: ?>>
      %extracted_slice_15 = tensor.extract_slice %51[0, 0] [%47, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
      %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[768, 1], offset: ?>> to memref<?x128xbf16, strided<[768, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[768, 1], offset: ?>>) -> ()
    }
    %21 = arith.cmpi sgt, %arg7, %c1_i32 : i32
    scf.if %21 {
      hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    }
    %22 = arith.remsi %8, %c2_i64 : i64
    %23 = arith.cmpi eq, %22, %c0_i64 : i64
    scf.if %23 {
      scf.for %arg19 = %4 to %arg7 step %2  : i32 {
        %25 = arith.cmpi ne, %arg19, %c1_i32 : i32
        scf.if %25 {
          %26 = arith.addi %13, %0 : i64
          %27 = arith.extsi %arg19 : i32 to i64
          %28 = arith.muli %27, %0 : i64
          scf.for %arg20 = %c0_i32 to %10 step %c780_i32  : i32 {
            %29 = arith.addi %arg20, %c780_i32 : i32
            %30 = arith.minsi %29, %10 : i32
            %31 = arith.subi %30, %arg20 : i32
            %32 = arith.subi %30, %c1_i32 : i32
            %33 = arith.muli %32, %c112_i32 : i32
            %34 = arith.subi %arg9, %33 : i32
            %35 = arith.minsi %34, %c112_i32 : i32
            %36 = arith.subi %31, %c1_i32 : i32
            %37 = arith.muli %36, %c112_i32 : i32
            %38 = arith.addi %37, %35 : i32
            %39 = arith.muli %arg20, %c86016_i32 : i32
            %40 = arith.index_cast %26 : i64 to index
            %41 = arith.index_cast %39 : i32 to index
            %42 = arith.addi %40, %41 : index
            %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%42], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
            %43 = arith.index_cast %28 : i64 to index
            %44 = arith.addi %43, %41 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
            %45 = arith.muli %38, %c1536_i32 : i32
            hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_getmem_bfloat16"} "dist.aclshmem_getmem_bfloat16" ins(%reinterpret_cast_0, %reinterpret_cast, %45, %arg19 : memref<1xbf16, strided<[1], offset: ?>>, memref<1xbf16, strided<[1], offset: ?>>, i32, i32)
          }
        }
      }
    }
    %24 = arith.cmpi eq, %22, %c1_i64 : i64
    scf.if %24 {
      %25 = arith.muli %10, %c2_i32 : i32
      scf.for %arg19 = %4 to %25 step %2  : i32 {
        %26 = arith.remsi %arg19, %c2_i32 : i32
        %27 = arith.divsi %arg19, %c2_i32 : i32
        %28 = arith.muli %27, %c112_i32 : i32
        %29 = arith.muli %26, %c128_i32 : i32
        %30 = arith.addi %26, %c2_i32 : i32
        %31 = arith.muli %30, %arg11 : i32
        %32 = arith.index_cast %28 : i32 to index
        %33 = arith.index_cast %arg10 : i32 to index
        %34 = arith.muli %32, %33 : index
        %35 = arith.index_cast %31 : i32 to index
        %36 = arith.addi %34, %35 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [112, 128], strides: [%33, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<112x128xbf16>
        %37 = arith.addi %32, %c112 : index
        %38 = arith.index_cast %arg9 : i32 to index
        %39 = arith.maxsi %32, %38 : index
        %40 = arith.minsi %37, %39 : index
        %41 = arith.subi %40, %32 : index
        %42 = arith.cmpi slt, %41, %c112 : index
        scf.if %42 {
          linalg.fill ins(%cst : bf16) outs(%alloc : memref<112x128xbf16>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_0 = memref.subview %alloc[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview, %subview_0 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %43 = bufferization.to_tensor %alloc restrict writable : memref<112x128xbf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [112, 128], strides: [%33, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
        %alloc_2 = memref.alloc() : memref<112x128xbf16>
        scf.if %42 {
          linalg.fill ins(%cst : bf16) outs(%alloc_2 : memref<112x128xbf16>)
        } {hivm.unlikely_condition}
        %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_4 = memref.subview %alloc_2[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_3, %subview_4 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %44 = bufferization.to_tensor %alloc_2 restrict writable : memref<112x128xbf16>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [112, 128], strides: [%33, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
        %alloc_6 = memref.alloc() : memref<112x128xbf16>
        scf.if %42 {
          linalg.fill ins(%cst : bf16) outs(%alloc_6 : memref<112x128xbf16>)
        } {hivm.unlikely_condition}
        %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_6[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_7, %subview_8 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %45 = bufferization.to_tensor %alloc_6 restrict writable : memref<112x128xbf16>
        %46 = arith.index_cast %arg12 : i32 to index
        %47 = arith.muli %32, %c768 : index
        %48 = arith.addi %46, %47 : index
        %49 = arith.index_cast %29 : i32 to index
        %50 = arith.addi %48, %49 : index
        %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%50], sizes: [112, 128], strides: [768, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[768, 1], offset: ?>>
        %extracted_slice = tensor.extract_slice %43[0, 0] [%41, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
        %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16, strided<[768, 1], offset: ?>> to memref<?x128xbf16, strided<[768, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[768, 1], offset: ?>>) -> ()
        %51 = arith.addi %50, %c256 : index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [112, 128], strides: [768, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[768, 1], offset: ?>>
        %extracted_slice_12 = tensor.extract_slice %44[0, 0] [%41, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
        %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16, strided<[768, 1], offset: ?>> to memref<?x128xbf16, strided<[768, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_12 in writable %subview_13 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[768, 1], offset: ?>>) -> ()
        %52 = arith.addi %50, %c512 : index
        %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [112, 128], strides: [768, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[768, 1], offset: ?>>
        %extracted_slice_15 = tensor.extract_slice %45[0, 0] [%41, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
        %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%41, 128] [1, 1] : memref<112x128xbf16, strided<[768, 1], offset: ?>> to memref<?x128xbf16, strided<[768, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[768, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
