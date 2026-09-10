// REQUIRES: hivmc, shmem
// RUN: bishengir-compile %s --target=Ascend950DT_9582 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=False --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o --enable-vf-merge-level=1

module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">} {
  func.func @kernel_qkv_fuse_a2a_head_grouped(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c200704 = arith.constant 200704 : index
    %c100352 = arith.constant 100352 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c112 = arith.constant 112 : index
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c14336_i32 = arith.constant 14336 : i32
    %c4_i32 = arith.constant 4 : i32
    %c112_i32 = arith.constant 112 : i32
    %c7_i32 = arith.constant 7 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c63_i32 = arith.constant 63 : i32
    %c301056_i32 = arith.constant 301056 : i32
    %c0_i32 = arith.constant 0 : i32
    %c111_i32 = arith.constant 111 : i32
    %0 = arith.muli %arg21, %c32_i32 : i32
    %1 = arith.addi %0, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg21, %c1_i32 : i32
    %4 = arith.muli %3, %c32_i32 : i32
    %5 = arith.addi %4, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.addi %arg13, %c111_i32 : i32
    %8 = arith.divsi %7, %c112_i32 : i32
    %9 = arith.muli %arg11, %c301056_i32 : i32
    %10 = arith.muli %arg11, %c16_i32 : i32
    %11 = arith.cmpi sgt, %6, %2 : i32
    %12 = arith.cmpi eq, %6, %2 : i32
    scf.for %arg24 = %c0_i32 to %8 step %c4_i32  : i32 {
      scf.if %11 {
        scf.for %arg25 = %2 to %10 step %c32_i32  : i32 {
          %13 = arith.remsi %arg25, %c4_i32 : i32
          %14 = arith.divsi %arg25, %c4_i32 : i32
          %15 = arith.remsi %14, %arg11 : i32
          %16 = arith.divsi %14, %arg11 : i32
          %17 = arith.addi %arg24, %13 : i32
          %18 = arith.cmpi slt, %17, %8 : i32
          scf.if %18 {
            %19 = arith.muli %17, %c112_i32 : i32
            %20 = arith.muli %arg10, %arg13 : i32
            %21 = arith.muli %15, %c7_i32 : i32
            %22 = arith.remsi %17, %arg12 : i32
            %23 = arith.muli %22, %9 : i32
            %24 = arith.muli %16, %c2_i32 : i32
            scf.for %arg26 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
              %25 = arith.addi %24, %arg26 : i32
              %26 = arith.cmpi slt, %25, %c7_i32 : i32
              scf.if %26 {
                %27 = arith.addi %21, %25 : i32
                %28 = arith.muli %27, %arg15 : i32
                %29 = arith.index_cast %19 : i32 to index
                %30 = arith.index_cast %arg14 : i32 to index
                %31 = arith.muli %29, %30 : index
                %32 = arith.index_cast %28 : i32 to index
                %33 = arith.addi %31, %32 : index
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [112, 128], strides: [%30, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<112x128xbf16>
                %34 = arith.addi %29, %c112 : index
                %35 = arith.index_cast %arg13 : i32 to index
                %36 = arith.maxsi %29, %35 : index
                %37 = arith.minsi %34, %36 : index
                %38 = arith.subi %37, %29 : index
                %39 = arith.cmpi slt, %38, %c112 : index
                scf.if %39 {
                  linalg.fill ins(%cst : bf16) outs(%alloc : memref<112x128xbf16>)
                } {hivm.unlikely_condition}
                %subview = memref.subview %reinterpret_cast[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                %subview_0 = memref.subview %alloc[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
                memref.copy %subview, %subview_0 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
                %40 = bufferization.to_tensor %alloc restrict writable : memref<112x128xbf16>
                %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [112, 128], strides: [%30, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<112x128xbf16>
                scf.if %39 {
                  linalg.fill ins(%cst : bf16) outs(%alloc_2 : memref<112x128xbf16>)
                } {hivm.unlikely_condition}
                %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                %subview_4 = memref.subview %alloc_2[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
                memref.copy %subview_3, %subview_4 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
                %41 = bufferization.to_tensor %alloc_2 restrict writable : memref<112x128xbf16>
                %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [112, 128], strides: [%30, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                %alloc_6 = memref.alloc() : memref<112x128xbf16>
                scf.if %39 {
                  linalg.fill ins(%cst : bf16) outs(%alloc_6 : memref<112x128xbf16>)
                } {hivm.unlikely_condition}
                %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                %subview_8 = memref.subview %alloc_6[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
                memref.copy %subview_7, %subview_8 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
                %42 = bufferization.to_tensor %alloc_6 restrict writable : memref<112x128xbf16>
                %43 = arith.cmpi eq, %15, %arg10 : i32
                scf.if %43 {
                  %44 = arith.muli %25, %arg17 : i32
                  %45 = arith.index_cast %20 : i32 to index
                  %46 = arith.addi %45, %29 : index
                  %47 = arith.index_cast %arg16 : i32 to index
                  %48 = arith.muli %46, %47 : index
                  %49 = arith.index_cast %44 : i32 to index
                  %50 = arith.addi %48, %49 : index
                  %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%50], sizes: [112, 128], strides: [%47, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                  %extracted_slice = tensor.extract_slice %40[0, 0] [%38, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                  %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                  bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%50], sizes: [112, 128], strides: [%47, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                  %extracted_slice_12 = tensor.extract_slice %41[0, 0] [%38, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                  %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                  bufferization.materialize_in_destination %extracted_slice_12 in writable %subview_13 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
                  %reinterpret_cast_14 = memref.reinterpret_cast %arg7 to offset: [%50], sizes: [112, 128], strides: [%47, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                  %extracted_slice_15 = tensor.extract_slice %42[0, 0] [%38, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                  %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                  bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
                } else {
                  %44 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_ptr_bfloat16"} "dist.aclshmem_ptr_bfloat16" ins(%arg8, %15 : memref<?xbf16>, i32) -> memref<?xbf16>
                  %45 = arith.muli %arg10, %c301056_i32 : i32
                  %46 = arith.addi %23, %45 : i32
                  %47 = arith.muli %25, %c14336_i32 : i32
                  %48 = arith.addi %46, %47 : i32
                  %49 = arith.index_cast %48 : i32 to index
                  %reinterpret_cast_9 = memref.reinterpret_cast %44 to offset: [%49], sizes: [112, 128], strides: [128, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[128, 1], offset: ?>>
                  %extracted_slice = tensor.extract_slice %40[0, 0] [%38, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                  %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
                  bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[128, 1], offset: ?>>) -> ()
                  %50 = arith.addi %49, %c100352 : index
                  %reinterpret_cast_11 = memref.reinterpret_cast %44 to offset: [%50], sizes: [112, 128], strides: [128, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[128, 1], offset: ?>>
                  %extracted_slice_12 = tensor.extract_slice %41[0, 0] [%38, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                  %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
                  bufferization.materialize_in_destination %extracted_slice_12 in writable %subview_13 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[128, 1], offset: ?>>) -> ()
                  %51 = arith.addi %49, %c200704 : index
                  %reinterpret_cast_14 = memref.reinterpret_cast %44 to offset: [%51], sizes: [112, 128], strides: [128, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[128, 1], offset: ?>>
                  %extracted_slice_15 = tensor.extract_slice %42[0, 0] [%38, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                  %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%38, 128] [1, 1] : memref<112x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
                  bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[128, 1], offset: ?>>) -> ()
                }
              }
            }
          }
        }
        hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_fence"} "dist.aclshmem_fence"
        scf.for %arg25 = %2 to %10 step %c32_i32  : i32 {
          %13 = arith.remsi %arg25, %c4_i32 : i32
          %14 = arith.divsi %arg25, %c4_i32 : i32
          %15 = arith.remsi %14, %arg11 : i32
          %16 = arith.divsi %14, %arg11 : i32
          %17 = arith.addi %arg24, %13 : i32
          %18 = arith.cmpi slt, %17, %8 : i32
          %19 = arith.cmpi ne, %15, %arg10 : i32
          %20 = arith.andi %18, %19 : i1
          scf.if %20 {
            %21 = arith.remsi %17, %arg12 : i32
            %22 = arith.muli %21, %arg11 : i32
            %23 = arith.addi %22, %arg10 : i32
            %24 = arith.muli %23, %c4_i32 : i32
            %25 = arith.addi %24, %16 : i32
            %26 = arith.muli %25, %c16_i32 : i32
            %27 = arith.index_cast %26 : i32 to index
            %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
            hivm.hir.custom {commScope = 3 : i32, gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmemx_signal_op"} "dist.aclshmemx_signal_op" ins(%reinterpret_cast, %c1_i32, %c0_i32, %15 : memref<1xi32, strided<[1], offset: ?>>, i32, i32, i32)
          }
        }
      }
      scf.if %12 {
        %13 = arith.subi %arg21, %2 : i32
        scf.for %arg25 = %13 to %10 step %c32_i32  : i32 {
          %14 = arith.remsi %arg25, %c4_i32 : i32
          %15 = arith.divsi %arg25, %c4_i32 : i32
          %16 = arith.remsi %15, %arg11 : i32
          %17 = arith.divsi %15, %arg11 : i32
          %18 = arith.addi %arg24, %14 : i32
          %19 = arith.cmpi slt, %18, %8 : i32
          %20 = arith.cmpi ne, %16, %arg10 : i32
          %21 = arith.andi %19, %20 : i1
          scf.if %21 {
            %22 = arith.muli %18, %c112_i32 : i32
            %23 = arith.muli %16, %arg13 : i32
            %24 = arith.remsi %18, %arg12 : i32
            %25 = arith.muli %24, %9 : i32
            %26 = arith.muli %24, %arg11 : i32
            %27 = arith.addi %26, %16 : i32
            %28 = arith.muli %27, %c4_i32 : i32
            %29 = arith.addi %28, %17 : i32
            %30 = arith.muli %29, %c16_i32 : i32
            %31 = arith.index_cast %30 : i32 to index
            %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%31], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
            %32 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, scope = 1 : i32, semantic = 2 : i32, symbol = "aclshmem_wait_int32"} "dist.aclshmem_wait_int32" ins(%reinterpret_cast, %c1_i32, %c1_i32 : memref<1xi32, strided<[1], offset: ?>>, i32, i32) -> i32
            %33 = hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "aclshmem_consume_token_bfloat16_ptr_1d"} "dist.aclshmem_consume_token_bfloat16_ptr_1d" ins(%arg8, %32 : memref<?xbf16>, i32) -> memref<?xbf16>
            %34 = arith.muli %17, %c2_i32 : i32
            scf.for %arg26 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
              %35 = arith.addi %34, %arg26 : i32
              %36 = arith.cmpi slt, %35, %c7_i32 : i32
              scf.if %36 {
                %37 = arith.muli %16, %c301056_i32 : i32
                %38 = arith.addi %25, %37 : i32
                %39 = arith.muli %35, %c14336_i32 : i32
                %40 = arith.addi %38, %39 : i32
                %41 = arith.index_cast %40 : i32 to index
                %reinterpret_cast_0 = memref.reinterpret_cast %33 to offset: [%41], sizes: [112, 128], strides: [128, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[128, 1], offset: ?>>
                %alloc = memref.alloc() : memref<112x128xbf16>
                %42 = arith.index_cast %22 : i32 to index
                %43 = arith.addi %42, %c112 : index
                %44 = arith.index_cast %arg13 : i32 to index
                %45 = arith.maxsi %42, %44 : index
                %46 = arith.minsi %43, %45 : index
                %47 = arith.subi %46, %42 : index
                %48 = arith.cmpi slt, %47, %c112 : index
                scf.if %48 {
                  linalg.fill ins(%cst : bf16) outs(%alloc : memref<112x128xbf16>)
                } {hivm.unlikely_condition}
                %subview = memref.subview %reinterpret_cast_0[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
                %subview_1 = memref.subview %alloc[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
                memref.copy %subview, %subview_1 : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
                %49 = bufferization.to_tensor %alloc restrict writable : memref<112x128xbf16>
                %50 = arith.addi %41, %c100352 : index
                %reinterpret_cast_2 = memref.reinterpret_cast %33 to offset: [%50], sizes: [112, 128], strides: [128, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[128, 1], offset: ?>>
                %alloc_3 = memref.alloc() : memref<112x128xbf16>
                scf.if %48 {
                  linalg.fill ins(%cst : bf16) outs(%alloc_3 : memref<112x128xbf16>)
                } {hivm.unlikely_condition}
                %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
                %subview_5 = memref.subview %alloc_3[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
                memref.copy %subview_4, %subview_5 : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
                %51 = bufferization.to_tensor %alloc_3 restrict writable : memref<112x128xbf16>
                %52 = arith.addi %41, %c200704 : index
                %reinterpret_cast_6 = memref.reinterpret_cast %33 to offset: [%52], sizes: [112, 128], strides: [128, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[128, 1], offset: ?>>
                %alloc_7 = memref.alloc() : memref<112x128xbf16>
                scf.if %48 {
                  linalg.fill ins(%cst : bf16) outs(%alloc_7 : memref<112x128xbf16>)
                } {hivm.unlikely_condition}
                %subview_8 = memref.subview %reinterpret_cast_6[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
                %subview_9 = memref.subview %alloc_7[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
                memref.copy %subview_8, %subview_9 : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
                %53 = bufferization.to_tensor %alloc_7 restrict writable : memref<112x128xbf16>
                %54 = arith.muli %35, %arg17 : i32
                %55 = arith.index_cast %23 : i32 to index
                %56 = arith.addi %55, %42 : index
                %57 = arith.index_cast %arg16 : i32 to index
                %58 = arith.muli %56, %57 : index
                %59 = arith.index_cast %54 : i32 to index
                %60 = arith.addi %58, %59 : index
                %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [112, 128], strides: [%57, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                %extracted_slice = tensor.extract_slice %49[0, 0] [%47, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                %subview_11 = memref.subview %reinterpret_cast_10[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                bufferization.materialize_in_destination %extracted_slice in writable %subview_11 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
                %reinterpret_cast_12 = memref.reinterpret_cast %arg6 to offset: [%60], sizes: [112, 128], strides: [%57, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                %extracted_slice_13 = tensor.extract_slice %51[0, 0] [%47, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                %subview_14 = memref.subview %reinterpret_cast_12[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                bufferization.materialize_in_destination %extracted_slice_13 in writable %subview_14 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
                %reinterpret_cast_15 = memref.reinterpret_cast %arg7 to offset: [%60], sizes: [112, 128], strides: [%57, 1] : memref<?xbf16> to memref<112x128xbf16, strided<[?, 1], offset: ?>>
                %extracted_slice_16 = tensor.extract_slice %53[0, 0] [%47, 128] [1, 1] : tensor<112x128xbf16> to tensor<?x128xbf16>
                %subview_17 = memref.subview %reinterpret_cast_15[0, 0] [%47, 128] [1, 1] : memref<112x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
                bufferization.materialize_in_destination %extracted_slice_16 in writable %subview_17 : (tensor<?x128xbf16>, memref<?x128xbf16, strided<[?, 1], offset: ?>>) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
