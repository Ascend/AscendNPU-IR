// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true \
// RUN:   --enable-tla-mode --disable-auto-inject-block-sync=true \
// RUN:   --save-temps=%t.saved -o %t.o
// RUN: test -s %t.o
// RUN: FileCheck %s --input-file=%t.saved/module.hivm.opt.mlir

// Check that the explicit Cube-to-Vector handoff survives kernel splitting.
// CHECK-LABEL: func.func @_sparse_mla_mix_aic(
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 1
// CHECK-LABEL: func.func @_sparse_mla_mix_aiv(
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 1

module attributes {hacc.mode = "tla", hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_sparse_mla(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16>, %arg3: memref<?xbf16>, %arg4: memref<?xi32>, %arg5: memref<?xi64>, %arg6: memref<?xbf16>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?xbf16>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1024_i64 = arith.constant 1024 : i64
    %c32 = arith.constant 32 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c448 = arith.constant 448 : index
    %c384 = arith.constant 384 : index
    %c320 = arith.constant 320 : index
    %c256 = arith.constant 256 : index
    %c192 = arith.constant 192 : index
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c28_i32 = arith.constant 28 : i32
    %c8_i64 = arith.constant 8 : i64
    %c15_i32 = arith.constant 15 : i32
    %c144 = arith.constant 144 : index
    %cst = arith.constant 0.0416666679 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c48_i64 = arith.constant 48 : i64
    %c64 = arith.constant 64 : index
    %false = arith.constant false
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %c272 = arith.constant 272 : index
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c576_i32 = arith.constant 576 : i32
    %c32_i64 = arith.constant 32 : i64
    %c32_i32 = arith.constant 32 : i32
    %c8191_i32 = arith.constant 8191 : i32
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %c2_i64 = arith.constant 2 : i64
    %c2_i32 = arith.constant 2 : i32
    %c128_i64 = arith.constant 128 : i64
    %c2048_i32 = arith.constant 2048 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c16_i64 = arith.constant 16 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8192_i32 = arith.constant 8192 : i32
    scf.for %arg16 = %arg13 to %c8192_i32 step %c28_i32  : i32 {
      %alloc = memref.alloc() : memref<32x1x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_3 = memref.alloc() : memref<4x1x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() : memref<96x8x16x16xbf16, #hivm.address_space<cbuf>>
      %collapse_shape = memref.collapse_shape %alloc_4 [[0], [1, 2, 3]] : memref<96x8x16x16xbf16, #hivm.address_space<cbuf>> into memref<96x2048xbf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
      %collapse_shape_6 = memref.collapse_shape %alloc_5 [[0], [1, 2, 3]] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> into memref<8x2048xbf16, #hivm.address_space<cbuf>>
      %alloc_7 = memref.alloc() : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>>
      %collapse_shape_8 = memref.collapse_shape %alloc_7 [[0], [1, 2, 3]] : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>> into memref<8x256xbf16, #hivm.address_space<cbuf>>
      %alloc_9 = memref.alloc() : memref<8x1x16x16xf32, #hivm.address_space<cc>>
      %alloc_10 = memref.alloc() : memref<32x1x16x16xf32, #hivm.address_space<cc>>
      %alloc_11 = memref.alloc() : memref<8x128xf32, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xf32, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x512xf32, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x512xbf16, #hivm.address_space<ub>>
      %alloc_15 = memref.alloc() : memref<8x144xbf16, #hivm.address_space<ub>>
      %alloc_16 = memref.alloc() : memref<256xi32, #hivm.address_space<ub>>
      %alloc_17 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %alloc_18 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %alloc_19 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
      %alloc_20 = memref.alloc() : memref<16x640xbf16, #hivm.address_space<ub>>
      %alloc_21 = memref.alloc() : memref<16x640xbf16, #hivm.address_space<ub>>
      %subview = memref.subview %alloc_20[0, 512] [16, 128] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<16x128xbf16, strided<[640, 1], offset: 512>, #hivm.address_space<ub>>
      %cast = memref.cast %subview : memref<16x128xbf16, strided<[640, 1], offset: 512>, #hivm.address_space<ub>> to memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_22 = memref.subview %alloc_21[0, 512] [16, 128] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<16x128xbf16, strided<[640, 1], offset: 512>, #hivm.address_space<ub>>
      %cast_23 = memref.cast %subview_22 : memref<16x128xbf16, strided<[640, 1], offset: 512>, #hivm.address_space<ub>> to memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>>
      %alloc_24 = memref.alloc() : memref<36x272xbf16, #hivm.address_space<ub>>
      %alloc_25 = memref.alloc() : memref<36x272xbf16, #hivm.address_space<ub>>
      %subview_26 = memref.subview %alloc_16[0] [128] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<128xi32, strided<[1]>, #hivm.address_space<ub>>
      scope.scope : () -> () {
        %0 = arith.muli %arg16, %c8192_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [16, 512], strides: [512, 1] : memref<?xbf16> to memref<16x512xbf16, strided<[512, 1], offset: ?>>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<16x512xbf16, strided<[512, 1], offset: ?>>) outs(%alloc : memref<32x1x16x16xbf16, #hivm.address_space<cbuf>>)
        %2 = arith.muli %arg16, %c1024_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %reinterpret_cast_27 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [16, 64], strides: [64, 1] : memref<?xbf16> to memref<16x64xbf16, strided<[64, 1], offset: ?>>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_27 : memref<16x64xbf16, strided<[64, 1], offset: ?>>) outs(%alloc_3 : memref<4x1x16x16xbf16, #hivm.address_space<cbuf>>)
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        scope.scope : () -> () {
          %0 = ave.hir.scalar_broadcast %cst_2 : f32 -> vector<64xf32>
          %1 = ave.hir.scalar_broadcast %cst_1 : f32 -> vector<64xf32>
          %2 = ave.hir.pge <ALL> : vector<64xi1>
          ave.hir.masked_store <NORM_B32> %alloc_17[%c0], %2, %1 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
          %3 = ave.hir.pge <ALL> : vector<64xi1>
          ave.hir.masked_store <NORM_B32> %alloc_18[%c0], %3, %0 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
          scf.for %arg17 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %4 = ave.hir.pge <ALL> : vector<64xi1>
            %5 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%5, %c0], %4, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %6 = ave.hir.pge <ALL> : vector<64xi1>
            %7 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%7, %c64], %6, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %8 = ave.hir.pge <ALL> : vector<64xi1>
            %9 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%9, %c128], %8, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %10 = ave.hir.pge <ALL> : vector<64xi1>
            %11 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%11, %c192], %10, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %12 = ave.hir.pge <ALL> : vector<64xi1>
            %13 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%13, %c256], %12, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %14 = ave.hir.pge <ALL> : vector<64xi1>
            %15 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%15, %c320], %14, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %16 = ave.hir.pge <ALL> : vector<64xi1>
            %17 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%17, %c384], %16, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %18 = ave.hir.pge <ALL> : vector<64xi1>
            %19 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%19, %c448], %18, %0 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
          }
          scope.return
        } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %0 = hivm.hir.get_sub_block_idx -> i64
        %1 = arith.muli %arg16, %c2048_i32 : i32
        %2 = arith.index_cast %1 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%2], sizes: [128], strides: [1] : memref<?xi32> to memref<128xi32, strided<[1], offset: ?>>
        hivm.hir.load ins(%reinterpret_cast : memref<128xi32, strided<[1], offset: ?>>) outs(%subview_26 : memref<128xi32, strided<[1]>, #hivm.address_space<ub>>)
        scf.for %arg17 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %3 = arith.muli %0, %c64_i64 : i64
          %4 = arith.muli %arg17, %c32_i32 : i32
          %5 = arith.extsi %4 : i32 to i64
          %6 = arith.addi %3, %5 : i64
          %7 = arith.muli %arg16, %c2048_i32 : i32
          %8 = arith.extsi %7 : i32 to i64
          %9 = arith.addi %8, %6 : i64
          %10 = arith.divsi %9, %c2_i64 : i64
          %11 = arith.index_cast %10 : i64 to index
          %reinterpret_cast_27 = memref.reinterpret_cast %arg5 to offset: [%11], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %12 = memref.load %reinterpret_cast_27[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %13 = arith.andi %12, %c4294967295_i64 : i64
          %14 = arith.trunci %13 : i64 to i32
          %15 = arith.andi %14, %c8191_i32 : i32
          %16 = arith.shrui %12, %c32_i64 : i64
          %17 = arith.trunci %16 : i64 to i32
          %18 = arith.andi %17, %c8191_i32 : i32
          %19 = arith.muli %15, %c576_i32 : i32
          %20 = arith.index_cast %19 : i32 to index
          %reinterpret_cast_28 = memref.reinterpret_cast %arg9 to offset: [%20], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_29 = memref.subview %alloc_20[0, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_28 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_29 : memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>)
          %21 = arith.muli %18, %c576_i32 : i32
          %22 = arith.index_cast %21 : i32 to index
          %reinterpret_cast_30 = memref.reinterpret_cast %arg9 to offset: [%22], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_31 = memref.subview %alloc_20[1, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_30 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_31 : memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>)
          %23 = arith.muli %arg16, %c2048_i32 : i32
          %24 = arith.extsi %23 : i32 to i64
          %25 = arith.addi %24, %6 : i64
          %26 = arith.divsi %25, %c2_i64 : i64
          %27 = arith.index_cast %26 : i64 to index
          %28 = arith.addi %27, %c1 : index
          %reinterpret_cast_32 = memref.reinterpret_cast %arg5 to offset: [%28], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %29 = memref.load %reinterpret_cast_32[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %30 = arith.andi %29, %c4294967295_i64 : i64
          %31 = arith.trunci %30 : i64 to i32
          %32 = arith.andi %31, %c8191_i32 : i32
          %33 = arith.shrui %29, %c32_i64 : i64
          %34 = arith.trunci %33 : i64 to i32
          %35 = arith.andi %34, %c8191_i32 : i32
          %36 = arith.muli %32, %c576_i32 : i32
          %37 = arith.index_cast %36 : i32 to index
          %reinterpret_cast_33 = memref.reinterpret_cast %arg9 to offset: [%37], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_34 = memref.subview %alloc_20[2, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_33 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_34 : memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>)
          %38 = arith.muli %35, %c576_i32 : i32
          %39 = arith.index_cast %38 : i32 to index
          %reinterpret_cast_35 = memref.reinterpret_cast %arg9 to offset: [%39], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_36 = memref.subview %alloc_20[3, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_35 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_36 : memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>)
          %40 = arith.muli %arg16, %c2048_i32 : i32
          %41 = arith.extsi %40 : i32 to i64
          %42 = arith.addi %41, %6 : i64
          %43 = arith.divsi %42, %c2_i64 : i64
          %44 = arith.index_cast %43 : i64 to index
          %45 = arith.addi %44, %c2 : index
          %reinterpret_cast_37 = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %46 = memref.load %reinterpret_cast_37[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %47 = arith.andi %46, %c4294967295_i64 : i64
          %48 = arith.trunci %47 : i64 to i32
          %49 = arith.andi %48, %c8191_i32 : i32
          %50 = arith.shrui %46, %c32_i64 : i64
          %51 = arith.trunci %50 : i64 to i32
          %52 = arith.andi %51, %c8191_i32 : i32
          %53 = arith.muli %49, %c576_i32 : i32
          %54 = arith.index_cast %53 : i32 to index
          %reinterpret_cast_38 = memref.reinterpret_cast %arg9 to offset: [%54], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_39 = memref.subview %alloc_20[4, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_38 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_39 : memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>)
          %55 = arith.muli %52, %c576_i32 : i32
          %56 = arith.index_cast %55 : i32 to index
          %reinterpret_cast_40 = memref.reinterpret_cast %arg9 to offset: [%56], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_41 = memref.subview %alloc_20[5, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_40 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_41 : memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>)
          %57 = arith.muli %arg16, %c2048_i32 : i32
          %58 = arith.extsi %57 : i32 to i64
          %59 = arith.addi %58, %6 : i64
          %60 = arith.divsi %59, %c2_i64 : i64
          %61 = arith.index_cast %60 : i64 to index
          %62 = arith.addi %61, %c3 : index
          %reinterpret_cast_42 = memref.reinterpret_cast %arg5 to offset: [%62], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %63 = memref.load %reinterpret_cast_42[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %64 = arith.andi %63, %c4294967295_i64 : i64
          %65 = arith.trunci %64 : i64 to i32
          %66 = arith.andi %65, %c8191_i32 : i32
          %67 = arith.shrui %63, %c32_i64 : i64
          %68 = arith.trunci %67 : i64 to i32
          %69 = arith.andi %68, %c8191_i32 : i32
          %70 = arith.muli %66, %c576_i32 : i32
          %71 = arith.index_cast %70 : i32 to index
          %reinterpret_cast_43 = memref.reinterpret_cast %arg9 to offset: [%71], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_44 = memref.subview %alloc_20[6, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_43 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_44 : memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>)
          %72 = arith.muli %69, %c576_i32 : i32
          %73 = arith.index_cast %72 : i32 to index
          %reinterpret_cast_45 = memref.reinterpret_cast %arg9 to offset: [%73], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_46 = memref.subview %alloc_20[7, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_45 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_46 : memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>)
          %74 = arith.muli %arg16, %c2048_i32 : i32
          %75 = arith.extsi %74 : i32 to i64
          %76 = arith.addi %75, %6 : i64
          %77 = arith.divsi %76, %c2_i64 : i64
          %78 = arith.index_cast %77 : i64 to index
          %79 = arith.addi %78, %c4 : index
          %reinterpret_cast_47 = memref.reinterpret_cast %arg5 to offset: [%79], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %80 = memref.load %reinterpret_cast_47[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %81 = arith.andi %80, %c4294967295_i64 : i64
          %82 = arith.trunci %81 : i64 to i32
          %83 = arith.andi %82, %c8191_i32 : i32
          %84 = arith.shrui %80, %c32_i64 : i64
          %85 = arith.trunci %84 : i64 to i32
          %86 = arith.andi %85, %c8191_i32 : i32
          %87 = arith.muli %83, %c576_i32 : i32
          %88 = arith.index_cast %87 : i32 to index
          %reinterpret_cast_48 = memref.reinterpret_cast %arg9 to offset: [%88], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_49 = memref.subview %alloc_20[8, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_48 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_49 : memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>)
          %89 = arith.muli %86, %c576_i32 : i32
          %90 = arith.index_cast %89 : i32 to index
          %reinterpret_cast_50 = memref.reinterpret_cast %arg9 to offset: [%90], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_51 = memref.subview %alloc_20[9, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_50 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_51 : memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>)
          %91 = arith.muli %arg16, %c2048_i32 : i32
          %92 = arith.extsi %91 : i32 to i64
          %93 = arith.addi %92, %6 : i64
          %94 = arith.divsi %93, %c2_i64 : i64
          %95 = arith.index_cast %94 : i64 to index
          %96 = arith.addi %95, %c5 : index
          %reinterpret_cast_52 = memref.reinterpret_cast %arg5 to offset: [%96], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %97 = memref.load %reinterpret_cast_52[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %98 = arith.andi %97, %c4294967295_i64 : i64
          %99 = arith.trunci %98 : i64 to i32
          %100 = arith.andi %99, %c8191_i32 : i32
          %101 = arith.shrui %97, %c32_i64 : i64
          %102 = arith.trunci %101 : i64 to i32
          %103 = arith.andi %102, %c8191_i32 : i32
          %104 = arith.muli %100, %c576_i32 : i32
          %105 = arith.index_cast %104 : i32 to index
          %reinterpret_cast_53 = memref.reinterpret_cast %arg9 to offset: [%105], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_54 = memref.subview %alloc_20[10, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_53 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_54 : memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>)
          %106 = arith.muli %103, %c576_i32 : i32
          %107 = arith.index_cast %106 : i32 to index
          %reinterpret_cast_55 = memref.reinterpret_cast %arg9 to offset: [%107], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_56 = memref.subview %alloc_20[11, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_55 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_56 : memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>)
          %108 = arith.muli %arg16, %c2048_i32 : i32
          %109 = arith.extsi %108 : i32 to i64
          %110 = arith.addi %109, %6 : i64
          %111 = arith.divsi %110, %c2_i64 : i64
          %112 = arith.index_cast %111 : i64 to index
          %113 = arith.addi %112, %c6 : index
          %reinterpret_cast_57 = memref.reinterpret_cast %arg5 to offset: [%113], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %114 = memref.load %reinterpret_cast_57[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %115 = arith.andi %114, %c4294967295_i64 : i64
          %116 = arith.trunci %115 : i64 to i32
          %117 = arith.andi %116, %c8191_i32 : i32
          %118 = arith.shrui %114, %c32_i64 : i64
          %119 = arith.trunci %118 : i64 to i32
          %120 = arith.andi %119, %c8191_i32 : i32
          %121 = arith.muli %117, %c576_i32 : i32
          %122 = arith.index_cast %121 : i32 to index
          %reinterpret_cast_58 = memref.reinterpret_cast %arg9 to offset: [%122], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_59 = memref.subview %alloc_20[12, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_58 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_59 : memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>)
          %123 = arith.muli %120, %c576_i32 : i32
          %124 = arith.index_cast %123 : i32 to index
          %reinterpret_cast_60 = memref.reinterpret_cast %arg9 to offset: [%124], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_61 = memref.subview %alloc_20[13, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_60 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_61 : memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>)
          %125 = arith.muli %arg16, %c2048_i32 : i32
          %126 = arith.extsi %125 : i32 to i64
          %127 = arith.addi %126, %6 : i64
          %128 = arith.divsi %127, %c2_i64 : i64
          %129 = arith.index_cast %128 : i64 to index
          %130 = arith.addi %129, %c7 : index
          %reinterpret_cast_62 = memref.reinterpret_cast %arg5 to offset: [%130], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %131 = memref.load %reinterpret_cast_62[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %132 = arith.andi %131, %c4294967295_i64 : i64
          %133 = arith.trunci %132 : i64 to i32
          %134 = arith.andi %133, %c8191_i32 : i32
          %135 = arith.shrui %131, %c32_i64 : i64
          %136 = arith.trunci %135 : i64 to i32
          %137 = arith.andi %136, %c8191_i32 : i32
          %138 = arith.muli %134, %c576_i32 : i32
          %139 = arith.index_cast %138 : i32 to index
          %reinterpret_cast_63 = memref.reinterpret_cast %arg9 to offset: [%139], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_64 = memref.subview %alloc_20[14, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_63 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_64 : memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>)
          %140 = arith.muli %137, %c576_i32 : i32
          %141 = arith.index_cast %140 : i32 to index
          %reinterpret_cast_65 = memref.reinterpret_cast %arg9 to offset: [%141], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_66 = memref.subview %alloc_20[15, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_65 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_66 : memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>)
          scope.scope : () -> () {
            %291 = ave.hir.pge <VL64> : vector<128xi1>
            scf.for %arg18 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
              %292 = arith.index_cast %arg18 : i32 to index
              %res = ave.hir.vload <NORM> %alloc_20[%292, %c0] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %293 = arith.muli %arg18, %c16_i32 : i32
              %294 = ave.hir.pge <ALL> : vector<128xi1>
              %295 = arith.index_cast %293 : i32 to index
              ave.hir.store_with_stride %alloc_24[%c0, %295], %c272, %294, %res : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %296 = arith.index_cast %arg18 : i32 to index
              %res_115 = ave.hir.vload <NORM> %alloc_20[%296, %c128] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %297 = arith.muli %arg18, %c16_i32 : i32
              %298 = ave.hir.pge <ALL> : vector<128xi1>
              %299 = arith.index_cast %297 : i32 to index
              ave.hir.store_with_stride %alloc_24[%c8, %299], %c272, %298, %res_115 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %300 = arith.index_cast %arg18 : i32 to index
              %res_116 = ave.hir.vload <NORM> %alloc_20[%300, %c256] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %301 = arith.muli %arg18, %c16_i32 : i32
              %302 = ave.hir.pge <ALL> : vector<128xi1>
              %303 = arith.index_cast %301 : i32 to index
              ave.hir.store_with_stride %alloc_24[%c16, %303], %c272, %302, %res_116 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %304 = arith.index_cast %arg18 : i32 to index
              %res_117 = ave.hir.vload <NORM> %alloc_20[%304, %c384] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %305 = arith.muli %arg18, %c16_i32 : i32
              %306 = ave.hir.pge <ALL> : vector<128xi1>
              %307 = arith.index_cast %305 : i32 to index
              ave.hir.store_with_stride %alloc_24[%c24, %307], %c272, %306, %res_117 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %308 = arith.index_cast %arg18 : i32 to index
              %res_118 = ave.hir.vload <NORM> %cast[%308, %c0] : memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
              %309 = arith.muli %arg18, %c16_i32 : i32
              %310 = arith.index_cast %309 : i32 to index
              ave.hir.store_with_stride %alloc_24[%c32, %310], %c272, %291, %res_118 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
            }
            scope.return
          } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
          %subview_67 = memref.subview %alloc_24[0, 0] [32, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>
          %142 = arith.muli %6, %c16_i64 : i64
          %143 = arith.index_cast %142 : i64 to index
          %subview_68 = memref.subview %collapse_shape[0, %143] [32, 256] [1, 1] : memref<96x2048xbf16, #hivm.address_space<cbuf>> to memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%subview_67 : memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>) outs(%subview_68 : memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
          %subview_69 = memref.subview %alloc_24[32, 0] [4, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>
          %144 = arith.muli %6, %c16_i64 : i64
          %145 = arith.index_cast %144 : i64 to index
          %subview_70 = memref.subview %collapse_shape_6[0, %145] [4, 256] [1, 1] : memref<8x2048xbf16, #hivm.address_space<cbuf>> to memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%subview_69 : memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>) outs(%subview_70 : memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
          %146 = arith.muli %0, %c64_i64 : i64
          %147 = arith.muli %arg17, %c2_i32 : i32
          %148 = arith.addi %147, %c1_i32 : i32
          %149 = arith.muli %148, %c16_i32 : i32
          %150 = arith.extsi %149 : i32 to i64
          %151 = arith.addi %146, %150 : i64
          %152 = arith.muli %arg16, %c2048_i32 : i32
          %153 = arith.extsi %152 : i32 to i64
          %154 = arith.addi %153, %151 : i64
          %155 = arith.divsi %154, %c2_i64 : i64
          %156 = arith.index_cast %155 : i64 to index
          %reinterpret_cast_71 = memref.reinterpret_cast %arg5 to offset: [%156], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %157 = memref.load %reinterpret_cast_71[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %158 = arith.andi %157, %c4294967295_i64 : i64
          %159 = arith.trunci %158 : i64 to i32
          %160 = arith.andi %159, %c8191_i32 : i32
          %161 = arith.shrui %157, %c32_i64 : i64
          %162 = arith.trunci %161 : i64 to i32
          %163 = arith.andi %162, %c8191_i32 : i32
          %164 = arith.muli %160, %c576_i32 : i32
          %165 = arith.index_cast %164 : i32 to index
          %reinterpret_cast_72 = memref.reinterpret_cast %arg9 to offset: [%165], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_73 = memref.subview %alloc_21[0, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_72 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_73 : memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>)
          %166 = arith.muli %163, %c576_i32 : i32
          %167 = arith.index_cast %166 : i32 to index
          %reinterpret_cast_74 = memref.reinterpret_cast %arg9 to offset: [%167], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_75 = memref.subview %alloc_21[1, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_74 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_75 : memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>)
          %168 = arith.muli %arg16, %c2048_i32 : i32
          %169 = arith.extsi %168 : i32 to i64
          %170 = arith.addi %169, %151 : i64
          %171 = arith.divsi %170, %c2_i64 : i64
          %172 = arith.index_cast %171 : i64 to index
          %173 = arith.addi %172, %c1 : index
          %reinterpret_cast_76 = memref.reinterpret_cast %arg5 to offset: [%173], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %174 = memref.load %reinterpret_cast_76[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %175 = arith.andi %174, %c4294967295_i64 : i64
          %176 = arith.trunci %175 : i64 to i32
          %177 = arith.andi %176, %c8191_i32 : i32
          %178 = arith.shrui %174, %c32_i64 : i64
          %179 = arith.trunci %178 : i64 to i32
          %180 = arith.andi %179, %c8191_i32 : i32
          %181 = arith.muli %177, %c576_i32 : i32
          %182 = arith.index_cast %181 : i32 to index
          %reinterpret_cast_77 = memref.reinterpret_cast %arg9 to offset: [%182], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_78 = memref.subview %alloc_21[2, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_77 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_78 : memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>)
          %183 = arith.muli %180, %c576_i32 : i32
          %184 = arith.index_cast %183 : i32 to index
          %reinterpret_cast_79 = memref.reinterpret_cast %arg9 to offset: [%184], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_80 = memref.subview %alloc_21[3, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_79 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_80 : memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>)
          %185 = arith.muli %arg16, %c2048_i32 : i32
          %186 = arith.extsi %185 : i32 to i64
          %187 = arith.addi %186, %151 : i64
          %188 = arith.divsi %187, %c2_i64 : i64
          %189 = arith.index_cast %188 : i64 to index
          %190 = arith.addi %189, %c2 : index
          %reinterpret_cast_81 = memref.reinterpret_cast %arg5 to offset: [%190], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %191 = memref.load %reinterpret_cast_81[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %192 = arith.andi %191, %c4294967295_i64 : i64
          %193 = arith.trunci %192 : i64 to i32
          %194 = arith.andi %193, %c8191_i32 : i32
          %195 = arith.shrui %191, %c32_i64 : i64
          %196 = arith.trunci %195 : i64 to i32
          %197 = arith.andi %196, %c8191_i32 : i32
          %198 = arith.muli %194, %c576_i32 : i32
          %199 = arith.index_cast %198 : i32 to index
          %reinterpret_cast_82 = memref.reinterpret_cast %arg9 to offset: [%199], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_83 = memref.subview %alloc_21[4, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_82 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_83 : memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>)
          %200 = arith.muli %197, %c576_i32 : i32
          %201 = arith.index_cast %200 : i32 to index
          %reinterpret_cast_84 = memref.reinterpret_cast %arg9 to offset: [%201], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_85 = memref.subview %alloc_21[5, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_84 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_85 : memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>)
          %202 = arith.muli %arg16, %c2048_i32 : i32
          %203 = arith.extsi %202 : i32 to i64
          %204 = arith.addi %203, %151 : i64
          %205 = arith.divsi %204, %c2_i64 : i64
          %206 = arith.index_cast %205 : i64 to index
          %207 = arith.addi %206, %c3 : index
          %reinterpret_cast_86 = memref.reinterpret_cast %arg5 to offset: [%207], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %208 = memref.load %reinterpret_cast_86[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %209 = arith.andi %208, %c4294967295_i64 : i64
          %210 = arith.trunci %209 : i64 to i32
          %211 = arith.andi %210, %c8191_i32 : i32
          %212 = arith.shrui %208, %c32_i64 : i64
          %213 = arith.trunci %212 : i64 to i32
          %214 = arith.andi %213, %c8191_i32 : i32
          %215 = arith.muli %211, %c576_i32 : i32
          %216 = arith.index_cast %215 : i32 to index
          %reinterpret_cast_87 = memref.reinterpret_cast %arg9 to offset: [%216], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_88 = memref.subview %alloc_21[6, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_87 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_88 : memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>)
          %217 = arith.muli %214, %c576_i32 : i32
          %218 = arith.index_cast %217 : i32 to index
          %reinterpret_cast_89 = memref.reinterpret_cast %arg9 to offset: [%218], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_90 = memref.subview %alloc_21[7, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_89 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_90 : memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>)
          %219 = arith.muli %arg16, %c2048_i32 : i32
          %220 = arith.extsi %219 : i32 to i64
          %221 = arith.addi %220, %151 : i64
          %222 = arith.divsi %221, %c2_i64 : i64
          %223 = arith.index_cast %222 : i64 to index
          %224 = arith.addi %223, %c4 : index
          %reinterpret_cast_91 = memref.reinterpret_cast %arg5 to offset: [%224], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %225 = memref.load %reinterpret_cast_91[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %226 = arith.andi %225, %c4294967295_i64 : i64
          %227 = arith.trunci %226 : i64 to i32
          %228 = arith.andi %227, %c8191_i32 : i32
          %229 = arith.shrui %225, %c32_i64 : i64
          %230 = arith.trunci %229 : i64 to i32
          %231 = arith.andi %230, %c8191_i32 : i32
          %232 = arith.muli %228, %c576_i32 : i32
          %233 = arith.index_cast %232 : i32 to index
          %reinterpret_cast_92 = memref.reinterpret_cast %arg9 to offset: [%233], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_93 = memref.subview %alloc_21[8, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_92 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_93 : memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>)
          %234 = arith.muli %231, %c576_i32 : i32
          %235 = arith.index_cast %234 : i32 to index
          %reinterpret_cast_94 = memref.reinterpret_cast %arg9 to offset: [%235], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_95 = memref.subview %alloc_21[9, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_94 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_95 : memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>)
          %236 = arith.muli %arg16, %c2048_i32 : i32
          %237 = arith.extsi %236 : i32 to i64
          %238 = arith.addi %237, %151 : i64
          %239 = arith.divsi %238, %c2_i64 : i64
          %240 = arith.index_cast %239 : i64 to index
          %241 = arith.addi %240, %c5 : index
          %reinterpret_cast_96 = memref.reinterpret_cast %arg5 to offset: [%241], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %242 = memref.load %reinterpret_cast_96[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %243 = arith.andi %242, %c4294967295_i64 : i64
          %244 = arith.trunci %243 : i64 to i32
          %245 = arith.andi %244, %c8191_i32 : i32
          %246 = arith.shrui %242, %c32_i64 : i64
          %247 = arith.trunci %246 : i64 to i32
          %248 = arith.andi %247, %c8191_i32 : i32
          %249 = arith.muli %245, %c576_i32 : i32
          %250 = arith.index_cast %249 : i32 to index
          %reinterpret_cast_97 = memref.reinterpret_cast %arg9 to offset: [%250], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_98 = memref.subview %alloc_21[10, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_97 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_98 : memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>)
          %251 = arith.muli %248, %c576_i32 : i32
          %252 = arith.index_cast %251 : i32 to index
          %reinterpret_cast_99 = memref.reinterpret_cast %arg9 to offset: [%252], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_100 = memref.subview %alloc_21[11, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_99 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_100 : memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>)
          %253 = arith.muli %arg16, %c2048_i32 : i32
          %254 = arith.extsi %253 : i32 to i64
          %255 = arith.addi %254, %151 : i64
          %256 = arith.divsi %255, %c2_i64 : i64
          %257 = arith.index_cast %256 : i64 to index
          %258 = arith.addi %257, %c6 : index
          %reinterpret_cast_101 = memref.reinterpret_cast %arg5 to offset: [%258], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %259 = memref.load %reinterpret_cast_101[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %260 = arith.andi %259, %c4294967295_i64 : i64
          %261 = arith.trunci %260 : i64 to i32
          %262 = arith.andi %261, %c8191_i32 : i32
          %263 = arith.shrui %259, %c32_i64 : i64
          %264 = arith.trunci %263 : i64 to i32
          %265 = arith.andi %264, %c8191_i32 : i32
          %266 = arith.muli %262, %c576_i32 : i32
          %267 = arith.index_cast %266 : i32 to index
          %reinterpret_cast_102 = memref.reinterpret_cast %arg9 to offset: [%267], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_103 = memref.subview %alloc_21[12, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_102 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_103 : memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>)
          %268 = arith.muli %265, %c576_i32 : i32
          %269 = arith.index_cast %268 : i32 to index
          %reinterpret_cast_104 = memref.reinterpret_cast %arg9 to offset: [%269], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_105 = memref.subview %alloc_21[13, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_104 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_105 : memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>)
          %270 = arith.muli %arg16, %c2048_i32 : i32
          %271 = arith.extsi %270 : i32 to i64
          %272 = arith.addi %271, %151 : i64
          %273 = arith.divsi %272, %c2_i64 : i64
          %274 = arith.index_cast %273 : i64 to index
          %275 = arith.addi %274, %c7 : index
          %reinterpret_cast_106 = memref.reinterpret_cast %arg5 to offset: [%275], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %276 = memref.load %reinterpret_cast_106[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %277 = arith.andi %276, %c4294967295_i64 : i64
          %278 = arith.trunci %277 : i64 to i32
          %279 = arith.andi %278, %c8191_i32 : i32
          %280 = arith.shrui %276, %c32_i64 : i64
          %281 = arith.trunci %280 : i64 to i32
          %282 = arith.andi %281, %c8191_i32 : i32
          %283 = arith.muli %279, %c576_i32 : i32
          %284 = arith.index_cast %283 : i32 to index
          %reinterpret_cast_107 = memref.reinterpret_cast %arg9 to offset: [%284], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_108 = memref.subview %alloc_21[14, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_107 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_108 : memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>)
          %285 = arith.muli %282, %c576_i32 : i32
          %286 = arith.index_cast %285 : i32 to index
          %reinterpret_cast_109 = memref.reinterpret_cast %arg9 to offset: [%286], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
          %subview_110 = memref.subview %alloc_21[15, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>
          hivm.hir.load ins(%reinterpret_cast_109 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_110 : memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>)
          scope.scope : () -> () {
            %291 = ave.hir.pge <VL64> : vector<128xi1>
            scf.for %arg18 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
              %292 = arith.index_cast %arg18 : i32 to index
              %res = ave.hir.vload <NORM> %alloc_21[%292, %c0] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %293 = arith.muli %arg18, %c16_i32 : i32
              %294 = ave.hir.pge <ALL> : vector<128xi1>
              %295 = arith.index_cast %293 : i32 to index
              ave.hir.store_with_stride %alloc_25[%c0, %295], %c272, %294, %res : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %296 = arith.index_cast %arg18 : i32 to index
              %res_115 = ave.hir.vload <NORM> %alloc_21[%296, %c128] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %297 = arith.muli %arg18, %c16_i32 : i32
              %298 = ave.hir.pge <ALL> : vector<128xi1>
              %299 = arith.index_cast %297 : i32 to index
              ave.hir.store_with_stride %alloc_25[%c8, %299], %c272, %298, %res_115 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %300 = arith.index_cast %arg18 : i32 to index
              %res_116 = ave.hir.vload <NORM> %alloc_21[%300, %c256] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %301 = arith.muli %arg18, %c16_i32 : i32
              %302 = ave.hir.pge <ALL> : vector<128xi1>
              %303 = arith.index_cast %301 : i32 to index
              ave.hir.store_with_stride %alloc_25[%c16, %303], %c272, %302, %res_116 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %304 = arith.index_cast %arg18 : i32 to index
              %res_117 = ave.hir.vload <NORM> %alloc_21[%304, %c384] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
              %305 = arith.muli %arg18, %c16_i32 : i32
              %306 = ave.hir.pge <ALL> : vector<128xi1>
              %307 = arith.index_cast %305 : i32 to index
              ave.hir.store_with_stride %alloc_25[%c24, %307], %c272, %306, %res_117 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              %308 = arith.index_cast %arg18 : i32 to index
              %res_118 = ave.hir.vload <NORM> %cast_23[%308, %c0] : memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
              %309 = arith.muli %arg18, %c16_i32 : i32
              %310 = arith.index_cast %309 : i32 to index
              ave.hir.store_with_stride %alloc_25[%c32, %310], %c272, %291, %res_118 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
            }
            scope.return
          } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
          %subview_111 = memref.subview %alloc_25[0, 0] [32, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>
          %287 = arith.muli %151, %c16_i64 : i64
          %288 = arith.index_cast %287 : i64 to index
          %subview_112 = memref.subview %collapse_shape[0, %288] [32, 256] [1, 1] : memref<96x2048xbf16, #hivm.address_space<cbuf>> to memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%subview_111 : memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>) outs(%subview_112 : memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
          %subview_113 = memref.subview %alloc_25[32, 0] [4, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>
          %289 = arith.muli %151, %c16_i64 : i64
          %290 = arith.index_cast %289 : i64 to index
          %subview_114 = memref.subview %collapse_shape_6[0, %290] [4, 256] [1, 1] : memref<8x2048xbf16, #hivm.address_space<cbuf>> to memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%subview_113 : memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>) outs(%subview_114 : memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
        }
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 0
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scf.for %arg17 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %0 = arith.remsi %arg17, %c3_i32 : i32
        %1 = arith.muli %0, %c32_i32 : i32
        %2 = arith.index_cast %1 : i32 to index
        %subview_27 = memref.subview %alloc_4[%2, 0, 0, 0] [32, 8, 16, 16] [1, 1, 1, 1] : memref<96x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        %3 = arith.remsi %arg17, %c2_i32 : i32
        %4 = arith.muli %3, %c4_i32 : i32
        %5 = arith.index_cast %4 : i32 to index
        %subview_28 = memref.subview %alloc_5[%5, 0, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<4x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        scope.scope : () -> () {
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 0
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %subview_27, %true, %c16, %c512, %c128 : memref<32x1x16x16xbf16, #hivm.address_space<cbuf>>, memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_9 : memref<8x1x16x16xf32, #hivm.address_space<cc>>)
          hivm.hir.mmadL1 {b_transpose} ins(%alloc_3, %subview_28, %false, %c16, %c64, %c128 : memref<4x1x16x16xbf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_9 : memref<8x1x16x16xf32, #hivm.address_space<cc>>)
          %6 = arith.cmpi sge, %arg17, %c1_i32 : i32
          scf.if %6 {
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 13
          }
          hivm.hir.fixpipe {c0_pad_en = true, dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_9 : memref<8x1x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_11 : memref<8x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 1
          %7 = arith.cmpi sge, %arg17, %c1_i32 : i32
          scf.if %7 {
            %8 = arith.subi %arg17, %c1_i32 : i32
            %9 = arith.remsi %8, %c3_i32 : i32
            %10 = arith.muli %9, %c32_i32 : i32
            %11 = arith.index_cast %10 : i32 to index
            %subview_29 = memref.subview %alloc_4[%11, 0, 0, 0] [32, 8, 16, 16] [1, 1, 1, 1] : memref<96x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 2
            hivm.hir.mmadL1 ins(%alloc_7, %subview_29, %true, %c16, %c128, %c512 : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>>, memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_10 : memref<32x1x16x16xf32, #hivm.address_space<cc>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_M>] flag = 6
            %12 = arith.cmpi sge, %arg17, %c2_i32 : i32
            scf.if %12 {
              hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 4
            }
            hivm.hir.fixpipe {c0_pad_en = true, dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_10 : memref<32x1x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_12 : memref<8x512xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 3
          }
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %6 = hivm.hir.get_sub_block_idx -> i64
          %7 = arith.addi %arg17, %c1_i32 : i32
          %8 = arith.cmpi slt, %7, %c16_i32 : i32
          scf.if %8 {
            %20 = arith.addi %arg17, %c1_i32 : i32
            %21 = arith.muli %20, %c128_i32 : i32
            %22 = arith.muli %arg16, %c2048_i32 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = arith.index_cast %21 : i32 to index
            %25 = arith.addi %23, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [128], strides: [1] : memref<?xi32> to memref<128xi32, strided<[1], offset: ?>>
            %26 = arith.addi %arg17, %c1_i32 : i32
            %27 = arith.remsi %26, %c2_i32 : i32
            %28 = arith.muli %27, %c128_i32 : i32
            %29 = arith.index_cast %28 : i32 to index
            %subview_33 = memref.subview %alloc_16[%29] [128] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast : memref<128xi32, strided<[1], offset: ?>>) outs(%subview_33 : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>)
            %30 = arith.muli %6, %c64_i64 : i64
            %31 = arith.muli %arg16, %c2048_i32 : i32
            %32 = arith.addi %31, %21 : i32
            %33 = arith.extsi %32 : i32 to i64
            %34 = arith.addi %33, %30 : i64
            %35 = arith.divsi %34, %c2_i64 : i64
            %36 = arith.index_cast %35 : i64 to index
            %reinterpret_cast_34 = memref.reinterpret_cast %arg5 to offset: [%36], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %37 = memref.load %reinterpret_cast_34[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %38 = arith.andi %37, %c4294967295_i64 : i64
            %39 = arith.trunci %38 : i64 to i32
            %40 = arith.andi %39, %c8191_i32 : i32
            %41 = arith.shrui %37, %c32_i64 : i64
            %42 = arith.trunci %41 : i64 to i32
            %43 = arith.andi %42, %c8191_i32 : i32
            %44 = arith.muli %40, %c576_i32 : i32
            %45 = arith.index_cast %44 : i32 to index
            %reinterpret_cast_35 = memref.reinterpret_cast %arg9 to offset: [%45], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_36 = memref.subview %alloc_20[0, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_35 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_36 : memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>)
            %46 = arith.muli %43, %c576_i32 : i32
            %47 = arith.index_cast %46 : i32 to index
            %reinterpret_cast_37 = memref.reinterpret_cast %arg9 to offset: [%47], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_38 = memref.subview %alloc_20[1, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_37 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_38 : memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>)
            %48 = arith.muli %arg16, %c2048_i32 : i32
            %49 = arith.addi %48, %21 : i32
            %50 = arith.extsi %49 : i32 to i64
            %51 = arith.addi %50, %30 : i64
            %52 = arith.divsi %51, %c2_i64 : i64
            %53 = arith.index_cast %52 : i64 to index
            %54 = arith.addi %53, %c1 : index
            %reinterpret_cast_39 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %55 = memref.load %reinterpret_cast_39[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %56 = arith.andi %55, %c4294967295_i64 : i64
            %57 = arith.trunci %56 : i64 to i32
            %58 = arith.andi %57, %c8191_i32 : i32
            %59 = arith.shrui %55, %c32_i64 : i64
            %60 = arith.trunci %59 : i64 to i32
            %61 = arith.andi %60, %c8191_i32 : i32
            %62 = arith.muli %58, %c576_i32 : i32
            %63 = arith.index_cast %62 : i32 to index
            %reinterpret_cast_40 = memref.reinterpret_cast %arg9 to offset: [%63], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_41 = memref.subview %alloc_20[2, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_40 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_41 : memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>)
            %64 = arith.muli %61, %c576_i32 : i32
            %65 = arith.index_cast %64 : i32 to index
            %reinterpret_cast_42 = memref.reinterpret_cast %arg9 to offset: [%65], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_43 = memref.subview %alloc_20[3, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_42 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_43 : memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>)
            %66 = arith.muli %arg16, %c2048_i32 : i32
            %67 = arith.addi %66, %21 : i32
            %68 = arith.extsi %67 : i32 to i64
            %69 = arith.addi %68, %30 : i64
            %70 = arith.divsi %69, %c2_i64 : i64
            %71 = arith.index_cast %70 : i64 to index
            %72 = arith.addi %71, %c2 : index
            %reinterpret_cast_44 = memref.reinterpret_cast %arg5 to offset: [%72], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %73 = memref.load %reinterpret_cast_44[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %74 = arith.andi %73, %c4294967295_i64 : i64
            %75 = arith.trunci %74 : i64 to i32
            %76 = arith.andi %75, %c8191_i32 : i32
            %77 = arith.shrui %73, %c32_i64 : i64
            %78 = arith.trunci %77 : i64 to i32
            %79 = arith.andi %78, %c8191_i32 : i32
            %80 = arith.muli %76, %c576_i32 : i32
            %81 = arith.index_cast %80 : i32 to index
            %reinterpret_cast_45 = memref.reinterpret_cast %arg9 to offset: [%81], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_46 = memref.subview %alloc_20[4, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_45 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_46 : memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>)
            %82 = arith.muli %79, %c576_i32 : i32
            %83 = arith.index_cast %82 : i32 to index
            %reinterpret_cast_47 = memref.reinterpret_cast %arg9 to offset: [%83], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_48 = memref.subview %alloc_20[5, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_47 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_48 : memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>)
            %84 = arith.muli %arg16, %c2048_i32 : i32
            %85 = arith.addi %84, %21 : i32
            %86 = arith.extsi %85 : i32 to i64
            %87 = arith.addi %86, %30 : i64
            %88 = arith.divsi %87, %c2_i64 : i64
            %89 = arith.index_cast %88 : i64 to index
            %90 = arith.addi %89, %c3 : index
            %reinterpret_cast_49 = memref.reinterpret_cast %arg5 to offset: [%90], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %91 = memref.load %reinterpret_cast_49[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %92 = arith.andi %91, %c4294967295_i64 : i64
            %93 = arith.trunci %92 : i64 to i32
            %94 = arith.andi %93, %c8191_i32 : i32
            %95 = arith.shrui %91, %c32_i64 : i64
            %96 = arith.trunci %95 : i64 to i32
            %97 = arith.andi %96, %c8191_i32 : i32
            %98 = arith.muli %94, %c576_i32 : i32
            %99 = arith.index_cast %98 : i32 to index
            %reinterpret_cast_50 = memref.reinterpret_cast %arg9 to offset: [%99], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_51 = memref.subview %alloc_20[6, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_50 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_51 : memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>)
            %100 = arith.muli %97, %c576_i32 : i32
            %101 = arith.index_cast %100 : i32 to index
            %reinterpret_cast_52 = memref.reinterpret_cast %arg9 to offset: [%101], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_53 = memref.subview %alloc_20[7, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_52 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_53 : memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>)
            %102 = arith.muli %arg16, %c2048_i32 : i32
            %103 = arith.addi %102, %21 : i32
            %104 = arith.extsi %103 : i32 to i64
            %105 = arith.addi %104, %30 : i64
            %106 = arith.divsi %105, %c2_i64 : i64
            %107 = arith.index_cast %106 : i64 to index
            %108 = arith.addi %107, %c4 : index
            %reinterpret_cast_54 = memref.reinterpret_cast %arg5 to offset: [%108], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %109 = memref.load %reinterpret_cast_54[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %110 = arith.andi %109, %c4294967295_i64 : i64
            %111 = arith.trunci %110 : i64 to i32
            %112 = arith.andi %111, %c8191_i32 : i32
            %113 = arith.shrui %109, %c32_i64 : i64
            %114 = arith.trunci %113 : i64 to i32
            %115 = arith.andi %114, %c8191_i32 : i32
            %116 = arith.muli %112, %c576_i32 : i32
            %117 = arith.index_cast %116 : i32 to index
            %reinterpret_cast_55 = memref.reinterpret_cast %arg9 to offset: [%117], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_56 = memref.subview %alloc_20[8, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_55 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_56 : memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>)
            %118 = arith.muli %115, %c576_i32 : i32
            %119 = arith.index_cast %118 : i32 to index
            %reinterpret_cast_57 = memref.reinterpret_cast %arg9 to offset: [%119], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_58 = memref.subview %alloc_20[9, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_57 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_58 : memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>)
            %120 = arith.muli %arg16, %c2048_i32 : i32
            %121 = arith.addi %120, %21 : i32
            %122 = arith.extsi %121 : i32 to i64
            %123 = arith.addi %122, %30 : i64
            %124 = arith.divsi %123, %c2_i64 : i64
            %125 = arith.index_cast %124 : i64 to index
            %126 = arith.addi %125, %c5 : index
            %reinterpret_cast_59 = memref.reinterpret_cast %arg5 to offset: [%126], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %127 = memref.load %reinterpret_cast_59[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %128 = arith.andi %127, %c4294967295_i64 : i64
            %129 = arith.trunci %128 : i64 to i32
            %130 = arith.andi %129, %c8191_i32 : i32
            %131 = arith.shrui %127, %c32_i64 : i64
            %132 = arith.trunci %131 : i64 to i32
            %133 = arith.andi %132, %c8191_i32 : i32
            %134 = arith.muli %130, %c576_i32 : i32
            %135 = arith.index_cast %134 : i32 to index
            %reinterpret_cast_60 = memref.reinterpret_cast %arg9 to offset: [%135], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_61 = memref.subview %alloc_20[10, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_60 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_61 : memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>)
            %136 = arith.muli %133, %c576_i32 : i32
            %137 = arith.index_cast %136 : i32 to index
            %reinterpret_cast_62 = memref.reinterpret_cast %arg9 to offset: [%137], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_63 = memref.subview %alloc_20[11, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_62 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_63 : memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>)
            %138 = arith.muli %arg16, %c2048_i32 : i32
            %139 = arith.addi %138, %21 : i32
            %140 = arith.extsi %139 : i32 to i64
            %141 = arith.addi %140, %30 : i64
            %142 = arith.divsi %141, %c2_i64 : i64
            %143 = arith.index_cast %142 : i64 to index
            %144 = arith.addi %143, %c6 : index
            %reinterpret_cast_64 = memref.reinterpret_cast %arg5 to offset: [%144], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %145 = memref.load %reinterpret_cast_64[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %146 = arith.andi %145, %c4294967295_i64 : i64
            %147 = arith.trunci %146 : i64 to i32
            %148 = arith.andi %147, %c8191_i32 : i32
            %149 = arith.shrui %145, %c32_i64 : i64
            %150 = arith.trunci %149 : i64 to i32
            %151 = arith.andi %150, %c8191_i32 : i32
            %152 = arith.muli %148, %c576_i32 : i32
            %153 = arith.index_cast %152 : i32 to index
            %reinterpret_cast_65 = memref.reinterpret_cast %arg9 to offset: [%153], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_66 = memref.subview %alloc_20[12, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_65 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_66 : memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>)
            %154 = arith.muli %151, %c576_i32 : i32
            %155 = arith.index_cast %154 : i32 to index
            %reinterpret_cast_67 = memref.reinterpret_cast %arg9 to offset: [%155], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_68 = memref.subview %alloc_20[13, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_67 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_68 : memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>)
            %156 = arith.muli %arg16, %c2048_i32 : i32
            %157 = arith.addi %156, %21 : i32
            %158 = arith.extsi %157 : i32 to i64
            %159 = arith.addi %158, %30 : i64
            %160 = arith.divsi %159, %c2_i64 : i64
            %161 = arith.index_cast %160 : i64 to index
            %162 = arith.addi %161, %c7 : index
            %reinterpret_cast_69 = memref.reinterpret_cast %arg5 to offset: [%162], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %163 = memref.load %reinterpret_cast_69[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %164 = arith.andi %163, %c4294967295_i64 : i64
            %165 = arith.trunci %164 : i64 to i32
            %166 = arith.andi %165, %c8191_i32 : i32
            %167 = arith.shrui %163, %c32_i64 : i64
            %168 = arith.trunci %167 : i64 to i32
            %169 = arith.andi %168, %c8191_i32 : i32
            %170 = arith.muli %166, %c576_i32 : i32
            %171 = arith.index_cast %170 : i32 to index
            %reinterpret_cast_70 = memref.reinterpret_cast %arg9 to offset: [%171], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_71 = memref.subview %alloc_20[14, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_70 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_71 : memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>)
            %172 = arith.muli %169, %c576_i32 : i32
            %173 = arith.index_cast %172 : i32 to index
            %reinterpret_cast_72 = memref.reinterpret_cast %arg9 to offset: [%173], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_73 = memref.subview %alloc_20[15, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_72 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_73 : memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>)
            %174 = arith.muli %6, %c64_i64 : i64
            %175 = arith.addi %174, %c16_i64 : i64
            %176 = arith.muli %arg16, %c2048_i32 : i32
            %177 = arith.addi %176, %21 : i32
            %178 = arith.extsi %177 : i32 to i64
            %179 = arith.addi %178, %175 : i64
            %180 = arith.divsi %179, %c2_i64 : i64
            %181 = arith.index_cast %180 : i64 to index
            %reinterpret_cast_74 = memref.reinterpret_cast %arg5 to offset: [%181], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %182 = memref.load %reinterpret_cast_74[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %183 = arith.andi %182, %c4294967295_i64 : i64
            %184 = arith.trunci %183 : i64 to i32
            %185 = arith.andi %184, %c8191_i32 : i32
            %186 = arith.shrui %182, %c32_i64 : i64
            %187 = arith.trunci %186 : i64 to i32
            %188 = arith.andi %187, %c8191_i32 : i32
            %189 = arith.muli %185, %c576_i32 : i32
            %190 = arith.index_cast %189 : i32 to index
            %reinterpret_cast_75 = memref.reinterpret_cast %arg9 to offset: [%190], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_76 = memref.subview %alloc_21[0, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_75 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_76 : memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>)
            %191 = arith.muli %188, %c576_i32 : i32
            %192 = arith.index_cast %191 : i32 to index
            %reinterpret_cast_77 = memref.reinterpret_cast %arg9 to offset: [%192], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_78 = memref.subview %alloc_21[1, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_77 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_78 : memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>)
            %193 = arith.muli %arg16, %c2048_i32 : i32
            %194 = arith.addi %193, %21 : i32
            %195 = arith.extsi %194 : i32 to i64
            %196 = arith.addi %195, %175 : i64
            %197 = arith.divsi %196, %c2_i64 : i64
            %198 = arith.index_cast %197 : i64 to index
            %199 = arith.addi %198, %c1 : index
            %reinterpret_cast_79 = memref.reinterpret_cast %arg5 to offset: [%199], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %200 = memref.load %reinterpret_cast_79[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %201 = arith.andi %200, %c4294967295_i64 : i64
            %202 = arith.trunci %201 : i64 to i32
            %203 = arith.andi %202, %c8191_i32 : i32
            %204 = arith.shrui %200, %c32_i64 : i64
            %205 = arith.trunci %204 : i64 to i32
            %206 = arith.andi %205, %c8191_i32 : i32
            %207 = arith.muli %203, %c576_i32 : i32
            %208 = arith.index_cast %207 : i32 to index
            %reinterpret_cast_80 = memref.reinterpret_cast %arg9 to offset: [%208], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_81 = memref.subview %alloc_21[2, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_80 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_81 : memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>)
            %209 = arith.muli %206, %c576_i32 : i32
            %210 = arith.index_cast %209 : i32 to index
            %reinterpret_cast_82 = memref.reinterpret_cast %arg9 to offset: [%210], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_83 = memref.subview %alloc_21[3, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_82 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_83 : memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>)
            %211 = arith.muli %arg16, %c2048_i32 : i32
            %212 = arith.addi %211, %21 : i32
            %213 = arith.extsi %212 : i32 to i64
            %214 = arith.addi %213, %175 : i64
            %215 = arith.divsi %214, %c2_i64 : i64
            %216 = arith.index_cast %215 : i64 to index
            %217 = arith.addi %216, %c2 : index
            %reinterpret_cast_84 = memref.reinterpret_cast %arg5 to offset: [%217], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %218 = memref.load %reinterpret_cast_84[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %219 = arith.andi %218, %c4294967295_i64 : i64
            %220 = arith.trunci %219 : i64 to i32
            %221 = arith.andi %220, %c8191_i32 : i32
            %222 = arith.shrui %218, %c32_i64 : i64
            %223 = arith.trunci %222 : i64 to i32
            %224 = arith.andi %223, %c8191_i32 : i32
            %225 = arith.muli %221, %c576_i32 : i32
            %226 = arith.index_cast %225 : i32 to index
            %reinterpret_cast_85 = memref.reinterpret_cast %arg9 to offset: [%226], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_86 = memref.subview %alloc_21[4, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_85 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_86 : memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>)
            %227 = arith.muli %224, %c576_i32 : i32
            %228 = arith.index_cast %227 : i32 to index
            %reinterpret_cast_87 = memref.reinterpret_cast %arg9 to offset: [%228], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_88 = memref.subview %alloc_21[5, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_87 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_88 : memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>)
            %229 = arith.muli %arg16, %c2048_i32 : i32
            %230 = arith.addi %229, %21 : i32
            %231 = arith.extsi %230 : i32 to i64
            %232 = arith.addi %231, %175 : i64
            %233 = arith.divsi %232, %c2_i64 : i64
            %234 = arith.index_cast %233 : i64 to index
            %235 = arith.addi %234, %c3 : index
            %reinterpret_cast_89 = memref.reinterpret_cast %arg5 to offset: [%235], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %236 = memref.load %reinterpret_cast_89[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %237 = arith.andi %236, %c4294967295_i64 : i64
            %238 = arith.trunci %237 : i64 to i32
            %239 = arith.andi %238, %c8191_i32 : i32
            %240 = arith.shrui %236, %c32_i64 : i64
            %241 = arith.trunci %240 : i64 to i32
            %242 = arith.andi %241, %c8191_i32 : i32
            %243 = arith.muli %239, %c576_i32 : i32
            %244 = arith.index_cast %243 : i32 to index
            %reinterpret_cast_90 = memref.reinterpret_cast %arg9 to offset: [%244], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_91 = memref.subview %alloc_21[6, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_90 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_91 : memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>)
            %245 = arith.muli %242, %c576_i32 : i32
            %246 = arith.index_cast %245 : i32 to index
            %reinterpret_cast_92 = memref.reinterpret_cast %arg9 to offset: [%246], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_93 = memref.subview %alloc_21[7, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_92 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_93 : memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>)
            %247 = arith.muli %arg16, %c2048_i32 : i32
            %248 = arith.addi %247, %21 : i32
            %249 = arith.extsi %248 : i32 to i64
            %250 = arith.addi %249, %175 : i64
            %251 = arith.divsi %250, %c2_i64 : i64
            %252 = arith.index_cast %251 : i64 to index
            %253 = arith.addi %252, %c4 : index
            %reinterpret_cast_94 = memref.reinterpret_cast %arg5 to offset: [%253], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %254 = memref.load %reinterpret_cast_94[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %255 = arith.andi %254, %c4294967295_i64 : i64
            %256 = arith.trunci %255 : i64 to i32
            %257 = arith.andi %256, %c8191_i32 : i32
            %258 = arith.shrui %254, %c32_i64 : i64
            %259 = arith.trunci %258 : i64 to i32
            %260 = arith.andi %259, %c8191_i32 : i32
            %261 = arith.muli %257, %c576_i32 : i32
            %262 = arith.index_cast %261 : i32 to index
            %reinterpret_cast_95 = memref.reinterpret_cast %arg9 to offset: [%262], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_96 = memref.subview %alloc_21[8, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_95 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_96 : memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>)
            %263 = arith.muli %260, %c576_i32 : i32
            %264 = arith.index_cast %263 : i32 to index
            %reinterpret_cast_97 = memref.reinterpret_cast %arg9 to offset: [%264], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_98 = memref.subview %alloc_21[9, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_97 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_98 : memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>)
            %265 = arith.muli %arg16, %c2048_i32 : i32
            %266 = arith.addi %265, %21 : i32
            %267 = arith.extsi %266 : i32 to i64
            %268 = arith.addi %267, %175 : i64
            %269 = arith.divsi %268, %c2_i64 : i64
            %270 = arith.index_cast %269 : i64 to index
            %271 = arith.addi %270, %c5 : index
            %reinterpret_cast_99 = memref.reinterpret_cast %arg5 to offset: [%271], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %272 = memref.load %reinterpret_cast_99[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %273 = arith.andi %272, %c4294967295_i64 : i64
            %274 = arith.trunci %273 : i64 to i32
            %275 = arith.andi %274, %c8191_i32 : i32
            %276 = arith.shrui %272, %c32_i64 : i64
            %277 = arith.trunci %276 : i64 to i32
            %278 = arith.andi %277, %c8191_i32 : i32
            %279 = arith.muli %275, %c576_i32 : i32
            %280 = arith.index_cast %279 : i32 to index
            %reinterpret_cast_100 = memref.reinterpret_cast %arg9 to offset: [%280], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_21[10, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_100 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_101 : memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>)
            %281 = arith.muli %278, %c576_i32 : i32
            %282 = arith.index_cast %281 : i32 to index
            %reinterpret_cast_102 = memref.reinterpret_cast %arg9 to offset: [%282], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_103 = memref.subview %alloc_21[11, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_102 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_103 : memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>)
            %283 = arith.muli %arg16, %c2048_i32 : i32
            %284 = arith.addi %283, %21 : i32
            %285 = arith.extsi %284 : i32 to i64
            %286 = arith.addi %285, %175 : i64
            %287 = arith.divsi %286, %c2_i64 : i64
            %288 = arith.index_cast %287 : i64 to index
            %289 = arith.addi %288, %c6 : index
            %reinterpret_cast_104 = memref.reinterpret_cast %arg5 to offset: [%289], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %290 = memref.load %reinterpret_cast_104[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %291 = arith.andi %290, %c4294967295_i64 : i64
            %292 = arith.trunci %291 : i64 to i32
            %293 = arith.andi %292, %c8191_i32 : i32
            %294 = arith.shrui %290, %c32_i64 : i64
            %295 = arith.trunci %294 : i64 to i32
            %296 = arith.andi %295, %c8191_i32 : i32
            %297 = arith.muli %293, %c576_i32 : i32
            %298 = arith.index_cast %297 : i32 to index
            %reinterpret_cast_105 = memref.reinterpret_cast %arg9 to offset: [%298], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_106 = memref.subview %alloc_21[12, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_105 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_106 : memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>)
            %299 = arith.muli %296, %c576_i32 : i32
            %300 = arith.index_cast %299 : i32 to index
            %reinterpret_cast_107 = memref.reinterpret_cast %arg9 to offset: [%300], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_108 = memref.subview %alloc_21[13, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_107 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_108 : memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>)
            %301 = arith.muli %arg16, %c2048_i32 : i32
            %302 = arith.addi %301, %21 : i32
            %303 = arith.extsi %302 : i32 to i64
            %304 = arith.addi %303, %175 : i64
            %305 = arith.divsi %304, %c2_i64 : i64
            %306 = arith.index_cast %305 : i64 to index
            %307 = arith.addi %306, %c7 : index
            %reinterpret_cast_109 = memref.reinterpret_cast %arg5 to offset: [%307], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %308 = memref.load %reinterpret_cast_109[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %309 = arith.andi %308, %c4294967295_i64 : i64
            %310 = arith.trunci %309 : i64 to i32
            %311 = arith.andi %310, %c8191_i32 : i32
            %312 = arith.shrui %308, %c32_i64 : i64
            %313 = arith.trunci %312 : i64 to i32
            %314 = arith.andi %313, %c8191_i32 : i32
            %315 = arith.muli %311, %c576_i32 : i32
            %316 = arith.index_cast %315 : i32 to index
            %reinterpret_cast_110 = memref.reinterpret_cast %arg9 to offset: [%316], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_111 = memref.subview %alloc_21[14, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_110 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_111 : memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>)
            %317 = arith.muli %314, %c576_i32 : i32
            %318 = arith.index_cast %317 : i32 to index
            %reinterpret_cast_112 = memref.reinterpret_cast %arg9 to offset: [%318], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_113 = memref.subview %alloc_21[15, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_112 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_113 : memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>)
            scope.scope : () -> () {
              %663 = ave.hir.pge <VL64> : vector<128xi1>
              scf.for %arg18 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
                %664 = arith.index_cast %arg18 : i32 to index
                %res = ave.hir.vload <NORM> %alloc_20[%664, %c0] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %665 = arith.muli %arg18, %c16_i32 : i32
                %666 = ave.hir.pge <ALL> : vector<128xi1>
                %667 = arith.index_cast %665 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c0, %667], %c272, %666, %res : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %668 = arith.index_cast %arg18 : i32 to index
                %res_210 = ave.hir.vload <NORM> %alloc_20[%668, %c128] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %669 = arith.muli %arg18, %c16_i32 : i32
                %670 = ave.hir.pge <ALL> : vector<128xi1>
                %671 = arith.index_cast %669 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c8, %671], %c272, %670, %res_210 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %672 = arith.index_cast %arg18 : i32 to index
                %res_211 = ave.hir.vload <NORM> %alloc_20[%672, %c256] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %673 = arith.muli %arg18, %c16_i32 : i32
                %674 = ave.hir.pge <ALL> : vector<128xi1>
                %675 = arith.index_cast %673 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c16, %675], %c272, %674, %res_211 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %676 = arith.index_cast %arg18 : i32 to index
                %res_212 = ave.hir.vload <NORM> %alloc_20[%676, %c384] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %677 = arith.muli %arg18, %c16_i32 : i32
                %678 = ave.hir.pge <ALL> : vector<128xi1>
                %679 = arith.index_cast %677 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c24, %679], %c272, %678, %res_212 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %680 = arith.index_cast %arg18 : i32 to index
                %res_213 = ave.hir.vload <NORM> %cast[%680, %c0] : memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
                %681 = arith.muli %arg18, %c16_i32 : i32
                %682 = arith.index_cast %681 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c32, %682], %c272, %663, %res_213 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              }
              scope.return
            } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
            %subview_114 = memref.subview %alloc_24[0, 0] [32, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>
            %319 = arith.addi %arg17, %c1_i32 : i32
            %320 = arith.remsi %319, %c3_i32 : i32
            %321 = arith.muli %320, %c32_i32 : i32
            %322 = arith.muli %6, %c1024_i64 : i64
            %323 = arith.index_cast %321 : i32 to index
            %324 = arith.index_cast %322 : i64 to index
            %subview_115 = memref.subview %collapse_shape[%323, %324] [32, 256] [1, 1] : memref<96x2048xbf16, #hivm.address_space<cbuf>> to memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_114 : memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>) outs(%subview_115 : memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %subview_116 = memref.subview %alloc_24[32, 0] [4, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>
            %325 = arith.addi %arg17, %c1_i32 : i32
            %326 = arith.remsi %325, %c2_i32 : i32
            %327 = arith.muli %326, %c4_i32 : i32
            %328 = arith.muli %6, %c1024_i64 : i64
            %329 = arith.index_cast %327 : i32 to index
            %330 = arith.index_cast %328 : i64 to index
            %subview_117 = memref.subview %collapse_shape_6[%329, %330] [4, 256] [1, 1] : memref<8x2048xbf16, #hivm.address_space<cbuf>> to memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_116 : memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>) outs(%subview_117 : memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %331 = arith.muli %6, %c64_i64 : i64
            %332 = arith.addi %331, %c32_i64 : i64
            %333 = arith.muli %arg16, %c2048_i32 : i32
            %334 = arith.addi %333, %21 : i32
            %335 = arith.extsi %334 : i32 to i64
            %336 = arith.addi %335, %332 : i64
            %337 = arith.divsi %336, %c2_i64 : i64
            %338 = arith.index_cast %337 : i64 to index
            %reinterpret_cast_118 = memref.reinterpret_cast %arg5 to offset: [%338], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %339 = memref.load %reinterpret_cast_118[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %340 = arith.andi %339, %c4294967295_i64 : i64
            %341 = arith.trunci %340 : i64 to i32
            %342 = arith.andi %341, %c8191_i32 : i32
            %343 = arith.shrui %339, %c32_i64 : i64
            %344 = arith.trunci %343 : i64 to i32
            %345 = arith.andi %344, %c8191_i32 : i32
            %346 = arith.muli %342, %c576_i32 : i32
            %347 = arith.index_cast %346 : i32 to index
            %reinterpret_cast_119 = memref.reinterpret_cast %arg9 to offset: [%347], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_120 = memref.subview %alloc_20[0, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_119 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_120 : memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>)
            %348 = arith.muli %345, %c576_i32 : i32
            %349 = arith.index_cast %348 : i32 to index
            %reinterpret_cast_121 = memref.reinterpret_cast %arg9 to offset: [%349], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_122 = memref.subview %alloc_20[1, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_121 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_122 : memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>)
            %350 = arith.muli %arg16, %c2048_i32 : i32
            %351 = arith.addi %350, %21 : i32
            %352 = arith.extsi %351 : i32 to i64
            %353 = arith.addi %352, %332 : i64
            %354 = arith.divsi %353, %c2_i64 : i64
            %355 = arith.index_cast %354 : i64 to index
            %356 = arith.addi %355, %c1 : index
            %reinterpret_cast_123 = memref.reinterpret_cast %arg5 to offset: [%356], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %357 = memref.load %reinterpret_cast_123[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %358 = arith.andi %357, %c4294967295_i64 : i64
            %359 = arith.trunci %358 : i64 to i32
            %360 = arith.andi %359, %c8191_i32 : i32
            %361 = arith.shrui %357, %c32_i64 : i64
            %362 = arith.trunci %361 : i64 to i32
            %363 = arith.andi %362, %c8191_i32 : i32
            %364 = arith.muli %360, %c576_i32 : i32
            %365 = arith.index_cast %364 : i32 to index
            %reinterpret_cast_124 = memref.reinterpret_cast %arg9 to offset: [%365], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_125 = memref.subview %alloc_20[2, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_124 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_125 : memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>)
            %366 = arith.muli %363, %c576_i32 : i32
            %367 = arith.index_cast %366 : i32 to index
            %reinterpret_cast_126 = memref.reinterpret_cast %arg9 to offset: [%367], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_127 = memref.subview %alloc_20[3, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_126 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_127 : memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>)
            %368 = arith.muli %arg16, %c2048_i32 : i32
            %369 = arith.addi %368, %21 : i32
            %370 = arith.extsi %369 : i32 to i64
            %371 = arith.addi %370, %332 : i64
            %372 = arith.divsi %371, %c2_i64 : i64
            %373 = arith.index_cast %372 : i64 to index
            %374 = arith.addi %373, %c2 : index
            %reinterpret_cast_128 = memref.reinterpret_cast %arg5 to offset: [%374], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %375 = memref.load %reinterpret_cast_128[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %376 = arith.andi %375, %c4294967295_i64 : i64
            %377 = arith.trunci %376 : i64 to i32
            %378 = arith.andi %377, %c8191_i32 : i32
            %379 = arith.shrui %375, %c32_i64 : i64
            %380 = arith.trunci %379 : i64 to i32
            %381 = arith.andi %380, %c8191_i32 : i32
            %382 = arith.muli %378, %c576_i32 : i32
            %383 = arith.index_cast %382 : i32 to index
            %reinterpret_cast_129 = memref.reinterpret_cast %arg9 to offset: [%383], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_130 = memref.subview %alloc_20[4, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_129 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_130 : memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>)
            %384 = arith.muli %381, %c576_i32 : i32
            %385 = arith.index_cast %384 : i32 to index
            %reinterpret_cast_131 = memref.reinterpret_cast %arg9 to offset: [%385], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_132 = memref.subview %alloc_20[5, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_131 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_132 : memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>)
            %386 = arith.muli %arg16, %c2048_i32 : i32
            %387 = arith.addi %386, %21 : i32
            %388 = arith.extsi %387 : i32 to i64
            %389 = arith.addi %388, %332 : i64
            %390 = arith.divsi %389, %c2_i64 : i64
            %391 = arith.index_cast %390 : i64 to index
            %392 = arith.addi %391, %c3 : index
            %reinterpret_cast_133 = memref.reinterpret_cast %arg5 to offset: [%392], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %393 = memref.load %reinterpret_cast_133[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %394 = arith.andi %393, %c4294967295_i64 : i64
            %395 = arith.trunci %394 : i64 to i32
            %396 = arith.andi %395, %c8191_i32 : i32
            %397 = arith.shrui %393, %c32_i64 : i64
            %398 = arith.trunci %397 : i64 to i32
            %399 = arith.andi %398, %c8191_i32 : i32
            %400 = arith.muli %396, %c576_i32 : i32
            %401 = arith.index_cast %400 : i32 to index
            %reinterpret_cast_134 = memref.reinterpret_cast %arg9 to offset: [%401], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_135 = memref.subview %alloc_20[6, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_134 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_135 : memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>)
            %402 = arith.muli %399, %c576_i32 : i32
            %403 = arith.index_cast %402 : i32 to index
            %reinterpret_cast_136 = memref.reinterpret_cast %arg9 to offset: [%403], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_137 = memref.subview %alloc_20[7, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_136 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_137 : memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>)
            %404 = arith.muli %arg16, %c2048_i32 : i32
            %405 = arith.addi %404, %21 : i32
            %406 = arith.extsi %405 : i32 to i64
            %407 = arith.addi %406, %332 : i64
            %408 = arith.divsi %407, %c2_i64 : i64
            %409 = arith.index_cast %408 : i64 to index
            %410 = arith.addi %409, %c4 : index
            %reinterpret_cast_138 = memref.reinterpret_cast %arg5 to offset: [%410], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %411 = memref.load %reinterpret_cast_138[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %412 = arith.andi %411, %c4294967295_i64 : i64
            %413 = arith.trunci %412 : i64 to i32
            %414 = arith.andi %413, %c8191_i32 : i32
            %415 = arith.shrui %411, %c32_i64 : i64
            %416 = arith.trunci %415 : i64 to i32
            %417 = arith.andi %416, %c8191_i32 : i32
            %418 = arith.muli %414, %c576_i32 : i32
            %419 = arith.index_cast %418 : i32 to index
            %reinterpret_cast_139 = memref.reinterpret_cast %arg9 to offset: [%419], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_140 = memref.subview %alloc_20[8, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_139 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_140 : memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>)
            %420 = arith.muli %417, %c576_i32 : i32
            %421 = arith.index_cast %420 : i32 to index
            %reinterpret_cast_141 = memref.reinterpret_cast %arg9 to offset: [%421], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_142 = memref.subview %alloc_20[9, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_141 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_142 : memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>)
            %422 = arith.muli %arg16, %c2048_i32 : i32
            %423 = arith.addi %422, %21 : i32
            %424 = arith.extsi %423 : i32 to i64
            %425 = arith.addi %424, %332 : i64
            %426 = arith.divsi %425, %c2_i64 : i64
            %427 = arith.index_cast %426 : i64 to index
            %428 = arith.addi %427, %c5 : index
            %reinterpret_cast_143 = memref.reinterpret_cast %arg5 to offset: [%428], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %429 = memref.load %reinterpret_cast_143[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %430 = arith.andi %429, %c4294967295_i64 : i64
            %431 = arith.trunci %430 : i64 to i32
            %432 = arith.andi %431, %c8191_i32 : i32
            %433 = arith.shrui %429, %c32_i64 : i64
            %434 = arith.trunci %433 : i64 to i32
            %435 = arith.andi %434, %c8191_i32 : i32
            %436 = arith.muli %432, %c576_i32 : i32
            %437 = arith.index_cast %436 : i32 to index
            %reinterpret_cast_144 = memref.reinterpret_cast %arg9 to offset: [%437], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_145 = memref.subview %alloc_20[10, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_144 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_145 : memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>)
            %438 = arith.muli %435, %c576_i32 : i32
            %439 = arith.index_cast %438 : i32 to index
            %reinterpret_cast_146 = memref.reinterpret_cast %arg9 to offset: [%439], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_147 = memref.subview %alloc_20[11, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_146 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_147 : memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>)
            %440 = arith.muli %arg16, %c2048_i32 : i32
            %441 = arith.addi %440, %21 : i32
            %442 = arith.extsi %441 : i32 to i64
            %443 = arith.addi %442, %332 : i64
            %444 = arith.divsi %443, %c2_i64 : i64
            %445 = arith.index_cast %444 : i64 to index
            %446 = arith.addi %445, %c6 : index
            %reinterpret_cast_148 = memref.reinterpret_cast %arg5 to offset: [%446], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %447 = memref.load %reinterpret_cast_148[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %448 = arith.andi %447, %c4294967295_i64 : i64
            %449 = arith.trunci %448 : i64 to i32
            %450 = arith.andi %449, %c8191_i32 : i32
            %451 = arith.shrui %447, %c32_i64 : i64
            %452 = arith.trunci %451 : i64 to i32
            %453 = arith.andi %452, %c8191_i32 : i32
            %454 = arith.muli %450, %c576_i32 : i32
            %455 = arith.index_cast %454 : i32 to index
            %reinterpret_cast_149 = memref.reinterpret_cast %arg9 to offset: [%455], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_150 = memref.subview %alloc_20[12, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_149 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_150 : memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>)
            %456 = arith.muli %453, %c576_i32 : i32
            %457 = arith.index_cast %456 : i32 to index
            %reinterpret_cast_151 = memref.reinterpret_cast %arg9 to offset: [%457], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_152 = memref.subview %alloc_20[13, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_151 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_152 : memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>)
            %458 = arith.muli %arg16, %c2048_i32 : i32
            %459 = arith.addi %458, %21 : i32
            %460 = arith.extsi %459 : i32 to i64
            %461 = arith.addi %460, %332 : i64
            %462 = arith.divsi %461, %c2_i64 : i64
            %463 = arith.index_cast %462 : i64 to index
            %464 = arith.addi %463, %c7 : index
            %reinterpret_cast_153 = memref.reinterpret_cast %arg5 to offset: [%464], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %465 = memref.load %reinterpret_cast_153[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %466 = arith.andi %465, %c4294967295_i64 : i64
            %467 = arith.trunci %466 : i64 to i32
            %468 = arith.andi %467, %c8191_i32 : i32
            %469 = arith.shrui %465, %c32_i64 : i64
            %470 = arith.trunci %469 : i64 to i32
            %471 = arith.andi %470, %c8191_i32 : i32
            %472 = arith.muli %468, %c576_i32 : i32
            %473 = arith.index_cast %472 : i32 to index
            %reinterpret_cast_154 = memref.reinterpret_cast %arg9 to offset: [%473], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_155 = memref.subview %alloc_20[14, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_154 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_155 : memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>)
            %474 = arith.muli %471, %c576_i32 : i32
            %475 = arith.index_cast %474 : i32 to index
            %reinterpret_cast_156 = memref.reinterpret_cast %arg9 to offset: [%475], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_157 = memref.subview %alloc_20[15, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_156 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_157 : memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>)
            %476 = arith.muli %6, %c64_i64 : i64
            %477 = arith.addi %476, %c16_i64 : i64
            scope.scope : () -> () {
              %663 = ave.hir.pge <VL64> : vector<128xi1>
              scf.for %arg18 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
                %664 = arith.index_cast %arg18 : i32 to index
                %res = ave.hir.vload <NORM> %alloc_21[%664, %c0] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %665 = arith.muli %arg18, %c16_i32 : i32
                %666 = ave.hir.pge <ALL> : vector<128xi1>
                %667 = arith.index_cast %665 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c0, %667], %c272, %666, %res : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %668 = arith.index_cast %arg18 : i32 to index
                %res_210 = ave.hir.vload <NORM> %alloc_21[%668, %c128] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %669 = arith.muli %arg18, %c16_i32 : i32
                %670 = ave.hir.pge <ALL> : vector<128xi1>
                %671 = arith.index_cast %669 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c8, %671], %c272, %670, %res_210 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %672 = arith.index_cast %arg18 : i32 to index
                %res_211 = ave.hir.vload <NORM> %alloc_21[%672, %c256] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %673 = arith.muli %arg18, %c16_i32 : i32
                %674 = ave.hir.pge <ALL> : vector<128xi1>
                %675 = arith.index_cast %673 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c16, %675], %c272, %674, %res_211 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %676 = arith.index_cast %arg18 : i32 to index
                %res_212 = ave.hir.vload <NORM> %alloc_21[%676, %c384] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %677 = arith.muli %arg18, %c16_i32 : i32
                %678 = ave.hir.pge <ALL> : vector<128xi1>
                %679 = arith.index_cast %677 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c24, %679], %c272, %678, %res_212 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %680 = arith.index_cast %arg18 : i32 to index
                %res_213 = ave.hir.vload <NORM> %cast_23[%680, %c0] : memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
                %681 = arith.muli %arg18, %c16_i32 : i32
                %682 = arith.index_cast %681 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c32, %682], %c272, %663, %res_213 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              }
              scope.return
            } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
            %subview_158 = memref.subview %alloc_25[0, 0] [32, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>
            %478 = arith.addi %arg17, %c1_i32 : i32
            %479 = arith.remsi %478, %c3_i32 : i32
            %480 = arith.muli %479, %c32_i32 : i32
            %481 = arith.muli %477, %c16_i64 : i64
            %482 = arith.index_cast %480 : i32 to index
            %483 = arith.index_cast %481 : i64 to index
            %subview_159 = memref.subview %collapse_shape[%482, %483] [32, 256] [1, 1] : memref<96x2048xbf16, #hivm.address_space<cbuf>> to memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_158 : memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>) outs(%subview_159 : memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %subview_160 = memref.subview %alloc_25[32, 0] [4, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>
            %484 = arith.addi %arg17, %c1_i32 : i32
            %485 = arith.remsi %484, %c2_i32 : i32
            %486 = arith.muli %485, %c4_i32 : i32
            %487 = arith.muli %477, %c16_i64 : i64
            %488 = arith.index_cast %486 : i32 to index
            %489 = arith.index_cast %487 : i64 to index
            %subview_161 = memref.subview %collapse_shape_6[%488, %489] [4, 256] [1, 1] : memref<8x2048xbf16, #hivm.address_space<cbuf>> to memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_160 : memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>) outs(%subview_161 : memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %490 = arith.muli %6, %c64_i64 : i64
            %491 = arith.addi %490, %c48_i64 : i64
            %492 = arith.muli %arg16, %c2048_i32 : i32
            %493 = arith.addi %492, %21 : i32
            %494 = arith.extsi %493 : i32 to i64
            %495 = arith.addi %494, %491 : i64
            %496 = arith.divsi %495, %c2_i64 : i64
            %497 = arith.index_cast %496 : i64 to index
            %reinterpret_cast_162 = memref.reinterpret_cast %arg5 to offset: [%497], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %498 = memref.load %reinterpret_cast_162[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %499 = arith.andi %498, %c4294967295_i64 : i64
            %500 = arith.trunci %499 : i64 to i32
            %501 = arith.andi %500, %c8191_i32 : i32
            %502 = arith.shrui %498, %c32_i64 : i64
            %503 = arith.trunci %502 : i64 to i32
            %504 = arith.andi %503, %c8191_i32 : i32
            %505 = arith.muli %501, %c576_i32 : i32
            %506 = arith.index_cast %505 : i32 to index
            %reinterpret_cast_163 = memref.reinterpret_cast %arg9 to offset: [%506], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_164 = memref.subview %alloc_21[0, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_163 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_164 : memref<1x576xbf16, strided<[640, 1]>, #hivm.address_space<ub>>)
            %507 = arith.muli %504, %c576_i32 : i32
            %508 = arith.index_cast %507 : i32 to index
            %reinterpret_cast_165 = memref.reinterpret_cast %arg9 to offset: [%508], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_166 = memref.subview %alloc_21[1, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_165 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_166 : memref<1x576xbf16, strided<[640, 1], offset: 640>, #hivm.address_space<ub>>)
            %509 = arith.muli %arg16, %c2048_i32 : i32
            %510 = arith.addi %509, %21 : i32
            %511 = arith.extsi %510 : i32 to i64
            %512 = arith.addi %511, %491 : i64
            %513 = arith.divsi %512, %c2_i64 : i64
            %514 = arith.index_cast %513 : i64 to index
            %515 = arith.addi %514, %c1 : index
            %reinterpret_cast_167 = memref.reinterpret_cast %arg5 to offset: [%515], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %516 = memref.load %reinterpret_cast_167[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %517 = arith.andi %516, %c4294967295_i64 : i64
            %518 = arith.trunci %517 : i64 to i32
            %519 = arith.andi %518, %c8191_i32 : i32
            %520 = arith.shrui %516, %c32_i64 : i64
            %521 = arith.trunci %520 : i64 to i32
            %522 = arith.andi %521, %c8191_i32 : i32
            %523 = arith.muli %519, %c576_i32 : i32
            %524 = arith.index_cast %523 : i32 to index
            %reinterpret_cast_168 = memref.reinterpret_cast %arg9 to offset: [%524], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_169 = memref.subview %alloc_21[2, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_168 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_169 : memref<1x576xbf16, strided<[640, 1], offset: 1280>, #hivm.address_space<ub>>)
            %525 = arith.muli %522, %c576_i32 : i32
            %526 = arith.index_cast %525 : i32 to index
            %reinterpret_cast_170 = memref.reinterpret_cast %arg9 to offset: [%526], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_171 = memref.subview %alloc_21[3, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_170 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_171 : memref<1x576xbf16, strided<[640, 1], offset: 1920>, #hivm.address_space<ub>>)
            %527 = arith.muli %arg16, %c2048_i32 : i32
            %528 = arith.addi %527, %21 : i32
            %529 = arith.extsi %528 : i32 to i64
            %530 = arith.addi %529, %491 : i64
            %531 = arith.divsi %530, %c2_i64 : i64
            %532 = arith.index_cast %531 : i64 to index
            %533 = arith.addi %532, %c2 : index
            %reinterpret_cast_172 = memref.reinterpret_cast %arg5 to offset: [%533], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %534 = memref.load %reinterpret_cast_172[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %535 = arith.andi %534, %c4294967295_i64 : i64
            %536 = arith.trunci %535 : i64 to i32
            %537 = arith.andi %536, %c8191_i32 : i32
            %538 = arith.shrui %534, %c32_i64 : i64
            %539 = arith.trunci %538 : i64 to i32
            %540 = arith.andi %539, %c8191_i32 : i32
            %541 = arith.muli %537, %c576_i32 : i32
            %542 = arith.index_cast %541 : i32 to index
            %reinterpret_cast_173 = memref.reinterpret_cast %arg9 to offset: [%542], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_174 = memref.subview %alloc_21[4, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_173 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_174 : memref<1x576xbf16, strided<[640, 1], offset: 2560>, #hivm.address_space<ub>>)
            %543 = arith.muli %540, %c576_i32 : i32
            %544 = arith.index_cast %543 : i32 to index
            %reinterpret_cast_175 = memref.reinterpret_cast %arg9 to offset: [%544], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_176 = memref.subview %alloc_21[5, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_175 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_176 : memref<1x576xbf16, strided<[640, 1], offset: 3200>, #hivm.address_space<ub>>)
            %545 = arith.muli %arg16, %c2048_i32 : i32
            %546 = arith.addi %545, %21 : i32
            %547 = arith.extsi %546 : i32 to i64
            %548 = arith.addi %547, %491 : i64
            %549 = arith.divsi %548, %c2_i64 : i64
            %550 = arith.index_cast %549 : i64 to index
            %551 = arith.addi %550, %c3 : index
            %reinterpret_cast_177 = memref.reinterpret_cast %arg5 to offset: [%551], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %552 = memref.load %reinterpret_cast_177[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %553 = arith.andi %552, %c4294967295_i64 : i64
            %554 = arith.trunci %553 : i64 to i32
            %555 = arith.andi %554, %c8191_i32 : i32
            %556 = arith.shrui %552, %c32_i64 : i64
            %557 = arith.trunci %556 : i64 to i32
            %558 = arith.andi %557, %c8191_i32 : i32
            %559 = arith.muli %555, %c576_i32 : i32
            %560 = arith.index_cast %559 : i32 to index
            %reinterpret_cast_178 = memref.reinterpret_cast %arg9 to offset: [%560], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_179 = memref.subview %alloc_21[6, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_178 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_179 : memref<1x576xbf16, strided<[640, 1], offset: 3840>, #hivm.address_space<ub>>)
            %561 = arith.muli %558, %c576_i32 : i32
            %562 = arith.index_cast %561 : i32 to index
            %reinterpret_cast_180 = memref.reinterpret_cast %arg9 to offset: [%562], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_181 = memref.subview %alloc_21[7, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_180 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_181 : memref<1x576xbf16, strided<[640, 1], offset: 4480>, #hivm.address_space<ub>>)
            %563 = arith.muli %arg16, %c2048_i32 : i32
            %564 = arith.addi %563, %21 : i32
            %565 = arith.extsi %564 : i32 to i64
            %566 = arith.addi %565, %491 : i64
            %567 = arith.divsi %566, %c2_i64 : i64
            %568 = arith.index_cast %567 : i64 to index
            %569 = arith.addi %568, %c4 : index
            %reinterpret_cast_182 = memref.reinterpret_cast %arg5 to offset: [%569], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %570 = memref.load %reinterpret_cast_182[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %571 = arith.andi %570, %c4294967295_i64 : i64
            %572 = arith.trunci %571 : i64 to i32
            %573 = arith.andi %572, %c8191_i32 : i32
            %574 = arith.shrui %570, %c32_i64 : i64
            %575 = arith.trunci %574 : i64 to i32
            %576 = arith.andi %575, %c8191_i32 : i32
            %577 = arith.muli %573, %c576_i32 : i32
            %578 = arith.index_cast %577 : i32 to index
            %reinterpret_cast_183 = memref.reinterpret_cast %arg9 to offset: [%578], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_184 = memref.subview %alloc_21[8, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_183 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_184 : memref<1x576xbf16, strided<[640, 1], offset: 5120>, #hivm.address_space<ub>>)
            %579 = arith.muli %576, %c576_i32 : i32
            %580 = arith.index_cast %579 : i32 to index
            %reinterpret_cast_185 = memref.reinterpret_cast %arg9 to offset: [%580], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_186 = memref.subview %alloc_21[9, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_185 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_186 : memref<1x576xbf16, strided<[640, 1], offset: 5760>, #hivm.address_space<ub>>)
            %581 = arith.muli %arg16, %c2048_i32 : i32
            %582 = arith.addi %581, %21 : i32
            %583 = arith.extsi %582 : i32 to i64
            %584 = arith.addi %583, %491 : i64
            %585 = arith.divsi %584, %c2_i64 : i64
            %586 = arith.index_cast %585 : i64 to index
            %587 = arith.addi %586, %c5 : index
            %reinterpret_cast_187 = memref.reinterpret_cast %arg5 to offset: [%587], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %588 = memref.load %reinterpret_cast_187[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %589 = arith.andi %588, %c4294967295_i64 : i64
            %590 = arith.trunci %589 : i64 to i32
            %591 = arith.andi %590, %c8191_i32 : i32
            %592 = arith.shrui %588, %c32_i64 : i64
            %593 = arith.trunci %592 : i64 to i32
            %594 = arith.andi %593, %c8191_i32 : i32
            %595 = arith.muli %591, %c576_i32 : i32
            %596 = arith.index_cast %595 : i32 to index
            %reinterpret_cast_188 = memref.reinterpret_cast %arg9 to offset: [%596], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_189 = memref.subview %alloc_21[10, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_188 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_189 : memref<1x576xbf16, strided<[640, 1], offset: 6400>, #hivm.address_space<ub>>)
            %597 = arith.muli %594, %c576_i32 : i32
            %598 = arith.index_cast %597 : i32 to index
            %reinterpret_cast_190 = memref.reinterpret_cast %arg9 to offset: [%598], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_191 = memref.subview %alloc_21[11, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_190 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_191 : memref<1x576xbf16, strided<[640, 1], offset: 7040>, #hivm.address_space<ub>>)
            %599 = arith.muli %arg16, %c2048_i32 : i32
            %600 = arith.addi %599, %21 : i32
            %601 = arith.extsi %600 : i32 to i64
            %602 = arith.addi %601, %491 : i64
            %603 = arith.divsi %602, %c2_i64 : i64
            %604 = arith.index_cast %603 : i64 to index
            %605 = arith.addi %604, %c6 : index
            %reinterpret_cast_192 = memref.reinterpret_cast %arg5 to offset: [%605], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %606 = memref.load %reinterpret_cast_192[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %607 = arith.andi %606, %c4294967295_i64 : i64
            %608 = arith.trunci %607 : i64 to i32
            %609 = arith.andi %608, %c8191_i32 : i32
            %610 = arith.shrui %606, %c32_i64 : i64
            %611 = arith.trunci %610 : i64 to i32
            %612 = arith.andi %611, %c8191_i32 : i32
            %613 = arith.muli %609, %c576_i32 : i32
            %614 = arith.index_cast %613 : i32 to index
            %reinterpret_cast_193 = memref.reinterpret_cast %arg9 to offset: [%614], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_194 = memref.subview %alloc_21[12, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_193 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_194 : memref<1x576xbf16, strided<[640, 1], offset: 7680>, #hivm.address_space<ub>>)
            %615 = arith.muli %612, %c576_i32 : i32
            %616 = arith.index_cast %615 : i32 to index
            %reinterpret_cast_195 = memref.reinterpret_cast %arg9 to offset: [%616], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_196 = memref.subview %alloc_21[13, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_195 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_196 : memref<1x576xbf16, strided<[640, 1], offset: 8320>, #hivm.address_space<ub>>)
            %617 = arith.muli %arg16, %c2048_i32 : i32
            %618 = arith.addi %617, %21 : i32
            %619 = arith.extsi %618 : i32 to i64
            %620 = arith.addi %619, %491 : i64
            %621 = arith.divsi %620, %c2_i64 : i64
            %622 = arith.index_cast %621 : i64 to index
            %623 = arith.addi %622, %c7 : index
            %reinterpret_cast_197 = memref.reinterpret_cast %arg5 to offset: [%623], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %624 = memref.load %reinterpret_cast_197[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %625 = arith.andi %624, %c4294967295_i64 : i64
            %626 = arith.trunci %625 : i64 to i32
            %627 = arith.andi %626, %c8191_i32 : i32
            %628 = arith.shrui %624, %c32_i64 : i64
            %629 = arith.trunci %628 : i64 to i32
            %630 = arith.andi %629, %c8191_i32 : i32
            %631 = arith.muli %627, %c576_i32 : i32
            %632 = arith.index_cast %631 : i32 to index
            %reinterpret_cast_198 = memref.reinterpret_cast %arg9 to offset: [%632], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_199 = memref.subview %alloc_21[14, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_198 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_199 : memref<1x576xbf16, strided<[640, 1], offset: 8960>, #hivm.address_space<ub>>)
            %633 = arith.muli %630, %c576_i32 : i32
            %634 = arith.index_cast %633 : i32 to index
            %reinterpret_cast_200 = memref.reinterpret_cast %arg9 to offset: [%634], sizes: [1, 576], strides: [576, 1] : memref<?xbf16> to memref<1x576xbf16, strided<[576, 1], offset: ?>>
            %subview_201 = memref.subview %alloc_21[15, 0] [1, 576] [1, 1] : memref<16x640xbf16, #hivm.address_space<ub>> to memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>
            hivm.hir.load ins(%reinterpret_cast_200 : memref<1x576xbf16, strided<[576, 1], offset: ?>>) outs(%subview_201 : memref<1x576xbf16, strided<[640, 1], offset: 9600>, #hivm.address_space<ub>>)
            %635 = arith.muli %6, %c64_i64 : i64
            %636 = arith.addi %635, %c32_i64 : i64
            scope.scope : () -> () {
              %663 = ave.hir.pge <VL64> : vector<128xi1>
              scf.for %arg18 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
                %664 = arith.index_cast %arg18 : i32 to index
                %res = ave.hir.vload <NORM> %alloc_20[%664, %c0] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %665 = arith.muli %arg18, %c16_i32 : i32
                %666 = ave.hir.pge <ALL> : vector<128xi1>
                %667 = arith.index_cast %665 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c0, %667], %c272, %666, %res : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %668 = arith.index_cast %arg18 : i32 to index
                %res_210 = ave.hir.vload <NORM> %alloc_20[%668, %c128] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %669 = arith.muli %arg18, %c16_i32 : i32
                %670 = ave.hir.pge <ALL> : vector<128xi1>
                %671 = arith.index_cast %669 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c8, %671], %c272, %670, %res_210 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %672 = arith.index_cast %arg18 : i32 to index
                %res_211 = ave.hir.vload <NORM> %alloc_20[%672, %c256] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %673 = arith.muli %arg18, %c16_i32 : i32
                %674 = ave.hir.pge <ALL> : vector<128xi1>
                %675 = arith.index_cast %673 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c16, %675], %c272, %674, %res_211 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %676 = arith.index_cast %arg18 : i32 to index
                %res_212 = ave.hir.vload <NORM> %alloc_20[%676, %c384] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %677 = arith.muli %arg18, %c16_i32 : i32
                %678 = ave.hir.pge <ALL> : vector<128xi1>
                %679 = arith.index_cast %677 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c24, %679], %c272, %678, %res_212 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %680 = arith.index_cast %arg18 : i32 to index
                %res_213 = ave.hir.vload <NORM> %cast[%680, %c0] : memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
                %681 = arith.muli %arg18, %c16_i32 : i32
                %682 = arith.index_cast %681 : i32 to index
                ave.hir.store_with_stride %alloc_24[%c32, %682], %c272, %663, %res_213 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              }
              scope.return
            } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
            %subview_202 = memref.subview %alloc_24[0, 0] [32, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>
            %637 = arith.addi %arg17, %c1_i32 : i32
            %638 = arith.remsi %637, %c3_i32 : i32
            %639 = arith.muli %638, %c32_i32 : i32
            %640 = arith.muli %636, %c16_i64 : i64
            %641 = arith.index_cast %639 : i32 to index
            %642 = arith.index_cast %640 : i64 to index
            %subview_203 = memref.subview %collapse_shape[%641, %642] [32, 256] [1, 1] : memref<96x2048xbf16, #hivm.address_space<cbuf>> to memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_202 : memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>) outs(%subview_203 : memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %subview_204 = memref.subview %alloc_24[32, 0] [4, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>
            %643 = arith.addi %arg17, %c1_i32 : i32
            %644 = arith.remsi %643, %c2_i32 : i32
            %645 = arith.muli %644, %c4_i32 : i32
            %646 = arith.muli %636, %c16_i64 : i64
            %647 = arith.index_cast %645 : i32 to index
            %648 = arith.index_cast %646 : i64 to index
            %subview_205 = memref.subview %collapse_shape_6[%647, %648] [4, 256] [1, 1] : memref<8x2048xbf16, #hivm.address_space<cbuf>> to memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_204 : memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>) outs(%subview_205 : memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %649 = arith.muli %6, %c64_i64 : i64
            %650 = arith.addi %649, %c48_i64 : i64
            scope.scope : () -> () {
              %663 = ave.hir.pge <VL64> : vector<128xi1>
              scf.for %arg18 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
                %664 = arith.index_cast %arg18 : i32 to index
                %res = ave.hir.vload <NORM> %alloc_21[%664, %c0] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %665 = arith.muli %arg18, %c16_i32 : i32
                %666 = ave.hir.pge <ALL> : vector<128xi1>
                %667 = arith.index_cast %665 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c0, %667], %c272, %666, %res : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %668 = arith.index_cast %arg18 : i32 to index
                %res_210 = ave.hir.vload <NORM> %alloc_21[%668, %c128] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %669 = arith.muli %arg18, %c16_i32 : i32
                %670 = ave.hir.pge <ALL> : vector<128xi1>
                %671 = arith.index_cast %669 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c8, %671], %c272, %670, %res_210 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %672 = arith.index_cast %arg18 : i32 to index
                %res_211 = ave.hir.vload <NORM> %alloc_21[%672, %c256] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %673 = arith.muli %arg18, %c16_i32 : i32
                %674 = ave.hir.pge <ALL> : vector<128xi1>
                %675 = arith.index_cast %673 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c16, %675], %c272, %674, %res_211 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %676 = arith.index_cast %arg18 : i32 to index
                %res_212 = ave.hir.vload <NORM> %alloc_21[%676, %c384] : memref<16x640xbf16, #hivm.address_space<ub>> into vector<128xbf16>
                %677 = arith.muli %arg18, %c16_i32 : i32
                %678 = ave.hir.pge <ALL> : vector<128xi1>
                %679 = arith.index_cast %677 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c24, %679], %c272, %678, %res_212 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
                %680 = arith.index_cast %arg18 : i32 to index
                %res_213 = ave.hir.vload <NORM> %cast_23[%680, %c0] : memref<16x128xbf16, strided<[640, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
                %681 = arith.muli %arg18, %c16_i32 : i32
                %682 = arith.index_cast %681 : i32 to index
                ave.hir.store_with_stride %alloc_25[%c32, %682], %c272, %663, %res_213 : memref<36x272xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
              }
              scope.return
            } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
            %subview_206 = memref.subview %alloc_25[0, 0] [32, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>
            %651 = arith.addi %arg17, %c1_i32 : i32
            %652 = arith.remsi %651, %c3_i32 : i32
            %653 = arith.muli %652, %c32_i32 : i32
            %654 = arith.muli %650, %c16_i64 : i64
            %655 = arith.index_cast %653 : i32 to index
            %656 = arith.index_cast %654 : i64 to index
            %subview_207 = memref.subview %collapse_shape[%655, %656] [32, 256] [1, 1] : memref<96x2048xbf16, #hivm.address_space<cbuf>> to memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_206 : memref<32x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>) outs(%subview_207 : memref<32x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %subview_208 = memref.subview %alloc_25[32, 0] [4, 256] [1, 1] : memref<36x272xbf16, #hivm.address_space<ub>> to memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>
            %657 = arith.addi %arg17, %c1_i32 : i32
            %658 = arith.remsi %657, %c2_i32 : i32
            %659 = arith.muli %658, %c4_i32 : i32
            %660 = arith.muli %650, %c16_i64 : i64
            %661 = arith.index_cast %659 : i32 to index
            %662 = arith.index_cast %660 : i64 to index
            %subview_209 = memref.subview %collapse_shape_6[%661, %662] [4, 256] [1, 1] : memref<8x2048xbf16, #hivm.address_space<cbuf>> to memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%subview_208 : memref<4x256xbf16, strided<[272, 1], offset: 8704>, #hivm.address_space<ub>>) outs(%subview_209 : memref<4x256xbf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 0
          }
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 1
          %9 = arith.remsi %arg17, %c2_i32 : i32
          %10 = arith.muli %9, %c128_i32 : i32
          %11 = arith.index_cast %10 : i32 to index
          %subview_29 = memref.subview %alloc_16[%11] [128] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %12 = arith.remsi %arg17, %c2_i32 : i32
          %13 = arith.muli %12, %c64_i32 : i32
          %14 = arith.index_cast %13 : i32 to index
          %subview_30 = memref.subview %alloc_19[%14] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          scope.scope : () -> () {
            %20 = ave.hir.scalar_broadcast %cst_0 : f32 -> vector<64xf32>
            %21 = ave.hir.scalar_broadcast %cst : f32 -> vector<64xf32>
            %22 = ave.hir.scalar_broadcast %c8192_i32 : i32 -> vector<64xi32>
            %res = ave.hir.vload <NORM> %subview_29[%c0] : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
            %res_33 = ave.hir.vload <NORM> %subview_29[%c64] : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
            %23 = ave.hir.pge <ALL> : vector<64xi1>
            %24 = ave.hir.vcmp <ULT> %res, %22, %23 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
            %25 = ave.hir.pge <ALL> : vector<64xi1>
            %26 = ave.hir.vcmp <ULT> %res_33, %22, %25 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
            scf.for %arg18 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %27 = arith.index_cast %arg18 : i32 to index
              %res_34 = ave.hir.vload <NORM> %alloc_11[%27, %c0] : memref<8x128xf32, #hivm.address_space<ub>> into vector<64xf32>
              %28 = ave.hir.pge <ALL> : vector<64xi1>
              %29 = ave.hir.vmul %res_34, %21, %28 : vector<64xf32>, vector<64xi1>
              %30 = arith.index_cast %arg18 : i32 to index
              %res_35 = ave.hir.vload <NORM> %alloc_11[%30, %c64] : memref<8x128xf32, #hivm.address_space<ub>> into vector<64xf32>
              %31 = ave.hir.pge <ALL> : vector<64xi1>
              %32 = ave.hir.vmul %res_35, %21, %31 : vector<64xf32>, vector<64xi1>
              %33 = ave.hir.vsel %24, %29, %20 : vector<64xi1>, vector<64xf32>
              %34 = ave.hir.vsel %26, %32, %20 : vector<64xi1>, vector<64xf32>
              %35 = ave.hir.pge <ALL> : vector<64xi1>
              %36 = ave.hir.vmax %33, %34, %35 : vector<64xf32>, vector<64xi1>
              %37 = arith.index_cast %arg18 : i32 to index
              %res_36 = ave.hir.vload <BRC_B32> %alloc_17[%37] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
              %38 = ave.hir.pge <ALL> : vector<64xi1>
              %39 = ave.hir.reduction <max>, %36, %38 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
              %40 = ave.hir.pge <ALL> : vector<64xi1>
              %41 = ave.hir.vector_broadcast %39, %40, true : vector<64xf32>, vector<64xi1> -> vector<64xf32>
              %42 = ave.hir.pge <ALL> : vector<64xi1>
              %43 = ave.hir.vmax %41, %res_36, %42 : vector<64xf32>, vector<64xi1>
              %44 = ave.hir.pge <ALL> : vector<64xi1>
              %45 = ave.hir.vsub %res_36, %43, %44 : vector<64xf32>, vector<64xi1>
              %46 = ave.hir.pge <ALL> : vector<64xi1>
              %47 = ave.hir.vexp %45, %46 : vector<64xf32>, vector<64xi1>
              %48 = ave.hir.pge <ALL> : vector<64xi1>
              %49 = ave.hir.vsub %33, %43, %48 : vector<64xf32>, vector<64xi1>
              %50 = ave.hir.pge <ALL> : vector<64xi1>
              %51 = ave.hir.vexp %49, %50 : vector<64xf32>, vector<64xi1>
              %52 = ave.hir.pge <ALL> : vector<64xi1>
              %53 = ave.hir.vsub %34, %43, %52 : vector<64xf32>, vector<64xi1>
              %54 = ave.hir.pge <ALL> : vector<64xi1>
              %55 = ave.hir.vexp %53, %54 : vector<64xf32>, vector<64xi1>
              %56 = ave.hir.pge <ALL> : vector<64xi1>
              %57 = ave.hir.vadd %51, %55, %56 : vector<64xf32>, vector<64xi1>
              %58 = ave.hir.pge <ALL> : vector<64xi1>
              %59 = ave.hir.reduction <add>, %57, %58 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
              %60 = ave.hir.pge <ALL> : vector<64xi1>
              %61 = ave.hir.vector_broadcast %59, %60, true : vector<64xf32>, vector<64xi1> -> vector<64xf32>
              %62 = arith.index_cast %arg18 : i32 to index
              %res_37 = ave.hir.vload <BRC_B32> %alloc_18[%62] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
              %63 = ave.hir.pge <ALL> : vector<64xi1>
              %64 = ave.hir.vmul %res_37, %47, %63 : vector<64xf32>, vector<64xi1>
              %65 = ave.hir.pge <ALL> : vector<64xi1>
              %66 = ave.hir.vadd %61, %64, %65 : vector<64xf32>, vector<64xi1>
              %67 = ave.hir.pge <ALL> : vector<64xi1>
              %68 = arith.index_cast %arg18 : i32 to index
              ave.hir.masked_store <ONEPT_B32> %alloc_17[%68], %67, %43 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
              %69 = ave.hir.pge <ALL> : vector<64xi1>
              %70 = arith.index_cast %arg18 : i32 to index
              ave.hir.masked_store <ONEPT_B32> %alloc_18[%70], %69, %66 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
              %71 = ave.hir.pge <ALL> : vector<64xi1>
              %72 = arith.index_cast %arg18 : i32 to index
              ave.hir.masked_store <ONEPT_B32> %subview_30[%72], %71, %47 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
              %73 = ave.hir.pge <ALL> : vector<64xi1>
              %74 = ave.hir.vtruncf %51, <round>, true, <part_even>, %73 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
              %75 = ave.hir.pge <ALL> : vector<64xi1>
              %76 = ave.hir.vtruncf %55, <round>, true, <part_even>, %75 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
              %res1, %res2 = ave.hir.vdintlv %74, %76 {layout_change = #ave<layout_change SPARSE>} : vector<128xbf16>, vector<128xbf16>
              %77 = arith.muli %arg18, %c16_i32 : i32
              %78 = ave.hir.pge <ALL> : vector<128xi1>
              %79 = arith.index_cast %77 : i32 to index
              ave.hir.store_with_stride %alloc_15[%c0, %79], %c144, %78, %res1 : memref<8x144xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
            }
            scope.return
          } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
          %15 = arith.cmpi slt, %arg17, %c15_i32 : i32
          scf.if %15 {
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 13
          }
          %16 = arith.cmpi sge, %arg17, %c1_i32 : i32
          scf.if %16 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 6
          }
          %subview_31 = memref.subview %alloc_15[0, 0] [8, 128] [1, 1] : memref<8x144xbf16, #hivm.address_space<ub>> to memref<8x128xbf16, strided<[144, 1]>, #hivm.address_space<ub>>
          %17 = arith.muli %6, %c128_i64 : i64
          %18 = arith.index_cast %17 : i64 to index
          %subview_32 = memref.subview %collapse_shape_8[0, %18] [8, 128] [1, 1] : memref<8x256xbf16, #hivm.address_space<cbuf>> to memref<8x128xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%subview_31 : memref<8x128xbf16, strided<[144, 1]>, #hivm.address_space<ub>>) outs(%subview_32 : memref<8x128xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 2
          %19 = arith.cmpi sge, %arg17, %c1_i32 : i32
          scf.if %19 {
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 3
            %20 = arith.subi %arg17, %c1_i32 : i32
            %21 = arith.remsi %20, %c2_i32 : i32
            %22 = arith.muli %21, %c64_i32 : i32
            %23 = arith.index_cast %22 : i32 to index
            %subview_33 = memref.subview %alloc_19[%23] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            scope.scope : () -> () {
              scf.for %arg18 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
                %24 = arith.index_cast %arg18 : i32 to index
                %res = ave.hir.vload <BRC_B32> %subview_33[%24] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
                %25 = arith.index_cast %arg18 : i32 to index
                %res_34 = ave.hir.vload <NORM> %alloc_12[%25, %c0] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %26 = arith.index_cast %arg18 : i32 to index
                %res_35 = ave.hir.vload <NORM> %alloc_13[%26, %c0] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %27 = ave.hir.pge <ALL> : vector<64xi1>
                %28 = ave.hir.vmula %res_34, %res_35, %res, %27 : vector<64xf32>, vector<64xi1>
                %29 = ave.hir.pge <ALL> : vector<64xi1>
                %30 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%30, %c0], %29, %28 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
                %31 = arith.index_cast %arg18 : i32 to index
                %res_36 = ave.hir.vload <NORM> %alloc_12[%31, %c64] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %32 = arith.index_cast %arg18 : i32 to index
                %res_37 = ave.hir.vload <NORM> %alloc_13[%32, %c64] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %33 = ave.hir.pge <ALL> : vector<64xi1>
                %34 = ave.hir.vmula %res_36, %res_37, %res, %33 : vector<64xf32>, vector<64xi1>
                %35 = ave.hir.pge <ALL> : vector<64xi1>
                %36 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%36, %c64], %35, %34 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
                %37 = arith.index_cast %arg18 : i32 to index
                %res_38 = ave.hir.vload <NORM> %alloc_12[%37, %c128] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %38 = arith.index_cast %arg18 : i32 to index
                %res_39 = ave.hir.vload <NORM> %alloc_13[%38, %c128] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %39 = ave.hir.pge <ALL> : vector<64xi1>
                %40 = ave.hir.vmula %res_38, %res_39, %res, %39 : vector<64xf32>, vector<64xi1>
                %41 = ave.hir.pge <ALL> : vector<64xi1>
                %42 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%42, %c128], %41, %40 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
                %43 = arith.index_cast %arg18 : i32 to index
                %res_40 = ave.hir.vload <NORM> %alloc_12[%43, %c192] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %44 = arith.index_cast %arg18 : i32 to index
                %res_41 = ave.hir.vload <NORM> %alloc_13[%44, %c192] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %45 = ave.hir.pge <ALL> : vector<64xi1>
                %46 = ave.hir.vmula %res_40, %res_41, %res, %45 : vector<64xf32>, vector<64xi1>
                %47 = ave.hir.pge <ALL> : vector<64xi1>
                %48 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%48, %c192], %47, %46 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
                %49 = arith.index_cast %arg18 : i32 to index
                %res_42 = ave.hir.vload <NORM> %alloc_12[%49, %c256] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %50 = arith.index_cast %arg18 : i32 to index
                %res_43 = ave.hir.vload <NORM> %alloc_13[%50, %c256] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %51 = ave.hir.pge <ALL> : vector<64xi1>
                %52 = ave.hir.vmula %res_42, %res_43, %res, %51 : vector<64xf32>, vector<64xi1>
                %53 = ave.hir.pge <ALL> : vector<64xi1>
                %54 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%54, %c256], %53, %52 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
                %55 = arith.index_cast %arg18 : i32 to index
                %res_44 = ave.hir.vload <NORM> %alloc_12[%55, %c320] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %56 = arith.index_cast %arg18 : i32 to index
                %res_45 = ave.hir.vload <NORM> %alloc_13[%56, %c320] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %57 = ave.hir.pge <ALL> : vector<64xi1>
                %58 = ave.hir.vmula %res_44, %res_45, %res, %57 : vector<64xf32>, vector<64xi1>
                %59 = ave.hir.pge <ALL> : vector<64xi1>
                %60 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%60, %c320], %59, %58 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
                %61 = arith.index_cast %arg18 : i32 to index
                %res_46 = ave.hir.vload <NORM> %alloc_12[%61, %c384] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %62 = arith.index_cast %arg18 : i32 to index
                %res_47 = ave.hir.vload <NORM> %alloc_13[%62, %c384] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %63 = ave.hir.pge <ALL> : vector<64xi1>
                %64 = ave.hir.vmula %res_46, %res_47, %res, %63 : vector<64xf32>, vector<64xi1>
                %65 = ave.hir.pge <ALL> : vector<64xi1>
                %66 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%66, %c384], %65, %64 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
                %67 = arith.index_cast %arg18 : i32 to index
                %res_48 = ave.hir.vload <NORM> %alloc_12[%67, %c448] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %68 = arith.index_cast %arg18 : i32 to index
                %res_49 = ave.hir.vload <NORM> %alloc_13[%68, %c448] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
                %69 = ave.hir.pge <ALL> : vector<64xi1>
                %70 = ave.hir.vmula %res_48, %res_49, %res, %69 : vector<64xf32>, vector<64xi1>
                %71 = ave.hir.pge <ALL> : vector<64xi1>
                %72 = arith.index_cast %arg18 : i32 to index
                ave.hir.masked_store <NORM_B32> %alloc_13[%72, %c448], %71, %70 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
              }
              scope.return
            } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 4
          }
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      }
      scope.scope : () -> () {
        %subview_27 = memref.subview %alloc_4[0, 0, 0, 0] [32, 8, 16, 16] [1, 1, 1, 1] : memref<96x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1]>, #hivm.address_space<cbuf>>
        %cast_28 = memref.cast %subview_27 : memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 2
        hivm.hir.mmadL1 ins(%alloc_7, %cast_28, %true, %c16, %c128, %c512 : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>>, memref<32x8x16x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_10 : memref<32x1x16x16xf32, #hivm.address_space<cc>>)
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 4
        hivm.hir.fixpipe {c0_pad_en = true, dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_10 : memref<32x1x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_12 : memref<8x512xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 3
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %0 = hivm.hir.get_sub_block_idx -> i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 3
        scope.scope : () -> () {
          scf.for %arg17 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %subview_31 = memref.subview %alloc_19[64] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>>
            %cast_32 = memref.cast %subview_31 : memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
            %17 = arith.index_cast %arg17 : i32 to index
            %res = ave.hir.vload <BRC_B32> %cast_32[%17] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
            %18 = arith.index_cast %arg17 : i32 to index
            %res_33 = ave.hir.vload <NORM> %alloc_12[%18, %c0] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %19 = arith.index_cast %arg17 : i32 to index
            %res_34 = ave.hir.vload <NORM> %alloc_13[%19, %c0] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %20 = ave.hir.pge <ALL> : vector<64xi1>
            %21 = ave.hir.vmula %res_33, %res_34, %res, %20 : vector<64xf32>, vector<64xi1>
            %22 = ave.hir.pge <ALL> : vector<64xi1>
            %23 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%23, %c0], %22, %21 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %24 = arith.index_cast %arg17 : i32 to index
            %res_35 = ave.hir.vload <NORM> %alloc_12[%24, %c64] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %25 = arith.index_cast %arg17 : i32 to index
            %res_36 = ave.hir.vload <NORM> %alloc_13[%25, %c64] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %26 = ave.hir.pge <ALL> : vector<64xi1>
            %27 = ave.hir.vmula %res_35, %res_36, %res, %26 : vector<64xf32>, vector<64xi1>
            %28 = ave.hir.pge <ALL> : vector<64xi1>
            %29 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%29, %c64], %28, %27 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %30 = arith.index_cast %arg17 : i32 to index
            %res_37 = ave.hir.vload <NORM> %alloc_12[%30, %c128] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %31 = arith.index_cast %arg17 : i32 to index
            %res_38 = ave.hir.vload <NORM> %alloc_13[%31, %c128] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %32 = ave.hir.pge <ALL> : vector<64xi1>
            %33 = ave.hir.vmula %res_37, %res_38, %res, %32 : vector<64xf32>, vector<64xi1>
            %34 = ave.hir.pge <ALL> : vector<64xi1>
            %35 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%35, %c128], %34, %33 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %36 = arith.index_cast %arg17 : i32 to index
            %res_39 = ave.hir.vload <NORM> %alloc_12[%36, %c192] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %37 = arith.index_cast %arg17 : i32 to index
            %res_40 = ave.hir.vload <NORM> %alloc_13[%37, %c192] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %38 = ave.hir.pge <ALL> : vector<64xi1>
            %39 = ave.hir.vmula %res_39, %res_40, %res, %38 : vector<64xf32>, vector<64xi1>
            %40 = ave.hir.pge <ALL> : vector<64xi1>
            %41 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%41, %c192], %40, %39 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %42 = arith.index_cast %arg17 : i32 to index
            %res_41 = ave.hir.vload <NORM> %alloc_12[%42, %c256] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %43 = arith.index_cast %arg17 : i32 to index
            %res_42 = ave.hir.vload <NORM> %alloc_13[%43, %c256] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %44 = ave.hir.pge <ALL> : vector<64xi1>
            %45 = ave.hir.vmula %res_41, %res_42, %res, %44 : vector<64xf32>, vector<64xi1>
            %46 = ave.hir.pge <ALL> : vector<64xi1>
            %47 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%47, %c256], %46, %45 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %48 = arith.index_cast %arg17 : i32 to index
            %res_43 = ave.hir.vload <NORM> %alloc_12[%48, %c320] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %49 = arith.index_cast %arg17 : i32 to index
            %res_44 = ave.hir.vload <NORM> %alloc_13[%49, %c320] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %50 = ave.hir.pge <ALL> : vector<64xi1>
            %51 = ave.hir.vmula %res_43, %res_44, %res, %50 : vector<64xf32>, vector<64xi1>
            %52 = ave.hir.pge <ALL> : vector<64xi1>
            %53 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%53, %c320], %52, %51 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %54 = arith.index_cast %arg17 : i32 to index
            %res_45 = ave.hir.vload <NORM> %alloc_12[%54, %c384] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %55 = arith.index_cast %arg17 : i32 to index
            %res_46 = ave.hir.vload <NORM> %alloc_13[%55, %c384] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %56 = ave.hir.pge <ALL> : vector<64xi1>
            %57 = ave.hir.vmula %res_45, %res_46, %res, %56 : vector<64xf32>, vector<64xi1>
            %58 = ave.hir.pge <ALL> : vector<64xi1>
            %59 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%59, %c384], %58, %57 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
            %60 = arith.index_cast %arg17 : i32 to index
            %res_47 = ave.hir.vload <NORM> %alloc_12[%60, %c448] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %61 = arith.index_cast %arg17 : i32 to index
            %res_48 = ave.hir.vload <NORM> %alloc_13[%61, %c448] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %62 = ave.hir.pge <ALL> : vector<64xi1>
            %63 = ave.hir.vmula %res_47, %res_48, %res, %62 : vector<64xf32>, vector<64xi1>
            %64 = ave.hir.pge <ALL> : vector<64xi1>
            %65 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_13[%65, %c448], %64, %63 : memref<8x512xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
          }
          scope.return
        } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_V>] flag = 4
        scope.scope : () -> () {
          scf.for %arg17 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg17 : i32 to index
            %res = ave.hir.vload <BRC_B32> %alloc_18[%17] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
            %18 = arith.index_cast %arg17 : i32 to index
            %res_31 = ave.hir.vload <NORM> %alloc_13[%18, %c0] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %19 = ave.hir.pge <ALL> : vector<64xi1>
            %20 = ave.hir.vdiv %res_31, %res, %19 : vector<64xf32>, vector<64xi1>
            %21 = ave.hir.pge <ALL> : vector<64xi1>
            %22 = ave.hir.vtruncf %20, <round>, true, <part_even>, %21 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %23 = arith.index_cast %arg17 : i32 to index
            %res_32 = ave.hir.vload <NORM> %alloc_13[%23, %c64] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %24 = ave.hir.pge <ALL> : vector<64xi1>
            %25 = ave.hir.vdiv %res_32, %res, %24 : vector<64xf32>, vector<64xi1>
            %26 = ave.hir.pge <ALL> : vector<64xi1>
            %27 = ave.hir.vtruncf %25, <round>, true, <part_even>, %26 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %res1, %res2 = ave.hir.vdintlv %22, %27 {layout_change = #ave<layout_change SPARSE>} : vector<128xbf16>, vector<128xbf16>
            %28 = ave.hir.pge <ALL> : vector<128xi1>
            %29 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_14[%29, %c0], %28, %res1 : memref<8x512xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
            %30 = arith.index_cast %arg17 : i32 to index
            %res_33 = ave.hir.vload <NORM> %alloc_13[%30, %c128] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %31 = ave.hir.pge <ALL> : vector<64xi1>
            %32 = ave.hir.vdiv %res_33, %res, %31 : vector<64xf32>, vector<64xi1>
            %33 = ave.hir.pge <ALL> : vector<64xi1>
            %34 = ave.hir.vtruncf %32, <round>, true, <part_even>, %33 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %35 = arith.index_cast %arg17 : i32 to index
            %res_34 = ave.hir.vload <NORM> %alloc_13[%35, %c192] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %36 = ave.hir.pge <ALL> : vector<64xi1>
            %37 = ave.hir.vdiv %res_34, %res, %36 : vector<64xf32>, vector<64xi1>
            %38 = ave.hir.pge <ALL> : vector<64xi1>
            %39 = ave.hir.vtruncf %37, <round>, true, <part_even>, %38 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %res1_35, %res2_36 = ave.hir.vdintlv %34, %39 {layout_change = #ave<layout_change SPARSE>} : vector<128xbf16>, vector<128xbf16>
            %40 = ave.hir.pge <ALL> : vector<128xi1>
            %41 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_14[%41, %c128], %40, %res1_35 : memref<8x512xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
            %42 = arith.index_cast %arg17 : i32 to index
            %res_37 = ave.hir.vload <NORM> %alloc_13[%42, %c256] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %43 = ave.hir.pge <ALL> : vector<64xi1>
            %44 = ave.hir.vdiv %res_37, %res, %43 : vector<64xf32>, vector<64xi1>
            %45 = ave.hir.pge <ALL> : vector<64xi1>
            %46 = ave.hir.vtruncf %44, <round>, true, <part_even>, %45 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %47 = arith.index_cast %arg17 : i32 to index
            %res_38 = ave.hir.vload <NORM> %alloc_13[%47, %c320] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %48 = ave.hir.pge <ALL> : vector<64xi1>
            %49 = ave.hir.vdiv %res_38, %res, %48 : vector<64xf32>, vector<64xi1>
            %50 = ave.hir.pge <ALL> : vector<64xi1>
            %51 = ave.hir.vtruncf %49, <round>, true, <part_even>, %50 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %res1_39, %res2_40 = ave.hir.vdintlv %46, %51 {layout_change = #ave<layout_change SPARSE>} : vector<128xbf16>, vector<128xbf16>
            %52 = ave.hir.pge <ALL> : vector<128xi1>
            %53 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_14[%53, %c256], %52, %res1_39 : memref<8x512xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
            %54 = arith.index_cast %arg17 : i32 to index
            %res_41 = ave.hir.vload <NORM> %alloc_13[%54, %c384] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %55 = ave.hir.pge <ALL> : vector<64xi1>
            %56 = ave.hir.vdiv %res_41, %res, %55 : vector<64xf32>, vector<64xi1>
            %57 = ave.hir.pge <ALL> : vector<64xi1>
            %58 = ave.hir.vtruncf %56, <round>, true, <part_even>, %57 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %59 = arith.index_cast %arg17 : i32 to index
            %res_42 = ave.hir.vload <NORM> %alloc_13[%59, %c448] : memref<8x512xf32, #hivm.address_space<ub>> into vector<64xf32>
            %60 = ave.hir.pge <ALL> : vector<64xi1>
            %61 = ave.hir.vdiv %res_42, %res, %60 : vector<64xf32>, vector<64xi1>
            %62 = ave.hir.pge <ALL> : vector<64xi1>
            %63 = ave.hir.vtruncf %61, <round>, true, <part_even>, %62 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %res1_43, %res2_44 = ave.hir.vdintlv %58, %63 {layout_change = #ave<layout_change SPARSE>} : vector<128xbf16>, vector<128xbf16>
            %64 = ave.hir.pge <ALL> : vector<128xi1>
            %65 = arith.index_cast %arg17 : i32 to index
            ave.hir.masked_store <NORM_B32> %alloc_14[%65, %c384], %64, %res1_43 : memref<8x512xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
          }
          scope.return
        } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
        %1 = arith.muli %arg16, %c16_i32 : i32
        %2 = arith.muli %0, %c8_i64 : i64
        %3 = arith.extsi %1 : i32 to i64
        %4 = arith.addi %3, %2 : i64
        %5 = arith.muli %4, %c512_i64 : i64
        %6 = arith.index_cast %5 : i64 to index
        %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%6], sizes: [8, 512], strides: [512, 1] : memref<?xbf16> to memref<8x512xbf16, strided<[512, 1], offset: ?>>
        hivm.hir.store ins(%alloc_14 : memref<8x512xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<8x512xbf16, strided<[512, 1], offset: ?>>)
        %subview_27 = memref.subview %alloc_17[0] [8] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
        %7 = arith.muli %arg16, %c16_i32 : i32
        %8 = arith.muli %0, %c8_i64 : i64
        %9 = arith.index_cast %7 : i32 to index
        %10 = arith.index_cast %8 : i64 to index
        %11 = arith.addi %9, %10 : index
        %reinterpret_cast_28 = memref.reinterpret_cast %arg7 to offset: [%11], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1], offset: ?>>
        hivm.hir.store ins(%subview_27 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_28 : memref<8xf32, strided<[1], offset: ?>>)
        %subview_29 = memref.subview %alloc_18[0] [8] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
        %12 = arith.muli %arg16, %c16_i32 : i32
        %13 = arith.muli %0, %c8_i64 : i64
        %14 = arith.index_cast %12 : i32 to index
        %15 = arith.index_cast %13 : i64 to index
        %16 = arith.addi %14, %15 : index
        %reinterpret_cast_30 = memref.reinterpret_cast %arg8 to offset: [%16], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1], offset: ?>>
        hivm.hir.store ins(%subview_29 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_30 : memref<8xf32, strided<[1], offset: ?>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 10
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 11
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 10
        hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 11
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 4
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    }
    return
  }
}
