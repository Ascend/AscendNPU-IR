// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend950PR_957c" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @_hstu_jagged_fwd_fused_0(
// CHECK: arith.constant
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: return

// CHECK-LABEL: func.func @_hstu_jagged_fwd(
// CHECK: arith.constant
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: scope.scope
// CHECK: hivm.hir.sync_block
// CHECK: scf.if
// CHECK: memref.alloc
// CHECK: scope.scope
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: hfusion.cast
// CHECK: bufferization.materialize_in_destination
// CHECK: scope.scope
// CHECK: scf.for
// CHECK: scf.if
// CHECK: tensor.collapse_shape
// CHECK: func.call @_hstu_jagged_fwd_fused_0
// CHECK: tensor.expand_shape
// CHECK: hivm.hir.copy
// CHECK: return

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_957c">, hivm.disable_auto_tile_and_bind_subblock} {
  func.func @_hstu_jagged_fwd(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: f32, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c256_i32 = arith.constant 256 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c128_i32 = arith.constant 128 : i32
    %c2_i64 = arith.constant 2 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %c128_i64 = arith.constant 128 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c255_i64 = arith.constant 255 : i64
    %c127_i32 = arith.constant 127 : i32
    %c127_i64 = arith.constant 127 : i64
    %c-1_i64 = arith.constant -1 : i64
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c3_i32 = arith.constant 3 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %0 = tensor.empty() : tensor<1xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1xi32>) -> tensor<1xi32>
    %2 = tensor.empty() : tensor<256x64xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
    %4 = tensor.empty() : tensor<256x128xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %6 = tensor.empty() : tensor<16384xf32>
    scope.scope : () -> () {
      %28 = arith.cmpi eq, %arg18, %c0_i32 : i32
      %29 = hivm.hir.get_sub_block_idx -> i64
      %30 = arith.cmpi eq, %29, %c0_i64 : i64
      %31 = arith.andi %28, %30 : i1
      %32 = arith.cmpi sgt, %arg12, %c256_i32 : i32
      %33 = arith.andi %31, %32 : i1
      scf.if %33 {
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
        bufferization.materialize_in_destination %1 in writable %reinterpret_cast : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
        %34 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1]>>
        %35:2 = scf.for %arg21 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg22 = %c0_i32, %arg23 = %34) -> (i32, i64)  : i32 {
          %36 = arith.index_cast %arg21 : i32 to index
          %37 = arith.addi %36, %c1 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %38 = memref.load %reinterpret_cast_4[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %39 = arith.subi %38, %arg23 : i64
          %40 = arith.addi %39, %c255_i64 : i64
          %41 = arith.divsi %40, %c256_i64 : i64
          %42 = arith.trunci %41 : i64 to i32
          %43 = arith.addi %arg22, %42 : i32
          %inserted = tensor.insert %43 into %0[%c0] : tensor<1xi32>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%37], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_5 : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
          scf.yield %43, %38 : i32, i64
        }
      }
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
    hivm.hir.sync_block[<ALL>, 0 : index] tcube_pipe = <PIPE_ALL> tvector_pipe = <PIPE_ALL>
    %7 = arith.cmpi sle, %arg12, %c256_i32 : i32
    %8 = scf.if %7 -> (i64) {
      scf.yield %c2_i64 : i64
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [2], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: 2>>
      %28 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: 2>>
      %29 = arith.extsi %28 : i32 to i64
      scf.yield %29 : i64
    }
    %9 = arith.muli %8, %c8_i64 : i64
    %10 = arith.extsi %arg15 : i32 to i64
    %11 = arith.minsi %10, %9 : i64
    %12 = arith.divsi %9, %11 : i64
    %13 = arith.addi %12, %c1_i64 : i64
    %14 = arith.remsi %9, %11 : i64
    %15 = arith.extsi %arg18 : i32 to i64
    %16 = arith.cmpi slt, %15, %11 : i64
    %17 = arith.cmpi slt, %15, %14 : i64
    %18 = arith.muli %15, %13 : i64
    %19 = arith.muli %14, %13 : i64
    %20 = arith.subi %15, %14 : i64
    %21 = arith.muli %20, %12 : i64
    %22 = arith.addi %19, %21 : i64
    %23 = arith.select %17, %18, %22 : i64
    %24 = arith.select %16, %23, %c0_i64 : i64
    %25 = arith.select %17, %13, %12 : i64
    %26 = arith.select %16, %25, %c0_i64 : i64
    %27 = arith.cmpi sge, %15, %11 : i64
    scf.if %27 {
    } else {
      %28 = arith.cmpi sle, %26, %c0_i64 : i64
      scf.if %28 {
      } else {
        %29 = arith.addi %arg13, %c127_i32 : i32
        %30 = arith.divsi %29, %c128_i32 : i32
        %31 = arith.extsi %30 : i32 to i64
        %32 = arith.muli %26, %31 : i64
        %alloc = memref.alloc() : memref<4x16x16x16xf32, #hivm.address_space<ub>>
        annotation.mark %alloc {effects = ["write", "read"]} : memref<4x16x16x16xf32, #hivm.address_space<ub>>
        %alloc_3 = memref.alloc() : memref<4x16x16x16xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_3 {effects = ["write", "read"]} : memref<4x16x16x16xf32, #hivm.address_space<ub>>
        %alloc_4 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_4 {effects = ["write", "read"]} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_5 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_5 {effects = ["write", "read"]} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        scope.scope : () -> () {
          %alloc_6 = memref.alloc() : memref<256x64xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_6 {effects = ["write", "read"]} : memref<256x64xf16, #hivm.address_space<cbuf>>
          %33 = arith.addi %32, %c2_i64 : i64
          %34:3 = scf.for %arg21 = %c0_i64 to %33 step %c1_i64 iter_args(%arg22 = %c-1_i64, %arg23 = %c-1_i64, %arg24 = %c-1_i64) -> (i64, i64, i64)  : i64 {
            %35 = arith.subi %arg21, %c2_i64 : i64
            %36 = arith.cmpi sge, %35, %c0_i64 : i64
            %37 = arith.cmpi slt, %35, %32 : i64
            %38 = arith.andi %36, %37 : i1
            scf.if %38 {
              %41 = arith.divsi %35, %31 : i64
              %42 = arith.addi %24, %41 : i64
              %43 = arith.remsi %35, %31 : i64
              %44 = arith.divsi %42, %8 : i64
              %45 = arith.remsi %42, %8 : i64
              %46:2 = scf.if %7 -> (i64, i64) {
                scf.yield %45, %c0_i64 : i64, i64
              } else {
                %69:2 = scf.for %arg25 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg26 = %c0_i32, %arg27 = %c3_i32) -> (i32, i32)  : i32 {
                  %76 = arith.addi %arg26, %arg27 : i32
                  %77 = arith.divsi %76, %c2_i32 : i32
                  %78 = arith.index_cast %77 : i32 to index
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%78], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
                  %79 = memref.load %reinterpret_cast_11[%c0] : memref<1xi32, strided<[1], offset: ?>>
                  %80 = arith.extsi %79 : i32 to i64
                  %81 = arith.cmpi sle, %80, %45 : i64
                  %82 = arith.select %81, %arg27, %77 : i32
                  %83 = scf.if %81 -> (i32) {
                    %84 = arith.addi %77, %c1_i32 : i32
                    scf.yield %84 : i32
                  } else {
                    scf.yield %arg26 : i32
                  }
                  scf.yield %83, %82 : i32, i32
                }
                %70 = arith.subi %69#0, %c1_i32 : i32
                %71 = arith.extsi %70 : i32 to i64
                %72 = arith.index_cast %70 : i32 to index
                %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%72], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
                %73 = memref.load %reinterpret_cast_10[%c0] : memref<1xi32, strided<[1], offset: ?>>
                %74 = arith.extsi %73 : i32 to i64
                %75 = arith.subi %45, %74 : i64
                scf.yield %71, %75 : i64, i64
              }
              %47 = arith.divsi %44, %c2_i64 : i64
              %48 = arith.index_cast %46#0 : i64 to index
              %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%48], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %49 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %50 = arith.addi %48, %c1 : index
              %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%50], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %51 = memref.load %reinterpret_cast_7[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%48], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %52 = memref.load %reinterpret_cast_8[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%50], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %53 = memref.load %reinterpret_cast_9[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %54 = arith.subi %51, %49 : i64
              %55 = arith.subi %53, %52 : i64
              %56 = arith.muli %47, %c64_i64 : i64
              %57 = arith.muli %52, %c256_i64 : i64
              %58 = arith.addi %56, %57 : i64
              %59 = arith.index_cast %58 : i64 to index
              %60 = arith.muli %44, %c64_i64 : i64
              %61 = arith.muli %49, %c512_i64 : i64
              %62 = arith.addi %60, %61 : i64
              %63 = arith.index_cast %62 : i64 to index
              %64 = arith.addi %55, %c127_i64 : i64
              %65 = arith.divsi %64, %c128_i64 : i64
              %66 = arith.muli %43, %c128_i64 : i64
              %67 = arith.remsi %35, %c2_i64 : i64
              %68 = arith.cmpi slt, %43, %65 : i64
              scf.if %68 {
                hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = %c1_i64
                %69 = arith.cmpi eq, %67, %c0_i64 : i64
                %70 = arith.select %69, %alloc_4, %alloc_5 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
                %71 = arith.muli %46#1, %c256_i64 : i64
                %72 = hivm.hir.convert_layout %70 output_shape [256, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x16x16x16xf16, #hivm.address_space<cbuf>>) -> memref<256x128xf16, #hivm.address_space<cbuf>>
                %memspacecast = memref.memory_space_cast %72 : memref<256x128xf16, #hivm.address_space<cbuf>> to memref<256x128xf16>
                %73 = bufferization.to_tensor %memspacecast restrict writable : memref<256x128xf16>
                %74 = arith.index_cast %66 : i64 to index
                %75 = arith.muli %74, %c256 : index
                %76 = arith.addi %59, %75 : index
                %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%76], sizes: [128, 64], strides: [256, 1] : memref<?xf16> to memref<128x64xf16, strided<[256, 1], offset: ?>>
                %alloc_11 = memref.alloc() : memref<128x64xf16>
                %77 = arith.addi %74, %c128 : index
                %78 = arith.index_cast %55 : i64 to index
                %79 = arith.maxsi %74, %78 : index
                %80 = arith.minsi %77, %79 : index
                %81 = arith.subi %80, %74 : index
                %82 = arith.cmpi slt, %81, %c128 : index
                scf.if %82 {
                  linalg.fill ins(%cst_0 : f16) outs(%alloc_11 : memref<128x64xf16>)
                } {hivm.unlikely_condition}
                %subview = memref.subview %reinterpret_cast_10[0, 0] [%81, 64] [1, 1] : memref<128x64xf16, strided<[256, 1], offset: ?>> to memref<?x64xf16, strided<[256, 1], offset: ?>>
                %subview_12 = memref.subview %alloc_11[0, 0] [%81, 64] [1, 1] : memref<128x64xf16> to memref<?x64xf16, strided<[64, 1]>>
                memref.copy %subview, %subview_12 : memref<?x64xf16, strided<[256, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
                %83 = bufferization.to_tensor %alloc_11 restrict writable : memref<128x64xf16>
                %84 = linalg.matmul {input_precision = "ieee"} ins(%73, %83 : tensor<256x128xf16>, tensor<128x64xf16>) outs(%3 : tensor<256x64xf32>) -> tensor<256x64xf32>
                %85 = arith.index_cast %71 : i64 to index
                %86 = arith.muli %85, %c512 : index
                %87 = arith.addi %63, %86 : index
                %reinterpret_cast_13 = memref.reinterpret_cast %arg9 to offset: [%87], sizes: [256, 64], strides: [512, 1] : memref<?xf16> to memref<256x64xf16, strided<[512, 1], offset: ?>>
                %88 = tensor.empty() : tensor<256x64xf16>
                %89 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%84 : tensor<256x64xf32>) outs(%88 : tensor<256x64xf16>) -> tensor<256x64xf16>
                %90 = arith.addi %85, %c256 : index
                %91 = arith.index_cast %54 : i64 to index
                %92 = arith.maxsi %85, %91 : index
                %93 = arith.minsi %90, %92 : index
                %94 = arith.subi %93, %85 : index
                %extracted_slice = tensor.extract_slice %89[0, 0] [%94, 64] [1, 1] : tensor<256x64xf16> to tensor<?x64xf16>
                %subview_14 = memref.subview %reinterpret_cast_13[0, 0] [%94, 64] [1, 1] : memref<256x64xf16, strided<[512, 1], offset: ?>> to memref<?x64xf16, strided<[512, 1], offset: ?>>
                bufferization.materialize_in_destination %extracted_slice in writable %subview_14 : (tensor<?x64xf16>, memref<?x64xf16, strided<[512, 1], offset: ?>>) -> ()
              }
            }
            %39 = arith.cmpi slt, %arg21, %32 : i64
            %40:3 = scf.if %39 -> (i64, i64, i64) {
              %41 = arith.divsi %arg21, %31 : i64
              %42 = arith.addi %24, %41 : i64
              %43 = arith.remsi %arg21, %31 : i64
              %44 = arith.divsi %42, %8 : i64
              %45 = arith.remsi %42, %8 : i64
              %46:2 = scf.if %7 -> (i64, i64) {
                scf.yield %45, %c0_i64 : i64, i64
              } else {
                %72:2 = scf.for %arg25 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg26 = %c0_i32, %arg27 = %c3_i32) -> (i32, i32)  : i32 {
                  %79 = arith.addi %arg26, %arg27 : i32
                  %80 = arith.divsi %79, %c2_i32 : i32
                  %81 = arith.index_cast %80 : i32 to index
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%81], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
                  %82 = memref.load %reinterpret_cast_11[%c0] : memref<1xi32, strided<[1], offset: ?>>
                  %83 = arith.extsi %82 : i32 to i64
                  %84 = arith.cmpi sle, %83, %45 : i64
                  %85 = arith.select %84, %arg27, %80 : i32
                  %86 = scf.if %84 -> (i32) {
                    %87 = arith.addi %80, %c1_i32 : i32
                    scf.yield %87 : i32
                  } else {
                    scf.yield %arg26 : i32
                  }
                  scf.yield %86, %85 : i32, i32
                }
                %73 = arith.subi %72#0, %c1_i32 : i32
                %74 = arith.extsi %73 : i32 to i64
                %75 = arith.index_cast %73 : i32 to index
                %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%75], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
                %76 = memref.load %reinterpret_cast_10[%c0] : memref<1xi32, strided<[1], offset: ?>>
                %77 = arith.extsi %76 : i32 to i64
                %78 = arith.subi %45, %77 : i64
                scf.yield %74, %78 : i64, i64
              }
              %47 = arith.divsi %44, %c2_i64 : i64
              %48 = arith.index_cast %46#0 : i64 to index
              %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%48], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %49 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %50 = arith.addi %48, %c1 : index
              %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%50], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %51 = memref.load %reinterpret_cast_7[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%48], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %52 = memref.load %reinterpret_cast_8[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%50], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %53 = memref.load %reinterpret_cast_9[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %54 = arith.subi %51, %49 : i64
              %55 = arith.subi %53, %52 : i64
              %56 = arith.muli %44, %c64_i64 : i64
              %57 = arith.muli %49, %c512_i64 : i64
              %58 = arith.addi %56, %57 : i64
              %59 = arith.index_cast %58 : i64 to index
              %60 = arith.muli %47, %c64_i64 : i64
              %61 = arith.muli %52, %c256_i64 : i64
              %62 = arith.addi %60, %61 : i64
              %63 = arith.index_cast %62 : i64 to index
              %64 = arith.addi %55, %c127_i64 : i64
              %65 = arith.divsi %64, %c128_i64 : i64
              %66 = arith.muli %43, %c128_i64 : i64
              %67 = arith.remsi %arg21, %c2_i64 : i64
              %68 = arith.cmpi slt, %43, %65 : i64
              %69 = arith.select %68, %44, %arg22 : i64
              %70 = arith.select %68, %46#0, %arg23 : i64
              %71 = arith.select %68, %46#1, %arg24 : i64
              scf.if %68 {
                %72 = arith.cmpi ne, %44, %arg22 : i64
                %73 = arith.cmpi ne, %46#0, %arg23 : i64
                %74 = arith.ori %72, %73 : i1
                %75 = arith.cmpi ne, %46#1, %arg24 : i64
                %76 = arith.ori %74, %75 : i1
                %77 = scf.if %76 -> (tensor<256x64xf16>) {
                  %92 = arith.muli %46#1, %c256_i64 : i64
                  %93 = arith.index_cast %92 : i64 to index
                  %94 = arith.muli %93, %c512 : index
                  %95 = arith.addi %59, %94 : index
                  %reinterpret_cast_13 = memref.reinterpret_cast %arg2 to offset: [%95], sizes: [256, 64], strides: [512, 1] : memref<?xf16> to memref<256x64xf16, strided<[512, 1], offset: ?>>
                  %alloc_14 = memref.alloc() : memref<256x64xf16>
                  %96 = arith.addi %93, %c256 : index
                  %97 = arith.index_cast %54 : i64 to index
                  %98 = arith.maxsi %93, %97 : index
                  %99 = arith.minsi %96, %98 : index
                  %100 = arith.subi %99, %93 : index
                  %101 = arith.cmpi slt, %100, %c256 : index
                  scf.if %101 {
                    linalg.fill ins(%cst_0 : f16) outs(%alloc_14 : memref<256x64xf16>)
                  } {hivm.unlikely_condition}
                  %subview_15 = memref.subview %reinterpret_cast_13[0, 0] [%100, 64] [1, 1] : memref<256x64xf16, strided<[512, 1], offset: ?>> to memref<?x64xf16, strided<[512, 1], offset: ?>>
                  %subview_16 = memref.subview %alloc_14[0, 0] [%100, 64] [1, 1] : memref<256x64xf16> to memref<?x64xf16, strided<[64, 1]>>
                  memref.copy %subview_15, %subview_16 : memref<?x64xf16, strided<[512, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
                  %102 = bufferization.to_tensor %alloc_14 restrict writable : memref<256x64xf16>
                  annotation.mark %102 keys = ["bind_buffer"] values = [%alloc_6 : memref<256x64xf16, #hivm.address_space<cbuf>>] : tensor<256x64xf16>
                  scf.yield %102 : tensor<256x64xf16>
                } else {
                  %memspacecast = memref.memory_space_cast %alloc_6 : memref<256x64xf16, #hivm.address_space<cbuf>> to memref<256x64xf16>
                  %92 = bufferization.to_tensor %memspacecast restrict writable : memref<256x64xf16>
                  scf.yield %92 : tensor<256x64xf16>
                }
                %78 = arith.cmpi eq, %67, %c0_i64 : i64
                %79 = arith.select %78, %alloc, %alloc_3 : memref<4x16x16x16xf32, #hivm.address_space<ub>>
                %80 = arith.index_cast %66 : i64 to index
                %81 = arith.muli %80, %c256 : index
                %82 = arith.addi %63, %81 : index
                %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%82], sizes: [128, 64], strides: [256, 1] : memref<?xf16> to memref<128x64xf16, strided<[256, 1], offset: ?>>
                %alloc_11 = memref.alloc() : memref<128x64xf16>
                %83 = arith.addi %80, %c128 : index
                %84 = arith.index_cast %55 : i64 to index
                %85 = arith.maxsi %80, %84 : index
                %86 = arith.minsi %83, %85 : index
                %87 = arith.subi %86, %80 : index
                %88 = arith.cmpi slt, %87, %c128 : index
                scf.if %88 {
                  linalg.fill ins(%cst_0 : f16) outs(%alloc_11 : memref<128x64xf16>)
                } {hivm.unlikely_condition}
                %subview = memref.subview %reinterpret_cast_10[0, 0] [%87, 64] [1, 1] : memref<128x64xf16, strided<[256, 1], offset: ?>> to memref<?x64xf16, strided<[256, 1], offset: ?>>
                %subview_12 = memref.subview %alloc_11[0, 0] [%87, 64] [1, 1] : memref<128x64xf16> to memref<?x64xf16, strided<[64, 1]>>
                memref.copy %subview, %subview_12 : memref<?x64xf16, strided<[256, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
                %89 = bufferization.to_tensor %alloc_11 restrict writable : memref<128x64xf16>
                %90 = tensor.empty() : tensor<64x128xf16>
                %transposed = linalg.transpose ins(%89 : tensor<128x64xf16>) outs(%90 : tensor<64x128xf16>) permutation = [1, 0] 
                %91 = linalg.matmul {input_precision = "ieee"} ins(%77, %transposed : tensor<256x64xf16>, tensor<64x128xf16>) outs(%5 : tensor<256x128xf32>) -> tensor<256x128xf32>
                hivm.hir.fixpipe ins(%91 : tensor<256x128xf32>) outs(%79 : memref<4x16x16x16xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
                hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = %c0_i64
              }
              scf.yield %69, %70, %71 : i64, i64, i64
            } else {
              scf.yield %arg22, %arg23, %arg24 : i64, i64, i64
            }
            scf.yield %40#0, %40#1, %40#2 : i64, i64, i64
          }
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
        scope.scope : () -> () {
          %33 = arith.addi %32, %c3_i64 : i64
          scf.for %arg21 = %c0_i64 to %33 step %c1_i64  : i64 {
            %34 = arith.subi %arg21, %c1_i64 : i64
            %35 = arith.cmpi sge, %34, %c0_i64 : i64
            %36 = arith.cmpi slt, %34, %32 : i64
            %37 = arith.andi %35, %36 : i1
            scf.if %37 {
              %38 = arith.divsi %34, %31 : i64
              %39 = arith.addi %24, %38 : i64
              %40 = arith.remsi %34, %31 : i64
              %41 = arith.remsi %39, %8 : i64
              %42 = scf.if %7 -> (i64) {
                scf.yield %41 : i64
              } else {
                %52:2 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %c0_i32, %arg24 = %c3_i32) -> (i32, i32)  : i32 {
                  %55 = arith.addi %arg23, %arg24 : i32
                  %56 = arith.divsi %55, %c2_i32 : i32
                  %57 = arith.index_cast %56 : i32 to index
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%57], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
                  %58 = memref.load %reinterpret_cast_7[%c0] : memref<1xi32, strided<[1], offset: ?>>
                  %59 = arith.extsi %58 : i32 to i64
                  %60 = arith.cmpi sle, %59, %41 : i64
                  %61 = arith.select %60, %arg24, %56 : i32
                  %62 = scf.if %60 -> (i32) {
                    %63 = arith.addi %56, %c1_i32 : i32
                    scf.yield %63 : i32
                  } else {
                    scf.yield %arg23 : i32
                  }
                  scf.yield %62, %61 : i32, i32
                }
                %53 = arith.subi %52#0, %c1_i32 : i32
                %54 = arith.extsi %53 : i32 to i64
                scf.yield %54 : i64
              }
              %43 = arith.index_cast %42 : i64 to index
              %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%43], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %44 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %45 = arith.addi %43, %c1 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%45], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %46 = memref.load %reinterpret_cast_6[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %47 = arith.subi %46, %44 : i64
              %48 = arith.addi %47, %c127_i64 : i64
              %49 = arith.divsi %48, %c128_i64 : i64
              %50 = arith.remsi %34, %c2_i64 : i64
              %51 = arith.cmpi slt, %40, %49 : i64
              scf.if %51 {
                hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = %c0_i64
                %52 = arith.cmpi eq, %50, %c0_i64 : i64
                %53 = arith.select %52, %alloc, %alloc_3 : memref<4x16x16x16xf32, #hivm.address_space<ub>>
                %54 = hivm.hir.get_sub_block_idx -> i64
                %55 = scf.if %52 -> (memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>) {
                  %66 = arith.muli %54, %c4_i64 : i64
                  %67 = arith.index_cast %66 : i64 to index
                  %subview = memref.subview %alloc_4[%67, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
                  scf.yield %subview : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
                } else {
                  %66 = arith.muli %54, %c4_i64 : i64
                  %67 = arith.index_cast %66 : i64 to index
                  %subview = memref.subview %alloc_5[%67, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
                  scf.yield %subview : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
                }
                %memspacecast = memref.memory_space_cast %53 : memref<4x16x16x16xf32, #hivm.address_space<ub>> to memref<4x16x16x16xf32>
                %56 = bufferization.to_tensor %memspacecast restrict writable : memref<4x16x16x16xf32>
                %collapsed = tensor.collapse_shape %56 [[0, 1, 2, 3]] : tensor<4x16x16x16xf32> into tensor<16384xf32>
                %57 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %arg10 : tensor<16384xf32>, f32) outs(%6 : tensor<16384xf32>) -> tensor<16384xf32>
                %58 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%57, %cst : tensor<16384xf32>, f32) outs(%6 : tensor<16384xf32>) -> tensor<16384xf32>
                %59 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%58 : tensor<16384xf32>) outs(%6 : tensor<16384xf32>) -> tensor<16384xf32>
                %60 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%59, %cst_2 : tensor<16384xf32>, f32) outs(%6 : tensor<16384xf32>) -> tensor<16384xf32>
                %61 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%57, %60 : tensor<16384xf32>, tensor<16384xf32>) outs(%6 : tensor<16384xf32>) -> tensor<16384xf32>
                %62 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%61, %arg11 : tensor<16384xf32>, f32) outs(%6 : tensor<16384xf32>) -> tensor<16384xf32>
                %63 = tensor.empty() : tensor<16384xf16>
                %64 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%62 : tensor<16384xf32>) outs(%63 : tensor<16384xf16>) -> tensor<16384xf16>
                %expanded = tensor.expand_shape %64 [[0, 1, 2, 3]] output_shape [4, 16, 16, 16] : tensor<16384xf16> into tensor<4x16x16x16xf16>
                %65 = bufferization.to_memref %expanded : memref<4x16x16x16xf16>
                %memspacecast_7 = memref.memory_space_cast %65 : memref<4x16x16x16xf16> to memref<4x16x16x16xf16, #hivm.address_space<ub>>
                hivm.hir.copy ins(%memspacecast_7 : memref<4x16x16x16xf16, #hivm.address_space<ub>>) outs(%55 : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
                hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = %c1_i64
              }
            }
          }
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
      }
    }
    return
  }
}