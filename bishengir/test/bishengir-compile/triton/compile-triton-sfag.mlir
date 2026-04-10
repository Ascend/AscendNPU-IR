// RUN: bishengir-compile %s --target=Ascend910_9589 --enable-auto-multi-buffer=false --enable-auto-bind-sub-block=true --disable-ffts --enable-flatten=false --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t--enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @sparse_flash_attention_grad_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg14: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg15: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg16: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg17: memref<?xf32> {tt.divisibility = 16 : i32}, %arg18: memref<?xf32> {tt.divisibility = 16 : i32}, %arg19: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg20: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg21: memref<?xf16> {tt.divisibility = 16 : i32}, %arg22: memref<?xf16> {tt.divisibility = 16 : i32}, %arg23: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: i32 {tt.divisibility = 16 : i32}, %arg43: i32 {tt.divisibility = 16 : i32}, %arg44: i32 {tt.divisibility = 16 : i32}, %arg45: i32 {tt.divisibility = 16 : i32}, %arg46: i32 {tt.divisibility = 16 : i32}, %arg47: i32 {tt.divisibility = 16 : i32}, %arg48: i32 {tt.divisibility = 16 : i32}, %arg49: i32 {tt.divisibility = 16 : i32}, %arg50: i32 {tt.divisibility = 16 : i32}, %arg51: i32 {tt.divisibility = 16 : i32}, %arg52: i32 {tt.divisibility = 16 : i32}, %arg53: i32 {tt.divisibility = 16 : i32}, %arg54: i32 {tt.divisibility = 16 : i32}, %arg55: i32 {tt.divisibility = 16 : i32}, %arg56: i32 {tt.divisibility = 16 : i32}, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32 {tt.divisibility = 16 : i32}, %arg61: i32 {tt.divisibility = 16 : i32}, %arg62: i32 {tt.divisibility = 16 : i32}, %arg63: i32 {tt.divisibility = 16 : i32}, %arg64: i32 {tt.divisibility = 16 : i32}, %arg65: i32 {tt.divisibility = 16 : i32}, %arg66: i32 {tt.divisibility = 16 : i32}, %arg67: i32 {tt.divisibility = 16 : i32}, %arg68: i32 {tt.divisibility = 16 : i32}, %arg69: i32 {tt.divisibility = 16 : i32}, %arg70: i32 {tt.divisibility = 16 : i32}, %arg71: i32 {tt.divisibility = 16 : i32}, %arg72: i32 {tt.divisibility = 16 : i32}, %arg73: i32 {tt.divisibility = 16 : i32}, %arg74: i32 {tt.divisibility = 16 : i32}, %arg75: i32 {tt.divisibility = 16 : i32}, %arg76: i32 {tt.divisibility = 16 : i32}, %arg77: i32 {tt.divisibility = 16 : i32}, %arg78: i32 {tt.divisibility = 16 : i32}, %arg79: f32, %arg80: i32, %arg81: i32, %arg82: i32, %arg83: i32, %arg84: i32, %arg85: i32, %arg86: i32, %arg87: i32, %arg88: i32, %arg89: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant dense<[16, 1]> : tensor<2xi64>
    %c128 = arith.constant 128 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c192 = arith.constant 192 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x64xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %2 = tensor.empty() : tensor<16xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %4 = tensor.empty() : tensor<16x128xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %6 = tensor.empty() : tensor<64x128xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
    %8 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1]>>
    %9 = arith.cmpi slt, %arg87, %8 : i32
    %reinterpret_cast_2 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
    %10 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1]>>
    %11 = arith.muli %arg87, %arg33 : i32
    %12 = arith.muli %arg87, %arg24 : i32
    %13 = arith.muli %arg87, %arg35 : i32
    %14 = arith.muli %arg87, %arg38 : i32
    %15 = arith.muli %arg87, %arg41 : i32
    %16 = arith.muli %arg87, %arg43 : i32
    %17 = arith.muli %arg87, %arg45 : i32
    %18 = arith.muli %arg87, %arg52 : i32
    %19 = arith.muli %arg87, %arg55 : i32
    %20 = arith.muli %arg87, %arg64 : i32
    %21 = arith.muli %arg87, %arg67 : i32
    %22 = arith.muli %arg87, %arg76 : i32
    %23 = arith.cmpi slt, %arg87, %arg80 : i32
    %24 = arith.andi %9, %23 : i1
    %25 = scf.if %24 -> (i32) {
      %27 = arith.index_cast %11 : i32 to index
      %28 = scf.for %arg90 = %c0_i32 to %c2048_i32 step %c1_i32 iter_args(%arg91 = %c0_i32) -> (i32)  : i32 {
        %29 = arith.index_cast %arg90 : i32 to index
        %30 = arith.addi %27, %29 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%30], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %31 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %32 = arith.cmpi sge, %31, %c0_i32 : i32
        %33 = scf.if %32 -> (i32) {
          %34 = arith.addi %arg91, %c1_i32 : i32
          scf.yield %34 : i32
        } else {
          scf.yield %arg91 : i32
        }
        scf.yield %33 : i32
      }
      scf.yield %28 : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    %26 = arith.cmpi ne, %25, %c0_i32 : i32
    scf.if %26 {
      %27 = arith.addi %25, %c63_i32 : i32
      %28 = arith.divsi %27, %c64_i32 : i32
      %29 = arith.index_cast %11 : i32 to index
      scf.for %arg90 = %c0_i32 to %25 step %c1_i32  : i32 {
        %43 = arith.index_cast %arg90 : i32 to index
        %44 = arith.addi %29, %43 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %45 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %46 = arith.cmpi sge, %45, %c0_i32 : i32
        scf.if %46 {
          %47 = arith.cmpi slt, %45, %10 : i32
          %48 = arith.muli %45, %arg27 : i32
          %49 = arith.muli %arg90, %arg54 : i32
          %50 = arith.addi %18, %49 : i32
          scf.for %arg91 = %c0_i32 to %c6_i32 step %c1_i32  : i32 {
            %51 = arith.muli %arg91, %c32_i32 : i32
            %52 = arith.index_cast %48 : i32 to index
            %53 = arith.index_cast %51 : i32 to index
            %54 = arith.addi %52, %53 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [1, 32], strides: [32, 1] : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
            %alloc = memref.alloc() : memref<1x32xf16>
            %55 = arith.index_castui %47 : i1 to index
            %56 = arith.muli %55, %c32 : index
            %57 = arith.addi %53, %c32 : index
            %58 = arith.maxsi %53, %c192 : index
            %59 = arith.minsi %57, %58 : index
            %60 = arith.subi %59, %53 : index
            %61 = arith.minsi %55, %c1 : index
            %62 = arith.minsi %56, %60 : index
            %63 = arith.cmpi slt, %61, %c1 : index
            %64 = arith.cmpi slt, %62, %c32 : index
            %65 = arith.ori %63, %64 : i1
            scf.if %65 {
              linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<1x32xf16>)
            } {hivm.unlikely_condition}
            %subview = memref.subview %reinterpret_cast_7[0, 0] [%61, %62] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
            %subview_8 = memref.subview %alloc[0, 0] [%61, %62] [1, 1] : memref<1x32xf16> to memref<?x?xf16, strided<[32, 1]>>
            memref.copy %subview, %subview_8 : memref<?x?xf16, strided<[32, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
            %66 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf16>
            %67 = arith.index_cast %50 : i32 to index
            %68 = arith.addi %67, %53 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg15 to offset: [%68], sizes: [1, 32], strides: [32, 1] : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
            %extracted_slice = tensor.extract_slice %66[0, 0] [1, %60] [1, 1] : tensor<1x32xf16> to tensor<1x?xf16>
            %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [1, %60] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x?xf16, strided<[32, 1], offset: ?>>
            bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<1x?xf16>, memref<1x?xf16, strided<[32, 1], offset: ?>>) -> ()
          }
        }
      }
      scf.for %arg90 = %c0_i32 to %25 step %c1_i32  : i32 {
        %43 = arith.index_cast %arg90 : i32 to index
        %44 = arith.addi %29, %43 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %45 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %46 = arith.cmpi sge, %45, %c0_i32 : i32
        scf.if %46 {
          %47 = arith.cmpi slt, %45, %10 : i32
          %48 = arith.muli %45, %arg30 : i32
          %49 = arith.muli %arg90, %arg57 : i32
          %50 = arith.addi %19, %49 : i32
          scf.for %arg91 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
            %51 = arith.muli %arg91, %c32_i32 : i32
            %52 = arith.index_cast %48 : i32 to index
            %53 = arith.index_cast %51 : i32 to index
            %54 = arith.addi %52, %53 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [1, 32], strides: [32, 1] : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
            %alloc = memref.alloc() : memref<1x32xf16>
            %55 = arith.index_castui %47 : i1 to index
            %56 = arith.muli %55, %c32 : index
            %57 = arith.addi %53, %c32 : index
            %58 = arith.maxsi %53, %c128 : index
            %59 = arith.minsi %57, %58 : index
            %60 = arith.subi %59, %53 : index
            %61 = arith.minsi %55, %c1 : index
            %62 = arith.minsi %56, %60 : index
            %63 = arith.cmpi slt, %61, %c1 : index
            %64 = arith.cmpi slt, %62, %c32 : index
            %65 = arith.ori %63, %64 : i1
            scf.if %65 {
              linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<1x32xf16>)
            } {hivm.unlikely_condition}
            %subview = memref.subview %reinterpret_cast_7[0, 0] [%61, %62] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
            %subview_8 = memref.subview %alloc[0, 0] [%61, %62] [1, 1] : memref<1x32xf16> to memref<?x?xf16, strided<[32, 1]>>
            memref.copy %subview, %subview_8 : memref<?x?xf16, strided<[32, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
            %66 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf16>
            %67 = arith.index_cast %50 : i32 to index
            %68 = arith.addi %67, %53 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg16 to offset: [%68], sizes: [1, 32], strides: [32, 1] : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
            %extracted_slice = tensor.extract_slice %66[0, 0] [1, %60] [1, 1] : tensor<1x32xf16> to tensor<1x?xf16>
            %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [1, %60] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x?xf16, strided<[32, 1], offset: ?>>
            bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<1x?xf16>, memref<1x?xf16, strided<[32, 1], offset: ?>>) -> ()
          }
        }
      }
      %30 = arith.index_cast %13 : i32 to index
      %31 = arith.index_cast %arg37 : i32 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%30], sizes: [16, 128], strides: [%31, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
      %32 = arith.index_cast %15 : i32 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg8 to offset: [%32], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %33 = arith.index_cast %16 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg9 to offset: [%33], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %34 = arith.index_cast %14 : i32 to index
      %35 = arith.index_cast %arg40 : i32 to index
      %36 = linalg.fill ins(%arg79 : f32) outs(%0 : tensor<16x64xf32>) -> tensor<16x64xf32>
      scf.for %arg90 = %c0_i32 to %28 step %c1_i32  : i32 {
        %43 = arith.muli %arg90, %c64_i32 : i32
        %44 = scf.for %arg91 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg92 = %1) -> (tensor<16x64xf32>)  : i32 {
          %72 = arith.muli %arg91, %c128_i32 : i32
          %73 = arith.index_cast %12 : i32 to index
          %74 = arith.index_cast %arg26 : i32 to index
          %75 = arith.index_cast %72 : i32 to index
          %76 = arith.addi %73, %75 : index
          %reinterpret_cast_18 = memref.reinterpret_cast %arg2 to offset: [%76], sizes: [16, 128], strides: [%74, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
          %alloc_19 = memref.alloc() : memref<16x128xf16>
          %77 = arith.addi %75, %c128 : index
          %78 = arith.maxsi %75, %c192 : index
          %79 = arith.minsi %77, %78 : index
          %80 = arith.subi %79, %75 : index
          %81 = arith.cmpi slt, %80, %c128 : index
          scf.if %81 {
            linalg.fill ins(%cst_0 : f16) outs(%alloc_19 : memref<16x128xf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_18[0, 0] [16, %80] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[?, 1], offset: ?>>
          %subview_20 = memref.subview %alloc_19[0, 0] [16, %80] [1, 1] : memref<16x128xf16> to memref<16x?xf16, strided<[128, 1]>>
          memref.copy %subview, %subview_20 : memref<16x?xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[128, 1]>>
          %82 = bufferization.to_tensor %alloc_19 restrict writable : memref<16x128xf16>
          %83 = arith.index_cast %18 : i32 to index
          %84 = arith.index_cast %43 : i32 to index
          %85 = arith.index_cast %arg54 : i32 to index
          %86 = arith.muli %84, %85 : index
          %87 = arith.addi %83, %86 : index
          %88 = arith.addi %87, %75 : index
          %reinterpret_cast_21 = memref.reinterpret_cast %arg15 to offset: [%88], sizes: [64, 128], strides: [%85, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
          %alloc_22 = memref.alloc() : memref<64x128xf16>
          scf.if %81 {
            linalg.fill ins(%cst_0 : f16) outs(%alloc_22 : memref<64x128xf16>)
          } {hivm.unlikely_condition}
          %subview_23 = memref.subview %reinterpret_cast_21[0, 0] [64, %80] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<64x?xf16, strided<[?, 1], offset: ?>>
          %subview_24 = memref.subview %alloc_22[0, 0] [64, %80] [1, 1] : memref<64x128xf16> to memref<64x?xf16, strided<[128, 1]>>
          memref.copy %subview_23, %subview_24 : memref<64x?xf16, strided<[?, 1], offset: ?>> to memref<64x?xf16, strided<[128, 1]>>
          %89 = bufferization.to_tensor %alloc_22 restrict writable : memref<64x128xf16>
          %90 = tensor.empty() : tensor<128x64xf16>
          %transposed_25 = linalg.transpose ins(%89 : tensor<64x128xf16>) outs(%90 : tensor<128x64xf16>) permutation = [1, 0]
          %91 = linalg.matmul {input_precison = "ieee"} ins(%82, %transposed_25 : tensor<16x128xf16>, tensor<128x64xf16>) outs(%arg92 : tensor<16x64xf32>) -> tensor<16x64xf32>
          scf.yield %91 : tensor<16x64xf32>
        }
        %45 = arith.index_cast %19 : i32 to index
        %46 = arith.index_cast %43 : i32 to index
        %47 = arith.index_cast %arg57 : i32 to index
        %48 = arith.muli %46, %47 : index
        %49 = arith.addi %45, %48 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg16 to offset: [%49], sizes: [64, 128], strides: [%47, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<64x128xf16>
        memref.copy %reinterpret_cast_6, %alloc : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<64x128xf16>
        %50 = bufferization.to_tensor %alloc restrict writable : memref<64x128xf16>
        %alloc_7 = memref.alloc() : memref<16x128xf16>
        memref.copy %reinterpret_cast_3, %alloc_7 : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<16x128xf16>
        %51 = bufferization.to_tensor %alloc_7 restrict writable : memref<16x128xf16>
        %52 = tensor.empty() : tensor<128x64xf16>
        %transposed = linalg.transpose ins(%50 : tensor<64x128xf16>) outs(%52 : tensor<128x64xf16>) permutation = [1, 0]
        %53 = linalg.matmul {input_precison = "ieee"} ins(%51, %transposed : tensor<16x128xf16>, tensor<128x64xf16>) outs(%1 : tensor<16x64xf32>) -> tensor<16x64xf32>
        %alloc_8 = memref.alloc() : memref<16xf32>
        memref.copy %reinterpret_cast_4, %alloc_8 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32>
        %54 = bufferization.to_tensor %alloc_8 restrict writable : memref<16xf32>
        %reshape = tensor.reshape %54(%cst) : (tensor<16xf32>, tensor<2xi64>) -> tensor<16x1xf32>
        %alloc_9 = memref.alloc() : memref<16xf32>
        memref.copy %reinterpret_cast_5, %alloc_9 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32>
        %55 = bufferization.to_tensor %alloc_9 restrict writable : memref<16xf32>
        %reshape_10 = tensor.reshape %55(%cst) : (tensor<16xf32>, tensor<2xi64>) -> tensor<16x1xf32>
        %56 = scf.for %arg91 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg92 = %3) -> (tensor<16xf32>)  : i32 {
          %72 = arith.muli %arg91, %c32_i32 : i32
          %73 = arith.index_cast %72 : i32 to index
          %74 = arith.addi %30, %73 : index
          %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [%74], sizes: [16, 32], strides: [%31, 1] : memref<?xf16> to memref<16x32xf16, strided<[?, 1], offset: ?>>
          %alloc_19 = memref.alloc() : memref<16x32xf16>
          %75 = arith.addi %73, %c32 : index
          %76 = arith.maxsi %73, %c128 : index
          %77 = arith.minsi %75, %76 : index
          %78 = arith.subi %77, %73 : index
          %79 = arith.cmpi slt, %78, %c32 : index
          scf.if %79 {
            linalg.fill ins(%cst_0 : f16) outs(%alloc_19 : memref<16x32xf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_18[0, 0] [16, %78] [1, 1] : memref<16x32xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[?, 1], offset: ?>>
          %subview_20 = memref.subview %alloc_19[0, 0] [16, %78] [1, 1] : memref<16x32xf16> to memref<16x?xf16, strided<[32, 1]>>
          memref.copy %subview, %subview_20 : memref<16x?xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[32, 1]>>
          %80 = bufferization.to_tensor %alloc_19 restrict writable : memref<16x32xf16>
          %81 = arith.addi %34, %73 : index
          %reinterpret_cast_21 = memref.reinterpret_cast %arg7 to offset: [%81], sizes: [16, 32], strides: [%35, 1] : memref<?xf16> to memref<16x32xf16, strided<[?, 1], offset: ?>>
          %alloc_22 = memref.alloc() : memref<16x32xf16>
          scf.if %79 {
            linalg.fill ins(%cst_0 : f16) outs(%alloc_22 : memref<16x32xf16>)
          } {hivm.unlikely_condition}
          %subview_23 = memref.subview %reinterpret_cast_21[0, 0] [16, %78] [1, 1] : memref<16x32xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[?, 1], offset: ?>>
          %subview_24 = memref.subview %alloc_22[0, 0] [16, %78] [1, 1] : memref<16x32xf16> to memref<16x?xf16, strided<[32, 1]>>
          memref.copy %subview_23, %subview_24 : memref<16x?xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[32, 1]>>
          %82 = bufferization.to_tensor %alloc_22 restrict writable : memref<16x32xf16>
          %83 = arith.mulf %80, %82 : tensor<16x32xf16>
          %84 = tensor.empty() : tensor<16xf16>
          %85 = linalg.fill ins(%cst_0 : f16) outs(%84 : tensor<16xf16>) -> tensor<16xf16>
          %reduced = linalg.reduce ins(%83 : tensor<16x32xf16>) outs(%85 : tensor<16xf16>) dimensions = [1]
            (%in: f16, %init: f16) {
              %88 = arith.addf %in, %init : f16
              linalg.yield %88 : f16
            }
          %86 = arith.extf %reduced : tensor<16xf16> to tensor<16xf32>
          %87 = arith.addf %arg92, %86 : tensor<16xf32>
          scf.yield %87 : tensor<16xf32>
        }
        %57 = arith.mulf %44, %36 : tensor<16x64xf32>
        %collapsed = tensor.collapse_shape %reshape [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<16xf32>) outs(%0 : tensor<16x64xf32>) dimensions = [1]
        %58 = arith.subf %57, %broadcasted : tensor<16x64xf32>
        %59 = math.exp %58 : tensor<16x64xf32>
        %collapsed_11 = tensor.collapse_shape %reshape_10 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
        %broadcasted_12 = linalg.broadcast ins(%collapsed_11 : tensor<16xf32>) outs(%0 : tensor<16x64xf32>) dimensions = [1]
        %60 = arith.divf %59, %broadcasted_12 : tensor<16x64xf32>
        %61 = arith.truncf %60 : tensor<16x64xf32> to tensor<16x64xf16>
        %broadcasted_13 = linalg.broadcast ins(%56 : tensor<16xf32>) outs(%0 : tensor<16x64xf32>) dimensions = [1]
        %62 = arith.subf %53, %broadcasted_13 : tensor<16x64xf32>
        %63 = arith.mulf %60, %62 : tensor<16x64xf32>
        %64 = arith.truncf %63 : tensor<16x64xf32> to tensor<16x64xf16>
        scf.for %arg91 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %72 = arith.muli %arg91, %c128_i32 : i32
          %73 = arith.index_cast %18 : i32 to index
          %74 = arith.index_cast %arg54 : i32 to index
          %75 = arith.muli %46, %74 : index
          %76 = arith.addi %73, %75 : index
          %77 = arith.index_cast %72 : i32 to index
          %78 = arith.addi %76, %77 : index
          %reinterpret_cast_18 = memref.reinterpret_cast %arg15 to offset: [%78], sizes: [64, 128], strides: [%74, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
          %alloc_19 = memref.alloc() : memref<64x128xf16>
          %79 = arith.addi %77, %c128 : index
          %80 = arith.maxsi %77, %c192 : index
          %81 = arith.minsi %79, %80 : index
          %82 = arith.subi %81, %77 : index
          %83 = arith.cmpi slt, %82, %c128 : index
          scf.if %83 {
            linalg.fill ins(%cst_0 : f16) outs(%alloc_19 : memref<64x128xf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_18[0, 0] [64, %82] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<64x?xf16, strided<[?, 1], offset: ?>>
          %subview_20 = memref.subview %alloc_19[0, 0] [64, %82] [1, 1] : memref<64x128xf16> to memref<64x?xf16, strided<[128, 1]>>
          memref.copy %subview, %subview_20 : memref<64x?xf16, strided<[?, 1], offset: ?>> to memref<64x?xf16, strided<[128, 1]>>
          %84 = bufferization.to_tensor %alloc_19 restrict writable : memref<64x128xf16>
          %85 = linalg.matmul {input_precison = "ieee"} ins(%64, %84 : tensor<16x64xf16>, tensor<64x128xf16>) outs(%5 : tensor<16x128xf32>) -> tensor<16x128xf32>
          %86 = arith.index_cast %20 : i32 to index
          %87 = arith.index_cast %arg66 : i32 to index
          %88 = arith.addi %86, %77 : index
          %reinterpret_cast_21 = memref.reinterpret_cast %arg19 to offset: [%88], sizes: [16, 128], strides: [%87, 1] : memref<?xf32> to memref<16x128xf32, strided<[?, 1], offset: ?>>
          %subview_22 = memref.subview %reinterpret_cast_21[0, 0] [16, %82] [1, 1] : memref<16x128xf32, strided<[?, 1], offset: ?>> to memref<16x?xf32, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %85[0, 0] [16, %82] [1, 1] : tensor<16x128xf32> to tensor<16x?xf32>
          hivm.hir.store ins(%extracted_slice : tensor<16x?xf32>) outs(%subview_22 : memref<16x?xf32, strided<[?, 1], offset: ?>>) atomic = <add>
        }
        %65 = tensor.empty() : tensor<64x16xf16>
        %transposed_14 = linalg.transpose ins(%64 : tensor<16x64xf16>) outs(%65 : tensor<64x16xf16>) permutation = [1, 0]
        scf.for %arg91 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %72 = arith.muli %arg91, %c128_i32 : i32
          %73 = arith.index_cast %12 : i32 to index
          %74 = arith.index_cast %arg26 : i32 to index
          %75 = arith.index_cast %72 : i32 to index
          %76 = arith.addi %73, %75 : index
          %reinterpret_cast_18 = memref.reinterpret_cast %arg2 to offset: [%76], sizes: [16, 128], strides: [%74, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
          %alloc_19 = memref.alloc() : memref<16x128xf16>
          %77 = arith.addi %75, %c128 : index
          %78 = arith.maxsi %75, %c192 : index
          %79 = arith.minsi %77, %78 : index
          %80 = arith.subi %79, %75 : index
          %81 = arith.cmpi slt, %80, %c128 : index
          scf.if %81 {
            linalg.fill ins(%cst_0 : f16) outs(%alloc_19 : memref<16x128xf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_18[0, 0] [16, %80] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[?, 1], offset: ?>>
          %subview_20 = memref.subview %alloc_19[0, 0] [16, %80] [1, 1] : memref<16x128xf16> to memref<16x?xf16, strided<[128, 1]>>
          memref.copy %subview, %subview_20 : memref<16x?xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[128, 1]>>
          %82 = bufferization.to_tensor %alloc_19 restrict writable : memref<16x128xf16>
          %83 = linalg.matmul {input_precison = "ieee"} ins(%transposed_14, %82 : tensor<64x16xf16>, tensor<16x128xf16>) outs(%7 : tensor<64x128xf32>) -> tensor<64x128xf32>
          %84 = arith.index_cast %21 : i32 to index
          %85 = arith.index_cast %arg69 : i32 to index
          %86 = arith.muli %46, %85 : index
          %87 = arith.addi %84, %86 : index
          %88 = arith.addi %87, %75 : index
          %reinterpret_cast_21 = memref.reinterpret_cast %arg20 to offset: [%88], sizes: [64, 128], strides: [%85, 1] : memref<?xf32> to memref<64x128xf32, strided<[?, 1], offset: ?>>
          %subview_22 = memref.subview %reinterpret_cast_21[0, 0] [64, %80] [1, 1] : memref<64x128xf32, strided<[?, 1], offset: ?>> to memref<64x?xf32, strided<[?, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %83[0, 0] [64, %80] [1, 1] : tensor<64x128xf32> to tensor<64x?xf32>
          hivm.hir.store ins(%extracted_slice : tensor<64x?xf32>) outs(%subview_22 : memref<64x?xf32, strided<[?, 1], offset: ?>>) atomic = <add>
        }
        %alloc_15 = memref.alloc() : memref<16x128xf16>
        memref.copy %reinterpret_cast_3, %alloc_15 : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<16x128xf16>
        %66 = bufferization.to_tensor %alloc_15 restrict writable : memref<16x128xf16>
        %transposed_16 = linalg.transpose ins(%61 : tensor<16x64xf16>) outs(%65 : tensor<64x16xf16>) permutation = [1, 0]
        %67 = linalg.matmul {input_precison = "ieee"} ins(%transposed_16, %66 : tensor<64x16xf16>, tensor<16x128xf16>) outs(%7 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %68 = arith.index_cast %22 : i32 to index
        %69 = arith.index_cast %arg78 : i32 to index
        %70 = arith.muli %46, %69 : index
        %71 = arith.addi %68, %70 : index
        %reinterpret_cast_17 = memref.reinterpret_cast %arg23 to offset: [%71], sizes: [64, 128], strides: [%69, 1] : memref<?xf32> to memref<64x128xf32, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %67 in writable %reinterpret_cast_17 : (tensor<64x128xf32>, memref<64x128xf32, strided<[?, 1], offset: ?>>) -> ()
      }
      %37 = tensor.empty() : tensor<16x32xf32>
      %38 = linalg.fill ins(%arg79 : f32) outs(%37 : tensor<16x32xf32>) -> tensor<16x32xf32>
      %39 = arith.index_cast %17 : i32 to index
      %40 = arith.index_cast %arg47 : i32 to index
      scf.for %arg90 = %c0_i32 to %c6_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg90, %c32_i32 : i32
        %44 = arith.index_cast %20 : i32 to index
        %45 = arith.index_cast %arg66 : i32 to index
        %46 = arith.index_cast %43 : i32 to index
        %47 = arith.addi %44, %46 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg19 to offset: [%47], sizes: [16, 32], strides: [%45, 1] : memref<?xf32> to memref<16x32xf32, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x32xf32>
        %48 = arith.addi %46, %c32 : index
        %49 = arith.maxsi %46, %c192 : index
        %50 = arith.minsi %48, %49 : index
        %51 = arith.subi %50, %46 : index
        %52 = arith.cmpi slt, %51, %c32 : index
        scf.if %52 {
          linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<16x32xf32>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast_6[0, 0] [16, %51] [1, 1] : memref<16x32xf32, strided<[?, 1], offset: ?>> to memref<16x?xf32, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, 0] [16, %51] [1, 1] : memref<16x32xf32> to memref<16x?xf32, strided<[32, 1]>>
        memref.copy %subview, %subview_7 : memref<16x?xf32, strided<[?, 1], offset: ?>> to memref<16x?xf32, strided<[32, 1]>>
        %53 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf32>
        %54 = arith.mulf %53, %38 : tensor<16x32xf32>
        %55 = arith.addi %39, %46 : index
        %reinterpret_cast_8 = memref.reinterpret_cast %arg12 to offset: [%55], sizes: [16, 32], strides: [%40, 1] : memref<?xf16> to memref<16x32xf16, strided<[?, 1], offset: ?>>
        %56 = arith.truncf %54 : tensor<16x32xf32> to tensor<16x32xf16>
        %extracted_slice = tensor.extract_slice %56[0, 0] [16, %51] [1, 1] : tensor<16x32xf16> to tensor<16x?xf16>
        %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [16, %51] [1, 1] : memref<16x32xf16, strided<[?, 1], offset: ?>> to memref<16x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_9 : (tensor<16x?xf16>, memref<16x?xf16, strided<[?, 1], offset: ?>>) -> ()
      }
      %41 = scf.for %arg90 = %c0_i32 to %c2048_i32 step %c1_i32 iter_args(%arg91 = %c0_i32) -> (i32)  : i32 {
        %43 = arith.index_cast %arg90 : i32 to index
        %44 = arith.addi %29, %43 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %45 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %46 = arith.cmpi sge, %45, %c0_i32 : i32
        %47 = scf.if %46 -> (i32) {
          %48 = arith.muli %arg91, %arg69 : i32
          %49 = arith.addi %21, %48 : i32
          %50 = arith.muli %45, %arg48 : i32
          %51 = tensor.empty() : tensor<1x32xf32>
          %52 = linalg.fill ins(%arg79 : f32) outs(%51 : tensor<1x32xf32>) -> tensor<1x32xf32>
          scf.for %arg92 = %c0_i32 to %c6_i32 step %c1_i32  : i32 {
            %54 = arith.muli %arg92, %c32_i32 : i32
            %55 = arith.index_cast %49 : i32 to index
            %56 = arith.index_cast %54 : i32 to index
            %57 = arith.addi %55, %56 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg20 to offset: [%57], sizes: [1, 32], strides: [32, 1] : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
            %alloc = memref.alloc() : memref<1x32xf32>
            memref.copy %reinterpret_cast_7, %alloc : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32>
            %58 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf32>
            %59 = arith.mulf %58, %52 : tensor<1x32xf32>
            %60 = arith.index_cast %50 : i32 to index
            %61 = arith.addi %60, %56 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg13 to offset: [%61], sizes: [1, 32], strides: [32, 1] : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
            hivm.hir.store ins(%59 : tensor<1x32xf32>) outs(%reinterpret_cast_8 : memref<1x32xf32, strided<[32, 1], offset: ?>>) atomic = <add>
          }
          %53 = arith.addi %arg91, %c1_i32 : i32
          scf.yield %53 : i32
        } else {
          scf.yield %arg91 : i32
        }
        scf.yield %47 : i32
      }
      %42 = scf.for %arg90 = %c0_i32 to %c2048_i32 step %c1_i32 iter_args(%arg91 = %c0_i32) -> (i32)  : i32 {
        %43 = arith.index_cast %arg90 : i32 to index
        %44 = arith.addi %29, %43 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %45 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %46 = arith.cmpi sge, %45, %c0_i32 : i32
        %47 = scf.if %46 -> (i32) {
          %48 = arith.muli %arg91, %arg78 : i32
          %49 = arith.addi %22, %48 : i32
          %50 = arith.muli %45, %arg50 : i32
          scf.for %arg92 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
            %52 = arith.muli %arg92, %c32_i32 : i32
            %53 = arith.index_cast %49 : i32 to index
            %54 = arith.index_cast %52 : i32 to index
            %55 = arith.addi %53, %54 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg23 to offset: [%55], sizes: [1, 32], strides: [32, 1] : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
            %alloc = memref.alloc() : memref<1x32xf32>
            memref.copy %reinterpret_cast_7, %alloc : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32>
            %56 = arith.index_cast %50 : i32 to index
            %57 = arith.addi %56, %54 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg14 to offset: [%57], sizes: [1, 32], strides: [32, 1] : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
            %cast = memref.cast %alloc : memref<1x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
            %58 = bufferization.to_tensor %cast restrict writable : memref<1x32xf32, strided<[32, 1], offset: ?>>
            hivm.hir.store ins(%58 : tensor<1x32xf32>) outs(%reinterpret_cast_8 : memref<1x32xf32, strided<[32, 1], offset: ?>>) atomic = <add>
          }
          %51 = arith.addi %arg91, %c1_i32 : i32
          scf.yield %51 : i32
        } else {
          scf.yield %arg91 : i32
        }
        scf.yield %47 : i32
      }
    }
    return
  }
}
