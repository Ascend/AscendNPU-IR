// RUN: bishengir-opt %s \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     func.func( \
// RUN:       linalg-fold-unit-extent-dims{fold-rank-reducing-slices=false use-rank-reducing-slices=false}, \
// RUN:       cse, canonicalize))" | FileCheck %s --check-prefix=CHECK-OPT

// RUN: bishengir-compile %s \
// RUN:   -enable-lir-compile=false \
// RUN:   -enable-hfusion-compile=true \
// RUN:   -enable-triton-kernel-compile=true \
// RUN:   -target=Ascend950PR_9589 | FileCheck %s --check-prefix=CHECK-COMPILE

// CHECK-OPT: arith.maxnumf
// CHECK-OPT-SAME: tensor<27x39xf32>
// CHECK-OPT: arith.addf
// CHECK-OPT-SAME: tensor<27x39xf32>
// CHECK-OPT-NOT: linalg.reduce
// CHECK-COMPILE: LLVMDialectModule
module {
  func.func @triton_softmax_dim1_3d(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [27, 1, 39], strides: [39, 39, 1] : memref<?xf16> to memref<27x1x39xf16, strided<[39, 39, 1]>>
    %alloc = memref.alloc() : memref<27x1x39xf16>
    memref.copy %reinterpret_cast, %alloc : memref<27x1x39xf16, strided<[39, 39, 1]>> to memref<27x1x39xf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<27x1x39xf16>
    %1 = arith.extf %0 : tensor<27x1x39xf16> to tensor<27x1x39xf32>
    %2 = tensor.empty() : tensor<27x39xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<27x39xf32>) -> tensor<27x39xf32>
    %reduced = linalg.reduce ins(%1 : tensor<27x1x39xf32>) outs(%3 : tensor<27x39xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %9 = arith.maxnumf %in, %init : f32
        linalg.yield %9 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [27, 1, 39] : tensor<27x39xf32> into tensor<27x1x39xf32>
    %4 = arith.subf %1, %expanded : tensor<27x1x39xf32>
    %5 = math.exp %4 : tensor<27x1x39xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%2 : tensor<27x39xf32>) -> tensor<27x39xf32>
    %reduced_1 = linalg.reduce ins(%5 : tensor<27x1x39xf32>) outs(%6 : tensor<27x39xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
    %expanded_2 = tensor.expand_shape %reduced_1 [[0], [1, 2]] output_shape [27, 1, 39] : tensor<27x39xf32> into tensor<27x1x39xf32>
    %7 = arith.divf %5, %expanded_2 : tensor<27x1x39xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [27, 1, 39], strides: [39, 39, 1] : memref<?xf16> to memref<27x1x39xf16, strided<[39, 39, 1]>>
    %8 = arith.truncf %7 : tensor<27x1x39xf32> to tensor<27x1x39xf16>
    bufferization.materialize_in_destination %8 in writable %reinterpret_cast_3 : (tensor<27x1x39xf16>, memref<27x1x39xf16, strided<[39, 39, 1]>>) -> ()
    return
  }
}

