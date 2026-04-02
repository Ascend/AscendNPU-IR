// RUN: bishengir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(hfusion-fold-unit-dims))' \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPT

module {
  // Positive test
  // maxnumf over a unit reduction dim should be rewritten into isnan/select +
  // collapse_shape, and the addf reduce should disappear

  // CHECK-OPT-LABEL: func.func @triton_softmax_dim1_3d(
  // CHECK-OPT: tensor.collapse_shape
  // CHECK-OPT-SAME: tensor<27x1x39xf32> into tensor<27x39xf32>
  // CHECK-OPT: hfusion.isnan %{{.*}} : tensor<27x39xf32> -> tensor<27x39xi1>
  // CHECK-OPT: hfusion.select
  // CHECK-OPT-SAME: tensor<27x39xi1>, tensor<27x39xf32>, tensor<27x39xf32>
  // CHECK-OPT-NOT: linalg.reduce
  // CHECK-OPT: return

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

  // Positive test
  // This case isolates the "reuse expand source" path:
  // if the reduce input is only an expand of an already-collapsed tensor,
  // the rewrite should use that source directly and avoid creating
  // collapse(expand(x)) again.

  // CHECK-OPT-LABEL: func.func @reduce_fold_reuses_expand_source(
  // CHECK-OPT-NOT: tensor.expand_shape
  // CHECK-OPT: hfusion.isnan %arg0 : tensor<27x39xf32> -> tensor<27x39xi1>
  // CHECK-OPT: hfusion.select
  // CHECK-OPT-SAME: tensor<27x39xi1>, tensor<27x39xf32>, tensor<27x39xf32>
  // CHECK-OPT-NOT: linalg.reduce
  // CHECK-OPT: return

  func.func @reduce_fold_reuses_expand_source(%arg0: tensor<27x39xf32>) -> tensor<27x39xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [27, 1, 39] : tensor<27x39xf32> into tensor<27x1x39xf32>
    %init = tensor.empty() : tensor<27x39xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<27x39xf32>) -> tensor<27x39xf32>
    %reduced = linalg.reduce ins(%expanded : tensor<27x1x39xf32>) outs(%fill : tensor<27x39xf32>) dimensions = [1]
      (%in: f32, %initv: f32) {
        %r = arith.maxnumf %in, %initv : f32
        linalg.yield %r : f32
      }
    return %reduced : tensor<27x39xf32>
  }

  // Positive test
  // This case isolates the "restore result shape" path:
  // even if the reduce result still keeps another unit dim, the no-op reduce
  // should disappear, NaN normalization should happen on the squeezed rank,
  // and only the final result type should be restored.

  // CHECK-OPT-LABEL: func.func @reduce_fold_with_extra_unit_dims(
  // CHECK-OPT: tensor.collapse_shape %arg0
  // CHECK-OPT-SAME: tensor<20x1x2x1x21xf32> into tensor<20x2x21xf32>
  // CHECK-OPT: hfusion.isnan %{{.*}} : tensor<20x2x21xf32> -> tensor<20x2x21xi1>
  // CHECK-OPT: hfusion.select
  // CHECK-OPT-SAME: tensor<20x2x21xi1>, tensor<20x2x21xf32>, tensor<20x2x21xf32>
  // CHECK-OPT: tensor.expand_shape
  // CHECK-OPT-SAME: tensor<20x2x21xf32> into tensor<20x2x1x21xf32>
  // CHECK-OPT-NOT: linalg.reduce
  // CHECK-OPT: return

  func.func @reduce_fold_with_extra_unit_dims(%arg0: tensor<20x1x2x1x21xf32>) -> tensor<20x2x1x21xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %init = tensor.empty() : tensor<20x2x1x21xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<20x2x1x21xf32>) -> tensor<20x2x1x21xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<20x1x2x1x21xf32>) outs(%fill : tensor<20x2x1x21xf32>) dimensions = [1]
      (%in: f32, %initv: f32) {
        %r = arith.maxnumf %in, %initv : f32
        linalg.yield %r : f32
      }
    return %reduced : tensor<20x2x1x21xf32>
  }

  // Negative test
  // Same reduction axis, but reduction extent is 2, so no fold should happen

  // CHECK-OPT-LABEL: func.func @triton_softmax_dim1_3d_no_fold_non_unit_reduce_dim(
  // CHECK-OPT: linalg.reduce ins(%{{.*}} : tensor<27x2x39xf32>)
  // CHECK-OPT-SAME: dimensions = [1]
  // CHECK-OPT: arith.maxnumf
  // CHECK-OPT: linalg.reduce ins(%{{.*}} : tensor<27x2x39xf32>)
  // CHECK-OPT-SAME: dimensions = [1]
  // CHECK-OPT: arith.addf
  // CHECK-OPT-NOT: hfusion.select
  // CHECK-OPT: return

  func.func @triton_softmax_dim1_3d_no_fold_non_unit_reduce_dim(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [27, 2, 39], strides: [78, 39, 1] : memref<?xf16> to memref<27x2x39xf16, strided<[78, 39, 1]>>
    %alloc = memref.alloc() : memref<27x2x39xf16>
    memref.copy %reinterpret_cast, %alloc : memref<27x2x39xf16, strided<[78, 39, 1]>> to memref<27x2x39xf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<27x2x39xf16>
    %1 = arith.extf %0 : tensor<27x2x39xf16> to tensor<27x2x39xf32>
    %2 = tensor.empty() : tensor<27x39xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<27x39xf32>) -> tensor<27x39xf32>
    %reduced = linalg.reduce ins(%1 : tensor<27x2x39xf32>) outs(%3 : tensor<27x39xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %9 = arith.maxnumf %in, %init : f32
        linalg.yield %9 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [27, 1, 39] : tensor<27x39xf32> into tensor<27x1x39xf32>
    %4 = tensor.extract_slice %1[0, 0, 0] [27, 1, 39] [1, 1, 1] : tensor<27x2x39xf32> to tensor<27x1x39xf32>
    %5 = arith.subf %4, %expanded : tensor<27x1x39xf32>
    %6 = math.exp %5 : tensor<27x1x39xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%2 : tensor<27x39xf32>) -> tensor<27x39xf32>
    %reduced_1 = linalg.reduce ins(%1 : tensor<27x2x39xf32>) outs(%7 : tensor<27x39xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
    %expanded_2 = tensor.expand_shape %reduced_1 [[0], [1, 2]] output_shape [27, 1, 39] : tensor<27x39xf32> into tensor<27x1x39xf32>
    %8 = arith.divf %6, %expanded_2 : tensor<27x1x39xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [27, 1, 39], strides: [39, 39, 1] : memref<?xf16> to memref<27x1x39xf16, strided<[39, 39, 1]>>
    %9 = arith.truncf %8 : tensor<27x1x39xf32> to tensor<27x1x39xf16>
    bufferization.materialize_in_destination %9 in writable %reinterpret_cast_3 : (tensor<27x1x39xf16>, memref<27x1x39xf16, strided<[39, 39, 1]>>) -> ()
    return
  }

  // Negative test
  // There is a unit dim in the shape, but reduction is on a different axis
  // The unit dim may still be collapsed before the reduce, so only check that
  // the reduce itself remains

  // CHECK-OPT-LABEL: func.func @reduce_no_fold_wrong_axis(
  // CHECK-OPT: tensor.collapse_shape %arg0
  // CHECK-OPT: linalg.reduce
  // CHECK-OPT-SAME: dimensions = [1]
  // CHECK-OPT: arith.maxnumf
  // CHECK-OPT: return

  func.func @reduce_no_fold_wrong_axis(%arg0: tensor<27x1x39xf32>) -> tensor<27x1xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %init = tensor.empty() : tensor<27x1xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<27x1xf32>) -> tensor<27x1xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<27x1x39xf32>) outs(%fill : tensor<27x1xf32>) dimensions = [2]
      (%in: f32, %initv: f32) {
        %r = arith.maxnumf %in, %initv : f32
        linalg.yield %r : f32
      }
    return %reduced : tensor<27x1xf32>
  }

  // Positive test for 2-result reduce
  // Unit reduction dim should become collapsed values + zero index tensor
  // No NaN normalization is expected for argmin/argmax

  // CHECK-OPT-LABEL: func.func @argmax_unit_dim_fold(
  // CHECK-OPT-NOT: hfusion.isnan
  // CHECK-OPT-NOT: hfusion.select
  // CHECK-OPT: tensor.collapse_shape
  // CHECK-OPT-SAME: tensor<8x1x16xf32> into tensor<8x16xf32>
  // CHECK-OPT: linalg.fill
  // CHECK-OPT-SAME: tensor<8x16xi32>
  // CHECK-OPT-NOT: linalg.reduce
  // CHECK-OPT: return

  func.func @argmax_unit_dim_fold(%arg0: tensor<8x1x16xf32>, %arg1: tensor<8x1x16xi32>) -> (tensor<8x16xf32>, tensor<8x16xi32>) {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : i32
    %initv = tensor.empty() : tensor<8x16xf32>
    %initi = tensor.empty() : tensor<8x16xi32>
    %fillv = linalg.fill ins(%cst : f32) outs(%initv : tensor<8x16xf32>) -> tensor<8x16xf32>
    %filli = linalg.fill ins(%c0 : i32) outs(%initi : tensor<8x16xi32>) -> tensor<8x16xi32>
    %r:2 = linalg.reduce ins(%arg0, %arg1 : tensor<8x1x16xf32>, tensor<8x1x16xi32>) outs(%fillv, %filli : tensor<8x16xf32>, tensor<8x16xi32>) dimensions = [1]
      (%vin: f32, %iin: i32, %vinit: f32, %iinit: i32) {
        %cmp = arith.cmpf ogt, %vin, %vinit : f32
        %v = arith.select %cmp, %vin, %vinit : f32
        %i = arith.select %cmp, %iin, %iinit : i32
        linalg.yield %v, %i : f32, i32
      }
    return %r#0, %r#1 : tensor<8x16xf32>, tensor<8x16xi32>
  }

  // Negative test for 2-result reduce
  // Non-unit reduction extent should keep the reduce

  // CHECK-OPT-LABEL: func.func @argmax_no_fold_non_unit_dim(
  // CHECK-OPT: linalg.reduce
  // CHECK-OPT-SAME: tensor<8x2x16xf32>
  // CHECK-OPT: arith.cmpf ogt
  // CHECK-OPT: return

  func.func @argmax_no_fold_non_unit_dim(%arg0: tensor<8x2x16xf32>, %arg1: tensor<8x2x16xi32>) -> (tensor<8x16xf32>, tensor<8x16xi32>) {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : i32
    %initv = tensor.empty() : tensor<8x16xf32>
    %initi = tensor.empty() : tensor<8x16xi32>
    %fillv = linalg.fill ins(%cst : f32) outs(%initv : tensor<8x16xf32>) -> tensor<8x16xf32>
    %filli = linalg.fill ins(%c0 : i32) outs(%initi : tensor<8x16xi32>) -> tensor<8x16xi32>
    %r:2 = linalg.reduce ins(%arg0, %arg1 : tensor<8x2x16xf32>, tensor<8x2x16xi32>) outs(%fillv, %filli : tensor<8x16xf32>, tensor<8x16xi32>) dimensions = [1]
      (%vin: f32, %iin: i32, %vinit: f32, %iinit: i32) {
        %cmp = arith.cmpf ogt, %vin, %vinit : f32
        %v = arith.select %cmp, %vin, %vinit : f32
        %i = arith.select %cmp, %iin, %iinit : i32
        linalg.yield %v, %i : f32, i32
      }
    return %r#0, %r#1 : tensor<8x16xf32>, tensor<8x16xi32>
  }
}
