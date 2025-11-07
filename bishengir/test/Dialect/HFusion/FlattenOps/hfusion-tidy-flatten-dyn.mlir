// RUN: bishengir-opt %s                              \
// RUN:   -pass-pipeline="builtin.module(func.func(   \
// RUN:      hfusion-flatten-ops{flatten-mode=tidy}), \
// RUN:      canonicalize-ext)"                           \
// RUN:   -split-input-file | FileCheck %s

// RUN: bishengir-opt %s                                                        \
// RUN:   -pass-pipeline="builtin.module(func.func(                             \
// RUN:      hfusion-flatten-ops{flatten-mode=tidy multi-dynamic-shape=false}), \
// RUN:      canonicalize-ext)"                                                     \
// RUN:   -split-input-file | FileCheck %s --check-prefix=MULTI-FALSE

// CHECK-LABEL: func.func @broadcast_mul_reduce(
// CHECK: linalg.broadcast
// CHECK-NOT: tensor.collapse_shape
func.func @broadcast_mul_reduce(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim1) : tensor<?x?xf32>
  %1 = tensor.empty(%dim) : tensor<?xf32>
  %2 = linalg.broadcast ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x?xf32>) dimensions = [1]
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.reduce ins(%3 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
  return %4 : tensor<?xf32>
}


// -----

// CHECK-LABEL: func.func @broadcast_mul_onebc(
// CHECK: tensor.collapse_shape
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2]]
// CHECK-SAME{LITERAL}: tensor<?x?xf32> into tensor<?x?x?xf32>
// return %[[EXPANDED]] : tensor<?x?x?xf32>
// func.func @broadcast_mul_onebc(%arg0: tensor<AxBxf32>, %arg1: tensor<CxBxExf32>) -> tensor<AxBxExf32>
// attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %dim = tensor.dim %arg0, %c0 : tensor<AxBxf32> // A
//   %dim1 = tensor.dim %arg1, %c1 : tensor<CxBxExf32> // B
//   %dim2 = tensor.dim %arg1, %c2 : tensor<CxBxExf32> // E
//   %0 = tensor.empty(%dim, %dim1, %dim2) : tensor<AxBxExf32>
//   %2 = linalg.broadcast ins(%arg0 : tensor<AxBxf32>) outs(%0 : tensor<AxBxExf32>) dimensions = [2]
//   return %2 : tensor<AxBxExf32>
// }

func.func @broadcast_mul_onebc(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %0 = tensor.empty(%dim, %dim1, %dim2) : tensor<?x?x?xf32>
  %2 = linalg.broadcast ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?x?xf32>) dimensions = [2]
  return %2 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_mul_onebc(
// CHECK: %[[COLLAPSED0:.*]] = tensor.collapse_shape
// CHECK-SAME{LITERAL}: [[0, 1]]
// CHECK-SAME: tensor<?x?xf32> into tensor<?xf32>
// CHECK: %[[OUT1:.*]] = tensor.collapse_shape
// CHECK-SAME: {{\[\[}}0, 1], {{\[}}2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
// CHECK: %[[BC1:.*]] = linalg.broadcast ins(%[[COLLAPSED0]] : tensor<?xf32>) outs(%[[OUT1]] : tensor<?x?xf32>)
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[BC1]]
// CHECK-SAME{LITERAL}: [[0, 1], [2]]
// CHECK-SAME{LITERAL}: tensor<?x?xf32> into tensor<?x?x?xf32>
// return %[[EXPANDED]] : tensor<?x?x?xf32>
func.func @broadcast_mul_onebc(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %0 = tensor.empty(%dim, %dim1, %dim2) : tensor<?x?x?xf32>
  %2 = linalg.broadcast ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?x?xf32>) dimensions = [2]
  return %2 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_1(
// CHECK: tensor<5x?x6xf32> into tensor<5x?xf32>
// CHECK: tensor<?x6xf32> into tensor<?xf32>
// CHECK: return
func.func @complex_ops_1(%arg0: tensor<5x?x6xf32>, %arg1: tensor<?x6xf32>, %arg2: tensor<5x?x6xf32>) -> (tensor<5x?x6xf32>, tensor<5x?x6xf32>)
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<5x?x6xf32>
  %0 = tensor.empty(%dim) : tensor<5x?x6xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg2 : tensor<5x?x6xf32>, tensor<5x?x6xf32>) outs(%0 : tensor<5x?x6xf32>) -> tensor<5x?x6xf32>
  
  %2 = tensor.empty(%dim) : tensor<?x6xf32>
  %3 = linalg.reduce ins(%arg0 : tensor<5x?x6xf32>) outs(%2 : tensor<?x6xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  %4 = tensor.empty(%dim) : tensor<?x6xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg1 : tensor<?x6xf32>, tensor<?x6xf32>) outs(%4 : tensor<?x6xf32>) -> tensor<?x6xf32>
  
  %6 = tensor.empty(%dim) : tensor<5x?x6xf32>
  %7 = linalg.broadcast ins(%5 : tensor<?x6xf32>) outs(%6 : tensor<5x?x6xf32>) dimensions = [0]
  
  return %1, %7 : tensor<5x?x6xf32>, tensor<5x?x6xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_add_mul(
// CHECK: tensor<?x?x2048x20xf16> into tensor<?x?x40960xf16>
// CHECK: return
func.func @matmul_add_mul(%arg0: tensor<?x?x2048x20xf16>, %tmp: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>, %arg3: tensor<?x?xf16>, %arg4: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x2048x20xf16>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x2048x20xf16>
  %0 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %reduced = linalg.reduce ins(%arg0 : tensor<?x?x2048x20xf16>) outs(%tmp : tensor<?x?xf16>) dimensions = [2, 3]
      (%in: f16, %init: f16) {
        %5 = arith.addf %in, %init : f16
        linalg.yield %5 : f16
      }
  %1 = linalg.matmul ins(%reduced, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %2 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, sub} ins(%3, %arg3 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg4 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %4 : tensor<?x?xf16>
}


// -----

// CHECK-LABEL: func.func @matmul_add_mul(
// CHECK: tensor<1024x?x2048x20xf16> into tensor<1024x?x40960xf16>
// CHECK: return
func.func @matmul_add_mul(%arg0: tensor<1024x?x2048x20xf16>, %tmp: tensor<?x?xf16>, %tmp2: tensor<1024x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>, %arg3: tensor<?x?xf16>, %arg4: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %tmp, %c0 : tensor<?x?xf16>
  %dim1 = tensor.dim %arg0, %c1 : tensor<1024x?x2048x20xf16>
  %0 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %reduced = linalg.reduce ins(%arg0 : tensor<1024x?x2048x20xf16>) outs(%tmp2 : tensor<1024x?xf16>) dimensions = [2, 3]
      (%in: f16, %init: f16) {
        %5 = arith.addf %in, %init : f16
        linalg.yield %5 : f16
      }
  %1 = linalg.matmul ins(%reduced, %arg1 : tensor<1024x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %2 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, sub} ins(%3, %arg3 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg4 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %4 : tensor<?x?xf16>
}

// -----

// CHECK-LABEL: func.func @matmul_transpose_a(
// CHECK: tensor<1024x?x2048x20xf16> into tensor<1024x?x40960xf16>
// CHECK: return
func.func @matmul_transpose_a(%arg0: tensor<1024x?x2048x20xf16>, %tmp: tensor<?x?xf16>, %tmp2: tensor<1024x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %tmp, %c0 : tensor<?x?xf16>
  %dim1 = tensor.dim %arg0, %c1 : tensor<1024x?x2048x20xf16>
  %0 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %reduced = linalg.reduce ins(%arg0 : tensor<1024x?x2048x20xf16>) outs(%tmp2 : tensor<1024x?xf16>) dimensions = [2, 3]
      (%in: f16, %init: f16) {
        %5 = arith.addf %in, %init : f16
        linalg.yield %5 : f16
      }
  %1 = linalg.matmul_transpose_a ins(%reduced, %arg1 : tensor<1024x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %2 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %3 : tensor<?x?xf16>
}

// -----

// CHECK-LABEL: func.func @matmul_transpose_b(
// CHECK: tensor<1024x?x2048x20xf16> into tensor<1024x?x40960xf16>
// CHECK: return
func.func @matmul_transpose_b(%arg0: tensor<1024x?x2048x20xf16>, %tmp: tensor<?x?xf16>, %tmp2: tensor<1024x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %tmp, %c0 : tensor<?x?xf16>
  %dim1 = tensor.dim %arg0, %c1 : tensor<1024x?x2048x20xf16>
  %0 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %reduced = linalg.reduce ins(%arg0 : tensor<1024x?x2048x20xf16>) outs(%tmp2 : tensor<1024x?xf16>) dimensions = [2, 3]
      (%in: f16, %init: f16) {
        %5 = arith.addf %in, %init : f16
        linalg.yield %5 : f16
      }
  %1 = linalg.matmul_transpose_b ins(%reduced, %arg1 : tensor<1024x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %2 = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>
  %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %3 : tensor<?x?xf16>
}

// -----
// CHECK-LABEL: func.func @test_extract_collapse(
// CHECK: tensor.extract
// CHECK-SAME: <?xf32>
// CHECK: return
// MULTI-FALSE-LABEL: func.func @test_extract_collapse(
// MULTI-FALSE: tensor.extract
// MULTI-FALSE-SAME: <?x?xf32>
// MULTI-FALSE: return
func.func @test_extract_collapse(%arg0: tensor<3x?x8x?xf32>) -> f32 {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %dim1 = tensor.dim %arg0, %c1 : tensor<3x?x8x?xf32>
  %dim3 = tensor.dim %arg0, %c3 : tensor<3x?x8x?xf32>

  // Original extract with indices [2, 1, 5, 3]
  %extracted = tensor.extract %arg0[%c2, %c1, %c5, %c3] : tensor<3x?x8x?xf32>

  return %extracted : f32
}

// -----

// CHECK-LABEL: gelu_withweight
// CHECK: tensor.dim
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: tensor.collapse_shape
// CHECK-SAME: into tensor<?xf16>

// MULTI-FALSE-LABEL: gelu_withweight
// MULTI-FALSE: tensor.dim
// MULTI-FALSE: tensor.dim
// MULTI-FALSE: tensor.empty
// MULTI-FALSE-NOT: tensor.collapse_shape
func.func @gelu_withweight(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>, %arg3: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
  %1 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %2 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %3 = hfusion.elemwise_unary {__intermediate_producer__, fun = #hfusion.unary_fn<erf>} ins(%2 : tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %4 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<add>} ins(%3, %arg3 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %5 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<mul>} ins(%1, %4 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %5 : tensor<?x?xf16>
}


// -----
// CHECK-LABEL: func.func @argbind_bug(
// CHECK: %{{.*}} = tensor.extract_slice %{{.*}}[0, 0] [%{{.*}}, 64] [1, 1] : tensor<?x128xbf16> to tensor<?x64xbf16>

// MULTI-FALSE-LABEL: func.func @argbind_bug(
func.func @argbind_bug(%arg0: tensor<?x32x128xbf16>, %arg1: i64) -> tensor<1x?x32x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x32x128xbf16>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [1, %dim, 32, 128] : tensor<?x32x128xbf16> into tensor<1x?x32x128xbf16>
  %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0] [1, %dim, 32, 64] [1, 1, 1, 1] : tensor<1x?x32x128xbf16> to tensor<1x?x32x64xbf16>
  %0 = tensor.empty(%dim) : tensor<1x?x32x64xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<1x?x32x64xbf16>) outs(%0 : tensor<1x?x32x64xf32>) -> tensor<1x?x32x64xf32>
  return %1 : tensor<1x?x32x64xf32>
}

// -----
// CHECK-LABEL: func.func @unit_bind_drop(
// CHECK: %{{.*}} = linalg.reduce ins(%{{.*}} : tensor<?x4096xf32>) outs(%{{.*}} : tensor<?xf32>) dimensions = [1]
// MULTI-FALSE-LABEL: func.func @unit_bind_drop(
func.func @unit_bind_drop(%arg0: tensor<1x?x4096xf32>, %arg1: tensor<1x24576xbf16>, %arg2: i64, %arg3: tensor<?x4096xbf16>, %arg4: tensor<?x4096xbf16>, %arg5: tensor<?x4096xbf16>, %arg6: tensor<1x1x4096xf32>, %arg7: tensor<1x1x4096xf32>, %arg8: tensor<1x24576xbf16>) -> tensor<1x?x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 0.699999988 : f32
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c1 : tensor<1x?x4096xf32>
  %0 = tensor.empty(%dim) : tensor<1x?x4096xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %cst : tensor<1x?x4096xf32>, f32) outs(%0 : tensor<1x?x4096xf32>) -> tensor<1x?x4096xf32>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, %dim] : tensor<?xf32> into tensor<1x?xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%expanded : tensor<1x?xf32>) -> tensor<1x?xf32>
  %reduced = linalg.reduce ins(%1 : tensor<1x?x4096xf32>) outs(%3 : tensor<1x?xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %4 = arith.addf %in, %init : f32
      linalg.yield %4 : f32
    }
  %expanded_1 = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [1, %dim, 1] : tensor<1x?xf32> into tensor<1x?x1xf32>
  return %expanded_1 : tensor<1x?x1xf32>
}

// -----
// CHECK-LABEL: func.func @dynamic_flatten_dependency(
// CHECK: %{{.*}} = linalg.transpose ins(%{{.*}} : tensor<?x1x16x128xbf16>) outs(%{{.*}} : tensor<1x16x?x128xbf16>) permutation = [1, 2, 0, 3]
func.func @dynamic_flatten_dependency(%arg0: tensor<?x4096xbf16>, %arg1: i64) -> tensor<1x16x?x128xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.29730177875068026 : f64
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xbf16>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3, 4]] output_shape [1, %dim, 2, 16, 128] : tensor<?x4096xbf16> into tensor<1x?x2x16x128xbf16>
  %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0, 0] [1, %dim, 1, 16, 128] [1, 1, 1, 1, 1] : tensor<1x?x2x16x128xbf16> to tensor<1x?x1x16x128xbf16>
  %0 = tensor.empty(%dim) : tensor<1x16x1x?x128xbf16>
  %1 = tensor.empty(%dim) : tensor<1x16x1x?x128xbf16>
  %transposed = linalg.transpose ins(%extracted_slice : tensor<1x?x1x16x128xbf16>) outs(%1 : tensor<1x16x1x?x128xbf16>) permutation = [2, 3, 0, 1, 4]
  %2 = arith.truncf %cst : f64 to f32
  %3 = tensor.empty(%dim) : tensor<1x16x1x?x128xf32>
  %4 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%transposed : tensor<1x16x1x?x128xbf16>) outs(%3 : tensor<1x16x1x?x128xf32>) -> tensor<1x16x1x?x128xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %2 : tensor<1x16x1x?x128xf32>, f32) outs(%3 : tensor<1x16x1x?x128xf32>) -> tensor<1x16x1x?x128xf32>
  %6 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%5 : tensor<1x16x1x?x128xf32>) outs(%0 : tensor<1x16x1x?x128xbf16>) -> tensor<1x16x1x?x128xbf16>
  %collapsed = tensor.collapse_shape %6 [[0], [1], [2, 3], [4]] : tensor<1x16x1x?x128xbf16> into tensor<1x16x?x128xbf16>
  return %collapsed : tensor<1x16x?x128xbf16>
}