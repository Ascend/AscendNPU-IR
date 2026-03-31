// RUN: bishengir-opt %s -hfusion-auto-schedule -hfusion-pre-vectorization-fusion | FileCheck %s

// CHECK-LABEL: func @score_kernel_new(
// CHECK-NOT: linalg.elemwise_binary
// CHECK: linalg.generic {{.*}} ins(%arg{{.*}} : tensor<64x64xf32>{{.*}} outs({{.*}} : tensor<64x64xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @score_kernel_new(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: i32) {
    %cst = arith.constant 0.693147182 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = tensor.empty() : tensor<64xi32>
    %5 = linalg.fill ins(%c1_i32 : i32) outs(%4 : tensor<64xi32>) -> tensor<64xi32>
    %inserted = tensor.insert %arg2 into %0[%c0] : tensor<1xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%inserted, %1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %6[%c0] : tensor<1xf32>
    %7 = arith.cmpi sge, %arg3, %arg0 : i32
    %8 = arith.cmpi sge, %arg3, %arg1 : i32
    %9 = arith.ori %7, %8 : i1
    scf.if %9 {
    } else {
      %alloc = memref.alloc() : memref<64x128xf16>
      %in_tensor = bufferization.to_tensor %alloc restrict writable : memref<64x128xf16>
      %out_tensor = tensor.empty() : tensor<128x64xf16>
      %transposed = linalg.transpose ins(%in_tensor : tensor<64x128xf16>) outs(%out_tensor : tensor<128x64xf16>) permutation = [1, 0]
      %out_tensor_1 = tensor.empty() : tensor<64x64xi32>
      %broadcasted = linalg.broadcast ins(%5 : tensor<64xi32>) outs(%out_tensor_1 : tensor<64x64xi32>) dimensions = [1]
      %broadcasted_8 = linalg.broadcast ins(%5 : tensor<64xi32>) outs(%out_tensor_1 : tensor<64x64xi32>) dimensions = [0]
      %10 = tensor.empty() : tensor<64x64xi1>
      %11 = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%broadcasted, %broadcasted_8 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%10 : tensor<64x64xi1>) -> tensor<64x64xi1>
      %12 = linalg.fill ins(%extracted : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %13 = scf.for %arg5 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg6 = %3) -> (tensor<64x64xf32>)  : i32 {
        %alloc_1 = memref.alloc() : memref<64x128xf16>
        %14 = arith.ori %7, %8 : i1
        scf.if %14 {
          linalg.fill ins(%cst_0 : f16) outs(%alloc_1 : memref<64x128xf16>)
        }
        %15 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x128xf16>
        %alloc_2 = memref.alloc() : memref<1x64xf32>
        %16 = arith.ori %7, %8 : i1
        scf.if %16 {
          linalg.fill ins(%cst_1 : f32) outs(%alloc_2 : memref<1x64xf32>)
        }
        %17 = bufferization.to_tensor %alloc_2 restrict writable : memref<1x64xf32>
        %18 = tensor.empty() : tensor<64x1xf32>
        %transposed_19 = linalg.transpose ins(%17 : tensor<1x64xf32>) outs(%18 : tensor<64x1xf32>) permutation = [1, 0]
        %19 = linalg.matmul {input_precison = "ieee"} ins(%15, %transposed : tensor<64x128xf16>, tensor<128x64xf16>) outs(%3 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %alloc_3 = memref.alloc() : memref<64x64xf16, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%19 : tensor<64x64xf32>) outs(%alloc_3 : memref<64x64xf16, #hivm.address_space<ub>>)
        %memspacecast = memref.memory_space_cast %alloc_3 : memref<64x64xf16, #hivm.address_space<ub>> to memref<64x64xf16>
        %20 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf16>
        %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%20 : tensor<64x64xf16>) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%21, %12 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %collapsed = tensor.collapse_shape %transposed_19 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %broadcasted_21 = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1]
        %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%22, %broadcasted_21 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %cst : tensor<64x64xf32>, f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %25 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%24 : tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %26 = hfusion.select ins(%11, %25, %3 : tensor<64x64xi1>, tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg6, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.yield %27 : tensor<64x64xf32>
      }
    }
    return
  }
}
