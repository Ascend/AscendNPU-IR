// RUN: (bishengir-compile --print-pass-id --target=Ascend950PR_9589 \
// RUN:   --enable-triton-kernel-compile=true --enable-lir-compile=false \
// RUN:   --enable-hfusion-compile=true --enable-hivm-compile=true --block-dim=1 \
// RUN:   %s 2>&1 || true) | FileCheck %s

// CHECK: [PassID]
// CHECK-NOT: hfusion-flatten-ops

module {
  func.func @triton_dot(%workspace: memref<?xi8>,
                        %out: memref<?xf32> {tt.divisibility = 16 : i32},
                        %lhs: memref<?xf16> {tt.divisibility = 16 : i32},
                        %rhs: memref<?xf16> {tt.divisibility = 16 : i32},
                        %pid0: i32, %pid1: i32, %pid2: i32,
                        %num0: i32, %num1: i32, %num2: i32)
      attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local",
                  mix_mode = "mix",
                  hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %cst = arith.constant 0.000000e+00 : f32
    %lhs_view = memref.reinterpret_cast %lhs to offset: [0],
      sizes: [32, 64], strides: [64, 1] :
      memref<?xf16> to memref<32x64xf16, strided<[64, 1]>>
    %rhs_view = memref.reinterpret_cast %rhs to offset: [0],
      sizes: [64, 16], strides: [16, 1] :
      memref<?xf16> to memref<64x16xf16, strided<[16, 1]>>
    %lhs_tensor = bufferization.to_tensor %lhs_view restrict writable :
      memref<32x64xf16, strided<[64, 1]>>
    %rhs_tensor = bufferization.to_tensor %rhs_view restrict writable :
      memref<64x16xf16, strided<[16, 1]>>
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%cst : f32)
      outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %result = linalg.matmul {input_precison = "ieee"}
      ins(%lhs_tensor, %rhs_tensor : tensor<32x64xf16>, tensor<64x16xf16>)
      outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %out_view = memref.reinterpret_cast %out to offset: [0],
      sizes: [32, 16], strides: [16, 1] :
      memref<?xf32> to memref<32x16xf32, strided<[16, 1]>>
    bufferization.materialize_in_destination %result in writable %out_view :
      (tensor<32x16xf32>, memref<32x16xf32, strided<[16, 1]>>) -> ()
    return
  }
}