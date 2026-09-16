// RUN: bishengir-opt %s --convert-hivm-to-tritongpu="allow-return-value=true" --rewrite-slice-op-to-triton | FileCheck %s --implicit-check-not=bufferization.alloc_tensor --implicit-check-not='tensor<i64>'
// RUN: bishengir-opt %s --convert-hivm-to-tritongpu="allow-return-value=true" --rewrite-slice-op-to-triton --convert-triton-to-tritongpu="target=cuda:80 num-warps=1 threads-per-warp=32 shared-memory-size=221184" -o /dev/null

// Reducing the only dimension must stay rank-1, and extracting the scalar must
// bypass the intermediate rank-0 reshape.
// CHECK-LABEL: func.func @scalar_reduce
// CHECK: %[[RESHAPED:.*]] = tt.reshape %arg0 : tensor<8xi64> -> tensor<1x8xi64>
// CHECK: %[[REDUCED:.*]] = "tt.reduce"(%[[RESHAPED]]) <{axis = 1 : i32}> ({
// CHECK: arith.addi
// CHECK: }) : (tensor<1x8xi64>) -> tensor<1xi64>
// CHECK: %[[SCALAR:.*]] = tt.unsplat %[[REDUCED]] : tensor<1xi64>
// CHECK: return %[[SCALAR]] : i64
func.func @scalar_reduce(%arg0: tensor<8xi64>) -> i64 {
  %c0 = arith.constant 0 : i64
  %initAlloc = bufferization.alloc_tensor() : tensor<i64>
  %initScalar = hivm.hir.vbrc ins(%c0 : i64)
      outs(%initAlloc : tensor<i64>) -> tensor<i64>
  %init = tensor.expand_shape %initScalar [] output_shape [1]
      : tensor<i64> into tensor<1xi64>
  %reduced = hivm.hir.vreduce <sum>
      ins(%arg0 : tensor<8xi64>)
      outs(%init : tensor<1xi64>)
      unsigned_src = false
      reduce_dims = [0] -> tensor<1xi64>
  %rank0 = tensor.collapse_shape %reduced []
      : tensor<1xi64> into tensor<i64>
  %scalar = tensor.extract %rank0[] : tensor<i64>
  return %scalar : i64
}
