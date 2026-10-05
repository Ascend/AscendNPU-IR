// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt -convert-hivm-to-tritongpu -rewrite-slice-op-to-triton %s -split-input-file | FileCheck %s --check-prefix=PIPELINE

// CHECK-LABEL: tt.func @test_tensor_extract_in_simt_func
// CHECK: tensor.extract
// CHECK: tt.splat

// PIPELINE-LABEL: tt.func @test_tensor_extract_in_simt_func
// PIPELINE-NOT: tensor.extract
// PIPELINE: tt.unsplat
// PIPELINE-NOT: tensor.extract
func.func @test_tensor_extract_in_simt_func(%out: memref<4xbf16>) attributes {
  hivm.func_core_type = #hivm.func_core_type<AIV>,
  hivm.vf_mode = #hivm.vf_mode<SIMT>
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<-1.000000e+09> : tensor<1xf32>
  %0 = arith.truncf %cst {round_mode = #hivm.round_mode<rint>} : tensor<1xf32> to tensor<1xbf16>
  %1 = tensor.extract %0[%c0] : tensor<1xbf16>
  %2 = tt.splat %1 : bf16 -> tensor<4xbf16>
  hivm.hir.local_store ins(%out : memref<4xbf16>, %2 : tensor<4xbf16>)
  return
}
