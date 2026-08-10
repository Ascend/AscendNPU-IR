// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: tt.func @test_tensor_extract_in_simt_func
func.func @test_tensor_extract_in_simt_func() attributes {
  hivm.func_core_type = #hivm.func_core_type<AIV>,
  hivm.vf_mode = #hivm.vf_mode<SIMT>
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<-1.000000e+09> : tensor<1xf32>
  %0 = arith.truncf %cst {round_mode = #hivm.round_mode<rint>} : tensor<1xf32> to tensor<1xbf16>
  // CHECK: tensor.extract
  %1 = tensor.extract %0[%c0] : tensor<1xbf16>
  // CHECK: tt.splat
  %2 = tt.splat %1 : bf16 -> tensor<1xbf16>
  return
}
