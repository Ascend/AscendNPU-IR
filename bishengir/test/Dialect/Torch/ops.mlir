// RUN: bishengir-opt %s --split-input-file | FileCheck %s


// CHECK-LABEL: test_aten.sort.stable
// CHECK: torch.aten.sort.stable
func.func @test_aten.sort.stable(%arg0: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>) {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %values, %indices = torch.aten.sort.stable %arg0, %false, %int0, %false : !torch.vtensor<[3,4],f32>, !torch.bool, !torch.int, !torch.bool -> !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
  return %values, %indices : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
  }


// -----

// CHECK-LABEL: test_aten.gmm
// CHECK: torch.aten.gmm
func.func @test_aten.gmm(%tokens : !torch.vtensor<[3,3],f32>, %weights : !torch.vtensor<[3,3,3],f32>, %tpe : !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,3],f32> {
  %result = torch.aten.gmm %tokens, %weights, %tpe : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3,3],f32>, !torch.vtensor<[3],si64> -> !torch.vtensor<[3,3],f32>
  return %result : !torch.vtensor<[3,3],f32>
}


// -----

// CHECK-LABEL: test_c10d_functional.all_to_all_v
// CHECK: torch._c10d_functional.all_to_all_v
func.func @test_c10d_functional.all_to_all_v(%arg0: !torch.vtensor<[4,4],f16>, %arg1: !torch.vtensor<[1],si32>) -> !torch.vtensor<[4,4],f16> {
    %int4 = torch.constant.int 4
    %str = torch.constant.str "0"
    %result = torch._c10d_functional.all_to_all_v %arg0, %int4, %arg1, %arg1, %str : !torch.vtensor<[4,4],f16>, !torch.int, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>, !torch.str -> !torch.vtensor<[4,4],f16>
    return %result : !torch.vtensor<[4,4],f16>
}

// -----

// CHECK-LABEL: test_c10d_functional.reduce_scatter_tensor
// CHECK: torch._c10d_functional.reduce_scatter_tensor
func.func @test_c10d_functional.reduce_scatter_tensor(%arg0: !torch.vtensor<[4096,4096],bf16>) -> !torch.vtensor<[4096,4096],bf16> {
  %int1 = torch.constant.int 1
  %str = torch.constant.str "sum"
  %str_0 = torch.constant.str "0"
  %0 = torch._c10d_functional.reduce_scatter_tensor %arg0, %str, %int1, %str_0 : !torch.vtensor<[4096,4096],bf16>, !torch.str, !torch.int, !torch.str -> !torch.vtensor<[4096,4096],bf16>
  return %0 : !torch.vtensor<[4096,4096],bf16>
}

// -----

// CHECK-LABEL: test_c10d_functional.all_gather_into_tensor
// CHECK: torch._c10d_functional.all_gather_into_tensor
func.func @test_c10d_functional.all_gather_into_tensor(%arg0: !torch.vtensor<[4096,2048],bf16>) -> !torch.vtensor<[4096,2048],bf16> {
  %int1 = torch.constant.int 1
  %str = torch.constant.str "0"
  %0 = torch._c10d_functional.all_gather_into_tensor %arg0, %int1, %str : !torch.vtensor<[4096,2048],bf16>, !torch.int, !torch.str -> !torch.vtensor<[4096,2048],bf16>
  return %0 : !torch.vtensor<[4096,2048],bf16>
}
