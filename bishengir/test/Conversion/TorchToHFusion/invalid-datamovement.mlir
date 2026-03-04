// RUN: bishengir-opt <%s --split-input-file -convert-torch-to-hfusion="ensure-no-implicit-broadcast" | FileCheck %s

// CHECK: broadcast_to_1
// expected-error@below {{unable to perform broadcast operation}}
// expected-error@below {{failed to legalize operation 'torch.aten.broadcast_to' that was explicitly marked illegal}}
func.func @torch.aten.broadcast_to_1() -> !torch.vtensor<[5,6],f32> {
  %0 = torch.vtensor.literal(dense<6.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
  %int5 = torch.constant.int 5
  %int6 = torch.constant.int 6
  %1 = torch.prim.ListConstruct %int5, %int6 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.broadcast_to %0, %1 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[5,6],f32>
  return %2 : !torch.vtensor<[5,6],f32>
}