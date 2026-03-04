// RUN: bishengir-opt <%s --split-input-file -convert-torch-to-hfusion="ensure-no-implicit-broadcast" | FileCheck %s

// CHECK-LABEL: @torch.aten.permute(
// CHECK: linalg.transpose {{.*}} permutation = [0, 3, 4, 1, 2]
func.func @torch.aten.permute(%arg0: !torch.vtensor<[64,32,16,8,4],f32>) -> !torch.vtensor<[64,8,4,32,16],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int0, %int3, %int4, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[64,32,16,8,4],f32>, !torch.list<int> -> !torch.vtensor<[64,8,4,32,16],f32>
  return %1 : !torch.vtensor<[64,8,4,32,16],f32>
}

// -----

// CHECK-LABEL: @torch.aten.broadcast_to(
// CHECK: linalg.broadcast {{.*}} dimensions = [1]
func.func @torch.aten.broadcast_to(%arg0: !torch.vtensor<[6,1,1],f32>) -> !torch.vtensor<[6,5,1],f32> {
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %int5 = torch.constant.int 5
  %0 = torch.prim.ListConstruct %int-1, %int5, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.broadcast_to %arg0, %0 : !torch.vtensor<[6,1,1],f32>, !torch.list<int> -> !torch.vtensor<[6,5,1],f32>
  return %1 : !torch.vtensor<[6,5,1],f32>
}

// -----

// CHECK-LABEL: @torch.aten.broadcast_to_1(
// CHECK-ENSURE-NO-IMPLICIT-BRC: linalg.broadcast {{.*}} dimensions = [0, 1]
func.func @torch.aten.broadcast_to_1() -> !torch.vtensor<[5,6],f32> {
  %0 = torch.vtensor.literal(dense<6.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
  %int5 = torch.constant.int 5
  %int6 = torch.constant.int 6
  %1 = torch.prim.ListConstruct %int5, %int6 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.broadcast_to %0, %1 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[5,6],f32>
  return %2 : !torch.vtensor<[5,6],f32>
}

// -----

// CHECK-LABEL: @torch.aten.broadcast_to_2(
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1, 2, 4]
func.func @torch.aten.broadcast_to_2(%arg0: !torch.vtensor<[1,?,1,?],f32>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor<[8,?,5,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %int5 = torch.constant.int 5
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int8, %arg1, %int5, %arg2, %arg3, %int-1 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.broadcast_to %arg0, %0 : !torch.vtensor<[1,?,1,?],f32>, !torch.list<int> -> !torch.vtensor<[8,?,5,?,?,?],f32>
  return %1 : !torch.vtensor<[8,?,5,?,?,?],f32>
}

// -----

// CHECK-LABEL: @torch.aten.broadcast_to_3(
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1, 2]
func.func @torch.aten.broadcast_to_3(%arg0: !torch.vtensor<[1,1,1],f32>, %arg1: !torch.int) -> !torch.vtensor<[8,?,?],f32> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %0 = torch.prim.ListConstruct %int8, %arg1, %arg1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.broadcast_to %arg0, %0 : !torch.vtensor<[1,1,1],f32>, !torch.list<int> -> !torch.vtensor<[8,?,?],f32>
  %2 = torch.aten.add.Scalar %1, %int1, %int1 : !torch.vtensor<[8,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[8,?,?],f32>
  return %2 : !torch.vtensor<[8,?,?],f32>
}