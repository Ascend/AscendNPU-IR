// RUN: bishengir-opt <%s --split-input-file -convert-torch-to-hfusion | FileCheck %s

// CHECK-LABEL: @torch.aten.any(
// CHECK: linalg.reduce {{.*}} dimensions = [0, 1]
// CHECK: arith.cmpf
// CHECK: arith.ori
func.func @torch.aten.any(%arg0: !torch.vtensor<[32,32],f32>) -> !torch.vtensor<[],i1> {
	%0 = torch.aten.any %arg0 : !torch.vtensor<[32,32],f32> -> !torch.vtensor<[],i1>
	return %0 : !torch.vtensor<[],i1>
}

// CHECK-LABEL: @torch.aten.any.dim(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.cmpf
// CHECK: arith.ori
func.func @torch.aten.any.dim(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[6,3],i1> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.any.dim %arg0, %int1, %false : !torch.vtensor<[6,4,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[6,3],i1>
  return %0 : !torch.vtensor<[6,3],i1>
}

// CHECK-LABEL: @torch.aten.any.dim.keepdim(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.cmpf
// CHECK: arith.ori
// CHECK: tensor.expand_shape
func.func @torch.aten.any.dim.keepdim(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[6,1,3],i1> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %0 = torch.aten.any.dim %arg0, %int1, %true : !torch.vtensor<[6,4,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[6,1,3],i1>
  return %0 : !torch.vtensor<[6,1,3],i1>
}

// CHECK-LABEL: @torch.aten.any.dims(
// CHECK: linalg.reduce {{.*}} dimensions = [1, 2]
// CHECK: arith.ori
func.func @torch.aten.any.dims(%arg0: !torch.vtensor<[4,3,2],i1>) -> !torch.vtensor<[4],i1> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %1 = torch.aten.any.dims %arg0, %0, %false : !torch.vtensor<[4,3,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],i1>
  return %1 : !torch.vtensor<[4],i1>
}

// CHECK-LABEL: @torch.aten.any.dims.keepdim(
// CHECK: linalg.reduce {{.*}} dimensions = [1, 2]
// CHECK: arith.ori
// CHECK: tensor.expand_shape
func.func @torch.aten.any.dims.keepdim(%arg0: !torch.vtensor<[4,3,2],i1>) -> !torch.vtensor<[4,1,1],i1> {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.any.dims %arg0, %0, %true : !torch.vtensor<[4,3,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,1,1],i1>
  return %1 : !torch.vtensor<[4,1,1],i1>
}

// CHECK-LABEL: @torch.aten.all(
// CHECK: linalg.reduce {{.*}} dimensions = [0, 1]
// CHECK: arith.cmpf
// CHECK: arith.andi
func.func @torch.aten.all(%arg0: !torch.vtensor<[32,32],f32>) -> !torch.vtensor<[],i1> {
	%0 = torch.aten.all %arg0 : !torch.vtensor<[32,32],f32> -> !torch.vtensor<[],i1>
	return %0 : !torch.vtensor<[],i1>
}

// CHECK-LABEL: @torch.aten.all.dim(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.cmpf
// CHECK: arith.andi
func.func @torch.aten.all.dim(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[6,3],i1> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.all.dim %arg0, %int1, %false : !torch.vtensor<[6,4,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[6,3],i1>
  return %0 : !torch.vtensor<[6,3],i1>
}

// CHECK-LABEL: @torch.aten.all.dim.keepdim(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.cmpf
// CHECK: arith.andi
// CHECK: tensor.expand_shape
func.func @torch.aten.all.dim.keepdim(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[6,1,3],i1> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %0 = torch.aten.all.dim %arg0, %int1, %true : !torch.vtensor<[6,4,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[6,1,3],i1>
  return %0 : !torch.vtensor<[6,1,3],i1>
}

// CHECK-LABEL: @torch.aten.sum(
// CHECK: linalg.reduce {{.*}} dimensions = [0, 1]
// CHECK: arith.addf
func.func @torch.aten.sum(%arg0: !torch.vtensor<[32,32],f32>) -> !torch.vtensor<[],f32> {
	%none = torch.constant.none
	%0 = torch.aten.sum %arg0, %none : !torch.vtensor<[32,32],f32>, !torch.none -> !torch.vtensor<[],f32>
	return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL: @torch.aten.sum.dim_IntList(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.addf
func.func @torch.aten.sum.dim_IntList(%arg0: !torch.vtensor<[32,32],f32>) -> !torch.vtensor<[32],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[32,32],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[32],f32>
  return %1 : !torch.vtensor<[32],f32>
}

// CHECK-LABEL: @torch.aten.sum.dim_IntList.keepdim(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.addf
// CHECK: tensor.expand_shape
func.func @torch.aten.sum.dim_IntList.keepdim(%arg0: !torch.vtensor<[32,32],f32>) -> !torch.vtensor<[32,1],f32> {
  %none = torch.constant.none
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[32,32],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[32,1],f32>
  return %1 : !torch.vtensor<[32,1],f32>
}

// CHECK-LABEL: @torch.aten.prod(
// CHECK: linalg.reduce {{.*}} dimensions = [0, 1, 2]
// CHECK: arith.mulf
func.func @torch.aten.prod(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[6,4,3],f32>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL: @torch.aten.prod.dim_int(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.mulf
func.func @torch.aten.prod.dim_int(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[6,3],f32> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[6,4,3],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[6,3],f32>
  return %0 : !torch.vtensor<[6,3],f32>
}

// CHECK-LABEL: @torch.aten.prod.dim_int.keepdim(
// CHECK: linalg.reduce {{.*}} dimensions = [1]
// CHECK: arith.mulf
// CHECK: tensor.expand_shape
func.func @torch.aten.prod.dim_int.keepdim(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[6,1,3],f32> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none : !torch.vtensor<[6,4,3],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[6,1,3],f32>
  return %0 : !torch.vtensor<[6,1,3],f32>
}

// CHECK-LABEL: @torch.aten.max(
// CHECK: linalg.reduce {{.*}} dimensions = [0, 1, 2]
// CHECK: arith.maximumf
func.func @torch.aten.max(%arg0: !torch.vtensor<[16,8,4],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.max %arg0 : !torch.vtensor<[16,8,4],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL: @torch.aten.min(
// CHECK: linalg.reduce {{.*}} dimensions = [0, 1, 2]
// CHECK: arith.minimumf
func.func @torch.aten.min(%arg0: !torch.vtensor<[16,8,4],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.min %arg0 : !torch.vtensor<[16,8,4],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL: @torch.aten.max.dim(
// CHECK: arith.constant 0xFF800000
// CHECK: linalg.fill
// CHECK: hfusion.reduce_with_index {{.*}} dimensions = [1]
func.func @torch.aten.max.dim(%arg0: !torch.vtensor<[6,3,2],f32>) -> !torch.vtensor<[6,2],f32> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %int1, %false : !torch.vtensor<[6,3,2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[6,2],f32>, !torch.vtensor<[6,2],si64>
  return %values : !torch.vtensor<[6,2],f32>
}

// CHECK-LABEL: @torch.aten.max.dim.keepdim(
// CHECK: arith.constant -2147483648
// CHECK: linalg.fill
// CHECK: hfusion.reduce_with_index {{.*}} dimensions = [1]
// CHECK: tensor.expand_shape
func.func @torch.aten.max.dim.keepdim(%arg0: !torch.vtensor<[6,3,2],si32>) -> !torch.vtensor<[6,1,2],si32> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %values, %indices = torch.aten.max.dim %arg0, %int1, %true : !torch.vtensor<[6,3,2],si32>, !torch.int, !torch.bool -> !torch.vtensor<[6,1,2],si32>, !torch.vtensor<[6,1,2],si64>
  return %values : !torch.vtensor<[6,1,2],si32>
}

// CHECK-LABEL: @torch.aten.min.dim(
// CHECK: arith.constant 0x7C00
// CHECK: linalg.fill
// CHECK: hfusion.reduce_with_index {{.*}} dimensions = [1]
func.func @torch.aten.min.dim(%arg0: !torch.vtensor<[6,3,2],f16>) -> !torch.vtensor<[6,2],f16> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %values, %indices = torch.aten.min.dim %arg0, %int1, %false : !torch.vtensor<[6,3,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[6,2],f16>, !torch.vtensor<[6,2],si64>
  return %values : !torch.vtensor<[6,2],f16>
}

// CHECK-LABEL: @torch.aten.min.dim.keepdim(
// CHECK: arith.constant 2147483647
// CHECK: linalg.fill
// CHECK: hfusion.reduce_with_index {{.*}} dimensions = [1]
// CHECK: tensor.expand_shape
func.func @torch.aten.min.dim.keepdim(%arg0: !torch.vtensor<[6,3,2],si32>) -> !torch.vtensor<[6,1,2],si32> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %values, %indices = torch.aten.min.dim %arg0, %int1, %true : !torch.vtensor<[6,3,2],si32>, !torch.int, !torch.bool -> !torch.vtensor<[6,1,2],si32>, !torch.vtensor<[6,1,2],si64>
  return %values : !torch.vtensor<[6,1,2],si32>
}