// RUN: bishengir-opt <%s --split-input-file -convert-torch-to-hfusion | FileCheck %s
// RUN: bishengir-opt <%s --split-input-file -convert-torch-to-hfusion -mlir-print-op-generic | FileCheck %s --check-prefix=CHECK-GENERIC

// CHECK-LABEL: @torch.aten.mul_tensor(
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
func.func @torch.aten.mul_tensor(%arg0: !torch.vtensor<[4096],f16>, %arg1: !torch.vtensor<[1,56,4096],f16>) -> !torch.vtensor<[1,56,4096],f16> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4096],f16>, !torch.vtensor<[1,56,4096],f16> -> !torch.vtensor<[1,56,4096],f16>
  return %0 : !torch.vtensor<[1,56,4096],f16>
}

// -----

// CHECK-LABEL: @torch.aten.mul_scalar(
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
func.func @torch.aten.mul_scalar(%arg0: !torch.vtensor<[1,32,56,128],f16>) -> !torch.vtensor<[1,32,56,128],f16> {
  %float2.973020e-01 = torch.constant.float 0.29730177875068026
  %0 = torch.aten.mul.Scalar %arg0, %float2.973020e-01 : !torch.vtensor<[1,32,56,128],f16>, !torch.float -> !torch.vtensor<[1,32,56,128],f16>
return %0 : !torch.vtensor<[1,32,56,128],f16>
}

// -----

// CHECK-LABEL: @torch.aten.div_tensor(
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
func.func @torch.aten.div_tensor(%arg0: !torch.vtensor<[4096],f16>, %arg1: !torch.vtensor<[1,56,4096],f16>) -> !torch.vtensor<[1,56,4096],f16> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[4096],f16>, !torch.vtensor<[1,56,4096],f16> -> !torch.vtensor<[1,56,4096],f16>
  return %0 : !torch.vtensor<[1,56,4096],f16>
}

// -----

// CHECK-LABEL: @torch.aten.div_scalar(
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
func.func @torch.aten.div_scalar(%arg0: !torch.vtensor<[1,32,56,128],f16>) -> !torch.vtensor<[1,32,56,128],f16> {
  %float2.973020e-01 = torch.constant.float 0.29730177875068026
  %0 = torch.aten.div.Scalar %arg0, %float2.973020e-01 : !torch.vtensor<[1,32,56,128],f16>, !torch.float -> !torch.vtensor<[1,32,56,128],f16>
  return %0 : !torch.vtensor<[1,32,56,128],f16>
}

// -----

// CHECK-LABEL: @torch.aten.clamp_min.Tensor
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1, 2]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
func.func @torch.aten.clamp_min.Tensor(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> {
  %0 = torch.aten.clamp_min.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[3,4,5],si8>
  return %0: !torch.vtensor<[3,4,5],si8>
}

// -----

// CHECK-LABEL: @torch.aten.clamp_min
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
func.func @torch.aten.clamp_min(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.float) -> !torch.vtensor<[3,4,5],f32> {
  %0 = torch.aten.clamp_min %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.float -> !torch.vtensor<[3,4,5],f32>
  return %0: !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @torch.aten.clamp_max.Tensor
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1, 2]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
func.func @torch.aten.clamp_max.Tensor(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> {
  %0 = torch.aten.clamp_max.Tensor %arg0, %arg1 : !torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[3,4,5],si8>
  return %0: !torch.vtensor<[3,4,5],si8>
}

// -----

// CHECK-LABEL: @torch.aten.clamp_max
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
func.func @torch.aten.clamp_max(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.float) -> !torch.vtensor<[3,4,5],f32> {
  %0 = torch.aten.clamp_max %arg0, %arg1 : !torch.vtensor<[3,4,5],f32>, !torch.float -> !torch.vtensor<[3,4,5],f32>
  return %0: !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK-LABEL: @torch.aten.maximum
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1, 2]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
func.func @torch.aten.maximum(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> {
  %0 = torch.aten.maximum %arg0, %arg1 : !torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[3,4,5],si8>
  return %0: !torch.vtensor<[3,4,5],si8>
}

// -----

// CHECK-LABEL: @torch.aten.minimum
// CHECK: linalg.broadcast {{.*}} dimensions = [0, 1, 2]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
func.func @torch.aten.minimum(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> {
  %0 = torch.aten.minimum %arg0, %arg1 : !torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[3,4,5],si8>
  return %0: !torch.vtensor<[3,4,5],si8>
}

// -----

// CHECK-LABEL: @torch.aten.add.Tensor
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
func.func @torch.aten.add.Tensor(%arg0: !torch.vtensor<[3, 4],f32>, %arg1: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[3, 4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3, 4],f32>, !torch.vtensor<[3, 4],f32>, !torch.int -> !torch.vtensor<[3, 4],f32>
  return %0 : !torch.vtensor<[3, 4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.add.Tensor.hfusion_cast
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
func.func @torch.aten.add.Tensor.hfusion_cast(%arg0: !torch.vtensor<[3, 4],f16>, %arg1: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[3, 4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3, 4],f16>, !torch.vtensor<[3, 4],f32>, !torch.int -> !torch.vtensor<[3, 4],f32>
  return %0 : !torch.vtensor<[3, 4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.add.Tensor.dynamic
// CHECK: tensor.dim
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
 func.func @torch.aten.add.Tensor.dynamic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL: @torch.aten.add.Scalar
// CHECK: linalg.fill
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
func.func @torch.aten.add.Scalar(%arg0: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[3, 4],f32> {
  %int1 = torch.constant.int 1
  %int256 = torch.constant.int 256
  %0 = torch.aten.add.Scalar %arg0, %int256, %int1 : !torch.vtensor<[3, 4],f32>, !torch.int, !torch.int -> !torch.vtensor<[3, 4],f32>
  return %0 : !torch.vtensor<[3, 4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.sub.Tensor
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @torch.aten.sub.Tensor(%arg0: !torch.vtensor<[3, 4],f32>, %arg1: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[3, 4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3, 4],f32>, !torch.vtensor<[3, 4],f32>, !torch.int -> !torch.vtensor<[3, 4],f32>
  return %0 : !torch.vtensor<[3, 4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.sub.Tensor_broadcast
// CHECK: tensor.collapse_shape
// CHECK: linalg.broadcast
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @torch.aten.sub.Tensor_broadcast(%arg0: !torch.vtensor<[3, 1],f32>, %arg1: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[3, 4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3, 1],f32>, !torch.vtensor<[3, 4],f32>, !torch.int -> !torch.vtensor<[3, 4],f32>
  return %0 : !torch.vtensor<[3, 4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.sub.Tensor_broadcast1
// CHECK: tensor.collapse_shape
// CHECK: linalg.broadcast
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @torch.aten.sub.Tensor_broadcast1(%arg0: !torch.vtensor<[3, 1, 5],f32>, %arg1: !torch.vtensor<[3, 4, 5],f32>) -> !torch.vtensor<[3, 4, 5],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[3, 1, 5],f32>, !torch.vtensor<[3, 4, 5],f32>, !torch.int -> !torch.vtensor<[3, 4, 5],f32>
  return %0 : !torch.vtensor<[3, 4, 5],f32>
}

// -----

// CHECK-LABEL: @torch.aten.sub.Scalar
// CHECK: linalg.fill
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @torch.aten.sub.Scalar(%arg0: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[3, 4],f32> {
  %int1 = torch.constant.int 1
  %int256 = torch.constant.int 256
  %0 = torch.aten.sub.Scalar %arg0, %int256, %int1 : !torch.vtensor<[3, 4],f32>, !torch.int, !torch.int -> !torch.vtensor<[3, 4],f32>
  return %0 : !torch.vtensor<[3, 4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.abs
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
func.func @torch.aten.abs(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.abs %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.ceil
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
func.func @torch.aten.ceil(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.ceil %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.floor
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
func.func @torch.aten.floor(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.floor %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.negf
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
func.func @torch.aten.negf(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.log
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
func.func @torch.aten.log(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.log %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.exp
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
func.func @torch.aten.exp(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.exp %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.reciprocal
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
func.func @torch.aten.reciprocal(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.reciprocal %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.relu
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
func.func @torch.aten.relu(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.rsqrt
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
func.func @torch.aten.rsqrt(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.rsqrt %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.sqrt
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
func.func @torch.aten.sqrt(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.sqrt %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.erf
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
func.func @torch.aten.erf(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.erf %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.tanh
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
func.func @torch.aten.tanh(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.sin
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
func.func @torch.aten.sin(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.sin %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.cos
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
func.func @torch.aten.cos(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.cos %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.vnot
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
func.func @torch.aten.vnot(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> {
  %0 = torch.aten.bitwise_not %arg0 : !torch.vtensor<[3,4],si32> -> !torch.vtensor<[3,4],si32>
  return %0 : !torch.vtensor<[3,4],si32>
}

// -----

// CHECK-LABEL: @torch.aten.sigmoid
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
func.func @torch.aten.sigmoid(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %0 = torch.aten.sigmoid %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.clamp
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
func.func @torch.aten.clamp(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %int5 = torch.constant.int 5
  %int10 = torch.constant.int 10
  %0 = torch.aten.clamp %arg0, %int5, %int10 : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.clamp1
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
func.func @torch.aten.clamp1(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %int5 = torch.constant.int 5
  %none = torch.constant.none
  %0 = torch.aten.clamp %arg0, %int5, %none : !torch.vtensor<[3,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: @torch.aten.to.dtype.f32_f16
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
func.func @torch.aten.to.dtype.f32_f16(%arg0: !torch.vtensor<[6,4,3],f32>) -> !torch.vtensor<[6,4,3],f16> {
  %int5 = torch.constant.int 5
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.to.dtype %arg0, %int5, %false, %false, %none : !torch.vtensor<[6,4,3],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6,4,3],f16>
  return %0 : !torch.vtensor<[6,4,3],f16>
}

// -----

// CHECK-LABEL: @torch.aten.to.dtype.i1_f32
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
// CHECK-GENERIC: arith.uitofp
func.func @torch.aten.to.dtype.i1_f32(%arg0: !torch.vtensor<[6,4,3],i1>) -> !torch.vtensor<[6,4,3],f32> {
  %int6 = torch.constant.int 6
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.to.dtype %arg0, %int6, %false, %false, %none : !torch.vtensor<[6,4,3],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6,4,3],f32>
  return %0 : !torch.vtensor<[6,4,3],f32>
}

// -----

// CHECK-LABEL: @torch.aten.to.dtype.f16_f32
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
func.func @torch.aten.to.dtype.f16_f32(%arg0: !torch.vtensor<[6,4,3],f16>) -> !torch.vtensor<[6,4,3],f32> {
  %int6 = torch.constant.int 6
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.to.dtype %arg0, %int6, %false, %false, %none : !torch.vtensor<[6,4,3],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6,4,3],f32>
  return %0 : !torch.vtensor<[6,4,3],f32>
}

// -----

// CHECK-LABEL: @torch.aten.to.dtype.f16_i32
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<trunc>}
func.func @torch.aten.to.dtype.f16_i32(%arg0: !torch.vtensor<[6,4,3],f16>) -> !torch.vtensor<[6,4,3],si32> {
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.to.dtype %arg0, %int3, %false, %false, %none : !torch.vtensor<[6,4,3],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6,4,3],si32>
  return %0 : !torch.vtensor<[6,4,3],si32>
}

// -----

// CHECK-LABEL: @torch.aten.to.dtype.i32_bf16
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
func.func @torch.aten.to.dtype.i32_bf16(%arg0: !torch.vtensor<[6,4,3],si32>) -> !torch.vtensor<[6,4,3],bf16> {
  %int15 = torch.constant.int 15
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.to.dtype %arg0, %int15, %false, %false, %none : !torch.vtensor<[6,4,3],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6,4,3],bf16>
  return %0 : !torch.vtensor<[6,4,3],bf16>
}

// -----

// CHECK-LABEL: @torch.aten.pow.Tensor_Scalar
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
func.func @torch.aten.pow.Tensor_Scalar(%arg0: !torch.vtensor<[1024,1024],f32>) -> !torch.vtensor<[1024,1024],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.pow.Tensor_Scalar %arg0, %int2 : !torch.vtensor<[1024,1024],f32>, !torch.int -> !torch.vtensor<[1024,1024],f32>
  return %0 : !torch.vtensor<[1024,1024],f32>
}

// -----

// CHECK-LABEL: @torch.aten.pow.Tensor_Tensor
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
func.func @torch.aten.pow.Tensor_Tensor(%arg0: !torch.vtensor<[1024,1024],f32>, %arg1: !torch.vtensor<[1024,1024],f32>) -> !torch.vtensor<[1024,1024],f32> {
  %0 = torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[1024,1024],f32>, !torch.vtensor<[1024,1024],f32> -> !torch.vtensor<[1024,1024],f32>
  return %0 : !torch.vtensor<[1024,1024],f32>
}

// -----

// CHECK-LABEL: @torch.aten.logical_and
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
func.func @torch.aten.logical_and(%arg0: !torch.vtensor<[1024,1024],i1>, %arg1: !torch.vtensor<[1024,1024],i1>) -> !torch.vtensor<[1024,1024],i1> {
  %0 = torch.aten.logical_and %arg0, %arg1 : !torch.vtensor<[1024,1024],i1>, !torch.vtensor<[1024,1024],i1> -> !torch.vtensor<[1024,1024],i1>
  return %0 : !torch.vtensor<[1024,1024],i1>
}

// -----

// CHECK-LABEL: @torch.aten.logical_or
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
func.func @torch.aten.logical_or(%arg0: !torch.vtensor<[1024,1024],i1>, %arg1: !torch.vtensor<[1024,1024],i1>) -> !torch.vtensor<[1024,1024],i1> {
  %0 = torch.aten.logical_or %arg0, %arg1 : !torch.vtensor<[1024,1024],i1>, !torch.vtensor<[1024,1024],i1> -> !torch.vtensor<[1024,1024],i1>
  return %0 : !torch.vtensor<[1024,1024],i1>
}

// -----

// CHECK-LABEL: @torch.aten.where.self.f32
// CHECK: hfusion.select
func.func @torch.aten.where.self.f32(%arg0: !torch.vtensor<[5,2],i1>, %arg1: !torch.vtensor<[5,2],f32>, %arg2: !torch.vtensor<[5,2],f32>) -> !torch.vtensor<[5,2],f32> {
  %0 = torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[5,2],i1>, !torch.vtensor<[5,2],f32>, !torch.vtensor<[5,2],f32> -> !torch.vtensor<[5,2],f32>
  return %0 : !torch.vtensor<[5,2],f32>
}

// -----

// CHECK-LABEL: @torch.aten.where.self.i32
// CHECK: hfusion.select
func.func @torch.aten.where.self.i32(%arg0: !torch.vtensor<[5,2],i1>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[5,2],si32>) -> !torch.vtensor<[5,2],si32> {
  %0 = torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[5,2],i1>, !torch.vtensor<[],si32>, !torch.vtensor<[5,2],si32> -> !torch.vtensor<[5,2],si32>
  return %0 : !torch.vtensor<[5,2],si32>
}

// -----

// CHECK-LABEL: @torch.aten.where.self_cast_broadcast
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
// CHECK: linalg.broadcast
// CHECK: hfusion.select
func.func @torch.aten.where.self_cast_broadcast(%arg0: !torch.vtensor<[5,2],i1>, %arg1: !torch.vtensor<[2],si32>, %arg2: !torch.vtensor<[5,2],f32>) -> !torch.vtensor<[5,2],f32> {
  %0 = torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[5,2],i1>, !torch.vtensor<[2],si32>, !torch.vtensor<[5,2],f32> -> !torch.vtensor<[5,2],f32>
  return %0 : !torch.vtensor<[5,2],f32>
}

// -----

// CHECK-LABEL: @torch.aten.gt.Scalar.f32_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
func.func @torch.aten.gt.Scalar.f32_i64(%arg0: !torch.vtensor<[200,200],f32>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.gt.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f32>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.gt.Scalar.f16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
func.func @torch.aten.gt.Scalar.f16_i64(%arg0: !torch.vtensor<[200,200],f16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.gt.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.gt.Scalar.bf16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
func.func @torch.aten.gt.Scalar.bf16_i64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.gt.Scalar %arg0, %int0 : !torch.vtensor<[200,200],bf16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.gt.Scalar.bf16_f64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
func.func @torch.aten.gt.Scalar.bf16_f64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %0 = torch.aten.gt.Scalar %arg0, %float1.000000e00 : !torch.vtensor<[200,200],bf16>, !torch.float -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.lt.Scalar.f32_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
func.func @torch.aten.lt.Scalar.f32_i64(%arg0: !torch.vtensor<[200,200],f32>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.lt.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f32>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.lt.Scalar.f16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
func.func @torch.aten.lt.Scalar.f16_i64(%arg0: !torch.vtensor<[200,200],f16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.lt.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.lt.Scalar.bf16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
func.func @torch.aten.lt.Scalar.bf16_i64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.lt.Scalar %arg0, %int0 : !torch.vtensor<[200,200],bf16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.lt.Scalar.bf16_f64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
func.func @torch.aten.lt.Scalar.bf16_f64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %0 = torch.aten.lt.Scalar %arg0, %float1.000000e00 : !torch.vtensor<[200,200],bf16>, !torch.float -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ne.Scalar.si8_int
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
func.func @torch.aten.ne.Scalar.si8_int(%arg0: !torch.vtensor<[200,200],si8>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.ne.Scalar %arg0, %int0 : !torch.vtensor<[200,200],si8>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ne.Scalar.ui8_int
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
func.func @torch.aten.ne.Scalar.ui8_int(%arg0: !torch.vtensor<[200,200],ui8>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.ne.Scalar %arg0, %int0 : !torch.vtensor<[200,200],ui8>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ne.Scalar.f32_float
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
func.func @torch.aten.ne.Scalar.f32_float(%arg0: !torch.vtensor<[200,200],f32>) -> !torch.vtensor<[200,200],i1> {
  %fpScalar = torch.constant.float 128.0
  %0 = torch.aten.ne.Scalar %arg0, %fpScalar : !torch.vtensor<[200,200],f32>, !torch.float -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.eq.Scalar.si8_int
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
func.func @torch.aten.eq.Scalar.si8_int(%arg0: !torch.vtensor<[200,200],si8>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.eq.Scalar %arg0, %int0 : !torch.vtensor<[200,200],si8>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.eq.Scalar.ui8_int
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
func.func @torch.aten.eq.Scalar.ui8_int(%arg0: !torch.vtensor<[200,200],ui8>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.eq.Scalar %arg0, %int0 : !torch.vtensor<[200,200],ui8>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.eq.Scalar.f32_float
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
func.func @torch.aten.eq.Scalar.f32_float(%arg0: !torch.vtensor<[200,200],f32>) -> !torch.vtensor<[200,200],i1> {
  %fpScalar = torch.constant.float 128.0
  %0 = torch.aten.eq.Scalar %arg0, %fpScalar : !torch.vtensor<[200,200],f32>, !torch.float -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ge.Scalar.f32_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
func.func @torch.aten.ge.Scalar.f32_i64(%arg0: !torch.vtensor<[200,200],f32>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.ge.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f32>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ge.Scalar.f16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
func.func @torch.aten.ge.Scalar.f16_i64(%arg0: !torch.vtensor<[200,200],f16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.ge.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ge.Scalar.bf16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
func.func @torch.aten.ge.Scalar.bf16_i64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.ge.Scalar %arg0, %int0 : !torch.vtensor<[200,200],bf16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ge.Scalar.bf16_f64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
func.func @torch.aten.ge.Scalar.bf16_f64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %0 = torch.aten.ge.Scalar %arg0, %float1.000000e00 : !torch.vtensor<[200,200],bf16>, !torch.float -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.le.Scalar.f32_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
func.func @torch.aten.le.Scalar.f32_i64(%arg0: !torch.vtensor<[200,200],f32>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.le.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f32>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.le.Scalar.f16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
func.func @torch.aten.le.Scalar.f16_i64(%arg0: !torch.vtensor<[200,200],f16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.le.Scalar %arg0, %int0 : !torch.vtensor<[200,200],f16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.le.Scalar.bf16_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
func.func @torch.aten.le.Scalar.bf16_i64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.le.Scalar %arg0, %int0 : !torch.vtensor<[200,200],bf16>, !torch.int -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.le.Scalar.bf16_f64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
func.func @torch.aten.le.Scalar.bf16_f64(%arg0: !torch.vtensor<[200,200],bf16>) -> !torch.vtensor<[200,200],i1> {
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %0 = torch.aten.le.Scalar %arg0, %float1.000000e00 : !torch.vtensor<[200,200],bf16>, !torch.float -> !torch.vtensor<[200,200],i1>
  return %0 : !torch.vtensor<[200,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.ne.Tensor.i32_i64
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
func.func @torch.aten.ne.Tensor.i32_i64(%arg0: !torch.vtensor<[2,200],si32>, %arg1: !torch.vtensor<[2,200],si64>) -> !torch.vtensor<[2,200],i1> {
  %int4 = torch.constant.int 4
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.to.dtype %arg0, %int4, %false, %false, %none : !torch.vtensor<[2,200],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,200],si64>
  %1 = torch.aten.ne.Tensor %0, %arg1 : !torch.vtensor<[2,200],si64>, !torch.vtensor<[2,200],si64> -> !torch.vtensor<[2,200],i1>
  return %1 : !torch.vtensor<[2,200],i1>
}

// -----

// CHECK-LABEL: @torch.aten.pow.Scalar(
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
func.func @torch.aten.pow.Scalar(%arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>)  {
  %float9.000000e-01 = torch.constant.float 9.000000e-01
  %1 = torch.aten.pow.Scalar %float9.000000e-01, %arg3 : !torch.float, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}
