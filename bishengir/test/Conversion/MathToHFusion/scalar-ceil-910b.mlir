// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B4-1 --convert-math-to-hfusion | FileCheck %s --check-prefixes=COMMON,MEM
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910_9382 --convert-math-to-hfusion | FileCheck %s --check-prefixes=COMMON,MEM
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B4-1 --convert-math-to-hfusion --hfusion-normalize-ops --canonicalize | FileCheck %s --check-prefix=NORM
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9589 --convert-math-to-hfusion | FileCheck %s --check-prefixes=COMMON,REG

module {
  // COMMON-LABEL: func.func @scalar_ceil_f32(
  // COMMON: tensor.from_elements
  // COMMON: linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // COMMON: tensor.extract
  // NORM-LABEL: func.func @scalar_ceil_f32(
  // NORM: tensor.from_elements
  // NORM: hfusion.cast {{.*}}round_mode = #hfusion.round_mode<ceil>
  // NORM: hfusion.bitcast
  // NORM: tensor.extract
  func.func @scalar_ceil_f32(%x: f32) -> f32 {
    %y = math.ceil %x : f32
    return %y : f32
  }

  // COMMON-LABEL: func.func @scalar_ceil_f16(
  // COMMON: tensor.from_elements {{.*}} : tensor<1xf16>
  // COMMON: linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // COMMON: tensor.extract
  func.func @scalar_ceil_f16(%x: f16) -> f16 {
    %y = math.ceil %x : f16
    return %y : f16
  }

  // COMMON-LABEL: func.func @scalar_ceil_bf16(
  // COMMON: arith.extf {{.*}} : bf16 to f32
  // COMMON: tensor.from_elements {{.*}} : tensor<1xf32>
  // COMMON: linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
  // COMMON: tensor.extract
  // COMMON: arith.truncf {{.*}} : f32 to bf16
  func.func @scalar_ceil_bf16(%x: bf16) -> bf16 {
    %y = math.ceil %x : bf16
    return %y : bf16
  }

  // MEM-LABEL: func.func @scalar_floor(
  // MEM: tensor.from_elements
  // MEM: linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
  // MEM: tensor.extract
  // REG-LABEL: func.func @scalar_floor(
  // REG: tensor.from_elements
  // REG: linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
  func.func @scalar_floor(%x: f32) -> f32 {
    %y = math.floor %x : f32
    return %y : f32
  }

  // COMMON-LABEL: func.func @inside_linalg(
  // COMMON: linalg.generic
  // COMMON-NOT: tensor.from_elements
  // COMMON: math.ceil
  // COMMON: linalg.yield
  func.func @inside_linalg(%x: tensor<8xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
    %y = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %rounded = math.ceil %v : f32
      linalg.yield %rounded : f32
    } -> tensor<8xf32>
    return %y : tensor<8xf32>
  }
}
