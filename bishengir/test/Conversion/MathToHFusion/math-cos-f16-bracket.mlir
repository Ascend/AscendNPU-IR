// RUN: bishengir-opt -convert-math-to-hfusion %s -split-input-file | FileCheck %s

// An f32 cos bracketed by f16 widening/narrowing keeps the low-precision
// Normalize recipe (pi reduction + Taylor degree 9) and skips both the 2/pi
// table and the SIMT call. This split covers every supported way the input can
// be derived from f16 and the result narrowed back to f16.
//
// CHECK-NOT: memref.global @tbl
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // arith.extf / arith.truncf bracket.
  func.func @bracket_extf_truncf(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // hfusion.cast bracket.
  func.func @bracket_hfusion_cast(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %e0 = tensor.empty() : tensor<8x16xf32>
    %0 = hfusion.cast ins(%arg0 : tensor<8x16xf16>) outs(%e0 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %e1 = tensor.empty() : tensor<8x16xf16>
    %2 = hfusion.cast ins(%1 : tensor<8x16xf32>) outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // f16 provenance through tensor.extract_slice.
  func.func @bracket_input_extract_slice(%arg0: tensor<16x16xf16>) -> tensor<8x16xf16> {
    %s = tensor.extract_slice %arg0[0, 0] [8, 16] [1, 1] : tensor<16x16xf16> to tensor<8x16xf16>
    %0 = arith.extf %s : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // f16 provenance through tensor.collapse_shape.
  func.func @bracket_input_collapse_shape(%arg0: tensor<8x1x16xf16>) -> tensor<8x16xf16> {
    %c = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<8x1x16xf16> into tensor<8x16xf16>
    %0 = arith.extf %c : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // f16 provenance through tensor.expand_shape.
  func.func @bracket_input_expand_shape(%arg0: tensor<8x16xf16>) -> tensor<8x16x1xf16> {
    %e = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [8, 16, 1] : tensor<8x16xf16> into tensor<8x16x1xf16>
    %0 = arith.extf %e : tensor<8x16x1xf16> to tensor<8x16x1xf32>
    %1 = math.cos %0 : tensor<8x16x1xf32>
    %2 = arith.truncf %1 : tensor<8x16x1xf32> to tensor<8x16x1xf16>
    return %2 : tensor<8x16x1xf16>
  }
  // f16 provenance through tensor.cast.
  func.func @bracket_input_tensor_cast(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %c = tensor.cast %arg0 : tensor<8x16xf16> to tensor<8x16xf16>
    %0 = arith.extf %c : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // f16 provenance through a linalg.generic body that widens f16.
  func.func @bracket_input_generic(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %e = tensor.empty() : tensor<8x16xf32>
    %g = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x16xf16>) outs(%e : tensor<8x16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %x = arith.extf %in : f16 to f32
      linalg.yield %x : f32
    } -> tensor<8x16xf32>
    %1 = math.cos %g : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // f16 provenance through a linalg.generic with two f16 inputs.
  func.func @bracket_input_generic_two_inputs(%arg0: tensor<8x16xf16>, %arg1: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %e = tensor.empty() : tensor<8x16xf32>
    %g = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x16xf16>, tensor<8x16xf16>) outs(%e : tensor<8x16xf32>) {
    ^bb0(%in: f16, %in1: f16, %out: f32):
      %x = arith.extf %in : f16 to f32
      %y = arith.extf %in1 : f16 to f32
      %z = arith.addf %x, %y : f32
      linalg.yield %z : f32
    } -> tensor<8x16xf32>
    %1 = math.cos %g : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // Narrowing through tensor.extract_slice before truncf.
  func.func @bracket_output_extract_slice(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %s = tensor.extract_slice %1[0, 0] [8, 16] [1, 1] : tensor<8x16xf32> to tensor<8x16xf32>
    %2 = arith.truncf %s : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // Narrowing through tensor.cast before truncf.
  func.func @bracket_output_tensor_cast(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %c = tensor.cast %1 : tensor<8x16xf32> to tensor<8x16xf32>
    %2 = arith.truncf %c : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
}

// The recipe computes k = round(x / pi + 0.5), range-reduces with the pi
// pieces, shifts by pi/2, evaluates the degree-9 Taylor series and applies the
// parity sign. Verify the full recipe once. The positive recipe checks fail if
// the SIMT custom op replaces the expansion, so no per-function CHECK-NOT is
// needed here.
// CHECK-LABEL: func.func @bracket_extf_truncf
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.constant 5.000000e-01 : f32
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<round>
// CHECK: arith.constant 3.140625 : f32
// CHECK: arith.constant 2.75573188E-6 : f32
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<floor>
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_hfusion_cast
// CHECK: arith.constant 0.318309873 : f32
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<round>
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<floor>
// CHECK: hfusion.cast ins({{.*}} : tensor<8x16xf32>) outs({{.*}} : tensor<8x16xf16>)
// CHECK: return

// CHECK-LABEL: func.func @bracket_input_extract_slice
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_input_collapse_shape
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_input_expand_shape
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_input_tensor_cast
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_input_generic
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_input_generic_two_inputs
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_output_extract_slice
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_output_tensor_cast
// CHECK: arith.constant 0.318309873 : f32
// CHECK: arith.truncf
// CHECK: return

// -----

// Cases that must not take the f16 recipe: every one still calls the SIMT
// kernel.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // f16-derived input but the result is not narrowed back to f16.
  func.func @f16_input_no_narrow(%arg0: tensor<8x16xf16>) -> tensor<8x16xf32> {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
  // Result narrowed to f16, but the input is not f16-derived.
  func.func @f32_input_narrow(%arg0: tensor<8x16xf32>) -> tensor<8x16xf16> {
    %1 = math.cos %arg0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // The cos result has another use, so it cannot be treated as a pure
  // f16 bracket.
  func.func @f16_derived_multi_use(%arg0: tensor<8x16xf16>) -> (tensor<8x16xf16>, tensor<8x16xf32>) {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2, %1 : tensor<8x16xf16>, tensor<8x16xf32>
  }
  // Arithmetic between the f16 widening and cos leaves the f16 range, so the
  // input is not considered f16-derived.
  func.func @derived_addf(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<8x16xf32>
    %a = arith.addf %0, %cst : tensor<8x16xf32>
    %1 = math.cos %a : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // A two-input linalg.generic without an f16 widening is not f16-derived.
  func.func @generic_two_f32(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %e = tensor.empty() : tensor<8x16xf32>
    %g = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e : tensor<8x16xf32>) {
    ^bb0(%in: f32, %in1: f32, %out: f32):
      %z = arith.addf %in, %in1 : f32
      linalg.yield %z : f32
    } -> tensor<8x16xf32>
    %1 = math.cos %g : tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
  // An hfusion.cast whose source is not f16 keeps walking the chain; here it
  // ends at an f32 argument, so the recipe is not used.
  func.func @cast_f32_not_f16(%arg0: tensor<8x16xf32>) -> tensor<8x16xf16> {
    %e = tensor.empty() : tensor<8x16xf32>
    %0 = hfusion.cast ins(%arg0 : tensor<8x16xf32>) outs(%e : tensor<8x16xf32>) -> tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // bf16 is not f16: widening from bf16 does not enable the recipe even when
  // the result is narrowed to f16.
  func.func @bf16_derived_f16_result(%arg0: tensor<8x16xbf16>) -> tensor<8x16xf16> {
    %0 = arith.extf %arg0 : tensor<8x16xbf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xf16>
    return %2 : tensor<8x16xf16>
  }
  // Narrowing through hfusion.cast to bf16 is not narrowing to f16.
  func.func @f16_derived_cast_bf16(%arg0: tensor<8x16xf16>) -> tensor<8x16xbf16> {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %e = tensor.empty() : tensor<8x16xbf16>
    %2 = hfusion.cast ins(%1 : tensor<8x16xf32>) outs(%e : tensor<8x16xbf16>) -> tensor<8x16xbf16>
    return %2 : tensor<8x16xbf16>
  }
  // Narrowing to bf16 is not narrowing to f16.
  func.func @f16_derived_bf16_result(%arg0: tensor<8x16xf16>) -> tensor<8x16xbf16> {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %1 = math.cos %0 : tensor<8x16xf32>
    %2 = arith.truncf %1 : tensor<8x16xf32> to tensor<8x16xbf16>
    return %2 : tensor<8x16xbf16>
  }
  // A single-input linalg.generic whose chain ends at a function argument is
  // not f16-derived either.
  func.func @generic_one_f32(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %e = tensor.empty() : tensor<8x16xf32>
    %g = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x16xf32>) outs(%e : tensor<8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %z = arith.addf %in, %in : f32
      linalg.yield %z : f32
    } -> tensor<8x16xf32>
    %1 = math.cos %g : tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
}

// CHECK-LABEL: func.func @f16_input_no_narrow
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @f32_input_narrow
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @f16_derived_multi_use
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @derived_addf
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @generic_two_f32
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @cast_f32_not_f16
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @bf16_derived_f16_result
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @f16_derived_cast_bf16
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @f16_derived_bf16_result
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// CHECK-LABEL: func.func @generic_one_f32
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK: return

// -----

// The f16 bracket optimization is subordinate to the rank limit: an
// f16-bracketed f32 cos above the template-library rank still uses the
// HFusion Normalize expansion.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @bracket_rank6(%arg0: tensor<1x1x1x1x1x2xf16>) -> tensor<1x1x1x1x1x2xf16> {
    %0 = arith.extf %arg0 : tensor<1x1x1x1x1x2xf16> to tensor<1x1x1x1x1x2xf32>
    %1 = math.cos %0 : tensor<1x1x1x1x1x2xf32>
    %2 = arith.truncf %1 : tensor<1x1x1x1x1x2xf32> to tensor<1x1x1x1x1x2xf16>
    return %2 : tensor<1x1x1x1x1x2xf16>
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @bracket_rank6
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins({{.*}} : tensor<1x1x1x1x1x2xf32>) outs({{.*}} : tensor<1x1x1x1x1x2xf32>)
// CHECK: arith.truncf
// CHECK: return
