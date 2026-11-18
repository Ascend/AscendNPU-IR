// RUN: bishengir-opt %s --convert-hivm-to-std | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  // RINT's numeric ABI must reach the template as CastMode::R == 0.
  // CHECK-LABEL: func.func @nearest_i32(
  // CHECK: %[[R:.*]] = arith.constant 0 : i32
  // CHECK-NOT: call @
  // CHECK: call @vcast_float_to_int32_t_1d_with_mode({{.*}}, %[[R]])
  // CHECK-NOT: call @
  // CHECK: return
  func.func @nearest_i32(%a: memref<16xf32, #hivm.address_space<ub>>, %out: memref<16xi32, #hivm.address_space<ub>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.vcast ins(%a : memref<16xf32, #hivm.address_space<ub>>) outs(%out : memref<16xi32, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<rint>
    return
  }
}
