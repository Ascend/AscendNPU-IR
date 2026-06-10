// RUN: bishengir-opt --hivm-insert-anchors-and-backup -verify-diagnostics %s | FileCheck %s
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
// CHECK-LABEL: func.func @outlined_vf_0_backup
// CHECK-LABEL: func.func @fused_2_backup
// CHECK: call @outlined_vf_0_backup
// CHECK-LABEL: func.func @calc_backup
// CHECK: call @fused_2_backup
  
  func.func @outlined_vf_0() -> tensor<11x33xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %0 = tensor.empty() : tensor<11x33xf16>
    return %0 : tensor<11x33xf16>
  }
  func.func  @fused_2() -> tensor<33xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "mix"} {
    %1 = call @outlined_vf_0() {hivm.vector_function, no_inline} : () -> tensor<11x33xf16>
    %2 = tensor.empty() : tensor<1x33xf16>
    %3 = tensor.empty() : tensor<1x33xi32>
    %4:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index> ins(%1 : tensor<11x33xf16>) outs(%2, %3 : tensor<1x33xf16>, tensor<1x33xi32>) unsigned_src = false tie_break_left = true reduce_dims = [0] -> tensor<1x33xf16>, tensor<1x33xi32>
    %collapsed = tensor.collapse_shape %4#1 [[0, 1]] : tensor<1x33xi32> into tensor<33xi32>
    return %collapsed : tensor<33xi32>
  }
  func.func @calc(%arg0: i32, %arg1: i32, %arg2: i32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode= "mix"} {
    %0 = arith.muli %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg2 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = call @fused_2() : () -> tensor<33xi32>
    return
  }
}