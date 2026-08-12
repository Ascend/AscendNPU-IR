// RUN: bishengir-opt -convert-hivm-to-std %s -split-input-file | FileCheck %s

// Verify FixpipeOp c0_pad_en lowers to an i1 library-call argument after
// channel_split. Omitted attr defaults to true; explicit false is preserved.

// CHECK-LABEL: func.func @fixpipe_c0_pad_default_true
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @fixpipe_c0_pad_default_true(
      %src: memref<1x1x16x16xi32, #hivm.address_space<cc>>,
      %dst: memref<1x1x16x16xi8, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // channel_split defaults false; c0_pad_en defaults true
    // CHECK: %[[FALSE:.*]] = arith.constant false
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: call @fixpipe_normal_int32_t_to_int8_t_4d_to_4d_cbuf({{.*}}, %[[FALSE]], %[[TRUE]], {{.*}})
    hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>}
        ins(%src : memref<1x1x16x16xi32, #hivm.address_space<cc>>)
        outs(%dst : memref<1x1x16x16xi8, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @fixpipe_c0_pad_explicit_false
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @fixpipe_c0_pad_explicit_false(
      %src: memref<1x1x16x16xi32, #hivm.address_space<cc>>,
      %dst: memref<1x1x16x16xi8, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // Both channel_split and c0_pad_en are false; CSE folds them to one constant.
    // CHECK: %[[FALSE:.*]] = arith.constant false
    // CHECK: call @fixpipe_normal_int32_t_to_int8_t_4d_to_4d_cbuf({{.*}}, %[[FALSE]], %[[FALSE]], {{.*}})
    hivm.hir.fixpipe {c0_pad_en = false, pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>}
        ins(%src : memref<1x1x16x16xi32, #hivm.address_space<cc>>)
        outs(%dst : memref<1x1x16x16xi8, #hivm.address_space<cbuf>>)
    return
  }
}
