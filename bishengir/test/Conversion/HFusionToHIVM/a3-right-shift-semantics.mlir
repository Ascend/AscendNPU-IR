// RUN: bishengir-opt %s -convert-hfusion-to-hivm | FileCheck %s --check-prefix=ATTR
// RUN: bishengir-opt %s -convert-hfusion-to-hivm -convert-hivm-to-std | FileCheck %s --check-prefix=CALL

// The signless i32 type must not erase the distinction between the two shifts.
// Neither source op requests rounding, even though HIVM defaults round to true.
module attributes {hacc.target = #hacc.target<"Ascend910_9391">} {
  // ATTR-LABEL: func.func @shrui
  // ATTR: hivm.hir.vshr
  // ATTR-SAME: round : false
  // ATTR-SAME: is_signed : false
  // ATTR: return
  // CALL-LABEL: func.func @shrui(
  // CALL: call @vshrs_vs_1d_uint32_t
  func.func @shrui(%a: memref<16xi32>, %b: i32, %out: memref<16xi32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>}
      ins(%a, %b : memref<16xi32>, i32) outs(%out : memref<16xi32>)
    return
  }
  // ATTR-LABEL: func.func @shrsi
  // ATTR: hivm.hir.vshr
  // ATTR-SAME: round : false
  // ATTR-NOT: is_signed : false
  // ATTR: return
  // CALL-LABEL: func.func @shrsi(
  // CALL: call @vshrs_vs_1d_int32_t
  func.func @shrsi(%a: memref<16xi32>, %b: i32, %out: memref<16xi32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>}
      ins(%a, %b : memref<16xi32>, i32) outs(%out : memref<16xi32>)
    return
  }
}
