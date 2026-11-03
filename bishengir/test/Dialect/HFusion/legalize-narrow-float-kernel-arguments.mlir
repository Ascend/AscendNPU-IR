// RUN: bishengir-opt %s -hfusion-legalize-narrow-float-kernel-arguments | FileCheck %s

// CHECK-LABEL: func.func @entry
// CHECK-SAME: %[[F16:.*]]: f32, %[[BF16:.*]]: f32
// CHECK: %[[F16_NARROW:.*]] = arith.truncf %[[F16]] : f32 to f16
// CHECK: %[[BF16_NARROW:.*]] = arith.truncf %[[BF16]] : f32 to bf16
// CHECK: "test.use"(%[[F16_NARROW]], %[[BF16_NARROW]]) : (f16, bf16) -> ()
func.func @entry(%f16: f16, %bf16: bf16)
    attributes {hacc.entry} {
  "test.use"(%f16, %bf16) : (f16, bf16) -> ()
  return
}

// -----

// CHECK-LABEL: func.func private @mix_entry
// CHECK-SAME: (f32, f32) attributes {hacc.mix_entry}
func.func private @mix_entry(%f16: f16, %bf16: bf16)
    attributes {hacc.mix_entry}
