// RUN: bishengir-opt -hfusion-auto-schedule -split-input-file -verify-diagnostics %s

module {
  // expected-error @+4 {{Unsupported i1 element type}}
  // expected-error @+3 {{Failed to analyze and verify kernel}}
  // expected-error @+2 {{'func.func' op Failed to create and apply schedule}}
  // expected-error @+1 {{'func.func' op Failed to run pre schedule procedure}}
  func.func @test_reduce_i1(%arg0: tensor<1024x4xi1>) -> tensor<1024xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %0 = tensor.empty() : tensor<1024xi1>
    %reduced = linalg.reduce { arith.ori } ins(%arg0 : tensor<1024x4xi1>) outs(%0 : tensor<1024xi1>) dimensions = [1] 
    return %reduced : tensor<1024xi1>
  }
}
