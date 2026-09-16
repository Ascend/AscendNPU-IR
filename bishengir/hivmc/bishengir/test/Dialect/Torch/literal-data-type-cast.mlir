// RUN: bishengir-opt %s -literal-data-type-cast -allow-unregistered-dialect | FileCheck %s

module {
  func.func @fp64_literal() {
    // CHECK: torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    // CHECK-NOT: f64
    %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f64>) : !torch.vtensor<[],f64>
    "some_use"(%0) : (!torch.vtensor<[],f64>) -> ()
    return
  }
}
