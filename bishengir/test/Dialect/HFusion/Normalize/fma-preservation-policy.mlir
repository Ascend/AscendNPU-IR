// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="target=Ascend950PR_9589" --mlir-print-ir-after=convert-math-to-hfusion --mlir-print-ir-after-change 2>&1 | FileCheck %s --check-prefix=PRESERVE
// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="target=Ascend950PR_9589 disable-hfusion-vectorize=true" --mlir-print-ir-after=convert-math-to-hfusion --mlir-print-ir-after-change 2>&1 | FileCheck %s --check-prefix=DECOMPOSE
// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="target=Ascend950PR_9589 enable-mixed-cv=true" --mlir-print-ir-after=convert-math-to-hfusion --mlir-print-ir-after-change 2>&1 | FileCheck %s --check-prefix=DECOMPOSE

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // PRESERVE: IR Dump After ConvertMathToHFusion
  // PRESERVE-LABEL: func.func @fma_policy
  // PRESERVE: hfusion.elemwise_ternary {fun = #hfusion.ternary_fn<fma>}
  // DECOMPOSE: IR Dump After ConvertMathToHFusion
  // DECOMPOSE-LABEL: func.func @fma_policy
  // DECOMPOSE: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
  // DECOMPOSE: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
  // DECOMPOSE-NOT: hfusion.elemwise_ternary
  func.func @fma_policy(%a: tensor<64xf32>, %b: tensor<64xf32>,
                        %c: tensor<64xf32>) -> tensor<64xf32> {
    %result = math.fma %a, %b, %c : tensor<64xf32>
    return %result : tensor<64xf32>
  }
}
