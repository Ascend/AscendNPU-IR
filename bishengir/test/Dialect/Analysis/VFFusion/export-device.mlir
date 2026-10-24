// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend950PR_9579" --vf-fusion %s | FileCheck %s

// An outlined helper must not inherit its parent's export boundary.
// CHECK-LABEL: func.func private @exported_fused
// CHECK-NOT: export_device
// CHECK: return
// CHECK-LABEL: func.func @exported(
// CHECK-SAME: export_device
// CHECK-SAME: hacc.function_kind = #hacc.function_kind<DEVICE>
// CHECK: call @exported_fused
func.func @exported(%a: tensor<16xi16>, %b: tensor<16xi16>,
                    %c: tensor<16xi16>, %out: tensor<16xi16>) -> tensor<16xi16>
    attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<16xi16>, tensor<16xi16>) outs(%out : tensor<16xi16>) -> tensor<16xi16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %c : tensor<16xi16>, tensor<16xi16>) outs(%out : tensor<16xi16>) -> tensor<16xi16>
  return %1 : tensor<16xi16>
}
