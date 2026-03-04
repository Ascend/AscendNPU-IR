// RUN: bishengir-compile %s -enable-lir-compile=false -enable-hfusion-compile=true -enable-torch-compile=true | FileCheck %s

// CHECK: LLVMDialectModule
func.func @torch.aten.mul_tensor(%arg0: !torch.vtensor<[4096],f16>, %arg1: !torch.vtensor<[1,56,4096],f16>) -> !torch.vtensor<[1,56,4096],f16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4096],f16>, !torch.vtensor<[1,56,4096],f16> -> !torch.vtensor<[1,56,4096],f16>
  return %0 : !torch.vtensor<[1,56,4096],f16>
}