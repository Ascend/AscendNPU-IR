// RUN: bishengir-opt <%s --split-input-file -convert-torch-to-hfusion | FileCheck %s

// CHECK-LABEL: @torch.arange.start_step(
// CHECK: hfusion.arange
func.func @torch.arange.start_step() -> !torch.vtensor<[10],si32> {
  %none = torch.constant.none
  %int10 = torch.constant.int 10
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %npu3A0 = torch.constant.device "npu:0"
  %0 = torch.aten.arange.start_step %int0, %int10, %int1, %int3, %none, %npu3A0, %none : !torch.int, !torch.int, !torch.int, !torch.int, !torch.none, !torch.Device, !torch.none -> !torch.vtensor<[10],si32>
  return %0 : !torch.vtensor<[10],si32>
}
