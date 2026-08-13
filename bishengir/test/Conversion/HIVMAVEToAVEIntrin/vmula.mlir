// RUN: bishengir-opt %s -ave-normalize-ops -convert-hivmave-to-ave-intrin | FileCheck %s

// CHECK-LABEL: func.func @lower_vmula_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vmula.s.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vmula_operand_order(
    %a: vector<64xf32>, %b: vector<64xf32>,
    %c: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vmula %c, %a, %b, %mask : vector<64xf32>, vector<64xi1>
  return %result : vector<64xf32>
}
