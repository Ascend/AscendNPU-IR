// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960 -ave-normalize-ops -convert-hivmave-to-ave-intrin | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910_9589 -ave-normalize-ops -convert-hivmave-to-ave-intrin | FileCheck %s --check-prefix=LEGACY

// The fused multiply-add families are Ascend960-only: other targets keep the
// original ave.hir.vfma/vfms/... ops untouched.
// LEGACY-LABEL: func.func @lower_vfma_operand_order
// LEGACY: ave.hir.vfma
// LEGACY-LABEL: func.func @lower_vnmadd_operand_order
// LEGACY: ave.hir.vnmadd

// CHECK-LABEL: func.func @lower_vfma_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vfma.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vfma_operand_order(
    %a: vector<64xf32>, %b: vector<64xf32>,
    %c: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vfma %c, %a, %b, %mask : vector<64xf32>, vector<64xi1>
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_vfms_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<128xf16>, %[[B:.*]]: vector<128xf16>, %[[C:.*]]: vector<128xf16>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<128xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vfms.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vfms_operand_order(
    %a: vector<128xf16>, %b: vector<128xf16>,
    %c: vector<128xf16>) -> vector<128xf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %result = ave.hir.vfms %c, %a, %b, %mask : vector<128xf16>, vector<128xi1>
  return %result : vector<128xf16>
}

// CHECK-LABEL: func.func @lower_vfnma_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<128xbf16>, %[[B:.*]]: vector<128xbf16>, %[[C:.*]]: vector<128xbf16>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<128xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vfnma.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vfnma_operand_order(
    %a: vector<128xbf16>, %b: vector<128xbf16>,
    %c: vector<128xbf16>) -> vector<128xbf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %result = ave.hir.vfnma %c, %a, %b, %mask : vector<128xbf16>, vector<128xi1>
  return %result : vector<128xbf16>
}

// CHECK-LABEL: func.func @lower_vfnms_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vfnms.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vfnms_operand_order(
    %a: vector<64xf32>, %b: vector<64xf32>,
    %c: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vfnms %c, %a, %b, %mask : vector<64xf32>, vector<64xi1>
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_vmadd_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vmadd.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vmadd_operand_order(
    %a: vector<64xf32>, %b: vector<64xf32>,
    %c: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vmadd %c, %a, %b, %mask : vector<64xf32>, vector<64xi1>
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_vmsub_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vmsub.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vmsub_operand_order(
    %a: vector<64xf32>, %b: vector<64xf32>,
    %c: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vmsub %c, %a, %b, %mask : vector<64xf32>, vector<64xi1>
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_vnmadd_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vnmadd.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vnmadd_operand_order(
    %a: vector<64xf32>, %b: vector<64xf32>,
    %c: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vnmadd %c, %a, %b, %mask : vector<64xf32>, vector<64xi1>
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_vnmsub_operand_order
// CHECK-SAME: (%[[A:.*]]: vector<64xf32>, %[[B:.*]]: vector<64xf32>, %[[C:.*]]: vector<64xf32>)
// CHECK: %[[MASK:.*]] = builtin.unrealized_conversion_cast {{.*}} : vector<64xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vnmsub.m"(%[[C]], %[[A]], %[[B]], %[[MASK]])
func.func @lower_vnmsub_operand_order(
    %a: vector<64xf32>, %b: vector<64xf32>,
    %c: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vnmsub %c, %a, %b, %mask : vector<64xf32>, vector<64xi1>
  return %result : vector<64xf32>
}
