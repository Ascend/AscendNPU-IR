// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins | FileCheck %s

// CHECK-LABEL: func.func @permute(
// CHECK-SAME: %[[SRC1:.*]]: i32, %[[SRC2:.*]]: i32, %[[SELECTOR:.*]]: i32
func.func @permute(%src1: i32, %src2: i32, %selector: i32) -> i32 {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.prmt"(%[[SRC1]], %[[SRC2]], %[[SELECTOR]]) : (i32, i32, i32) -> i32
  %res = ascend_dpx.permute %src1, %src2, %selector : (i32, i32, i32) -> i32
  // CHECK: return %[[RES]] : i32
  return %res : i32
}
