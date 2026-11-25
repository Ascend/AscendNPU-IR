// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivmave-to-ave-intrin | FileCheck %s

// Unsupported BF16 ULE keeps its result intact during partial conversion.
// CHECK-LABEL: func.func @bf16_ule_invalid
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[RES:.*]] = ave.hir.vcmp <ULE> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xbf16>, vector<64xi1> -> vector<64xi1>
// CHECK-NEXT: return %[[RES]] : vector<64xi1>
func.func @bf16_ule_invalid(%lhs: vector<64xbf16>, %rhs: vector<64xbf16>, %mask: vector<64xi1>) -> vector<64xi1> {
  %res = ave.hir.vcmp <ULE> %lhs, %rhs, %mask : vector<64xbf16>, vector<64xi1> -> vector<64xi1>
  return %res : vector<64xi1>
}

// BF16 equality still lowers to the compare intrinsic.
// CHECK-LABEL: func.func @bf16_eq
// CHECK: %[[EQ:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.eq.s.z"
// CHECK-NEXT: return %[[EQ]] : vector<256xi1>
func.func @bf16_eq(%lhs: vector<128xbf16>, %rhs: vector<128xbf16>, %mask: vector<256xi1>) -> vector<256xi1> {
  %res = ave.hir.vcmp <EQ> %lhs, %rhs, %mask : vector<128xbf16>, vector<256xi1> -> vector<256xi1>
  return %res : vector<256xi1>
}

// Unsigned comparison still lowers for signless integer operands.
// CHECK-LABEL: func.func @i8_ule
// CHECK: %[[ULE:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.le.u.z"
// CHECK-NEXT: return %[[ULE]] : vector<256xi1>
func.func @i8_ule(%lhs: vector<256xi8>, %rhs: vector<256xi8>, %mask: vector<256xi1>) -> vector<256xi1> {
  %res = ave.hir.vcmp <ULE> %lhs, %rhs, %mask : vector<256xi8>, vector<256xi1> -> vector<256xi1>
  return %res : vector<256xi1>
}
