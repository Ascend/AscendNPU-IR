// RUN: bishengir-opt %s -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s

// CHECK-LABEL: func.func @scalar_broadcast_e4m3fn(
// CHECK-SAME: %[[SRC:.*]]: f8E4M3FN)
// CHECK: %[[BITS:.*]] = llvm.bitcast %[[SRC]] : f8E4M3FN to i8
// CHECK-NEXT: %[[VBR:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BITS]]) : (i8) -> vector<256xi8>
// CHECK-NEXT: %[[FP8:.*]] = llvm.bitcast %[[VBR]] : vector<256xi8> to vector<256xf8E4M3FN>
func.func @scalar_broadcast_e4m3fn(%arg0: f8E4M3FN) -> vector<256xf8E4M3FN> {
  %0 = ave.hir.scalar_broadcast %arg0 : f8E4M3FN -> vector<256xf8E4M3FN>
  return %0 : vector<256xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @scalar_broadcast_e5m2_narrow(
// CHECK-SAME: %[[SRC:.*]]: f8E5M2)
// CHECK: %[[BITS:.*]] = llvm.bitcast %[[SRC]] : f8E5M2 to i8
// CHECK-NEXT: %[[VBR:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BITS]]) : (i8) -> vector<256xi8>
// CHECK-NEXT: %[[FP8:.*]] = llvm.bitcast %[[VBR]] : vector<256xi8> to vector<256xf8E5M2>
// CHECK-NEXT: %[[NARROW:.*]] = builtin.unrealized_conversion_cast %[[FP8]] : vector<256xf8E5M2> to vector<64xf8E5M2>
func.func @scalar_broadcast_e5m2_narrow(%arg0: f8E5M2) -> vector<64xf8E5M2> {
  %0 = ave.hir.scalar_broadcast %arg0 : f8E5M2 -> vector<64xf8E5M2>
  return %0 : vector<64xf8E5M2>
}
