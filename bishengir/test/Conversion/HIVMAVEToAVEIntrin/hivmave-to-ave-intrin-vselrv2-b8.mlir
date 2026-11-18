// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend960 \
// RUN:   -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s \
// RUN:   --check-prefix=V2
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN:   -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s \
// RUN:   --check-prefix=LEGACY

// vselrv2 (V121, Ascend960) is a b8 masked move: dst = (mask_lane != 0) ?
// src0 : 0. It is represented by the Ascend960-only ave.hir.vselrv2 op. The
// legacy predicate select (ave.hir.vsel with an i1 mask) is unaffected on
// every target, including A5/A3.

// V2-LABEL: func.func @vselrv2_u8_ui8
// V2: %[[TRUE:.*]] = builtin.unrealized_conversion_cast %arg1 : vector<256xui8> to vector<256xi8>
// V2: %[[MSK:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<256xui8> to vector<256xi8>
// V2: "hivm_regbaseintrins.intr.hivm.vselrv2"(%[[TRUE]], %[[MSK]]) : (vector<256xi8>, vector<256xi8>) -> vector<256xui8>

// LEGACY-LABEL: func.func @vselrv2_u8_ui8
// LEGACY-NOT: intr.hivm.vselrv2
// LEGACY: ave.hir.vselrv2
func.func @vselrv2_u8_ui8(%mask: vector<256xui8>, %true: vector<256xui8>) -> vector<256xui8> {
  %r = ave.hir.vselrv2 %mask, %true : vector<256xui8>, vector<256xui8>
  return %r : vector<256xui8>
}

// V2-LABEL: func.func @vselrv2_i8
// V2: "hivm_regbaseintrins.intr.hivm.vselrv2"(%arg1, %arg0) : (vector<256xi8>, vector<256xi8>) -> vector<256xi8>

// LEGACY-LABEL: func.func @vselrv2_i8
// LEGACY-NOT: intr.hivm.vselrv2
// LEGACY: ave.hir.vselrv2
func.func @vselrv2_i8(%mask: vector<256xi8>, %true: vector<256xi8>) -> vector<256xi8> {
  %r = ave.hir.vselrv2 %mask, %true : vector<256xi8>, vector<256xi8>
  return %r : vector<256xi8>
}

// An i1 predicate mask keeps the three-input predicate vsel on every target.

// V2-LABEL: func.func @keep_predicate_vsel_i1
// V2: "hivm_regbaseintrins.intr.hivm.vsel"(%arg1, %arg2, %arg0) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi8>

// LEGACY-LABEL: func.func @keep_predicate_vsel_i1
// LEGACY: "hivm_regbaseintrins.intr.hivm.vsel"(%arg1, %arg2, %arg0) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi8>
func.func @keep_predicate_vsel_i1(%pred: vector<256xi1>, %true: vector<256xi8>, %false: vector<256xi8>) -> vector<256xi8> {
  %r = ave.hir.vsel %pred, %true, %false : vector<256xi1>, vector<256xi8>
  return %r : vector<256xi8>
}

// The byte-mask masked move is only expressible through ave.hir.vselrv2; the
// predicate vsel never accepts a data mask, so A5/A3 keep exactly the
// original op set and lowering.
