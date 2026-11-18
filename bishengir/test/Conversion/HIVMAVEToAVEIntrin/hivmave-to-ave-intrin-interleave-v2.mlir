// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend960 \
// RUN:   -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s \
// RUN:   --check-prefix=V2
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN:   -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s \
// RUN:   --check-prefix=LEGACY

// UC-A6-04: Ascend960 (V121) lowers dual-output interleave/de-interleave to the
// v2 single-output part-select form vintlvv2/vdintlvv2 with part=0 (LOWER) and
// part=1 (HIGHER); the old dual-output vintlv/vdintlv form is no longer emitted.
// The A5/A3 (910B) legacy path stays on vintlv/vdintlv.

// V2-LABEL: func.func @test_vintlv
// V2: %[[A0:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<256xi8> to vector<256xi8>
// V2: %[[A1:.*]] = builtin.unrealized_conversion_cast %arg1 : vector<256xi8> to vector<256xi8>
// V2: %[[P0:.*]] = llvm.mlir.constant(0 : i32) : i32
// V2: %[[P1:.*]] = llvm.mlir.constant(1 : i32) : i32
// V2-NOT: intr.hivm.vintlv"
// V2: %[[LO:.*]] = "hivm_regbaseintrins.intr.hivm.vintlvv2"(%[[A0]], %[[A1]], %[[P0]]) : (vector<256xi8>, vector<256xi8>, i32) -> vector<256xi8>
// V2: %[[HI:.*]] = "hivm_regbaseintrins.intr.hivm.vintlvv2"(%[[A0]], %[[A1]], %[[P1]]) : (vector<256xi8>, vector<256xi8>, i32) -> vector<256xi8>

// LEGACY-LABEL: func.func @test_vintlv
// LEGACY: %[[S:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%{{.*}}, %{{.*}}) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// LEGACY: llvm.extractvalue %[[S]][0]
// LEGACY: llvm.extractvalue %[[S]][1]

func.func @test_vintlv(%arg0: vector<256xi8>, %arg1: vector<256xi8>) -> (vector<256xi8>, vector<256xi8>) {
  %r0, %r1 = ave.hir.vintlv %arg0, %arg1 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
  return %r0, %r1 : vector<256xi8>, vector<256xi8>
}

// V2-LABEL: func.func @test_vdintlv
// V2: %[[A0:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<256xi8> to vector<256xi8>
// V2: %[[A1:.*]] = builtin.unrealized_conversion_cast %arg1 : vector<256xi8> to vector<256xi8>
// V2: %[[P0:.*]] = llvm.mlir.constant(0 : i32) : i32
// V2: %[[P1:.*]] = llvm.mlir.constant(1 : i32) : i32
// V2-NOT: intr.hivm.vdintlv"
// V2: %[[LO:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlvv2"(%[[A0]], %[[A1]], %[[P0]]) : (vector<256xi8>, vector<256xi8>, i32) -> vector<256xi8>
// V2: %[[HI:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlvv2"(%[[A0]], %[[A1]], %[[P1]]) : (vector<256xi8>, vector<256xi8>, i32) -> vector<256xi8>

// LEGACY-LABEL: func.func @test_vdintlv
// LEGACY: %[[S:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%{{.*}}, %{{.*}}) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// LEGACY: llvm.extractvalue %[[S]][0]
// LEGACY: llvm.extractvalue %[[S]][1]

func.func @test_vdintlv(%arg0: vector<256xi8>, %arg1: vector<256xi8>) -> (vector<256xi8>, vector<256xi8>) {
  %r0, %r1 = ave.hir.vdintlv %arg0, %arg1 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
  return %r0, %r1 : vector<256xi8>, vector<256xi8>
}

// The v2 path also covers the full data-movement type set; f32 is shown as the
// representative wide type.

// V2-LABEL: func.func @test_vintlv_f32
// V2: %[[F0:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xf32> to vector<64xf32>
// V2: %[[F1:.*]] = builtin.unrealized_conversion_cast %arg1 : vector<64xf32> to vector<64xf32>
// V2: %[[FP0:.*]] = llvm.mlir.constant(0 : i32) : i32
// V2: %[[FP1:.*]] = llvm.mlir.constant(1 : i32) : i32
// V2: "hivm_regbaseintrins.intr.hivm.vintlvv2"(%[[F0]], %[[F1]], %[[FP0]]) : (vector<64xf32>, vector<64xf32>, i32) -> vector<64xf32>
// V2: "hivm_regbaseintrins.intr.hivm.vintlvv2"(%[[F0]], %[[F1]], %[[FP1]]) : (vector<64xf32>, vector<64xf32>, i32) -> vector<64xf32>

// LEGACY-LABEL: func.func @test_vintlv_f32
// LEGACY: "hivm_regbaseintrins.intr.hivm.vintlv"(%{{.*}}, %{{.*}}) : (vector<64xf32>, vector<64xf32>) -> !llvm.struct<(vector<64xf32>, vector<64xf32>)>

func.func @test_vintlv_f32(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> (vector<64xf32>, vector<64xf32>) {
  %r0, %r1 = ave.hir.vintlv %arg0, %arg1 : vector<64xf32>, vector<64xf32>
  return %r0, %r1 : vector<64xf32>, vector<64xf32>
}

// VSELR b8: u8/s8 mask drives u8/s8 data via vsel, available on both the V121
// and legacy paths.

// V2-LABEL: func.func @test_vsel_u8
// V2: "hivm_regbaseintrins.intr.hivm.vsel"(%arg1, %arg2, %arg0) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi8>

// LEGACY-LABEL: func.func @test_vsel_u8
// LEGACY: "hivm_regbaseintrins.intr.hivm.vsel"(%arg1, %arg2, %arg0) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi8>

func.func @test_vsel_u8(%arg0: vector<256xi1>, %arg1: vector<256xi8>, %arg2: vector<256xi8>) -> vector<256xi8> {
  %r = ave.hir.vsel %arg0, %arg1, %arg2 : vector<256xi1>, vector<256xi8>
  return %r : vector<256xi8>
}
