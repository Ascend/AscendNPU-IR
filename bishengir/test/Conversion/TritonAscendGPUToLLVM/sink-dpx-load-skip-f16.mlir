// RUN: bishengir-opt --sink-dpx-load %s | FileCheck %s
//
// Regression test: the SinkDPXLoad scheduling phase must skip segments that
// contain vector<2xf16> operations (f16 element packing).  The downstream
// backend (hivmc) is sensitive to instruction ordering for these patterns,
// producing incorrect code when the scheduler reorders them.
//
// SROA (Phase 1) should still run — struct bypass is always safe.
// Scheduling (Phase 2) must be skipped for f16-packed segments, so the
// original instruction ordering is preserved.

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @test_f16_no_scheduling
  llvm.func @test_f16_no_scheduling(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %mask: i1) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f16) : f16

    // Input address.
    %in0 = llvm.getelementptr %arg0[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %in1 = llvm.getelementptr %arg0[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // Output address — computed BEFORE loads in the original IR.
    %out0 = llvm.getelementptr %arg1[%c8] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %out1 = llvm.getelementptr %arg1[%c16] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8

    // Default value for masked load (vector<2xf16>).
    %vdef = llvm.mlir.undef : vector<2xf16>
    %vdef0 = llvm.insertelement %zero, %vdef[%c0 : i32] : vector<2xf16>
    %vdef1 = llvm.insertelement %zero, %vdef0[%c1 : i32] : vector<2xf16>

    // Masked loads returning vector<2xf16>.
    %ld0 = ascend_dpx.load %in0, %mask, %vdef1 : (!llvm.ptr<1>, i1, vector<2xf16>) -> vector<2xf16>
    %ld1 = ascend_dpx.load %in1, %mask, %vdef1 : (!llvm.ptr<1>, i1, vector<2xf16>) -> vector<2xf16>

    // Unpack into struct (the dependency bottleneck that SROA eliminates).
    %s = llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16)>
    %e0 = llvm.extractelement %ld0[%c0 : i32] : vector<2xf16>
    %s0 = llvm.insertvalue %e0, %s[0] : !llvm.struct<(f16, f16, f16, f16)>
    %e1 = llvm.extractelement %ld0[%c1 : i32] : vector<2xf16>
    %s1 = llvm.insertvalue %e1, %s0[1] : !llvm.struct<(f16, f16, f16, f16)>
    %e2 = llvm.extractelement %ld1[%c0 : i32] : vector<2xf16>
    %s2 = llvm.insertvalue %e2, %s1[2] : !llvm.struct<(f16, f16, f16, f16)>
    %e3 = llvm.extractelement %ld1[%c1 : i32] : vector<2xf16>
    %s3 = llvm.insertvalue %e3, %s2[3] : !llvm.struct<(f16, f16, f16, f16)>

    // Extract from struct (SROA will bypass these).
    %x0 = llvm.extractvalue %s3[0] : !llvm.struct<(f16, f16, f16, f16)>
    %x1 = llvm.extractvalue %s3[1] : !llvm.struct<(f16, f16, f16, f16)>
    %x2 = llvm.extractvalue %s3[2] : !llvm.struct<(f16, f16, f16, f16)>
    %x3 = llvm.extractvalue %s3[3] : !llvm.struct<(f16, f16, f16, f16)>

    // Compute (atan on each element).
    %r0 = ascend_dpx.atan %x0 : (f16) -> f16
    %r1 = ascend_dpx.atan %x1 : (f16) -> f16
    %r2 = ascend_dpx.atan %x2 : (f16) -> f16
    %r3 = ascend_dpx.atan %x3 : (f16) -> f16

    // Pack results into vector<2xf16> for stores.
    %v0 = llvm.mlir.undef : vector<2xf16>
    %v0a = llvm.insertelement %r0, %v0[%c0 : i32] : vector<2xf16>
    %v0b = llvm.insertelement %r1, %v0a[%c1 : i32] : vector<2xf16>
    %p0 = llvm.bitcast %v0b : vector<2xf16> to i32
    %sv0 = llvm.mlir.undef : vector<1xi32>
    %sv0a = llvm.insertelement %p0, %sv0[%c0 : i32] : vector<1xi32>

    %v1 = llvm.mlir.undef : vector<2xf16>
    %v1a = llvm.insertelement %r2, %v1[%c0 : i32] : vector<2xf16>
    %v1b = llvm.insertelement %r3, %v1a[%c1 : i32] : vector<2xf16>
    %p1 = llvm.bitcast %v1b : vector<2xf16> to i32
    %sv1 = llvm.mlir.undef : vector<1xi32>
    %sv1a = llvm.insertelement %p1, %sv1[%c0 : i32] : vector<1xi32>

    // Stores.
    ascend_dpx.store %out0, %sv0a, %mask : <1>, vector<1xi32>
    ascend_dpx.store %out1, %sv1a, %mask : <1>, vector<1xi32>

    llvm.return
  }
}

// Scheduling must be SKIPPED for this f16-packed segment.  The key check is
// that the original instruction ordering is preserved: output GEPs appear
// BEFORE the loads (their original position), NOT pulled down next to the
// stores.  If scheduling ran, the output GEPs would be moved next to stores.

// Output GEPs appear before loads (original order preserved).
// CHECK:       llvm.getelementptr %arg1[{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK:       llvm.getelementptr %arg1[{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK:       ascend_dpx.load
// CHECK:       ascend_dpx.load
// SROA must still run — atan uses extractelement results directly, not extractvalue.
// CHECK-NOT:   llvm.extractvalue
// All four atans appear grouped together (original order, not interleaved with stores).
// CHECK:       ascend_dpx.atan
// CHECK:       ascend_dpx.atan
// CHECK:       ascend_dpx.atan
// CHECK:       ascend_dpx.atan
// Both stores appear at the end.
// CHECK:       ascend_dpx.store
// CHECK:       ascend_dpx.store
