// RUN: bishengir-opt %s -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s

// Test VaddV128F16XInstrOp
// CHECK-LABEL: test_vadd_v128_f16
func.func @test_vadd_v128_f16(%arg0: vector<128 x f16>, %arg1: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<128 x f16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VaddV128S16XInstrOp
// CHECK-LABEL: test_vadd_v128_s16
func.func @test_vadd_v128_s16(%arg0: vector<128 x si16>, %arg1: vector<128 x si16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xsi16>, vector<128xsi16>, vector<256xi1>) -> vector<128xsi16>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<128 x si16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x si16>) -> ()
  return
}

// -----

// Test VaddV128U16XInstrOp
// CHECK-LABEL: test_vadd_v128_u16
func.func @test_vadd_v128_u16(%arg0: vector<128 x ui16>, %arg1: vector<128 x ui16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xui16>, vector<128xui16>, vector<256xi1>) -> vector<128xui16>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<128 x ui16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x ui16>) -> ()
  return
}

// -----

// Test VaddV256S8XInstrOp
// CHECK-LABEL: test_vadd_v256_s8
func.func @test_vadd_v256_s8(%arg0: vector<256 x si8>, %arg1: vector<256 x si8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<256xsi8>, vector<256xsi8>, vector<256xi1>) -> vector<256xsi8>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<256 x si8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x si8>) -> ()
  return
}

// -----

// Test VaddV256U8XInstrOp
// CHECK-LABEL: test_vadd_v256_u8
func.func @test_vadd_v256_u8(%arg0: vector<256 x ui8>, %arg1: vector<256 x ui8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<256xui8>, vector<256xui8>, vector<256xi1>) -> vector<256xui8>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<256 x ui8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x ui8>) -> ()
  return
}

// -----

// Test VaddV64F32XInstrOp
// CHECK-LABEL: test_vadd_v64_f32
func.func @test_vadd_v64_f32(%arg0: vector<64 x f32>, %arg1: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<64 x f32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test VaddV64S32XInstrOp
// CHECK-LABEL: test_vadd_v64_s32
func.func @test_vadd_v64_s32(%arg0: vector<64 x si32>, %arg1: vector<64 x si32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xsi32>, vector<64xsi32>, vector<256xi1>) -> vector<64xsi32>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<64 x si32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VaddV64U32XInstrOp
// CHECK-LABEL: test_vadd_v64_u32
func.func @test_vadd_v64_u32(%arg0: vector<64 x ui32>, %arg1: vector<64 x ui32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xui32>, vector<64xui32>, vector<256xi1>) -> vector<64xui32>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<64 x ui32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x ui32>) -> ()
  return
}

// -----

// Test VmulV128F16XInstrOp
// CHECK-LABEL: test_vmul_v128_f16
func.func @test_vmul_v128_f16(%arg0: vector<128 x f16>, %arg1: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<128 x f16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VmulV128S16XInstrOp
// CHECK-LABEL: test_vmul_v128_s16
func.func @test_vmul_v128_s16(%arg0: vector<128 x si16>, %arg1: vector<128 x si16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xsi16>, vector<128xsi16>, vector<256xi1>) -> vector<128xsi16>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<128 x si16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x si16>) -> ()
  return
}

// -----

// Test VmulV128U16XInstrOp
// CHECK-LABEL: test_vmul_v128_u16
func.func @test_vmul_v128_u16(%arg0: vector<128 x ui16>, %arg1: vector<128 x ui16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xui16>, vector<128xui16>, vector<256xi1>) -> vector<128xui16>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<128 x ui16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x ui16>) -> ()
  return
}

// -----

// Test VmulV256S8XInstrOp
// CHECK-LABEL: test_vmul_v256_s8
func.func @test_vmul_v256_s8(%arg0: vector<256 x si8>, %arg1: vector<256 x si8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<256xsi8>, vector<256xsi8>, vector<256xi1>) -> vector<256xsi8>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<256 x si8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x si8>) -> ()
  return
}

// -----

// Test VmulV256U8XInstrOp
// CHECK-LABEL: test_vmul_v256_u8
func.func @test_vmul_v256_u8(%arg0: vector<256 x ui8>, %arg1: vector<256 x ui8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<256xui8>, vector<256xui8>, vector<256xi1>) -> vector<256xui8>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<256 x ui8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x ui8>) -> ()
  return
}

// -----

// Test VmulV64F32XInstrOp
// CHECK-LABEL: test_vmul_v64_f32
func.func @test_vmul_v64_f32(%arg0: vector<64 x f32>, %arg1: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<64 x f32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test VmulV64S32XInstrOp
// CHECK-LABEL: test_vmul_v64_s32
func.func @test_vmul_v64_s32(%arg0: vector<64 x si32>, %arg1: vector<64 x si32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xsi32>, vector<64xsi32>, vector<256xi1>) -> vector<64xsi32>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<64 x si32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VmulV64U32XInstrOp
// CHECK-LABEL: test_vmul_v64_u32
func.func @test_vmul_v64_u32(%arg0: vector<64 x ui32>, %arg1: vector<64 x ui32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xui32>, vector<64xui32>, vector<256xi1>) -> vector<64xui32>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<64 x ui32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x ui32>) -> ()
  return
}

// -----

// Test VmulV128F16XInstrOp
// CHECK-LABEL: test_vsub_v128_f16
func.func @test_vsub_v128_f16(%arg0: vector<128 x f16>, %arg1: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<128 x f16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VsubV128S16XInstrOp
// CHECK-LABEL: test_vsub_v128_s16
func.func @test_vsub_v128_s16(%arg0: vector<128 x si16>, %arg1: vector<128 x si16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xsi16>, vector<128xsi16>, vector<256xi1>) -> vector<128xsi16>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<128 x si16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x si16>) -> ()
  return
}

// -----

// Test VsubV128U16XInstrOp
// CHECK-LABEL: test_vsub_v128_u16
func.func @test_vsub_v128_u16(%arg0: vector<128 x ui16>, %arg1: vector<128 x ui16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xui16>, vector<128xui16>, vector<256xi1>) -> vector<128xui16>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<128 x ui16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x ui16>) -> ()
  return
}

// -----

// Test VsubV256S8XInstrOp
// CHECK-LABEL: test_vsub_v256_s8
func.func @test_vsub_v256_s8(%arg0: vector<256 x si8>, %arg1: vector<256 x si8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<256xsi8>, vector<256xsi8>, vector<256xi1>) -> vector<256xsi8>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<256 x si8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x si8>) -> ()
  return
}

// -----

// Test VsubV256U8XInstrOp
// CHECK-LABEL: test_vsub_v256_u8
func.func @test_vsub_v256_u8(%arg0: vector<256 x ui8>, %arg1: vector<256 x ui8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<256xui8>, vector<256xui8>, vector<256xi1>) -> vector<256xui8>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<256 x ui8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x ui8>) -> ()
  return
}

// -----

// Test VsubV64F32XInstrOp
// CHECK-LABEL: test_vsub_v64_f32
func.func @test_vsub_v64_f32(%arg0: vector<64 x f32>, %arg1: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<64 x f32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test VsubV64S32XInstrOp
// CHECK-LABEL: test_vsub_v64_s32
func.func @test_vsub_v64_s32(%arg0: vector<64 x si32>, %arg1: vector<64 x si32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xsi32>, vector<64xsi32>, vector<256xi1>) -> vector<64xsi32>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<64 x si32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VmulV64U32XInstrOp
// CHECK-LABEL: test_vsub_v64_u32
func.func @test_vsub_v64_u32(%arg0: vector<64 x ui32>, %arg1: vector<64 x ui32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xui32>, vector<64xui32>, vector<256xi1>) -> vector<64xui32>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<64 x ui32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x ui32>) -> ()
  return
}

// -----

// Test VreluXInstrOp
// CHECK-LABEL: test_vrelu_v128_f16
func.func @test_vrelu_v128_f16(%arg0: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vrelu.x"([[V0:%.*]], [[V1:%.*]]) : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vrelu %arg0, %mask : vector<128 x f16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VreluXInstrOp
// CHECK-LABEL: test_vrelu_v64_f32
func.func @test_vrelu_v64_f32(%arg0: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vrelu.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vrelu %arg0, %mask : vector<64 x f32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test VreluXInstrOp
// CHECK-LABEL: test_vrelu_v128_s32
func.func @test_vrelu_v128_s32(%arg0: vector<64 x si32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vrelu.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xsi32>, vector<256xi1>) -> vector<64xsi32>
  %0 = ave.hir.vrelu %arg0, %mask : vector<64 x si32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VabsXInstrOp
// CHECK-LABEL: test_vabs_v256_s8
func.func @test_vabs_v256_s8(%arg0: vector<256 x si8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vabs.x"([[V0:%.*]], [[V1:%.*]]) : (vector<256xsi8>, vector<256xi1>) -> vector<256xsi8>
  %0 = ave.hir.vabs %arg0, %mask : vector<256 x si8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x si8>) -> ()
  return
}

// -----

// Test VabsXInstrOp
// CHECK-LABEL: test_vabs_v128_s16
func.func @test_vabs_v128_s16(%arg0: vector<128 x si16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vabs.x"([[V0:%.*]], [[V1:%.*]]) : (vector<128xsi16>, vector<256xi1>) -> vector<128xsi16>
  %0 = ave.hir.vabs %arg0, %mask : vector<128 x si16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x si16>) -> ()
  return
}

// -----

// Test VabsXInstrOp
// CHECK-LABEL: test_vabs_v128_f16
func.func @test_vabs_v128_f16(%arg0: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vabs.x"([[V0:%.*]], [[V1:%.*]]) : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vabs %arg0, %mask : vector<128 x f16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VabsXInstrOp
// CHECK-LABEL: test_vabs_v64_s32
func.func @test_vabs_v64_s32(%arg0: vector<64 x si32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vabs.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xsi32>, vector<256xi1>) -> vector<64xsi32>
  %0 = ave.hir.vabs %arg0, %mask : vector<64 x si32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VabsXInstrOp
// CHECK-LABEL: test_vabs_v64_f32
func.func @test_vabs_v64_f32(%arg0: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vabs.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vabs %arg0, %mask : vector<64 x f32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test VNegXInstrOp
// CHECK-LABEL: test_vneg_v256_s8
func.func @test_vneg_v256_s8(%arg0: vector<256 x si8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vneg.x"([[V0:%.*]], [[V1:%.*]]) : (vector<256xsi8>, vector<256xi1>) -> vector<256xsi8>
  %0 = ave.hir.vneg %arg0, %mask : vector<256 x si8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x si8>) -> ()
  return
}

// -----


// Test VNegXInstrOp
// CHECK-LABEL: test_vneg_v128_s16
func.func @test_vneg_v128_s16(%arg0: vector<128 x si16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vneg.x"([[V0:%.*]], [[V1:%.*]]) : (vector<128xsi16>, vector<256xi1>) -> vector<128xsi16>
  %0 = ave.hir.vneg %arg0, %mask : vector<128 x si16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x si16>) -> ()
  return
}

// -----

// Test VNegXInstrOp
// CHECK-LABEL: test_vneg_v128_f16
func.func @test_vneg_v128_f16(%arg0: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vneg.x"([[V0:%.*]], [[V1:%.*]]) : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vneg %arg0, %mask : vector<128 x f16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VNegXInstrOp
// CHECK-LABEL: test_vneg_v64_s32
func.func @test_vneg_v64_s32(%arg0: vector<64 x si32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vneg.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xsi32>, vector<256xi1>) -> vector<64xsi32>
  %0 = ave.hir.vneg %arg0, %mask : vector<64 x si32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VNegXInstrOp
// CHECK-LABEL: test_vneg_v64_f32
func.func @test_vneg_v64_f32(%arg0: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vneg.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vneg %arg0, %mask : vector<64 x f32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test VNotXInstrOp
// CHECK-LABEL: test_vnot_v256_s8
func.func @test_vnot_v256_s8(%arg0: vector<256 x si8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vnot.x"([[V0:%.*]], [[V1:%.*]]) : (vector<256xsi8>, vector<256xi1>) -> vector<256xsi8>
  %0 = ave.hir.vnot %arg0, %mask : vector<256 x si8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x si8>) -> ()
  return
}

// -----

// Test VNotXInstrOp
// CHECK-LABEL: test_vnot_v128_s16
func.func @test_vnot_v128_s16(%arg0: vector<128 x si16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vnot.x"([[V0:%.*]], [[V1:%.*]]) : (vector<128xsi16>, vector<256xi1>) -> vector<128xsi16>
  %0 = ave.hir.vnot %arg0, %mask : vector<128 x si16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x si16>) -> ()
  return
}

// -----

// Test VNotXInstrOp
// CHECK-LABEL: test_vnot_v128_f16
func.func @test_vnot_v128_f16(%arg0: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vnot.x"([[V0:%.*]], [[V1:%.*]]) : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vnot %arg0, %mask : vector<128 x f16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VNotXInstrOp
// CHECK-LABEL: test_vnot_v64_s32
func.func @test_vnot_v64_s32(%arg0: vector<64 x si32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vnot.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xsi32>, vector<256xi1>) -> vector<64xsi32>
  %0 = ave.hir.vnot %arg0, %mask : vector<64 x si32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VNotXInstrOp
// CHECK-LABEL: test_vnot_v64_f32
func.func @test_vnot_v64_f32(%arg0: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vnot.x"([[V0:%.*]], [[V1:%.*]]) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vnot %arg0, %mask : vector<64 x f32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test Vldsx1V64F32InstrOp
// CHECK-LABEL: test_load_v64_f32
func.func @test_load_v64_f32(%arg0:memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
  "test.test"(%1) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test Vldsx1V32U64InstrOp
// CHECK-LABEL: test_load_v32_u64
func.func @test_load_v32_u64(%arg0:memref<32xui64, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v32u64"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<32xi64>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<32xui64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<32xi64>
  "test.test"(%1) : (vector<32 x i64>) -> ()
  return
}

// -----

// Test Vldsx1V32S64InstrOp
// CHECK-LABEL: test_load_v32_s64
func.func @test_load_v32_s64(%arg0:memref<32xsi64, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v32s64"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<32xi64>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<32xsi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<32xi64>
  "test.test"(%1) : (vector<32 x i64>) -> ()
  return
}

// -----

// Test Vldsx1V64S32InstrOp
// CHECK-LABEL: test_load_v64_s32
func.func @test_load_v64_s32(%arg0:memref<64xsi32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xi32>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<64xsi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
  "test.test"(%1) : (vector<64 x i32>) -> ()
  return
}

// -----

// Test Vldsx1V64U32InstrOp
// CHECK-LABEL: test_load_v64_u32
func.func @test_load_v64_u32(%arg0:memref<64xui32, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v64u32"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xi32>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<64xui32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
  "test.test"(%1) : (vector<64 x i32>) -> ()
  return
}

// -----

// Test Vldsx1V128S16InstrOp
// CHECK-LABEL: test_load_v128_s16
func.func @test_load_v128_s16(%arg0:memref<128xsi16, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v128s16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xi16>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<128xsi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xi16>
  "test.test"(%1) : (vector<128 x i16>) -> ()
  return
}

// -----

// Test Vldsx1V128U16InstrOp
// CHECK-LABEL: test_load_v128_u16
func.func @test_load_v128_u16(%arg0:memref<128xui16, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v128u16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xi16>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<128xui16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xi16>
  "test.test"(%1) : (vector<128 x i16>) -> ()
  return
}

// -----

// Test Vldsx1V256S8InstrOp
// CHECK-LABEL: test_load_v256_s8
func.func @test_load_v256_s8(%arg0:memref<256xsi8, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<256xsi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
  "test.test"(%1) : (vector<256 x i8>) -> ()
  return
}

// -----

// Test Vldsx1V256U8InstrOp
// CHECK-LABEL: test_load_v256_u8
func.func @test_load_v256_u8(%arg0:memref<256xui8, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256u8"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<256xui8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
  "test.test"(%1) : (vector<256 x i8>) -> ()
  return
}

// -----

// Test Vldsx1V128F16InstrOp
// CHECK-LABEL: test_load_v128_f16
func.func @test_load_v128_f16(%arg0:memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xf16>
  "test.test"(%1) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test Vldsx1V128BF16InstrOp
// CHECK-LABEL: test_load_v128_bf16
func.func @test_load_v128_bf16(%arg0:memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
  %1 = ave.hir.vload <NORM> %arg0[%0] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
  "test.test"(%1) : (vector<128 x bf16>) -> ()
  return
}

// -----

// Test Vstsx1V64F32InstrOp
// CHECK-LABEL: test_store_v64_f32
func.func @test_store_v64_f32(%arg0:memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, %res: vector<64xf32>, %mask:vector<256xi1>) -> () {
  %0 = arith.constant 0 : index
  // CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]], [[V3:%.*]], [[V4:%.*]], [[V5:%.*]]) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
  ave.hir.masked_store <NORM_B32> %arg0[%0], %mask, %res: memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<64xf32>
  return
}

// -----

// Test VGatherV64S32InstrOp
// CHECK-LABEL: test_gather_v64_s32
func.func @test_gather_v64_s32(%arg0 : memref<64x1xsi32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<64xi32>, %arg3 : vector<64xi1>) -> () {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v64s32"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<64x1xsi32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xsi32>
  "test.test"(%0) : (vector<64 x si32>) -> ()
  return
}

// -----

// Test VGatherV64U32InstrOp
// CHECK-LABEL: test_gather_v64_u32
func.func @test_gather_v64_u32(%arg0 : memref<64x1xui32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<64xi32>, %arg3 : vector<64xi1>) -> () {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v64u32"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<64x1xui32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xui32>
  "test.test"(%0) : (vector<64 x ui32>) -> ()
  return
}

// -----

// Test VGatherV128S16InstrOp
// CHECK-LABEL: test_gather_v128_s16
func.func @test_gather_v128_s16(%arg0 : memref<128x1xsi16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<128xi16>, %arg3 : vector<128xi1>) -> () {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128s16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xi16>
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<128x1xsi16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xi16>
  "test.test"(%0) : (vector<128 x i16>) -> ()
  return
}

// -----

// Test VGatherV128U16InstrOp
// CHECK-LABEL: test_gather_v128_u16
func.func @test_gather_v128_u16(%arg0 : memref<128x1xui16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<128xi16>, %arg3 : vector<128xi1>) -> () {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128u16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xi16>
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<128x1xui16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xi16>
  "test.test"(%0) : (vector<128 x i16>) -> ()
  return
}

// -----

// Test VGatherV256S8InstrOp
// CHECK-LABEL:     func.func @test_gather_v256_s8
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256s8"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xi16>
// CHECK: llvm.bitcast {{.*}} : vector<128xi16> to vector<256xi8>
func.func @test_gather_v256_s8(%arg0 : memref<256x1xsi8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<256xi16>, %arg3 : vector<256xi1>) -> () {
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<256x1xsi8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<256xi16>, vector<256xi1> into vector<256xi8>
  "test.test"(%0) : (vector<256 x i8>) -> ()
  return
}

// -----

// Test VGatherV256U8InstrOp
// CHECK-LABEL:   func.func @test_gather_v256_u8
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256u8"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xi16>
// CHECK: llvm.bitcast {{.*}} : vector<128xi16> to vector<256xi8>
func.func @test_gather_v256_u8(%arg0 : memref<256x1xui8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<256xi16>, %arg3 : vector<256xi1>) -> () {
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<256x1xui8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<256xi16>, vector<256xi1> into vector<256xi8>
  "test.test"(%0) : (vector<256 x i8>) -> ()
  return
}

// -----

// Test VGatherV64F32InstrOp
// CHECK-LABEL: test_gather_v64_f32
func.func @test_gather_v64_f32(%arg0 : memref<64x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<64xi32>, %arg3 : vector<64xi1>) -> () {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v64f32"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<64xf32>
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<64x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xf32>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  return
}

// -----

// Test VGatherV128F16InstrOp
// CHECK-LABEL: test_gather_v128_f16
func.func @test_gather_v128_f16(%arg0 : memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<128xi16>, %arg3 : vector<128xi1>) -> () {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128f16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xf16>
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xf16>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  return
}

// -----

// Test VGatherV128BF16InstrOp
// CHECK-LABEL: test_gather_v128_bf16
func.func @test_gather_v128_bf16(%arg0 : memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<128xi16>, %arg3 : vector<128xi1>) -> () {
  // CHECK: %[[RES:.*]] = "hivm_regbaseintrins.intr.hivm.vgather2_v300.v128bf16"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<128x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xbf16>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// -----

// Test VGatherV256F8E5InstrOp
// CHECK-LABEL: func.func @test_gather_v256_f8e5m2
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256f8e5m2"({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<6>, vector<64xi32>, vector<256xi1>) -> vector<128xi16>
func.func @test_gather_v256_f8e5m2(%arg0 : memref<256x1xf8E5M2, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, %arg1 : index, %arg2 : vector<256xi16>, %arg3 : vector<256xi1>) -> () {
  %0 = ave.hir.vgather %arg0[%arg1, %arg1] [%arg2], %arg3 : memref<256x1xf8E5M2, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<256xi16>, vector<256xi1> into vector<256xf8E5M2>
  "test.test"(%0) : (vector<256 x f8E5M2>) -> ()
  return
}

// CHECK-LABEL: test_vsstb_v128_f16
func.func @test_vsstb_v128_f16(%arg0 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>, %arg1 : vector<64xi1>, %arg2 : vector<64xf16>) -> () {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[STRIDECONFIG:.*]] = llvm.mlir.constant(4194304 : i32) : i32
  // CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128f16"([[V0:%.*]], [[V1:%.*]], %[[STRIDECONFIG:.*]], [[V3:%.*]], [[V4:%.*]]) : (vector<128xf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>)
  ave.hir.store_with_stride %arg0[%c0, %c0], %c1024, %arg1, %arg2 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  return
}

// CHECK-LABEL: test_vsstb_v128_bf16
func.func @test_vsstb_v128_bf16(%arg0 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>, %arg1 : vector<64xi1>, %arg2 : vector<64xbf16>) -> () {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[STRIDECONFIG:.*]] = llvm.mlir.constant(4194304 : i32) : i32
  // CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"([[V0:%.*]], [[V1:%.*]], %[[STRIDECONFIG:.*]], [[V3:%.*]], [[V4:%.*]]) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, vector<256xi1>)
  ave.hir.store_with_stride %arg0[%c0, %c0], %c1024, %arg1, %arg2 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
  return
}

// CHECK-LABEL: test_vsstb_v64_f32
func.func @test_vsstb_v64_f32(%arg0 : memref<8x8xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 + s0)>, #hivm.address_space<ub>>, %arg1 : vector<64xi1>, %arg2 : vector<64xf32>) -> () {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[STRIDECONFIG:.*]] = llvm.mlir.constant(4194304 : i32) : i32
  // CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v64f32"([[V0:%.*]], [[V1:%.*]], %[[STRIDECONFIG:.*]], [[V3:%.*]], [[V4:%.*]]) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, vector<256xi1>) -> ()
  ave.hir.store_with_stride %arg0[%c0, %c0], %c512, %arg1, %arg2 : memref<8x8xf32, affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_vshls$i8$subvl
// CHECK-NOT:     ave.hir.vshls
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i8) : i8
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_4:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b16"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 16 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_5:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b8"(%[[VAL_5]], %[[VAL_6]]) {mask_bit_width = 8 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
// CHECK:         %[[VAL_9:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_10:.*]] = "hivm_regbaseintrins.intr.hivm.vshls.u.x"(%[[VAL_8]], %[[VAL_9]], %[[VAL_7]]) : (vector<256xi8>, i16, vector<256xi1>) -> vector<256xi8>
// CHECK:         %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_10]] : vector<256xi8> to vector<128xi8>
// CHECK:         return %[[VAL_11]] : vector<128xi8>
func.func @test_vshls$i8$subvl(%arg0: vector<128xi8>) -> vector<128xi8> {
  %false = arith.constant false
  %c7_i8 = arith.constant 7 : i8
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<128xi1>
  %1 = ave.hir.vshls %arg0, %c7_i8, %0, %false : vector<128xi8>, i8, vector<128xi1>
  return %1 : vector<128xi8>
}

// -----

// CHECK-LABEL: func.func @test_vshrs$i8$subvl
// CHECK-NOT:     ave.hir.vshrs
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i8) : i8
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_4:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b16"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 16 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_5:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b8"(%[[VAL_5]], %[[VAL_6]]) {mask_bit_width = 8 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
// CHECK:         %[[VAL_9:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_10:.*]] = "hivm_regbaseintrins.intr.hivm.vshrs.u.x"(%[[VAL_8]], %[[VAL_9]], %[[VAL_7]]) : (vector<256xi8>, i16, vector<256xi1>) -> vector<256xi8>
// CHECK:         %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_10]] : vector<256xi8> to vector<128xi8>
// CHECK:         return %[[VAL_11]] : vector<128xi8>
func.func @test_vshrs$i8$subvl(%arg0: vector<128xi8>) -> vector<128xi8> {
  %false = arith.constant false
  %c7_i8 = arith.constant 7 : i8
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<128xi1>
  %1 = ave.hir.vshrs %arg0, %c7_i8, %0, %false : vector<128xi8>, i8, vector<128xi1>
  return %1 : vector<128xi8>
}

// -----

// CHECK-LABEL: func.func @test_vshls$i16$subvl
// CHECK-NOT:     ave.hir.vshls
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_4:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_5:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b16"(%[[VAL_5]], %[[VAL_6]]) {mask_bit_width = 16 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
// CHECK:         %[[VAL_9:.*]] = "hivm_regbaseintrins.intr.hivm.vshls.u.x"(%[[VAL_8]], %[[VAL_1]], %[[VAL_7]]) : (vector<128xi16>, i16, vector<256xi1>) -> vector<128xi16>
// CHECK:         %[[VAL_10:.*]] = builtin.unrealized_conversion_cast %[[VAL_9]] : vector<128xi16> to vector<64xi16>
// CHECK:         return %[[VAL_10]] : vector<64xi16>
func.func @test_vshls$i16$subvl(%arg0: vector<64xi16>) -> vector<64xi16> {
  %false = arith.constant false
  %c7_i16 = arith.constant 7 : i16
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<64xi1>
  %1 = ave.hir.vshls %arg0, %c7_i16, %0, %false : vector<64xi16>, i16, vector<64xi1>
  return %1 : vector<64xi16>
}

// -----

// CHECK-LABEL: func.func @test_vshrs$i16$subvl
// CHECK-NOT:     ave.hir.vshrs
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_4:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_5:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b16"(%[[VAL_5]], %[[VAL_6]]) {mask_bit_width = 16 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
// CHECK:         %[[VAL_9:.*]] = "hivm_regbaseintrins.intr.hivm.vshrs.u.x"(%[[VAL_8]], %[[VAL_1]], %[[VAL_7]]) : (vector<128xi16>, i16, vector<256xi1>) -> vector<128xi16>
// CHECK:         %[[VAL_10:.*]] = builtin.unrealized_conversion_cast %[[VAL_9]] : vector<128xi16> to vector<64xi16>
// CHECK:         return %[[VAL_10]] : vector<64xi16>
func.func @test_vshrs$i16$subvl(%arg0: vector<64xi16>) -> vector<64xi16> {
  %false = arith.constant false
  %c7_i16 = arith.constant 7 : i16
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<64xi1>
  %1 = ave.hir.vshrs %arg0, %c7_i16, %0, %false : vector<64xi16>, i16, vector<64xi1>
  return %1 : vector<64xi16>
}

// -----

// CHECK-LABEL: func.func @test_vshls$i32$subvl
// CHECK-NOT:     ave.hir.vshls
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_4:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<32xi32> to vector<64xi32>
// CHECK:         %[[VAL_6:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.vshls.u.x"(%[[VAL_5]], %[[VAL_6]], %[[VAL_4]]) : (vector<64xi32>, i16, vector<256xi1>) -> vector<64xi32>
// CHECK:         %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %[[VAL_7]] : vector<64xi32> to vector<32xi32>
// CHECK:         return %[[VAL_8]] : vector<32xi32>
func.func @test_vshls$i32$subvl(%arg0: vector<32xi32>) -> vector<32xi32> {
  %false = arith.constant false
  %c7_i32 = arith.constant 7 : i32
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<32xi1>
  %1 = ave.hir.vshls %arg0, %c7_i32, %0, %false : vector<32xi32>, i32, vector<32xi1>
  return %1 : vector<32xi32>
}

// -----

// CHECK-LABEL: func.func @test_vshrs$i32$subvl
// CHECK-NOT:     ave.hir.vshrs
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_4:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<32xi32> to vector<64xi32>
// CHECK:         %[[VAL_6:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.vshrs.u.x"(%[[VAL_5]], %[[VAL_6]], %[[VAL_4]]) : (vector<64xi32>, i16, vector<256xi1>) -> vector<64xi32>
// CHECK:         %[[VAL_8:.*]] = builtin.unrealized_conversion_cast %[[VAL_7]] : vector<64xi32> to vector<32xi32>
// CHECK:         return %[[VAL_8]] : vector<32xi32>
func.func @test_vshrs$i32$subvl(%arg0: vector<32xi32>) -> vector<32xi32> {
  %false = arith.constant false
  %c7_i32 = arith.constant 7 : i32
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<32xi1>
  %1 = ave.hir.vshrs %arg0, %c7_i32, %0, %false : vector<32xi32>, i32, vector<32xi1>
  return %1 : vector<32xi32>
}

// -----

// CHECK:         func.func @test_vshls$i8$eqvl(%[[ARG_0:.*]]: vector<256xi8>) -> vector<256xi8> {
// CHECK-NOT:     ave.hir.vshls
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i8) : i8
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b8"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 8 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_8:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_9:.*]] = "hivm_regbaseintrins.intr.hivm.vshls.u.x"(%[[ARG_0]], %[[VAL_8]], %[[VAL_7]]) : (vector<256xi8>, i16, vector<256xi1>) -> vector<256xi8>
// CHECK:         return %[[VAL_9]] : vector<256xi8>
func.func @test_vshls$i8$eqvl(%arg0: vector<256xi8>) -> vector<256xi8> {
  %false = arith.constant false
  %c7_i8 = arith.constant 7 : i8
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<256xi1>
  %1 = ave.hir.vshls %arg0, %c7_i8, %0, %false : vector<256xi8>, i8, vector<256xi1>
  return %1 : vector<256xi8>
}

// -----

// CHECK:         func.func @test_vshls$i32$eqvl(%[[ARG_0:.*]]: vector<64xi32>) -> vector<64xi32> {
// CHECK-NOT:     ave.hir.vshls
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_2:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_7:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b8"(%[[VAL_2]], %[[VAL_3]]) {mask_bit_width = 8 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
// CHECK:         %[[VAL_8:.*]] = llvm.mlir.constant(7 : i16) : i16
// CHECK:         %[[VAL_9:.*]] = "hivm_regbaseintrins.intr.hivm.vshls.u.x"(%[[ARG_0]], %[[VAL_8]], %[[VAL_7]]) : (vector<64xi32>, i16, vector<256xi1>) -> vector<64xi32>
// CHECK:         return %[[VAL_9]] : vector<64xi32>
func.func @test_vshls$i32$eqvl(%arg0: vector<64xi32>) -> vector<64xi32> {
  %false = arith.constant false
  %c7_i32 = arith.constant 7 : i32
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<256xi1>
  %1 = ave.hir.vshls %arg0, %c7_i32, %0, %false : vector<64xi32>, i32, vector<256xi1>
  return %1 : vector<64xi32>
}
