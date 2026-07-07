// RUN: bishengir-opt %s -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s

// Test VaddV128BF16XInstrOp
// CHECK-LABEL: test_vadd_v128_bf16
func.func @test_vadd_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VmulV128BF16XInstrOp
// CHECK-LABEL: test_vmul_v128_bf16
func.func @test_vmul_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmul.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vmul %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VsubV128BF16XInstrOp
// CHECK-LABEL: test_vsub_v128_bf16
func.func @test_vsub_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vsub.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vsub %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VmaxV128BF16XInstrOp
// CHECK-LABEL: test_vmax_v128_bf16
func.func @test_vmax_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmax.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vmax %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VminV128BF16XInstrOp
// CHECK-LABEL: test_vmin_v128_bf16
func.func @test_vmin_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmin.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vmin %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VorXInstrOp
// CHECK-LABEL: test_vor_v128_bf16
func.func @test_vor_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vor.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vor %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VandXInstrOp
// CHECK-LABEL: test_vand_v128_bf16
func.func @test_vand_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vand.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vand %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VxorXInstrOp
// CHECK-LABEL: test_vxor_v128_bf16
func.func @test_vxor_v128_bf16(%arg0: vector<128 x bf16>, %arg1: vector<128 x bf16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vxor.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xbf16>, vector<128xbf16>, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vxor %arg0, %arg1, %mask : vector<128 x bf16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x bf16>) -> ()
  return
}

// Test VaddV128S16XInstrOp
// CHECK-LABEL: test_vadd_v128_i16
func.func @test_vadd_v128_i16(%arg0: vector<128 x i16>, %arg1: vector<128 x i16>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<128xi16>, vector<128xi16>, vector<256xi1>) -> vector<128xi16>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<128 x i16>, vector<256 x i1>
  "test.test"(%0) : (vector<128 x i16>) -> ()
  return
}

// Test VaddV256S8XInstrOp 
// CHECK-LABEL: test_vadd_v256_i8
func.func @test_vadd_v256_i8(%arg0: vector<256 x i8>, %arg1: vector<256 x i8>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<256xi8>, vector<256xi8>, vector<256xi1>) -> vector<256xi8>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<256 x i8>, vector<256 x i1>
  "test.test"(%0) : (vector<256 x i8>) -> ()
  return
}

// Test VmaxV64U32XInstrOp
// CHECK-LABEL: test_vumax_v64_i32
func.func @test_vumax_v64_i32(%arg0: vector<64 x i32>, %arg1: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmax.u.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
  %0 = ave.hir.vumax %arg0, %arg1, %mask : vector<64 x i32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x i32>) -> ()
  return
}
 
// Test VmaxV64S32XInstrOp
// CHECK-LABEL: test_vsmax_v64_i32
func.func @test_vsmax_v64_i32(%arg0: vector<64 x i32>, %arg1: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vmax.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
  %0 = ave.hir.vsmax %arg0, %arg1, %mask : vector<64 x i32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x i32>) -> ()
  return
}

// Test VaddV64S32XInstrOp
// CHECK-LABEL: test_vadd_v64_i32
func.func @test_vadd_v64_i32(%arg0: vector<64 x i32>, %arg1: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  // CHECK: [[R:%.*]] = "hivm_regbaseintrins.intr.hivm.vadd.s.x"([[V0:%.*]], [[V1:%.*]], [[V2:%.*]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<64 x i32>, vector<256 x i1>
  "test.test"(%0) : (vector<64 x i32>) -> ()
  return
}

// Test VciV64S32InstrOp
// CHECK-LABEL: test_vci_v64_i32
func.func @test_vci_v64_i32(%arg0: i32) -> () {
  %c0 = arith.constant 0 : i32
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i32, i32) -> vector<64xi32>
  %0 = ave.hir.vci %arg0, <INCREASE> : i32, vector<64xi32>
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i32, i32) -> vector<64xi32>
  %1 = ave.hir.vci %c0, <INCREASE> : i32, vector<64xi32>
  "test.test"(%0) : (vector<64 x i32>) -> ()
  "test.test"(%1) : (vector<64 x i32>) -> ()
  return
}

// Test VciV64S32InstrOp
// CHECK-LABEL: test_vci_v64_i32
func.func @test_vci_v64_i32_16(%arg0: i32) -> () {
  %c0 = arith.constant 0 : i32
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i32, i32) -> vector<64xi32>
  %0 = ave.hir.vci %arg0, <INCREASE> : i32, vector<16xi32>
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i32, i32) -> vector<64xi32>
  %1 = ave.hir.vci %c0, <INCREASE> : i32, vector<16xi32>
  "test.test"(%0) : (vector<16 x i32>) -> ()
  "test.test"(%1) : (vector<16 x i32>) -> ()
  return
}

// Test VciV128S16InstrOp
// CHECK-LABEL: test_vci_v64_i16
func.func @test_vci_v64_i16(%arg0: i16) -> () {
  %c0 = arith.constant 0 : i16
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i16, i32) -> vector<128xi16>
  %0 = ave.hir.vci %arg0, <INCREASE> : i16, vector<128xi16>
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i16, i32) -> vector<128xi16>
  %1 = ave.hir.vci %c0, <INCREASE> : i16, vector<128xi16>
  "test.test"(%0) : (vector<128 x i16>) -> ()
  "test.test"(%1) : (vector<128 x i16>) -> ()
  return
}

// Test VciV128F16InstrOp
// CHECK-LABEL: test_vci_f16
func.func @test_vci_f16(%arg0: f16) -> () {
  %c0 = arith.constant 0.0 : f16
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (f16, i32) -> vector<128xf16>
  %0 = ave.hir.vci %arg0, <INCREASE> : f16, vector<128xf16>
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (f16, i32) -> vector<128xf16>
  %1 = ave.hir.vci %c0, <INCREASE> : f16, vector<128xf16>
  // CHECK: [[R2:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (f16, i32) -> vector<128xf16>
  // CHECK: [[R3:%.*]] = builtin.unrealized_conversion_cast [[R2]] : vector<128xf16> to vector<64xf16>
  %2 = ave.hir.vci %c0, <INCREASE> : f16, vector<64xf16>
  "test.test"(%0) : (vector<128 x f16>) -> ()
  "test.test"(%1) : (vector<128 x f16>) -> ()
  "test.test"(%2) : (vector<64 x f16>) -> ()
  return
}

// Test VciV64F32InstrOp
// CHECK-LABEL: test_vci_f32
func.func @test_vci_f32(%arg0: f32) -> () {
  %c0 = arith.constant 0.0 : f32
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (f32, i32) -> vector<64xf32>
  %0 = ave.hir.vci %arg0, <INCREASE> : f32, vector<64xf32>
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (f32, i32) -> vector<64xf32>
  %1 = ave.hir.vci %c0, <INCREASE> : f32, vector<64xf32>
  // CHECK: [[R2:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (f32, i32) -> vector<64xf32>
  // CHECK: [[R3:%.*]] = builtin.unrealized_conversion_cast [[R2]] : vector<64xf32> to vector<32xf32>
  %2 = ave.hir.vci %c0, <INCREASE> : f32, vector<32xf32>
  "test.test"(%0) : (vector<64 x f32>) -> ()
  "test.test"(%1) : (vector<64 x f32>) -> ()
  "test.test"(%2) : (vector<32 x f32>) -> ()
  return
}

// Test VciV256S8InstrOp
// CHECK-LABEL: test_vci_v64_i8
func.func @test_vci_v64_i8(%arg0: i8) -> () {
  %c0 = arith.constant 0 : i8
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i8, i32) -> vector<256xi8>
  %0 = ave.hir.vci %arg0, <INCREASE> : i8, vector<256xi8>
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i8, i32) -> vector<256xi8>
  %1 = ave.hir.vci %c0, <INCREASE> : i8, vector<256xi8>
  // CHECK: [[R1:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"([[V0:%.*]], [[V1:%.*]]) : (i8, i32) -> vector<256xi8>
  %2 = ave.hir.vci %c0, <INCREASE> : i8, vector<64xi8>
  "test.test"(%0) : (vector<256 x i8>) -> ()
  "test.test"(%1) : (vector<256 x i8>) -> ()
  "test.test"(%2) : (vector<64 x i8>) -> ()
  return
}
