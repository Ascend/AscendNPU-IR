// RUN: bishengir-opt %s  -convert-hivmave-to-ave-intrin -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: @test_extf_f8E5M2
func.func @test_extf_f8E5M2(%arg0: vector<64xf8E5M2>) -> vector<64xf32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xf8E5M2> to vector<256xf8E5M2>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC_I8:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC:.*]] = llvm.bitcast %[[BRC_I8]] : vector<256xi8> to vector<256xf8E5M2>
  // CHECK-NEXT: %[[INTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xf8E5M2>, vector<256xf8E5M2>) -> !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[INTLV1]][0] : !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  // CHECK-NEXT: %[[INTLV2:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[EXTRACT1]], %[[BRC]]) : (vector<256xf8E5M2>, vector<256xf8E5M2>) -> !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = llvm.extractvalue %[[INTLV2]][0] : !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22f32.x"(%[[EXTRACT2]], %{{.*}}, %[[EVENPART]]) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<64xf32>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xf8E5M2>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_truncf_f8E5M2
func.func @test_truncf_f8E5M2(%arg0: vector<64xf32>) -> vector<64xf8E5M2> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"({{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC_I8:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC:.*]] = llvm.bitcast %[[BRC_I8]] : vector<256xi8> to vector<256xf8E5M2>
  // CHECK-NEXT: %[[DINTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<256xf8E5M2>, vector<256xf8E5M2>) -> !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[DINTLV1]][0] : !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  // CHECK-NEXT: %[[DINTLV2:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[EXTRACT1]], %[[BRC]]) : (vector<256xf8E5M2>, vector<256xf8E5M2>) -> !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = llvm.extractvalue %[[DINTLV2]][0] : !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0 : vector<64xf32>, vector<64xf8E5M2>, vector<64xi1>
  return %1 : vector<64xf8E5M2>
}

// -----

// CHECK-LABEL: @test_extf_f8E4M3
func.func @test_extf_f8E4M3(%arg0: vector<64xf8E4M3FN>) -> vector<64xf32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xf8E4M3FN> to vector<256xf8E4M3FN>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC_I8:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC:.*]] = llvm.bitcast %[[BRC_I8]] : vector<256xi8> to vector<256xf8E4M3FN>
  // CHECK-NEXT: %[[INTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xf8E4M3FN>, vector<256xf8E4M3FN>) -> !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[INTLV1]][0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  // CHECK-NEXT: %[[INTLV2:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[EXTRACT1]], %[[BRC]]) : (vector<256xf8E4M3FN>, vector<256xf8E4M3FN>) -> !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = llvm.extractvalue %[[INTLV2]][0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%[[EXTRACT2]], %{{.*}}, %[[EVENPART]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xf8E4M3FN>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_truncf_f8E4M3
func.func @test_truncf_f8E4M3(%arg0: vector<64xf32>) -> vector<64xf8E4M3FN> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"({{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC_I8:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC:.*]] = llvm.bitcast %[[BRC_I8]] : vector<256xi8> to vector<256xf8E4M3FN>
  // CHECK-NEXT: %[[DINTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<256xf8E4M3FN>, vector<256xf8E4M3FN>) -> !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[DINTLV1]][0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  // CHECK-NEXT: %[[DINTLV2:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[EXTRACT1]], %[[BRC]]) : (vector<256xf8E4M3FN>, vector<256xf8E4M3FN>) -> !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = llvm.extractvalue %[[DINTLV2]][0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0 : vector<64xf32>, vector<64xf8E4M3FN>, vector<64xi1>
  return %1 : vector<64xf8E4M3FN>
}

// -----

// CHECK-LABEL: @test_extf_f16
func.func @test_extf_f16(%arg0: vector<64xf16>) -> vector<64xf32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xf16> to vector<128xf16>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (f16) -> vector<128xf16>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xf16>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_extf_bf16
func.func @test_extf_bf16(%arg0: vector<64xbf16>) -> vector<64xf32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xbf16> to vector<128xbf16>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (bf16) -> vector<128xbf16>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_truncf_f16
func.func @test_truncf_f16(%arg0: vector<64xf32>) -> vector<64xf16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"({{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xf16>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (f16) -> vector<128xf16>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)>
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0 : vector<64xf32>, vector<64xf16>, vector<64xi1>
  return %1 : vector<64xf16>
}

// -----

// CHECK-LABEL: @test_truncf_bf16
func.func @test_truncf_bf16(%arg0: vector<64xf32>) -> vector<64xbf16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"({{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (bf16) -> vector<128xbf16>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
  %1 = ave.hir.vtruncf %arg0, <round>, true, <part_even>, %0 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
  return %1 : vector<64xbf16>
}

// -----

// CHECK-LABEL: @test_extui_i82i32
func.func @test_extui_i82i32(%arg0: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xi8> to vector<256xi8>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[INTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[INTLV1]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[INTLV2:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[EXTRACT1]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = llvm.extractvalue %[[INTLV2]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[PP0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.u82u32.x"(%[[EXTRACT2]], %{{.*}}, %[[PP0]]) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
  %1 = ave.hir.vextui %arg0, %0 {pp = #ave.vcvt_pp_type<pp0>, element_alignment_bit_width = -1 : i32} : vector<64xi8>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_extui_i162i32
func.func @test_extui_i162i32(%arg0: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i16) -> vector<128xi16>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<128xi16>, vector<128xi16>) -> !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.u162u32.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<128xi16>, vector<256xi1>, i32) -> vector<64xi32>
  %1 = ave.hir.vextui %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi16>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_extui_i82i16
func.func @test_extui_i82i16(%arg0: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.u82u16.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xi16>
  %1 = ave.hir.vextui %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi8>, vector<128xi16>, vector<128xi1>
  return %1 : vector<128xi16>
}

// -----

// CHECK-LABEL: @test_extsi_i82i32
func.func @test_extsi_i82i32(%arg0: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xi8> to vector<256xi8>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[INTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[INTLV1]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[INTLV2:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[EXTRACT1]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = llvm.extractvalue %[[INTLV2]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[PP0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x"(%[[EXTRACT2]], %{{.*}}, %[[PP0]]) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
  %1 = ave.hir.vextsi %arg0, %0 {pp = #ave.vcvt_pp_type<pp0>, element_alignment_bit_width = -1 : i32} : vector<64xi8>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_extsi_i162i32
func.func @test_extsi_i162i32(%arg0: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i16) -> vector<128xi16>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<128xi16>, vector<128xi16>) -> !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s162s32.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<128xi16>, vector<256xi1>, i32) -> vector<64xi32>
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi16>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_extsi_i82i16
func.func @test_extsi_i82i16(%arg0: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s16.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xi16>
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi8>, vector<128xi16>, vector<128xi1>
  return %1 : vector<128xi16>
}

// -----

// CHECK-LABEL: @test_extsi_i322i64
func.func @test_extsi_i322i64(%arg0: vector<32xi32>) -> vector<32xi64> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<32xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<32xi32> to vector<64xi32>
  // CHECK: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%{{.*}}) : (i32) -> vector<64xi32>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<64xi32>, vector<64xi32>) -> !llvm.struct<(vector<64xi32>, vector<64xi32>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<64xi32>, vector<64xi32>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s322s64.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<64xi32>, vector<256xi1>, i32) -> vector<32xi64>
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<32xi32>, vector<32xi64>, vector<32xi1>
  return %1 : vector<32xi64>
}

// -----

// CHECK-LABEL: @test_trunci_i642i32
func.func @test_trunci_i642i32(%arg0: vector<32xi64>, %mask: vector<256xi1>) -> vector<32xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s642s32.x"({{.*}}) : (vector<32xi64>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i32) -> vector<64xi32>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<64xi32>, vector<64xi32>) -> !llvm.struct<(vector<64xi32>, vector<64xi32>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<64xi32>, vector<64xi32>)>
  %1 = ave.hir.vtrunci %arg0, true, %mask {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<32xi64>, vector<32xi32>, vector<256xi1>
  return %1 : vector<32xi32>
}

// -----

// CHECK-LABEL: @test_trunci_i32i16
func.func @test_trunci_i32i16(%arg0: vector<64xi32>) -> vector<64xi16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s322s16.x"({{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i16) -> vector<128xi16>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<128xi16>, vector<128xi16>) -> !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  %1 = ave.hir.vtrunci %arg0, true, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi32>, vector<64xi16>, vector<64xi1>
  return %1 : vector<64xi16>
}

// -----

// CHECK-LABEL: @test_trunci_i162i8
func.func @test_trunci_i162i8(%arg0: vector<128xi16>) -> vector<128xi8> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s162u8.x"({{.*}}) : (vector<128xi16>, vector<256xi1>, i32, i32) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  %1 = ave.hir.vtrunci %arg0, true, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi16>, vector<128xi8>, vector<128xi1>
  return %1 : vector<128xi8>
}

// -----

// CHECK-LABEL: @test_trunci_i322i8
func.func @test_trunci_i322i8(%arg0: vector<64xi32>) -> vector<64xi8> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s322u8.x"({{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[DINTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[DINTLV1]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[DINTLV2:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[EXTRACT1]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT1:.*]] = llvm.extractvalue %[[DINTLV2]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  %1 = ave.hir.vtrunci %arg0, true, %0 {pp = #ave.vcvt_pp_type<pp0>, element_alignment_bit_width = -1 : i32} : vector<64xi32>, vector<64xi8>, vector<64xi1>
  return %1 : vector<64xi8>
}

// -----

// CHECK-LABEL: @test_fp2si_f322i64
func.func @test_fp2si_f322i64(%arg0: vector<32xf32>) -> vector<32xi64> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<32xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<32xf32> to vector<64xf32>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (f32) -> vector<64xf32>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<64xf32>, vector<64xf32>) -> !llvm.struct<(vector<64xf32>, vector<64xf32>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<64xf32>, vector<64xf32>)>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s64.x"(%[[EXTRACT]], %{{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<32xi64>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<32xf32>, vector<32xi1>, vector<32xi64>
  return %1 : vector<32xi64>
}

// -----

// CHECK-LABEL: @test_fp2si_f322i32
func.func @test_fp2si_f322i32(%arg0: vector<64xf32>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f322s32.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true, element_alignment_bit_width = -1 : i32} : vector<64xf32>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_fp2si_f322i32
func.func @test_fp2si_f322i32(%arg0: vector<64xf32>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f322s32.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true, element_alignment_bit_width = -1 : i32} : vector<64xf32>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_fp2si_f322i16
func.func @test_fp2si_f322i16(%arg0: vector<64xf32>) -> vector<64xi16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s16.x"({{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xi16>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i16) -> vector<128xi16>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<128xi16>, vector<128xi16>) -> !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<64xf32>, vector<64xi1>, vector<64xi16>
  return %1 : vector<64xi16>
}

// -----

// CHECK-LABEL: @test_fp2si_f162i32
func.func @test_fp2si_f162i32(%arg0: vector<64xf16>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xf16> to vector<128xf16>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (f16) -> vector<128xf16>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x"(%[[EXTRACT]], %{{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<64xi32>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xf16>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_fp2si_f162i16
func.func @test_fp2si_f162i16(%arg0: vector<128xf16>) -> vector<128xi16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f162s16.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true, element_alignment_bit_width = -1 : i32} : vector<128xf16>, vector<128xi1>, vector<128xi16>
  return %1 : vector<128xi16>
}

// -----

// CHECK-LABEL: @test_fp2si_f162i8
func.func @test_fp2si_f162i8(%arg0: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s8.x"({{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<128xf16>, vector<128xi1>, vector<128xi8>
  return %1 : vector<128xi8>
}

// -----

// CHECK-LABEL: @test_fp2si_bf162i32
func.func @test_fp2si_bf162i32(%arg0: vector<64xbf16>) -> vector<64xi32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xbf16> to vector<128xbf16>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (bf16) -> vector<128xbf16>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<128xbf16>, vector<128xbf16>) -> !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtfi.bf162s32.x"(%[[EXTRACT]], %{{.*}}) : (vector<128xbf16>, vector<256xi1>, i32, i32, i32) -> vector<64xi32>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<64xbf16>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}

// -----

// CHECK-LABEL: @test_fp2ui_f162i8
func.func @test_fp2ui_f162i8(%arg0: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162u8.x"({{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  %1 = ave.hir.vfptoui %arg0, <round>, true, <part_even>, %0 : vector<128xf16>, vector<128xi1>, vector<128xi8>
  return %1 : vector<128xi8>
}

// -----

// CHECK-LABEL: @test_si2fp_i82f16
func.func @test_si2fp_i82f16(%arg0: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtif.s82f16.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xf16>
  %1 = ave.hir.vsitofp %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi8>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}

// -----

// CHECK-LABEL: @test_si2fp_i162f16
func.func @test_si2fp_i162f16(%arg0: vector<128xi16>) -> vector<128xf16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.s162f16.x
  %1 = ave.hir.vsitofp %arg0, %0 {rnd = #hivm.round_mode<round>, element_alignment_bit_width = -1 : i32} : vector<128xi16>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}

// -----

// CHECK-LABEL: @test_si2fp_i162f32
func.func @test_si2fp_i162f32(%arg0: vector<64xi16>) -> vector<64xf32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i16) -> vector<128xi16>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<128xi16>, vector<128xi16>) -> !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtif.s162f32.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<128xi16>, vector<256xi1>, i32) -> vector<64xf32>
  %1 = ave.hir.vsitofp %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi16>, vector<64xi1>, vector<64xf32>
  return %1 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_si2fp_i322f32
func.func @test_si2fp_i322f32(%arg0: vector<64xi32>) -> vector<64xf32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.s322f32.x
  %1 = ave.hir.vsitofp %arg0, %0 {rnd = #hivm.round_mode<round>, element_alignment_bit_width = -1 : i32} : vector<64xi32>, vector<64xi1>, vector<64xf32>
  return %1 : vector<64xf32>
}

// -----

// CHECK-LABEL: @test_si2fp_i642f32
func.func @test_si2fp_i642f32(%arg0: vector<32xi64>, %mask: vector<256xi1>) -> vector<32xf32> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  // CHECK: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtif.s642f32.x"({{.*}}) : (vector<32xi64>, vector<256xi1>, i32, i32) -> vector<64xf32>
  // CHECK-NEXT: %[[BRC0:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (f32) -> vector<64xf32>
  // CHECK-NEXT: %[[DINTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[CVT]], %[[BRC]]) : (vector<64xf32>, vector<64xf32>) -> !llvm.struct<(vector<64xf32>, vector<64xf32>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[DINTLV]][0] : !llvm.struct<(vector<64xf32>, vector<64xf32>)>
  %1 = ave.hir.vsitofp %arg0, %mask {part = #ave.vcvt_part_type<part_even>, rnd = #hivm.round_mode<round>, element_alignment_bit_width = -1 : i32} : vector<32xi64>, vector<256xi1>, vector<32xf32>
  return %1 : vector<32xf32>
}

// -----

// CHECK-LABEL: @test_ui2fp_i82f16
func.func @test_ui2fp_i82f16(%arg0: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: %[[BRC0:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NEXT: %[[BRC:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[BRC0]]) : (i8) -> vector<256xi8>
  // CHECK-NEXT: %[[INTLV:.*]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[SRC]], %[[BRC]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EXTRACT:.*]] = llvm.extractvalue %[[INTLV]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  // CHECK-NEXT: %[[EVENPART:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[CVT:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtif.u82f16.x"(%[[EXTRACT]], %{{.*}}, %[[EVENPART]]) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xf16>
  %1 = ave.hir.vuitofp %arg0, <part_even>, %0 : vector<128xi8>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}