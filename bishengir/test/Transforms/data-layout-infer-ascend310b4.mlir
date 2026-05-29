// RUN: bishengir-opt --arith-vector-mask-analyze --convert-vector-to-hivmave  \
// RUN: --convert-arith-to-hivmave --annotation-lowering  \
// RUN: --analyze-vector-layout --analyze-alignment-bitwidth --remove-vector-layout-attr \
// RUN: -ave-normalize-ops --convert-hivmave-to-ave-intrin -reconcile-unrealized-casts -cse  \
// RUN: -split-input-file %s | FileCheck %s

func.func @test_i8_cast_i16(%arg0: memref<5xi8, #hivm.address_space<ub>>, %arg1: memref<5xi8, #hivm.address_space<ub>>, %arg2: memref<5xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [5] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<5xi8, #hivm.address_space<ub>>, vector<128xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<5xi8, #hivm.address_space<ub>>, vector<128xi8>
  %3 = arith.extsi %2 {round_mode = #hfusion.round_mode<rint>} : vector<128xi8> to vector<128xi16>
  %4 = arith.extsi %1 {round_mode = #hfusion.round_mode<rint>} : vector<128xi8> to vector<128xi16>
  %5 = arith.divsi %4, %3 : vector<128xi16>
  %6 = arith.trunci %5 {round_mode = #hfusion.round_mode<trunc>} : vector<128xi16> to vector<128xi8>
  vector.transfer_write %6, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xi8>, memref<5xi8, #hivm.address_space<ub>>
  return
}

// CHECK: func.func @test_i8_cast_i16{{.*}} attributes {hivm.vector_function}
// CHECK-NEXT:   %[[UCC2:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   %[[UCC1:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   %[[UCC0:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:   %[[c5:.+]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT:   %[[PLT:.+]] = "hivm_regbaseintrins.intr.hivm.plt.b16.v300"(%[[c5]]) : (i32) -> !llvm.struct<(vector<256xi1>, i32)>
// CHECK-NEXT:   %[[EXTRACT_PLT:.+]] = llvm.extractvalue %[[PLT]][0] {mask_bit_width = 16 : i32, mask_op_idx = 0 : i32} : !llvm.struct<(vector<256xi1>, i32)>
// CHECK-NEXT:   %[[EXTRACT_UCC0:.+]] = llvm.extractvalue %[[UCC0]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC0:.+]] = llvm.getelementptr %[[EXTRACT_UCC0]]
// CHECK-NEXT:   llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:   %[[c13:.+]] = llvm.mlir.constant(13 : i32) : i32
// CHECK-NEXT:   %[[LOAD0:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%[[GET_UCC0]], %{{.*}}, %[[c13]], %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[EXTRACT_UCC1:.+]] = llvm.extractvalue %[[UCC1]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC1:.+]] = llvm.getelementptr %[[EXTRACT_UCC1]]
// CHECK-NEXT:   %[[LOAD1:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%[[GET_UCC1]], %{{.*}}, %[[c13]], %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[cvt1:.+]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s16.x"(%[[LOAD1]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xi16>
// CHECK-NEXT:   %[[cvt0:.+]] =  "hivm_regbaseintrins.intr.hivm.vcvtii.s82s16.x"(%[[LOAD0]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xi16>
// CHECK-NEXT:   %[[div:.+]] = "hivm_regbaseintrins.intr.hivm.vdiv.s.x"(%[[cvt0]], %[[cvt1]], %[[EXTRACT_PLT]]) : (vector<128xi16>, vector<128xi16>, vector<256xi1>) -> vector<128xi16>
// CHECK-NEXT:   %[[cvt2:.+]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s162u8.x"(%[[div]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<128xi16>, vector<256xi1>, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[EXTRACT_UCC2:.+]] = llvm.extractvalue %[[UCC2]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC2:.+]] = llvm.getelementptr %[[EXTRACT_UCC2]]
// CHECK-NEXT:   %[[c6:.+]] = llvm.mlir.constant(6 : i32) : i32
// CHECK-NEXT:   "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"(%[[cvt2]], %[[GET_UCC2]], %{{.*}}, %[[c6]], %{{.*}}, %[[EXTRACT_PLT]]) : (vector<256xi8>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()

func.func @test_i8_cast_i32(%arg0: memref<5xi8, #hivm.address_space<ub>>, %arg1: memref<5xi8, #hivm.address_space<ub>>, %arg2: memref<5xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [5] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<5xi8, #hivm.address_space<ub>>, vector<64xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<5xi8, #hivm.address_space<ub>>, vector<64xi8>
  %3 = arith.extsi %2 {round_mode = #hfusion.round_mode<rint>} : vector<64xi8> to vector<64xi32>
  %4 = arith.extsi %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xi8> to vector<64xi32>
  %5 = arith.divsi %4, %3 : vector<64xi32>
  %6 = arith.trunci %5 {round_mode = #hfusion.round_mode<trunc>} : vector<64xi32> to vector<64xi8>
  vector.transfer_write %6, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xi8>, memref<5xi8, #hivm.address_space<ub>>
  return
}

// CHECK: func.func @test_i8_cast_i32{{.*}} attributes {hivm.vector_function}
// CHECK-NEXT:   %[[UCC2:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   %[[UCC1:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   %[[UCC0:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:   %[[c5:.+]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT:   %[[PLT:.+]] = "hivm_regbaseintrins.intr.hivm.plt.b32.v300"(%[[c5]]) : (i32) -> !llvm.struct<(vector<256xi1>, i32)>
// CHECK-NEXT:   %[[EXTRACT_PLT:.+]] = llvm.extractvalue %[[PLT]][0] {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : !llvm.struct<(vector<256xi1>, i32)>
// CHECK-NEXT:   %[[EXTRACT_UCC0:.+]] = llvm.extractvalue %[[UCC0]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC0:.+]] = llvm.getelementptr %[[EXTRACT_UCC0]]
// CHECK-NEXT:   %[[c0_i32:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:   %[[LOAD0:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%[[GET_UCC0]], %[[c0_i32]], %[[c0_i32]], %[[c0_i32]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[INTLV0_0:.+]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[LOAD0]], %[[LOAD0]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_INTLV0_0:.+]] = llvm.extractvalue %[[INTLV0_0]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[INTLV0_1:.+]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[EXTRACT_INTLV0_0]], %[[EXTRACT_INTLV0_0]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_INTLV0_1:.+]] = llvm.extractvalue %[[INTLV0_1]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_UCC1:.+]] = llvm.extractvalue %[[UCC1]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC1:.+]] = llvm.getelementptr %[[EXTRACT_UCC1]]
// CHECK-NEXT:   %[[LOAD1:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%[[GET_UCC1]], %[[c0_i32]], %[[c0_i32]], %[[c0_i32]]) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[INTLV1_0:.+]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[LOAD1]], %[[LOAD1]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_INTLV1_0:.+]] = llvm.extractvalue %[[INTLV1_0]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[INTLV1_1:.+]] = "hivm_regbaseintrins.intr.hivm.vintlv"(%[[EXTRACT_INTLV1_0]], %[[EXTRACT_INTLV1_0]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_INTLV1_1:.+]] = llvm.extractvalue %[[INTLV1_1]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[cvt1:.+]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x"(%[[EXTRACT_INTLV1_1]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
// CHECK-NEXT:   %[[cvt0:.+]] =  "hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x"(%[[EXTRACT_INTLV0_1]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
// CHECK-NEXT:   %[[div:.+]] = "hivm_regbaseintrins.intr.hivm.vdiv.s.x"(%[[cvt0]], %[[cvt1]], %[[EXTRACT_PLT]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
// CHECK-NEXT:   %[[cvt2:.+]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s322u8.x"(%[[div]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[DINTLV0:.+]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[cvt2]], %[[cvt2]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_DINTLV0:.+]] = llvm.extractvalue %[[DINTLV0]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[DINTLV1:.+]] = "hivm_regbaseintrins.intr.hivm.vdintlv"(%[[EXTRACT_DINTLV0]], %[[EXTRACT_DINTLV0]]) : (vector<256xi8>, vector<256xi8>) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_DINTLV1:.+]] = llvm.extractvalue %[[DINTLV1]][0] : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK-NEXT:   %[[EXTRACT_UCC2:.+]] = llvm.extractvalue %[[UCC2]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC2:.+]] = llvm.getelementptr %[[EXTRACT_UCC2]]
// CHECK-NEXT:   "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"(%[[EXTRACT_DINTLV1]], %[[GET_UCC2]], %{{.*}}, %[[c0_i32]], %{{.*}}, %[[EXTRACT_PLT]]) : (vector<256xi8>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()