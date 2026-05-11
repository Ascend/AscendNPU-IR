// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 --arith-vector-mask-analyze --convert-vector-to-hivmave  \
// RUN: --convert-arith-to-hivmave --annotation-lowering --data-layout-analyze \
// RUN: --append-vector-layout --annotate-dist-op-layout --eliminate-vector-layout \
// RUN: --ave-normalize-ops --convert-hivmave-to-ave-intrin -reconcile-unrealized-casts -cse  \
// RUN: -split-input-file %s | FileCheck %s

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_i8_cast_i32_ascend910d(%arg0: memref<5xi8, #hivm.address_space<ub>>, %arg1: memref<5xi8, #hivm.address_space<ub>>, %arg2: memref<5xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
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
}

// CHECK: func.func @test_i8_cast_i32_ascend910d{{.*}} attributes {element_alignment_bit_width = 32 : i32, hivm.vector_function} {
// CHECK-NEXT:   %[[UCC2:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   %[[UCC1:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   %[[UCC0:.+]] = builtin.unrealized_conversion_cast
// CHECK-NEXT:   llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:   %[[c5:.+]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT:   %[[PLT:.+]] = "hivm_regbaseintrins.intr.hivm.plt.b32.v300"(%[[c5]]) : (i32) -> !llvm.struct<(vector<256xi1>, i32)>
// CHECK-NEXT:   %[[EXTRACT_PLT:.+]] = llvm.extractvalue %[[PLT]][0] {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : !llvm.struct<(vector<256xi1>, i32)>
// CHECK-NEXT:   %[[EXTRACT_UCC0:.+]] = llvm.extractvalue %[[UCC0]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC0:.+]] = llvm.getelementptr %[[EXTRACT_UCC0]]
// CHECK-NEXT:   llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:   %[[c20:.+]] = llvm.mlir.constant(20 : i32) : i32
// CHECK-NEXT:   %[[LOAD0:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%[[GET_UCC0]], %{{.*}}, %[[c20]], %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[EXTRACT_UCC1:.+]] = llvm.extractvalue %[[UCC1]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC1:.+]] = llvm.getelementptr %[[EXTRACT_UCC1]]
// CHECK-NEXT:   %[[LOAD1:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%[[GET_UCC1]], %{{.*}}, %[[c20]], %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[cvt1:.+]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x"(%[[LOAD1]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
// CHECK-NEXT:   %[[cvt0:.+]] =  "hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x"(%[[LOAD0]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
// CHECK-NEXT:   %[[div:.+]] = "hivm_regbaseintrins.intr.hivm.vdiv.s.x"(%[[cvt0]], %[[cvt1]], %[[EXTRACT_PLT]]) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
// CHECK-NEXT:   %[[cvt2:.+]] = "hivm_regbaseintrins.intr.hivm.vcvtii.s322u8.x"(%[[div]], %[[EXTRACT_PLT]], %{{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<256xi8>
// CHECK-NEXT:   %[[EXTRACT_UCC2:.+]] = llvm.extractvalue %[[UCC2]][1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:   %[[GET_UCC2:.+]] = llvm.getelementptr %[[EXTRACT_UCC2]]
// CHECK-NEXT:   %[[c12:.+]] = llvm.mlir.constant(12 : i32) : i32
// CHECK-NEXT:   "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"(%[[cvt2]], %[[GET_UCC2]], %{{.*}}, %[[c12]], %{{.*}}, %[[EXTRACT_PLT]]) : (vector<256xi8>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()

// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_i8_cast_i32_brc_onept_dist(%arg0: memref<1xi8, #hivm.address_space<ub>>, %arg1: memref<1xi8, #hivm.address_space<ub>>, %arg2: memref<1xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1] : vector<64xi1>
    %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<1xi8, #hivm.address_space<ub>>, vector<64xi8>
    %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<1xi8, #hivm.address_space<ub>>, vector<64xi8>
    %3 = arith.extsi %2 {round_mode = #hfusion.round_mode<rint>} : vector<64xi8> to vector<64xi32>
    %4 = arith.extsi %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xi8> to vector<64xi32>
    %5 = arith.divsi %4, %3 : vector<64xi32>
    %6 = arith.trunci %5 {round_mode = #hfusion.round_mode<trunc>} : vector<64xi32> to vector<64xi8>
    vector.transfer_write %6, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xi8>, memref<1xi8, #hivm.address_space<ub>>
    return
  }
}

// CHECK: func.func @test_i8_cast_i32_brc_onept_dist{{.*}} attributes {element_alignment_bit_width = 32 : i32, hivm.vector_function} {
// CHECK:   %[[c1:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:   %[[LOAD0:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%{{.*}}, %[[c1]], %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK:   %[[LOAD1:.+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%{{.*}}, %[[c1]], %{{.*}}) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
// CHECK:   %[[c3:.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:   "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"(%{{.*}}, %[[c3]], %{{.*}}) : (vector<256xi8>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()