// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 \
// RUN:   -normalize-vector -convert-vector-to-hivmave -canonicalize -cse \
// RUN:   -convert-arith-to-hivmave -analyze-vector-layout -ave-normalize-ops \
// RUN:   -convert-hivmave-to-ave-intrin -split-input-file | FileCheck %s

// Lower prefix masks through layout analysis to native predicates. Gather
// predicates use B16, independently of the B8 predicate on the final store.
// CHECK-LABEL: func.func @dual_gather_partial_200
// CHECK: "hivm_regbaseintrins.intr.hivm.plt.b8.v300"
// CHECK: %[[NE:.*]] = llvm.mlir.constant(100 : i32)
// CHECK: %[[PE:.*]] = "hivm_regbaseintrins.intr.hivm.plt.b16.v300"(%[[NE]])
// CHECK: %[[BE:.*]] = llvm.extractvalue %[[PE]][0] {mask_bit_width = 16 : i32}
// CHECK: %[[ME:.*]] = builtin.unrealized_conversion_cast %[[BE]] : vector<256xi1> to vector<128xi1>
// CHECK: %[[G0M:.*]] = builtin.unrealized_conversion_cast %[[ME]] : vector<128xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256s8"(%{{.*}}, %{{.*}}, %[[G0M]])
// CHECK: %[[G1M:.*]] = builtin.unrealized_conversion_cast %[[ME]] : vector<128xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256s8"(%{{.*}}, %{{.*}}, %[[G1M]])
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_partial_200(%src: memref<256xi8, #hivm.address_space<ub>>, %dst: memref<256xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %even = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [200] : vector<256xi1>
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    vector.maskedstore %dst[%c0], %mask, %res : memref<256xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
    return
  }
}

// -----
// CHECK-LABEL: func.func @dual_gather_partial_199
// CHECK: "hivm_regbaseintrins.intr.hivm.plt.b8.v300"
// CHECK: %[[NE:.*]] = llvm.mlir.constant(100 : i32)
// CHECK: %[[PE:.*]] = "hivm_regbaseintrins.intr.hivm.plt.b16.v300"(%[[NE]])
// CHECK: %[[BE:.*]] = llvm.extractvalue %[[PE]][0] {mask_bit_width = 16 : i32}
// CHECK: %[[ME:.*]] = builtin.unrealized_conversion_cast %[[BE]] : vector<256xi1> to vector<128xi1>
// CHECK: %[[NO:.*]] = llvm.mlir.constant(99 : i32)
// CHECK: %[[PO:.*]] = "hivm_regbaseintrins.intr.hivm.plt.b16.v300"(%[[NO]])
// CHECK: %[[BO:.*]] = llvm.extractvalue %[[PO]][0] {mask_bit_width = 16 : i32}
// CHECK: %[[MO:.*]] = builtin.unrealized_conversion_cast %[[BO]] : vector<256xi1> to vector<128xi1>
// CHECK: %[[G0M:.*]] = builtin.unrealized_conversion_cast %[[ME]] : vector<128xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256s8"(%{{.*}}, %{{.*}}, %[[G0M]])
// CHECK: %[[G1M:.*]] = builtin.unrealized_conversion_cast %[[MO]] : vector<128xi1> to vector<256xi1>
// CHECK: "hivm_regbaseintrins.intr.hivm.vgather2_v300.v256s8"(%{{.*}}, %{{.*}}, %[[G1M]])
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_partial_199(%src: memref<256xi8, #hivm.address_space<ub>>, %dst: memref<256xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %even = arith.constant dense<0> : vector<128xi16>
    %mask = vector.constant_mask [199] : vector<256xi1>
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    vector.maskedstore %dst[%c0], %mask, %res : memref<256xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
    return
  }
}
