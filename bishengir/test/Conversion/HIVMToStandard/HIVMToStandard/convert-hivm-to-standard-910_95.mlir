// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 %s -convert-hivm-to-std  -split-input-file| FileCheck %s

module {
  // CHECK-LABEL: test_gather_1d_int8_t
  func.func @test_gather_1d_int8_t() {
    %c96_i64 = arith.constant 96 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64

    %alloc = memref.alloc() : memref<1xi8, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xi32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xi8, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<1xi32, #hivm.address_space<ub>>
    // CHECK: call @gather_1d_int8_t
    hivm.hir.vgather ins(%alloc : memref<1xi8, #hivm.address_space<ub>>) 
                     indices(%alloc_1 : memref<1xi32, #hivm.address_space<ub>>) 
                     outs(%alloc_2 : memref<1xi8, #hivm.address_space<ub>>) 
                     temp_buffer(%alloc_3 : memref<1xi32, #hivm.address_space<ub>>)
    return
 }
}