// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-lir-compile=false -enable-static-bare-ptr=false %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// CHECK: LLVMDialectModule
// CHECK: @test_basic__kernel0(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4,
// CHECK-SAME:                 ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9,
// CHECK-SAME:                 ptr addrspace(1) %10, ptr addrspace(1) %11, i64 %12, i64 %13, i64 %14)
module {
  func.func @test_basic__kernel0(%valueA: memref<16xf16, #hivm.address_space<gm>>,
                                 %valueB: memref<16xf16, #hivm.address_space<gm>>,
                                 %valueC: memref<16xf16, #hivm.address_space<gm>>)
                                 attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}
  {
    %ubA = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueA : memref<16xf16, #hivm.address_space<gm>>) outs(%ubA : memref<16xf16, #hivm.address_space<ub>>)

    %ubB = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueB : memref<16xf16, #hivm.address_space<gm>>) outs(%ubB : memref<16xf16, #hivm.address_space<ub>>)

    %ubC = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%ubA, %ubB: memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%ubC: memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%ubC : memref<16xf16, #hivm.address_space<ub>>) outs(%valueC : memref<16xf16, #hivm.address_space<gm>>)
    return
  }
}