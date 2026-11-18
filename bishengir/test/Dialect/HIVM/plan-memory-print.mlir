// RUN: mkdir -p %t && cd %t && bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -hivm-plan-memory=enable-memory-display=true -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CHECK-UB
// RUN: mkdir -p %t && cd %t && bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=enable-memory-display=true -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CHECK-UB
// RUN: mkdir -p %t && cd %t && bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -hivm-plan-memory=enable-memory-display=true -o /dev/null && FileCheck %s --check-prefix=CHECK-MEMDISPLAY < memory_info.json
// RUN: mkdir -p %t && cd %t && bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=enable-memory-display=true -o /dev/null && FileCheck %s --check-prefix=CHECK-MEMDISPLAY < memory_info.json

// CHECK-UB: [AscendNPU IR] Allocated UB size = {{[0-9]+}} bits

// CHECK-MEMDISPLAY: "KernelName": "{{.*}}"
// CHECK-MEMDISPLAY: "memory_info_array": [
// CHECK-MEMDISPLAY: "extent": {{[0-9]+}}
// CHECK-MEMDISPLAY: "offset": [
// CHECK-MEMDISPLAY: "scope": "ub"
// CHECK-MEMDISPLAY: "status": "success"

module {
  func.func @test_mem_allocate_basic(%src : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                    %dst3 : memref<16x16x16xf16, #hivm.address_space<gm>>) {
    %copy_in_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    %dst1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    %dst2 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    %copy_out_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%copy_in_ub : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%copy_in_ub, %copy_in_ub : memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%dst1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%dst1, %dst1 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%dst2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%dst2, %dst2 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%copy_out_ub : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%copy_out_ub : memref<16x16x16xf16,#hivm.address_space<ub>>)
                   outs(%dst3: memref<16x16x16xf16,#hivm.address_space<gm>>)
    return
  }
}
