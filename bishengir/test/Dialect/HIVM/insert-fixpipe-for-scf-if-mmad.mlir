// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// A mmad in each branch of an scf.if yields a single merged result. The fixpipe
// must be inserted once after the scf.if, so that the merged value keeps
// pointing at the L0C buffer for annotation.mark bind_buffer.
// CHECK-LABEL: func.func @single_fixpipe_after_scf_if
// CHECK: %[[IF:.*]] = scf.if
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: scf.yield
// CHECK: } else {
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: scf.yield
// CHECK: }
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[IF]]
// CHECK: hivm.hir.store ins(%[[FIX]]
// CHECK: annotation.mark %[[IF]]
func.func @single_fixpipe_after_scf_if(
    %cond: i1,
    %arg0: memref<128x128xf16>,
    %arg1: memref<128x128xf16>,
    %arg3: memref<128x128xf32>) {
  %true = arith.constant true
  %false = arith.constant false
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<128x128xf32, #hivm.address_space<cc>>
  %a = bufferization.to_tensor %arg0 restrict writable : memref<128x128xf16>
  %b = bufferization.to_tensor %arg1 restrict writable : memref<128x128xf16>
  %res = scf.if %cond -> (tensor<128x128xf32>) {
    %memspacecast = memref.memory_space_cast %alloc : memref<128x128xf32, #hivm.address_space<cc>> to memref<128x128xf32>
    %acc = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn}
        ins(%a, %b, %false, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
        outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %mmad : tensor<128x128xf32>
  } else {
    %init = tensor.empty() : tensor<128x128xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn}
        ins(%a, %b, %true, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %mmad : tensor<128x128xf32>
  }
  hivm.hir.store ins(%res : tensor<128x128xf32>) outs(%arg3 : memref<128x128xf32>) atomic = <add>
  annotation.mark %res keys = ["bind_buffer"] values = [%alloc : memref<128x128xf32, #hivm.address_space<cc>>] : tensor<128x128xf32>
  return
}
}
