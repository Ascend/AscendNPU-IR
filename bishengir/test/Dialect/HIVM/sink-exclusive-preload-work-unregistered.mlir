// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work | FileCheck %s

func.func @vf_consume(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  return
}

// Off-registry functions must keep exclusive dest / mb=1 allocs in the producer.
// CHECK-LABEL: func.func @keep_returned_dest_tensor_in_producer
// CHECK:         scope.scope
// CHECK:           %[[SRC:.*]] = bufferization.to_tensor
// CHECK:           %[[EMPTY:.*]] = tensor.empty
// CHECK:           %[[DST:.*]] = hivm.hir.vcast {{.*}} ins(%[[SRC]] {{.*}} outs(%[[EMPTY]]
// CHECK:           scope.return %[[DST]]
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           hivm.hir.vexp
func.func @keep_returned_dest_tensor_in_producer(%arg0: memref<32x128xbf16, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<32x128xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// CHECK-LABEL: func.func @keep_single_slot_preload_alloc_in_loop
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         annotation.mark %[[BUF]] {{.*}}multi_buffer = 1{{.*}}preload_local_buffer
// CHECK:         scope.scope
// CHECK:           call @vf_consume(%[[BUF]])
func.func @keep_single_slot_preload_alloc_in_loop()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}
