// RUN: bishengir-opt %s --one-shot-bufferize | FileCheck %s --check-prefix=RECEIVER --implicit-check-not=memref.copy
// RUN: bishengir-opt %s --one-shot-bufferize -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std="mark-libcall-noinline=true" | FileCheck %s --check-prefix=SYMBOL
// SYMBOL: func.func private @online_softmax_nz_f16_f8e4m3fn_causal_if
// SYMBOL: call @online_softmax_nz_f16_f8e4m3fn_causal_if
// SYMBOL-SAME: i1

// RECEIVER: %[[A:.*]] = memref.alloc() {{.*}} : memref<65536xi8, #hivm.address_space<ub>>
// RECEIVER: %[[B:.*]] = memref.alloc() {{.*}} : memref<65536xi8, #hivm.address_space<ub>>
// RECEIVER-NOT: memref.alloc
// RECEIVER: %[[ROOT:.*]] = arith.select %arg0, %[[A]], %[[B]]
// RECEIVER: %[[CAST:.*]] = memref.memory_space_cast %[[ROOT]]
// RECEIVER: %[[SCORE:.*]] = memref.view %[[CAST]]{{.*}} to memref<32x4x16x16xf16
// RECEIVER: scf.if %arg0
// RECEIVER: call @observe(%arg1)
// RECEIVER: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
// RECEIVER: } else {
// RECEIVER: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
// RECEIVER: }
// RECEIVER-NOT: memref.alloc
// RECEIVER: %[[FLAT:.*]] = memref.view %[[CAST]]{{.*}} to memref<65536xf8E4M3FN
// RECEIVER: %[[P:.*]] = memref.reinterpret_cast %[[FLAT]]
// RECEIVER-SAME: strides: [4096, 512, 32, 1]
// RECEIVER-NOT: memref.alloc
// RECEIVER: hivm.hir.custom
// RECEIVER-SAME: ins(%[[SCORE]],
// RECEIVER-SAME: memref<64xi32>, i1) outs(%[[P]],
// RECEIVER-SAME: memref<64xf16>, memref<64xf32>
// RECEIVER-NOT: memref.alloc
// RECEIVER: call @observe_sum
// RECEIVER: return

func.func private @observe(i1)
func.func private @observe_sum(f32)
func.func @selected(%cond: i1, %other: i1, %m: memref<64xf16>, %lut: memref<256xi8>, %mn: memref<64xf16>, %sum: memref<64xf32>, %limits: memref<64xi32>, %required: i1) {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<32x4x16x16xf16, #hivm.address_space<ub>>
  annotation.mark %a {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<32x4x16x16xf16, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<32x4x16x16xf16, #hivm.address_space<ub>>
  annotation.mark %b {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<32x4x16x16xf16, #hivm.address_space<ub>>
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
  %s = scf.if %cond -> tensor<32x4x16x16xf16> {
    func.call @observe(%other) : (i1) -> ()
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    %ac = memref.memory_space_cast %a : memref<32x4x16x16xf16, #hivm.address_space<ub>> to memref<32x4x16x16xf16>
    %at = bufferization.to_tensor %ac restrict writable : memref<32x4x16x16xf16>
    scf.yield %at : tensor<32x4x16x16xf16>
  } else {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
    %bc = memref.memory_space_cast %b : memref<32x4x16x16xf16, #hivm.address_space<ub>> to memref<32x4x16x16xf16>
    %bt = bufferization.to_tensor %bc restrict writable : memref<32x4x16x16xf16>
    scf.yield %bt : tensor<32x4x16x16xf16>
  }
  %mt = bufferization.to_tensor %m restrict : memref<64xf16>
  %lt = bufferization.to_tensor %lut restrict : memref<256xi8>
  %mnt = bufferization.to_tensor %mn restrict writable : memref<64xf16>
  %st = bufferization.to_tensor %sum restrict writable : memref<64xf32>
  %ct = bufferization.to_tensor %limits restrict : memref<64xi32>
  %p = tensor.empty() : tensor<16x4x16x32xf8E4M3FN>
  %r:3 = hivm.hir.custom {symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%s, %mt, %lt, %ct, %required : tensor<32x4x16x16xf16> {inplace_outs = [0 : i32, affine_map<(d0, d1, d2, d3) -> (d0 * 2, d1, d2, d3)>]}, tensor<64xf16>, tensor<256xi8>, tensor<64xi32>, i1) outs(%p, %mnt, %st : tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>) -> (tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>)
    %sum0 = tensor.extract %r#2[%c0] : tensor<64xf32>
    func.call @observe_sum(%sum0) : (f32) -> ()
    scf.yield
  }
  return
}
