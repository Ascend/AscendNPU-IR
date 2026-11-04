// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" \
// RUN:   --convert-hfusion-to-hivm --cv-pipelining="pipeline-mode=skew" \
// RUN:   -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @preload_non_nan_fill
// CHECK: scope.scope
// CHECK:   hivm.hir.mmadL1
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scope.scope
// CHECK:   %[[ZERO:.*]] = hivm.hir.vbrc
// CHECK:   %[[MASKED:.*]] = hivm.hir.vsel
// CHECK-NOT:   hivm.hir.vsel
// CHECK:   hivm.hir.vmax ins(%[[MASKED]], %[[ZERO]]
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preload_non_nan_fill(
      %workspace: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %a: tensor<16x16xf16>,
      %b: tensor<16x16xf16>,
      %out: memref<16x16xf32>) attributes {
        WorkspaceArgIdx = 0 : i16,
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>,
        mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %bound step %step : i32 {
      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws = memref_ext.alloc_workspace() from %workspace : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws {hivm.multi_buffer = 2 : i32} : memref<16x16xf32>
      %ws_tensor = bufferization.to_tensor %ws restrict writable : memref<16x16xf32>
      %fix = hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%ws_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %load_init = tensor.empty() : tensor<16x16xf32>
      %load = hivm.hir.load ins(%fix : tensor<16x16xf32>) outs(%load_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %zero_init = tensor.empty() : tensor<16x16xf32>
      %zero_fill = linalg.fill ins(%zero : f32) outs(%zero_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %max_init = tensor.empty() : tensor<16x16xf32>
      %max = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%load, %zero_fill : tensor<16x16xf32>, tensor<16x16xf32>) outs(%max_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.store ins(%max : tensor<16x16xf32>) outs(%out : memref<16x16xf32>)
      scf.yield
    }
    return
  }
}
