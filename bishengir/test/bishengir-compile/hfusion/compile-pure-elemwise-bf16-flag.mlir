// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true \
// RUN:   --target=Ascend950PR_9579 -block-dim=20 --save-temps=%t.off.saved %s -o %t.off.o
// RUN: FileCheck %s --check-prefix=DISABLED --input-file=%t.off.saved/module.hivm.opt.mlir
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true \
// RUN:   --target=Ascend950PR_9579 -block-dim=20 --enable-bf16 --save-temps=%t.on.saved %s -o %t.on.o
// RUN: FileCheck %s --check-prefix=ENABLED --input-file=%t.on.saved/module.hivm.opt.mlir

// Without --enable-bf16 the bf16 elementwise add is legalized to f32 and the
// regbase vadd intrinsic is emitted with f32 vectors; with --enable-bf16 the
// native bf16 vectors reach the vadd instruction directly.
// DISABLED: "hivm_regbaseintrins.intr.hivm.vadd.s.x"
// DISABLED-SAME: vector<64xf32>, vector<64xf32>, vector<256xi1>
// ENABLED: "hivm_regbaseintrins.intr.hivm.vadd.s.x"
// ENABLED-SAME: vector<128xbf16>, vector<128xbf16>, vector<256xi1>
func.func @add_2d_bf16(%arg0: tensor<1024x1024xbf16>, %arg1: tensor<1024x1024xbf16>, %arg2: tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
attributes {hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<1024x1024xbf16>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) outs(%1 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
  return %2 : tensor<1024x1024xbf16>
}
