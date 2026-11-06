// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefixes=CHECK,FULL
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pre-vectorize-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefixes=CHECK,PRE --implicit-check-not=hivm-vectorize --implicit-check-not=bufferiz
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" | FileCheck %s --check-prefix=SPLIT
// RUN: bishengir-opt %s --hivm-v2-split-kernel | FileCheck %s --check-prefix=SPLIT
// RUN: echo 'module {}' | bishengir-opt --lower-hfusion-regbase-pipeline="target=Ascend960 enable-auto-vectorize-v2=true disable-hfusion-vectorize=false" --hivm-v2-pipeline="target=Ascend960" --mlir-print-ir-before=hivm-vectorize --mlir-print-ir-before=hfusion-auto-vectorize-v2 --mlir-print-ir-before=hfusion-select-hivm-vectorize -o /dev/null 2>&1 | FileCheck %s --check-prefix=ORDER

// The v2 pipeline reuses InsertLoadStoreForMixCV after TCB insertion.
// CHECK: hivm-insert-cv-tight-coupled-buffer
// CHECK: hivm-insert-load-store-for-mix-cv
// CHECK: hivm-mark-tightly-coupled-buffer
// CHECK: hivm-insert-fixpipe
// CHECK: hivm-inline-fixpipe
// CHECK-NOT: hivm-insert-cv-data-movement
// PRE: hivm-insert-convert-layout
// PRE: hivm-propagate-convert-layout
// PRE: hivm-convert-layout-to-transpose
// PRE: cv-pipelining
// PRE: hivm-normalize-matmul
// FULL: hivm-vectorize{{[[:space:]]*$}}
// FULL: hfusion-simplify-vf-arg
// FULL: one-shot-bufferize
// FULL: hivm-cross-core-gss
// FULL: hivm-v2-split-kernel

// Driver order: HFusion auto-vectorize, then hivm-vectorize.
// Use an empty module here: HFusion auto-scheduling requires tensor work.
// ORDER: IR Dump Before SelectHIVMVectorize (hfusion-select-hivm-vectorize)
// ORDER: IR Dump Before AutoVectorizeV2 (hfusion-auto-vectorize-v2)
// ORDER: IR Dump Before HIVMVectorize (hivm-vectorize)

// SPLIT-LABEL: func.func @empty_mix_mix_aic()
// SPLIT-LABEL: func.func @empty_mix_mix_aiv()
module {
  func.func @empty_mix() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    return
  }
}
