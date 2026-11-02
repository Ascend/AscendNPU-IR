// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=PIPE
// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="target=Ascend960 enable-auto-vectorize-v2=true disable-hfusion-vectorize=false" --hivm-v2-pipeline="target=Ascend960" --mlir-print-ir-before=hivm-vectorize --mlir-print-ir-before=hfusion-auto-vectorize-v2 --mlir-print-ir-before=hfusion-select-hivm-vectorize -o /dev/null 2>&1 | FileCheck %s --check-prefix=ORDER

// The v2 pipeline reuses InsertLoadStoreForMixCV after TCB insertion.
// PIPE: hivm-insert-cv-tight-coupled-buffer
// PIPE: hivm-insert-load-store-for-mix-cv
// PIPE: hivm-mark-tightly-coupled-buffer
// PIPE: hivm-insert-fixpipe
// PIPE: hivm-inline-fixpipe
// PIPE-NOT: hivm-insert-cv-data-movement
// PIPE: hivm-vectorize
// PIPE: hfusion-simplify-vf-arg

// Driver order: HFusion auto-vectorize, then hivm-vectorize.
// ORDER: IR Dump Before SelectHIVMVectorize (hfusion-select-hivm-vectorize)
// ORDER: IR Dump Before AutoVectorizeV2 (hfusion-auto-vectorize-v2)
// ORDER: IR Dump Before HIVMVectorize (hivm-vectorize)

module {}
