// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefixes=CHECK,FULL --implicit-check-not=hivm-inject-sync --implicit-check-not=hivm-inject-block-sync --implicit-check-not=hivm-delayed-cross-core-gss --implicit-check-not=hivm-insert-anchors-and-backup
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-MERGE --implicit-check-not=hfusion-merge-vf
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pre-vectorize-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefixes=CHECK,PRE --implicit-check-not=hivm-vectorize --implicit-check-not=bufferiz
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" | FileCheck %s --check-prefix=SPLIT
// RUN: bishengir-opt %s --hivm-v2-split-kernel | FileCheck %s --check-prefix=SPLIT
// RUN: echo 'module {}' | bishengir-opt --lower-hfusion-regbase-pipeline="target=Ascend960 enable-auto-vectorize-v2=true disable-hfusion-vectorize=false" --hivm-v2-pipeline="target=Ascend960" --mlir-print-ir-before=hivm-vectorize --mlir-print-ir-before=hfusion-auto-vectorize-v2 --mlir-print-ir-before=hfusion-select-hivm-vectorize -o /dev/null 2>&1 | FileCheck %s --check-prefix=ORDER
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 hivm-sync-solver-version=v1 enable-hivm-unit-flag-sync=true})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=SYNC-OPTIONS
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 disable-hivm-auto-inject-sync=true})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=NO-INTRA --implicit-check-not=hivm-graph-sync-solver
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 disable-auto-inject-block-sync=true})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=NO-CROSS --implicit-check-not=hivm-cross-core-gss
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 disable-hivm-auto-inject-sync=true disable-auto-inject-block-sync=true})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=NO-SYNC --implicit-check-not=hivm-graph-sync-solver --implicit-check-not=hivm-cross-core-gss
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 enable-hivm-inject-barrier-all-sync=true enable-hivm-unit-flag-sync=true})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=BARRIER-ALL --implicit-check-not=hivm-graph-sync-solver --implicit-check-not=hivm-inject-block-sync
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 enable-hivm-inject-block-all-sync=true})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=BLOCK-ALL --implicit-check-not=hivm-inject-block-sync --implicit-check-not=hivm-inject-sync
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960 disable-hivm-auto-inject-sync=true disable-auto-inject-block-sync=true enable-hivm-inject-barrier-all-sync=true enable-hivm-inject-block-all-sync=true})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s --check-prefix=NO-SYNC --implicit-check-not=hivm-graph-sync-solver --implicit-check-not=hivm-cross-core-gss --implicit-check-not=hivm-inject-sync --implicit-check-not=hivm-inject-block-sync
// The driver may fail later if template bitcode is unavailable. Check the v2
// sync stages before backend compilation, as in a5-code-motion-before-cross-core.
// RUN: (bishengir-compile %s --target=Ascend960 --use-pipeline-version=2 --enable-lir-compile=false --mlir-disable-threading --enable-hivm-graph-sync-solver=false --enable-hivm-cross-core-gss=false --enable-hivm-delayed-cross-core-gss=true --mlir-print-ir-before=hivm-cross-core-gss,hivm-graph-sync-solver -o %t.ll 2>&1 || true) | FileCheck %s --check-prefix=DRIVER-GSS
// RUN: (bishengir-compile %s --target=Ascend960 --use-pipeline-version=2 --enable-lir-compile=false --mlir-disable-threading --disable-hivm-auto-inject-sync=true --mlir-print-ir-before=hivm-cross-core-gss,hivm-graph-sync-solver,hivm-v2-split-kernel -o %t.ll 2>&1 || true) | FileCheck %s --check-prefix=DRIVER-NO-INTRA --implicit-check-not="IR Dump Before GraphSyncSolver"
// RUN: (bishengir-compile %s --target=Ascend960 --use-pipeline-version=2 --enable-lir-compile=false --mlir-disable-threading --disable-auto-inject-block-sync=true --mlir-print-ir-before=hivm-cross-core-gss,hivm-graph-sync-solver,hivm-v2-split-kernel -o %t.ll 2>&1 || true) | FileCheck %s --check-prefix=DRIVER-NO-CROSS --implicit-check-not="IR Dump Before CrossCoreGSS"
// RUN: (bishengir-compile %s --target=Ascend960 --use-pipeline-version=2 --enable-lir-compile=false --mlir-disable-threading --enable-hivm-inject-barrier-all-sync=true --enable-hivm-inject-block-all-sync=true --mlir-print-ir-before=hivm-inject-block-sync,hivm-inject-sync,hivm-cross-core-gss,hivm-graph-sync-solver -o %t.ll 2>&1 || true) | FileCheck %s --check-prefix=DRIVER-ALL --implicit-check-not="IR Dump Before InjectBlockSync" --implicit-check-not="IR Dump Before GraphSyncSolver"

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
// FULL: hivm-vectorize{phase=tensor}
// FULL-NEXT: canonicalize
// FULL-NOT: hfusion-merge-vf
// FULL: hfusion-simplify-vf-arg
// FULL: tensor-copy-insertion
// FULL: bufferize-module-op
// FULL: hivm-cross-core-gss
// FULL-SAME: solver-version=v2
// FULL: hivm-v2-split-kernel
// FULL: hivm-graph-sync-solver
// FULL-SAME: enable-unit-flag=false
// FULL-SAME: solver-version=v2
// FULL: hivm-vectorize{phase=tail}

// The same solver version reaches both solves, with unit flags on the per-core
// solver. Each disable flag affects only its own solve.
// SYNC-OPTIONS: hivm-cross-core-gss
// SYNC-OPTIONS-SAME: force-is-reg-based=true
// SYNC-OPTIONS-SAME: solver-version=v1
// SYNC-OPTIONS: hivm-v2-split-kernel
// SYNC-OPTIONS: hivm-graph-sync-solver
// SYNC-OPTIONS-SAME: enable-unit-flag=true
// SYNC-OPTIONS-SAME: solver-version=v1
// NO-INTRA: hivm-cross-core-gss
// NO-INTRA: hivm-v2-split-kernel
// NO-CROSS: hivm-v2-split-kernel
// NO-CROSS: hivm-graph-sync-solver
// NO-SYNC: hivm-v2-split-kernel

// Barrier-all selects InjectSync; block-all configures CrossCoreGSS.
// The disable flags take precedence over both all-sync modes.
// BARRIER-ALL: hivm-cross-core-gss
// BARRIER-ALL: hivm-v2-split-kernel
// BARRIER-ALL: hivm-inject-sync
// BARRIER-ALL-SAME: enable-unit-flag=true
// BARRIER-ALL-SAME: sync-mode=barrier-all
// BLOCK-ALL: hivm-cross-core-gss
// BLOCK-ALL-SAME: block-all-sync=true
// BLOCK-ALL: hivm-v2-split-kernel
// BLOCK-ALL: hivm-graph-sync-solver

// Driver algorithm-selection flags do not restore inject or delayed modes.
// DRIVER-GSS: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)
// DRIVER-GSS: IR Dump Before GraphSyncSolver (hivm-graph-sync-solver)
// DRIVER-NO-INTRA: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)
// DRIVER-NO-INTRA: IR Dump Before V2SplitKernel (hivm-v2-split-kernel)
// DRIVER-NO-CROSS: IR Dump Before V2SplitKernel (hivm-v2-split-kernel)
// DRIVER-NO-CROSS: IR Dump Before GraphSyncSolver (hivm-graph-sync-solver)
// DRIVER-ALL: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)
// DRIVER-ALL: IR Dump Before InjectSync (hivm-inject-sync)

// NO-MERGE: hivm-vectorize
// NO-MERGE: hfusion-simplify-vf-arg

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
