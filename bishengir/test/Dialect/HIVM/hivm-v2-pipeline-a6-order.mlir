// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | tr ',' '\n' | FileCheck %s

// The v2 pipeline reuses InsertLoadStoreForMixCV after TCB insertion.
// CHECK: hivm-insert-cv-tight-coupled-buffer
// CHECK: hivm-insert-load-store-for-mix-cv
// CHECK: hivm-mark-tightly-coupled-buffer
// CHECK: hivm-insert-fixpipe
// CHECK: hivm-inline-fixpipe
// CHECK-NOT: hivm-insert-cv-data-movement

module {}
