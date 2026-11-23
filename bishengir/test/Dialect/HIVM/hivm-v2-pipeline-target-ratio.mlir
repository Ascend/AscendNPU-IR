// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend960})" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=A6
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend950PR_9579})" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=A5

// The target spec states CubeCoreCount:VectorCoreCount; both current
// Ascend950 and Ascend960 specs reduce to 1:2, so the CV 1:2 stages are in
// the pipeline.
// A6: hivm-partition-and-bind-sub-block
// A6: hivm-sub-block-guard-cleanup
// A5: hivm-partition-and-bind-sub-block
// A5: hivm-sub-block-guard-cleanup

module {}
