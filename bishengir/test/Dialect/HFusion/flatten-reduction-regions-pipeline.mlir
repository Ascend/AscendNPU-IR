// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="enable-triton-kernel-compile target=Ascend950PR_9579" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="enable-triton-kernel-compile target=Ascend950PR_9579 enable-flatten=false" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="enable-triton-kernel-compile target=Ascend950PR_9579 enable-auto-vectorize-v2=false" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: bishengir-opt %s --lower-hfusion-regbase-pipeline="enable-triton-kernel-compile target=Ascend950PR_9579 enable-mixed-cv=true" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONE

// The non-Triton path is not dumped: the default OpFusion fusion mode cannot be
// printed there. The pass is gated on SIMD VF fusion, which requires Triton.

// Preserve Tidy and its global legality checks. The local pass runs only after
// payload fusion and i1 preparation, with canonicalization before V2 consumes
// the new domains.
// ENABLED: hfusion-flatten-ops{{\{[^}]*}}flatten-mode=tidy
// ENABLED-SAME: vf-fusion{
// ENABLED-SAME: hfusion-flatten-ops{{\{[^}]*}}flatten-mode=tidy
// ENABLED-SAME: hfusion-pre-vectorization-fusion
// ENABLED-SAME: hfusion-prepare-i1-nx1-for-vectorization
// ENABLED-SAME: hfusion-flatten-reduction-regions
// ENABLED-SAME: canonicalize
// ENABLED-SAME: hfusion-auto-vectorize-v2
// DISABLED-NOT: hfusion-flatten-reduction-regions
// DISABLED: hfusion-pre-vectorization-fusion
// DISABLED-NOT: hfusion-flatten-reduction-regions

// MixedCV does not run the RegBase auto-vectorization pipeline at all.
// NONE-NOT: hfusion-flatten-reduction-regions

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {}
