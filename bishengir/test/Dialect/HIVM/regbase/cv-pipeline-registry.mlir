// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9579 --lower-hivm-pipeline="target=Ascend950PR_9579 enable-mixed-cv=true enable-preload=true set-workspace-multibuffer=2 set-cv-pipeline-mode=skew" --mlir-disable-threading --mlir-print-ir-before=cv-pipelining,normalize-workspace-store,hivm-reposition-preload-work,hivm-merge-same-preload-scopes -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9579 --lower-hivm-pipeline="target=Ascend950PR_9579 enable-mixed-cv=true enable-preload=true set-workspace-multibuffer=2 set-cv-pipeline-mode=skew use-pipelinev2=true" --mlir-disable-threading --mlir-print-ir-before=cv-pipelining,normalize-workspace-store,hivm-reposition-preload-work,hivm-merge-same-preload-scopes -o /dev/null 2>&1 | FileCheck %s --check-prefix=V2 --implicit-check-not=cv-pipelining --implicit-check-not=hivm-reposition-preload-work --implicit-check-not=hivm-merge-same-preload-scopes
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9579 --lower-hivm-pipeline="target=Ascend950PR_9579 enable-mixed-cv=false enable-preload=true set-workspace-multibuffer=2 set-cv-pipeline-mode=off use-pipelinev2=true" --mlir-disable-threading --mlir-print-ir-before=cv-pipelining,normalize-workspace-store,hivm-reposition-preload-work,hivm-merge-same-preload-scopes -o /dev/null 2>&1 | FileCheck %s --check-prefix=V2 --implicit-check-not=cv-pipelining --implicit-check-not=hivm-reposition-preload-work --implicit-check-not=hivm-merge-same-preload-scopes
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9579 --lower-hivm-pipeline="target=Ascend950PR_9579 enable-mixed-cv=true enable-preload=false set-workspace-multibuffer=2 set-cv-pipeline-mode=skew" --mlir-disable-threading --mlir-print-ir-before=cv-pipelining,normalize-workspace-store,hivm-reposition-preload-work,hivm-merge-same-preload-scopes -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-PRELOAD --implicit-check-not=normalize-workspace-store --implicit-check-not=hivm-reposition-preload-work --implicit-check-not=hivm-merge-same-preload-scopes

// Select the route per function, including decorated registry names. A
// registered function defaults to v1 without a v2 fallback. Explicit v2 takes
// precedence for both functions and skips the legacy reposition/merge passes.
// Unregistered functions retain the legacy-then-v2 fallback by default, but
// NormalizeWorkspaceStore stays off because CVPipelining already ran.
// DEFAULT: IR Dump Before {{.*}}(cv-pipelining)
// DEFAULT: func.func @kernel_sdpa_bwd_q_decorated
// DEFAULT-NOT: (normalize-workspace-store)
// DEFAULT: IR Dump Before {{.*}}(cv-pipelining)
// DEFAULT: func.func @unregistered_kernel
// DEFAULT-NOT: (normalize-workspace-store)
// DEFAULT: IR Dump Before {{.*}}(hivm-reposition-preload-work)
// DEFAULT: IR Dump Before {{.*}}(hivm-merge-same-preload-scopes)
// DEFAULT-NOT: IR Dump Before

// V2: IR Dump Before {{.*}}(normalize-workspace-store)
// V2: func.func @kernel_sdpa_bwd_q_decorated
// V2: IR Dump Before {{.*}}(normalize-workspace-store)
// V2: func.func @unregistered_kernel
// V2-NOT: IR Dump Before

// NO-PRELOAD-COUNT-2: IR Dump Before {{.*}}(cv-pipelining)
// NO-PRELOAD-NOT: IR Dump Before

module {
  func.func @kernel_sdpa_bwd_q_decorated() attributes {hacc.entry} {
    return
  }
  func.func @unregistered_kernel() attributes {hacc.entry} {
    return
  }
}
