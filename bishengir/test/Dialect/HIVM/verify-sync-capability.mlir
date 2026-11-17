// RUN: bishengir-opt --hivm-verify-capability --split-input-file --verify-diagnostics %s

module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @native_c310(%flag: i64) -> i64 {
    %0 = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    hivm.hir.get_buf[<PIPE_V>, 3, false]
    hivm.hir.rls_buf[<PIPE_V>, 3, false]
    return %0 : i64
  }
}

// -----

// dav-920r1 is not surveyed for the sync intrinsics: the confirmed rows
// stay open under the forward-compatibility policy.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @native_920r1(%flag: i64) -> i64 {
    %0 = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    hivm.hir.get_buf[<PIPE_V>, 3, false]
    hivm.hir.rls_buf[<PIPE_V>, 3, false]
    return %0 : i64
  }
}

// -----

// Memory-based architectures keep surveyed empty coverages: Unknown like
// the old gates. The coverage exists, so it reports partial, not none.
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @unknown_c220(%flag: i64) -> i64 {
    // expected-error @below {{result is unknown; target is Ascend910B1 (dav-c220); coverage is partial}}
    %0 = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    // expected-error @below {{result is unknown; target is Ascend910B1 (dav-c220); coverage is partial}}
    hivm.hir.get_buf[<PIPE_V>, 3, false]
    // expected-error @below {{result is unknown; target is Ascend910B1 (dav-c220); coverage is partial}}
    hivm.hir.rls_buf[<PIPE_V>, 3, false]
    return %0 : i64
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend310B1">} {
  func.func @surveyed_empty_m300(%flag: i64) -> i64 {
    // expected-error @below {{result is unknown; target is Ascend310B1 (dav-m300); coverage is partial}}
    %0 = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    return %0 : i64
  }
}

// -----

module {
  func.func @missing_target(%flag: i64) -> i64 {
    // expected-error @below {{no #hacc.target on this operation or an enclosing module; this check does not assume Ascend910B}}
    %0 = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    // expected-error @below {{no #hacc.target on this operation or an enclosing module; this check does not assume Ascend910B}}
    hivm.hir.get_buf[<PIPE_V>, 3, false]
    return %0 : i64
  }
}
