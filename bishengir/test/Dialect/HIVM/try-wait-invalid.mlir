// RUN: bishengir-opt %s -verify-diagnostics -split-input-file

module {
  func.func @out_of_range_id() -> i64 {
    // expected-error @+1 {{static flag ID must fit the 5-bit Xt field (0..31)}}
    %counter = hivm.hir.try_wait flag = 32, sync_mode = <intra_block> : i64
    return %counter : i64
  }
}

// -----
func.func @negative_flag() -> i64 {
  // expected-error @+1 {{static flag ID must fit the 5-bit Xt field (0..31)}}
  %v = hivm.hir.try_wait flag = -1, sync_mode = <intra_block> : i64
  return %v : i64
}

// -----
func.func @missing_flag() -> i64 {
  // expected-error @+1 {{flag ID is needed}}
  %v = "hivm.hir.try_wait"() {sync_mode = #hivm.try_wait_sync_mode<intra_block>} : () -> i64
  return %v : i64
}

// -----
func.func @two_flags(%flag: i64) -> i64 {
  // expected-error @+1 {{only one flag ID is supported}}
  %v = "hivm.hir.try_wait"(%flag) {static_flag_id = 0 : i64, sync_mode = #hivm.try_wait_sync_mode<intra_block>} : (i64) -> i64
  return %v : i64
}

// -----
func.func @wide_static_flag() -> i64 {
  // expected-error @+1 {{static flag ID must fit the 5-bit Xt field (0..31)}}
  %v = "hivm.hir.try_wait"() {static_flag_id = 18446744073709551616 : i128, sync_mode = #hivm.try_wait_sync_mode<intra_block>} : () -> i64
  return %v : i64
}
