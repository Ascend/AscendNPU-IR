// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-vector-to-hivmave -split-input-file -verify-diagnostics

// 127 * 517 = 65659 wraps to 123 in a B16 scatter index.
func.func @halfword_overflow(%dst: memref<128xf16, strided<[517]>, #hivm.address_space<ub>>, %value: vector<128xf16>, %i: index) {
  // expected-error@+2 {{strided UB store element offset exceeds the unsigned 16-bit scatter index}}
  // expected-error@+1 {{failed to legalize operation 'vector.transfer_write'}}
  vector.transfer_write %value, %dst[%i] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, strided<[517]>, #hivm.address_space<ub>>
  return
}

// -----

// 63 * 68174085 = 4294967355 also exceeds the B32 index range.
func.func @word_overflow(%dst: memref<64xf32, strided<[68174085]>, #hivm.address_space<ub>>, %value: vector<64xf32>, %i: index) {
  // expected-error@+2 {{strided UB store element offset exceeds the unsigned 32-bit scatter index}}
  // expected-error@+1 {{failed to legalize operation 'vector.transfer_write'}}
  vector.transfer_write %value, %dst[%i] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[68174085]>, #hivm.address_space<ub>>
  return
}
