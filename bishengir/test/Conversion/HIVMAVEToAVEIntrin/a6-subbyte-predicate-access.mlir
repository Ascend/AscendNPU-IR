// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-hivmave-to-ave-intrin -split-input-file -verify-diagnostics %s

func.func @read_partial_byte(%src: memref<128xi1, #hivm.address_space<ub>>) -> vector<64xi1> {
  %four = arith.constant 4 : index
  // expected-error@+1 {{packed predicate access starts inside a byte}}
  %data = ave.hir.vload <NORM> %src[%four] {functionType = #ave.func_dist_type<pb8>} : memref<128xi1, #hivm.address_space<ub>> into vector<64xi1>
  return %data : vector<64xi1>
}

// -----

func.func @write_partial_byte(%dst: memref<128xi1, strided<[1], offset: 4>, #hivm.address_space<ub>>, %data: vector<64xi1>) {
  %zero = arith.constant 0 : index
  %mask = ave.hir.pge <VL8> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  // expected-error@+1 {{packed predicate access starts inside a byte}}
  ave.hir.masked_store <NORM_B8> %dst[%zero], %mask, %data {functionType = #ave.func_dist_type<pb32>} : memref<128xi1, strided<[1], offset: 4>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}

// -----

// A tag copied from one call cannot prove that every address starts inside a byte.
func.func @tagged_index(%dst: memref<128xi1, #hivm.address_space<ub>>, %index: index {hivm.constant_value = 4 : i64}, %data: vector<64xi1>) {
  %mask = ave.hir.pge <VL8> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %dst[%index], %mask, %data {functionType = #ave.func_dist_type<pb32>} : memref<128xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
