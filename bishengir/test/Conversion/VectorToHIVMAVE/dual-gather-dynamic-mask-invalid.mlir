// RUN: bishengir-opt %s -convert-vector-to-hivmave -split-input-file -verify-diagnostics

// Unsupported predicates must fail instead of being reused for both gathers.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_dynamic_mask(%src: memref<256xi8, #hivm.address_space<ub>>, %mask: vector<256xi1>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %even = arith.constant dense<0> : vector<128xi16>
    // expected-error @+2 {{dual B8 gather requires a constant mask}}
    // expected-error @+1 {{failed to legalize operation 'vector.gather' that was explicitly marked illegal}}
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
// Also reject a dynamic prefix after its producer becomes ave.hir.plt.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_gather_dynamic_prefix(%src: memref<256xi8, #hivm.address_space<ub>>, %extent: index) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %mask = vector.create_mask %extent : vector<256xi1>
    %even = arith.constant dense<0> : vector<128xi16>
    // expected-error @+2 {{dual B8 gather requires a constant mask}}
    // expected-error @+1 {{failed to legalize operation 'vector.gather' that was explicitly marked illegal}}
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // H depends on predicate layout and must not be guessed from vector length.
  func.func @layout_dependent_h(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %mask = ave.hir.pge <H> : vector<256xi1>
    %even = arith.constant dense<0> : vector<128xi16>
    // expected-error @+2 {{dual B8 gather requires a constant mask}}
    // expected-error @+1 {{failed to legalize operation 'vector.gather' that was explicitly marked illegal}}
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Unwrapping shape_cast must not treat a rank-2 producer as a 1-D prefix.
  func.func @multidimensional_source(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %original = arith.constant dense<false> : vector<16x16xi1>
    %mask = vector.shape_cast %original : vector<16x16xi1> to vector<256xi1>
    %even = arith.constant dense<0> : vector<128xi16>
    // expected-error @+2 {{dual B8 gather requires a constant mask}}
    // expected-error @+1 {{failed to legalize operation 'vector.gather' that was explicitly marked illegal}}
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // M3 is not supported by the constant-prefix analysis, even with a known width.
  func.func @unsupported_m3(%src: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %pass = arith.constant dense<0> : vector<256xi8>
    %mask = ave.hir.pge <M3> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
    %even = arith.constant dense<0> : vector<128xi16>
    // expected-error @+2 {{dual B8 gather requires a constant mask}}
    // expected-error @+1 {{failed to legalize operation 'vector.gather' that was explicitly marked illegal}}
    %res = vector.gather %src[%c0] [%even], %mask, %pass {secondary_index = dense<1> : vector<128xi16>} : memref<256xi8, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    return %res : vector<256xi8>
  }
}
