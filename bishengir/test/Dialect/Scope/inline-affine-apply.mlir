// RUN: bishengir-opt %s --inline-scope --inline --split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: bishengir-opt %s --inline-scope='lower-affine-apply-for-inlining=true' --split-input-file | FileCheck %s --check-prefix=LOWER

// A helper called from SCF must inline without lowering affine in the caller.
// DEFAULT: func.func private @helper
// DEFAULT: affine.apply
// DEFAULT: func.func @caller
// DEFAULT: func.call @helper
// LOWER-NOT: func.func private @helper
// LOWER-LABEL: func.func @caller
// LOWER: scf.for
// LOWER-NOT: func.call
// LOWER: %[[OFFSET:.*]] = arith.addi
// LOWER: memref.store %{{.*}}, %{{.*}}[%[[OFFSET]]]
// LOWER: %[[CALLER_OFFSET:.*]] = affine.apply
// LOWER: memref.load %{{.*}}[%[[CALLER_OFFSET]]]
module {
  func.func private @helper(%offset: index) -> memref<16xf32> {
    %one = arith.constant 1.0 : f32
    %buffer = memref.alloc() : memref<16xf32>
    %next = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%offset]
    memref.store %one, %buffer[%next] : memref<16xf32>
    return %buffer : memref<16xf32>
  }
  func.func @caller(%output: memref<16xf32>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %upper step %c1 {
      %buffer = func.call @helper(%i) : (index) -> memref<16xf32>
      %next = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%i]
      %value = memref.load %buffer[%next] : memref<16xf32>
      memref.store %value, %output[%i] : memref<16xf32>
    }
    return
  }
}

// -----

// Ordinary inlining must preserve affine operations where it already works.
// DEFAULT-NOT: func.func private @helper
// DEFAULT-LABEL: func.func @ordinary_caller
// DEFAULT: affine.apply
// LOWER-NOT: func.func private @helper
// LOWER-LABEL: func.func @ordinary_caller
// LOWER: affine.apply
module {
  func.func private @helper(%offset: index) -> index {
    %next = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%offset]
    return %next : index
  }
  func.func @ordinary_caller(%offset: index) -> index {
    %next = func.call @helper(%offset) : (index) -> index
    return %next : index
  }
}

// -----

// Keep affine addressing and loops in functions explicitly excluded from
// inlining, including call-site exclusions.
// DEFAULT-LABEL: func.func private @vf
// DEFAULT: affine.apply
// DEFAULT: affine.for
// DEFAULT-LABEL: func.func private @call_no_inline
// DEFAULT: affine.apply
// DEFAULT-LABEL: func.func @no_inline_caller
// LOWER-LABEL: func.func private @vf
// LOWER: affine.apply
// LOWER: affine.for
// LOWER-LABEL: func.func private @call_no_inline
// LOWER: affine.apply
// LOWER-LABEL: func.func @no_inline_caller
// LOWER: func.call @vf
// LOWER: func.call @call_no_inline
module {
  func.func private @vf(%offset: index, %output: memref<32xindex>)
      attributes {no_inline} {
    %next = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%offset]
    affine.for %i = 0 to 16 {
      memref.store %next, %output[%i] : memref<32xindex>
    }
    return
  }
  func.func private @call_no_inline(%offset: index) -> index {
    %next = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%offset]
    return %next : index
  }
  func.func @no_inline_caller(%output: memref<32xindex>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %upper step %c1 {
      func.call @vf(%i, %output) : (index, memref<32xindex>) -> ()
      %next = func.call @call_no_inline(%i) {no_inline} : (index) -> index
      memref.store %next, %output[%i] : memref<32xindex>
    }
    return
  }
}
