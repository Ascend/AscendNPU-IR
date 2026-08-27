// RUN: bishengir-opt %s --tree-reduce-v2="enable-ra=true enable-ar=false only-marked=true direct-register-ra=true" | FileCheck %s

// The marked loop is produced only for canonical RA reductions selected by
// AutoVectorizeV2.  The restricted pass must keep all balanced-tree partials
// in vector SSA rather than materializing intermediate tensors.

// CHECK-LABEL: func.func @marked_register_tree(
// CHECK-NOT: vector.multi_reduction
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor<8x64xf32>
// CHECK-NOT: tensor<4x64xf32>
// CHECK: %[[R0:[0-9]+]] = vector.mask {{.*}}tensor<9x512xf32>, vector<64xf32>
// CHECK: %[[R8:[0-9]+]] = vector.mask {{.*}}tensor<9x512xf32>, vector<64xf32>
// CHECK: %[[PAIR0:[0-9]+]] = arith.addf %[[R8]], %[[R0]] : vector<64xf32>
// CHECK: arith.addf
// CHECK: vector.mask {{.*}} { vector.transfer_write
// CHECK: return

func.func @marked_register_tree(
    %input: tensor<9x512xf32>, %init: tensor<512xf32>)
    -> tensor<512xf32> attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  %0 = scf.for %a = %c0 to %c512 step %c64
      iter_args(%outer_acc = %init) -> tensor<512xf32> {
    %1 = scf.for %r = %c0 to %c9 step %c1
        iter_args(%inner_acc = %outer_acc) -> tensor<512xf32> {
      %input_slice = tensor.extract_slice %input[%r, %a] [1, 64] [1, 1]
          : tensor<9x512xf32> to tensor<1x64xf32>
      %acc_slice = tensor.extract_slice %inner_acc[%a] [64] [1]
          : tensor<512xf32> to tensor<64xf32>
      %zero = arith.constant 0.0 : f32
      %row = vector.transfer_read %input_slice[%c0, %c0], %zero
          : tensor<1x64xf32>, vector<1x64xf32>
      %acc = vector.transfer_read %acc_slice[%c0], %zero
          : tensor<64xf32>, vector<64xf32>
      %sum = vector.multi_reduction <add>, %row, %acc [0]
          : vector<1x64xf32> to vector<64xf32>
      %written = vector.transfer_write %sum, %acc_slice[%c0]
          : vector<64xf32>, tensor<64xf32>
      %next = tensor.insert_slice %written into %inner_acc[%a] [64] [1]
          : tensor<64xf32> into tensor<512xf32>
      scf.yield %next : tensor<512xf32>
    }
    scf.yield %1 : tensor<512xf32>
  } {hfusion.register_tree_reduction}
  return %0 : tensor<512xf32>
}

// The marker is a strict contract: an otherwise identical unmarked loop must
// not be consumed by this specialized post-vectorization pass.

// CHECK-LABEL: func.func @unmarked_register_tree(
// CHECK: vector.multi_reduction <add>
// CHECK: return

func.func @unmarked_register_tree(
    %input: tensor<9x512xf32>, %init: tensor<512xf32>)
    -> tensor<512xf32> attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  %0 = scf.for %a = %c0 to %c512 step %c64
      iter_args(%outer_acc = %init) -> tensor<512xf32> {
    %1 = scf.for %r = %c0 to %c9 step %c1
        iter_args(%inner_acc = %outer_acc) -> tensor<512xf32> {
      %input_slice = tensor.extract_slice %input[%r, %a] [1, 64] [1, 1]
          : tensor<9x512xf32> to tensor<1x64xf32>
      %acc_slice = tensor.extract_slice %inner_acc[%a] [64] [1]
          : tensor<512xf32> to tensor<64xf32>
      %zero = arith.constant 0.0 : f32
      %row = vector.transfer_read %input_slice[%c0, %c0], %zero
          : tensor<1x64xf32>, vector<1x64xf32>
      %acc = vector.transfer_read %acc_slice[%c0], %zero
          : tensor<64xf32>, vector<64xf32>
      %sum = vector.multi_reduction <add>, %row, %acc [0]
          : vector<1x64xf32> to vector<64xf32>
      %written = vector.transfer_write %sum, %acc_slice[%c0]
          : vector<64xf32>, tensor<64xf32>
      %next = tensor.insert_slice %written into %inner_acc[%a] [64] [1]
          : tensor<64xf32> into tensor<512xf32>
      scf.yield %next : tensor<512xf32>
    }
    scf.yield %1 : tensor<512xf32>
  }
  return %0 : tensor<512xf32>
}
