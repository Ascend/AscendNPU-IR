// Regression test for the tree-reduce-v2 accumulator merge shape.
//
// Origin: an A5 masked slice+sum kernel produced all-zero output -- the
// reduction VF collapsed the accumulator to a 1D tensor and indexed it
// linearly (y0*W + chunk), baking in a row-major-contiguous layout contract;
// when stride-align pads the accumulator buffer, the linear addressing no
// longer matches the strided view the GM copy reads and the padded rows are
// never written. The merge must keep the accumulator at its natural rank and
// delinearize the linear offset instead.

// RUN: bishengir-opt --tree-reduce-v2="enable-ar enable-ra ascend-style-ra=true" %s | FileCheck %s

#map = affine_map<(d0) -> (-d0 + 4, 64)>

func.func @reduce_2d_acc(%input: tensor<2x20x4xf32>, %acc: tensor<2x4xf32>) -> tensor<2x4xf32> attributes {hivm.vector_function, no_inline} {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %y0 = %c0 to %c2 step %c1 iter_args(%acc_a = %acc) -> (tensor<2x4xf32>) {
    %1 = scf.for %r2 = %c0 to %c20 step %c1 iter_args(%acc_b = %acc_a) -> (tensor<2x4xf32>) {
      %2 = scf.for %vc = %c0 to %c4 step %c64 iter_args(%acc_c = %acc_b) -> (tensor<2x4xf32>) {
        %3 = affine.min #map(%vc)
        %in_slice = tensor.extract_slice %input[%y0, %r2, %vc] [1, 1, %3] [1, 1, 1] : tensor<2x20x4xf32> to tensor<1x1x?xf32>
        %acc_slice = tensor.extract_slice %acc_c[%y0, %vc] [1, %3] [1, 1] : tensor<2x4xf32> to tensor<1x?xf32>
        %c1_1 = arith.constant 1 : index
        %c0_3 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %4 = vector.create_mask %c1_1, %c1_1, %3 : vector<1x1x64xi1>
        %5 = vector.mask %4 { vector.transfer_read %in_slice[%c0_3, %c0_3, %c0_3], %cst {in_bounds = [true, true, true]} : tensor<1x1x?xf32>, vector<1x1x64xf32> } : vector<1x1x64xi1> -> vector<1x1x64xf32>
        %6 = vector.create_mask %c1_1, %3 : vector<1x64xi1>
        %7 = vector.mask %6 { vector.transfer_read %acc_slice[%c0_3, %c0_3], %cst {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %8 = vector.mask %4 { vector.multi_reduction <add>, %5, %7 [1] : vector<1x1x64xf32> to vector<1x64xf32> } : vector<1x1x64xi1> -> vector<1x64xf32>
        %9 = vector.mask %6 { vector.transfer_write %8, %acc_slice[%c0_3, %c0_3] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted = tensor.insert_slice %9 into %acc_c[%y0, %vc] [1, %3] [1, 1] : tensor<1x?xf32> into tensor<2x4xf32>
        scf.yield %inserted : tensor<2x4xf32>
      }
      scf.yield %2 : tensor<2x4xf32>
    }
    scf.yield %1 : tensor<2x4xf32>
  }
  return %0 : tensor<2x4xf32>
}

// The accumulator stays 2D through the batched-RA merge: no collapse before
// the loops, no expand after, and the linear offset is split with div/rem.
// CHECK-LABEL: func.func @reduce_2d_acc
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<2x4xf32>, tensor<2x20x4xf32>)
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<2x4xf32>, tensor<2x20x4xf32>)
// CHECK: vector.transfer_read {{.*}} : tensor<2x4xf32>, vector<64xf32>
// CHECK: vector.transfer_write {{.*}} tensor<2x4xf32>
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape

// A staged long-AR final fold with a rank-2 accumulator (unit trailing dim):
// each outer index merges one scalar, so the merge must delinearize that
// index against the accumulator's natural shape and address the rank-2 slice
// with as many indices as its rank. A single-index transfer_write is rejected
// by the verifier.
#map1 = affine_map<(d0) -> (-d0 + 1, 64)>

func.func @reduce_ar_rank2_acc(%input: tensor<2x100x1xf32>, %acc: tensor<2x1xf32>) -> tensor<2x1xf32> attributes {hivm.vector_function, no_inline} {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %y0 = %c0 to %c2 step %c1 iter_args(%acc_a = %acc) -> (tensor<2x1xf32>) {
    %1 = scf.for %r2 = %c0 to %c100 step %c1 iter_args(%acc_b = %acc_a) -> (tensor<2x1xf32>) {
      %2 = scf.for %vc = %c0 to %c1 step %c64 iter_args(%acc_c = %acc_b) -> (tensor<2x1xf32>) {
        %3 = affine.min #map1(%vc)
        %in_slice = tensor.extract_slice %input[%y0, %r2, %vc] [1, 1, %3] [1, 1, 1] : tensor<2x100x1xf32> to tensor<1x1x?xf32>
        %acc_slice = tensor.extract_slice %acc_c[%y0, %vc] [1, %3] [1, 1] : tensor<2x1xf32> to tensor<1x?xf32>
        %c1_1 = arith.constant 1 : index
        %c0_3 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %4 = vector.create_mask %c1_1, %c1_1, %3 : vector<1x1x64xi1>
        %5 = vector.mask %4 { vector.transfer_read %in_slice[%c0_3, %c0_3, %c0_3], %cst {in_bounds = [true, true, true]} : tensor<1x1x?xf32>, vector<1x1x64xf32> } : vector<1x1x64xi1> -> vector<1x1x64xf32>
        %6 = vector.create_mask %c1_1, %3 : vector<1x64xi1>
        %7 = vector.mask %6 { vector.transfer_read %acc_slice[%c0_3, %c0_3], %cst {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %8 = vector.mask %4 { vector.multi_reduction <add>, %5, %7 [1] : vector<1x1x64xf32> to vector<1x64xf32> } : vector<1x1x64xi1> -> vector<1x64xf32>
        %9 = vector.mask %6 { vector.transfer_write %8, %acc_slice[%c0_3, %c0_3] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted = tensor.insert_slice %9 into %acc_c[%y0, %vc] [1, %3] [1, 1] : tensor<1x?xf32> into tensor<2x1xf32>
        scf.yield %inserted : tensor<2x1xf32>
      }
      scf.yield %2 : tensor<2x1xf32>
    }
    scf.yield %1 : tensor<2x1xf32>
  }
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: func.func @reduce_ar_rank2_acc
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<2x1xf32>) {
// CHECK: tensor.extract_slice {{.*}} : tensor<2x1xf32> to tensor<1x1xf32>
// CHECK: vector.transfer_read {{.*}}[%c{{[0-9_]+}}, %c{{[0-9_]+}}], {{.*}} : tensor<1x1xf32>, vector<1xf32>
// CHECK: vector.transfer_write {{.*}}[%c{{[0-9_]+}}, %c{{[0-9_]+}}] : vector<1xf32>, tensor<1x1xf32>
// CHECK: tensor.insert_slice {{.*}} : tensor<1x1xf32> into tensor<2x1xf32>

// Two non-unit dims after the reduction group (inner = 4*4 = 16 exceeds the
// accumulator's last dim): the merge vector cannot span the last dim, so the
// merge falls back to the flat linear form -- the accumulator is collapsed
// before the loops, merged through one linear offset, and expanded back.
func.func @reduce_ra_inner16_lastdim4(%input: tensor<20x4x4xf32>, %acc: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {hivm.vector_function, no_inline} {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %y0 = %c0 to %c4 step %c1 iter_args(%acc_a = %acc) -> (tensor<4x4xf32>) {
    %1 = scf.for %r2 = %c0 to %c20 step %c1 iter_args(%acc_b = %acc_a) -> (tensor<4x4xf32>) {
      %2 = scf.for %vc = %c0 to %c4 step %c64 iter_args(%acc_c = %acc_b) -> (tensor<4x4xf32>) {
        %3 = affine.min #map(%vc)
        %in_slice = tensor.extract_slice %input[%r2, %y0, %vc] [1, 1, %3] [1, 1, 1] : tensor<20x4x4xf32> to tensor<1x1x?xf32>
        %acc_slice = tensor.extract_slice %acc_c[%y0, %vc] [1, %3] [1, 1] : tensor<4x4xf32> to tensor<1x?xf32>
        %c1_1 = arith.constant 1 : index
        %c0_3 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %4 = vector.create_mask %c1_1, %c1_1, %3 : vector<1x1x64xi1>
        %5 = vector.mask %4 { vector.transfer_read %in_slice[%c0_3, %c0_3, %c0_3], %cst {in_bounds = [true, true, true]} : tensor<1x1x?xf32>, vector<1x1x64xf32> } : vector<1x1x64xi1> -> vector<1x1x64xf32>
        %6 = vector.create_mask %c1_1, %3 : vector<1x64xi1>
        %7 = vector.mask %6 { vector.transfer_read %acc_slice[%c0_3, %c0_3], %cst {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32> } : vector<1x64xi1> -> vector<1x64xf32>
        %8 = vector.mask %4 { vector.multi_reduction <add>, %5, %7 [0] : vector<1x1x64xf32> to vector<1x64xf32> } : vector<1x1x64xi1> -> vector<1x64xf32>
        %9 = vector.mask %6 { vector.transfer_write %8, %acc_slice[%c0_3, %c0_3] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32> } : vector<1x64xi1> -> tensor<1x?xf32>
        %inserted = tensor.insert_slice %9 into %acc_c[%y0, %vc] [1, %3] [1, 1] : tensor<1x?xf32> into tensor<4x4xf32>
        scf.yield %inserted : tensor<4x4xf32>
      }
      scf.yield %2 : tensor<4x4xf32>
    }
    scf.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @reduce_ra_inner16_lastdim4
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<4x4xf32>, tensor<20x16xf32>)
// CHECK: vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<20x16xf32>, vector<64xf32> }
// CHECK: vector.mask {{.*}} { vector.transfer_write {{.*}} : vector<64xf32>, tensor<20x16xf32> }
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK: vector.transfer_read {{.*}} : tensor<4x4xf32>, vector<64xf32>
// CHECK: vector.transfer_write {{.*}} tensor<4x4xf32>
// CHECK-NOT: tensor.collapse_shape
