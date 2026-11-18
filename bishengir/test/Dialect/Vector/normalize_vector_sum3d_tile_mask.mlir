// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hacc-append-device-spec -normalize-vector \
// RUN: -convert-vector-to-hivmave -split-input-file | FileCheck %s --check-prefix=AVE

// Regression for the `triton_sum_3D_dim1` failure reported against !4100.
//
// The kernel reduces a 7x17x15 f32 tensor along dim 1.  Inside the fused
// loop nest the tiling emits, per iteration, a mask whose two outer dims have
// extent 1 but a *dynamic* bound:
//
//   %b0 = select(7 - %i < 1, 7 - %i, 1)     // clamped to [.., 1]
//   %b1 = select(17 - %j < 1, 17 - %j, 1)
//   %m  = vector.create_mask %b0, %b1, %c15 : vector<1x1x64xi1>
//
// That mask *is* a row-major prefix: the outer dims hold a single slice each,
// so the active lanes are 0..14 whatever the bounds evaluate to.  The guard
// added by !4100 nevertheless demanded a statically-known bound of 1 and
// rejected it, leaving a rank-3 `create_mask` for
// -convert-vector-to-hivmave, which only accepts rank-1 masks and aborted
// with "mask should be a 1-D vector.".
//
// Whether the bounds fold to constants depends on OptimizeScalarTransfers,
// which bails out on `scf` control flow -- so the loop nest and the
// affine.apply/cmpi/select clamp chain below are both load-bearing here and
// must not be simplified away.

#map0 = affine_map<()[s0] -> (-s0 + 7)>
#map1 = affine_map<()[s0] -> (-s0 + 17)>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @sum_3d_dim1_tile_mask
  // The rank-3 mask must be flattened, with each unit-extent bound clamped
  // into the length so a zero bound still yields an empty mask.
  // NORMALIZE: scf.for
  // NORMALIZE: scf.for
  // NORMALIZE: scf.for
  // NORMALIZE-DAG: arith.maxsi
  // NORMALIZE-DAG: arith.minsi
  // NORMALIZE: vector.create_mask %{{.*}} : vector<64xi1>
  // NORMALIZE-NOT: vector.create_mask %{{.*}}, %{{.*}}, %{{.*}} : vector<1x1x64xi1>

  // AVE-LABEL: func.func @sum_3d_dim1_tile_mask
  // AVE-NOT: vector.create_mask
  // AVE: ave.hir.plt
  func.func @sum_3d_dim1_tile_mask(
      %src: memref<7x17x15xf32, #hivm.address_space<ub>>,
      %dst: memref<7x17x15xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c15 = arith.constant 15 : index
    %c17 = arith.constant 17 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.for %arg3 = %c0 to %c7 step %c1 {
      scf.for %arg4 = %c0 to %c7 step %c1 {
        scf.for %arg5 = %c0 to %c17 step %c1 {
          %0 = affine.apply #map0()[%arg4]
          %1 = arith.cmpi slt, %0, %c1 : index
          %2 = arith.select %1, %0, %c1 : index
          %3 = affine.apply #map1()[%arg5]
          %4 = arith.cmpi slt, %3, %c1 : index
          %5 = arith.select %4, %3, %c1 : index
          %6 = vector.create_mask %2, %5, %c15 : vector<1x1x64xi1>
          %7 = vector.transfer_read %src[%arg4, %arg5, %c0], %cst, %6
              : memref<7x17x15xf32, #hivm.address_space<ub>>, vector<1x1x64xf32>
          vector.transfer_write %7, %dst[%arg4, %arg5, %c0], %6
              : vector<1x1x64xf32>, memref<7x17x15xf32, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
}

// -----

// The same shape outside any loop: the clamp chain still produces dynamic
// bounds, so this pins the fix independently of the control flow that makes
// OptimizeScalarTransfers give up.

#map2 = affine_map<()[s0] -> (-s0 + 7)>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // NORMALIZE-LABEL: func.func @unit_extent_dynamic_bound_no_loop
  // NORMALIZE: vector.create_mask %{{.*}} : vector<64xi1>
  // The flattened mask is cast back to the read's shape on purpose, so only
  // the rank-3 `create_mask` itself must be gone.
  // NORMALIZE-NOT: vector.create_mask %{{.*}}, %{{.*}}, %{{.*}} : vector<1x1x64xi1>

  // AVE-LABEL: func.func @unit_extent_dynamic_bound_no_loop
  // AVE-NOT: vector.create_mask
  // AVE: ave.hir.plt
  func.func @unit_extent_dynamic_bound_no_loop(
      %src: memref<7x17x15xf32, #hivm.address_space<ub>>,
      %dst: memref<7x17x15xf32, #hivm.address_space<ub>>,
      %i: index)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.apply #map2()[%i]
    %1 = arith.cmpi slt, %0, %c1 : index
    %2 = arith.select %1, %0, %c1 : index
    %3 = vector.create_mask %2, %c1, %c15 : vector<1x1x64xi1>
    %4 = vector.transfer_read %src[%i, %c0, %c0], %cst, %3
        : memref<7x17x15xf32, #hivm.address_space<ub>>, vector<1x1x64xf32>
    vector.transfer_write %4, %dst[%i, %c0, %c0], %3
        : vector<1x1x64xf32>, memref<7x17x15xf32, #hivm.address_space<ub>>
    return
  }
}
