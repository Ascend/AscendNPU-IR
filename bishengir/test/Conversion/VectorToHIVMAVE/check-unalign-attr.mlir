// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @triton_sum_3D_dim0_outlined_vf_1
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @triton_sum_3D_dim0_outlined_vf_1(%arg0: memref<3x514xi32, #hivm.address_space<ub>>, %arg1: memref<514xi32, #hivm.address_space<ub>>, %arg2: memref<256xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<0> : vector<64xi32>
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %c514 = arith.constant 514 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c514 step %c64 {
      %0 = affine.min affine_map<(d0) -> (-d0 + 514, 64)>(%arg3)
      %subview = memref.subview %arg1[%arg3] [%0] [1] : memref<514xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %view = memref.view %arg2[%c0][%0] : memref<256xi8, #hivm.address_space<ub>> to memref<1x?xi32, #hivm.address_space<ub>>
      %1 = vector.create_mask %0 : vector<64xi1>
      annotation.mark %1 {mask_op_idx = 0 : i32} : vector<64xi1>
      %subview_0 = memref.subview %view[0, 0] [1, %0] [1, 1] : memref<1x?xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      vector.transfer_write %cst, %subview_0[%c0], %1 {in_bounds = [true]} : vector<64xi32>, memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %view[0, 0] [1, %0] [1, 1] : memref<1x?xi32, #hivm.address_space<ub>> to memref<1x?xi32, strided<[?, 1]>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_1[0, 0] [1, %0] [1, 1] : memref<1x?xi32, strided<[?, 1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %2 = vector.transfer_read %subview_2[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi32>
      annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %3 = vector.shape_cast %2 : vector<64xi32> to vector<1x64xi32>
      annotation.mark %3 {reached_mask_ops_idx = 0 : i32} : vector<1x64xi32>
      %4:2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %3, %arg6 = %2) -> (vector<1x64xi32>, vector<64xi32>) {
        %subview_3 = memref.subview %arg0[%arg4, %arg3] [1, %0] [1, 1] : memref<3x514xi32, #hivm.address_space<ub>> to memref<1x?xi32, strided<[514, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_4 = memref.subview %subview_3[0, 0] [1, %0] [1, 1] : memref<1x?xi32, strided<[514, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: %[[LOAD:.*]] = ave.hir.vload <NORM> %{{.*}}[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access}
        %9 = vector.transfer_read %subview_4[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi32>
        %10 = arith.addi %9, %arg6 : vector<64xi32>
        %11 = vector.shape_cast %10 : vector<64xi32> to vector<1x64xi32>
        scf.yield %11, %10 : vector<1x64xi32>, vector<64xi32>
      }
      annotation.mark %4#1 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      vector.transfer_write %4#1, %subview_2[%c0], %1 {in_bounds = [true]} : vector<64xi32>, memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview_0[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi32>
      annotation.mark %5 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %6 = vector.transfer_read %subview[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>
      annotation.mark %6 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %7 = arith.addi %6, %5 : vector<64xi32>
      annotation.mark %7 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %8 = arith.select %1, %7, %5 : vector<64xi1>, vector<64xi32>
      annotation.mark %8 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      vector.transfer_write %8, %subview[%c0], %1 {in_bounds = [true]} : vector<64xi32>, memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_for_iv_offset
  func.func @aligned_for_iv_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c128 step %c64 {
        %subview = memref.subview %arg0[%i, %j] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %0 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %0, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_constant_offset
  func.func @aligned_affine_constant_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      %0 = affine.apply affine_map<() -> (64)>()
      %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
      %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
      vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_dim_offset
  func.func @aligned_affine_dim_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c128 step %c64 {
        %0 = affine.apply affine_map<(d0) -> (d0)>(%j)
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_symbol_offset
  func.func @aligned_affine_symbol_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c128 step %c64 {
        %0 = affine.apply affine_map<()[s0] -> (s0)>()[%j]
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_add_offset
  func.func @aligned_affine_add_offset(%arg0: memref<64x256xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c128 step %c64 {
        %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%j)[%c64]
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_sub_offset
  func.func @aligned_affine_sub_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c64 to %c128 step %c64 {
        %0 = affine.apply affine_map<(d0)[s0] -> (d0 - s0)>(%j)[%c64]
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_mul_offset
  func.func @aligned_affine_mul_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c8 step %c4 {
        %0 = affine.apply affine_map<(d0) -> (d0 * 16)>(%j)
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_floordiv_offset
  func.func @aligned_affine_floordiv_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c256 step %c128 {
        %0 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%j)
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_ceildiv_offset
  func.func @aligned_affine_ceildiv_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c256 step %c128 {
        %0 = affine.apply affine_map<(d0) -> (d0 ceildiv 2)>(%j)
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_affine_mod_offset
  func.func @aligned_affine_mod_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c128 step %c64 {
        %0 = affine.apply affine_map<(d0) -> (d0 mod 64)>(%j)
        %subview = memref.subview %arg0[%i, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @aligned_nested_affine_apply_offset
  func.func @aligned_nested_affine_apply_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c8 step %c4 {
        %0 = affine.apply affine_map<(d0) -> (d0 * 16)>(%j)
        %1 = affine.apply affine_map<(d0) -> (d0)>(%0)
        %subview = memref.subview %arg0[%i, %1] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] : memref<64xf32
        %2 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
        vector.transfer_write %2, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      }
    }
    return
  }

  // CHECK-LABEL: func.func @unaligned_view_byte_shift
  func.func @unaligned_view_byte_shift(%arg0: memref<512xi8, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %view = memref.view %arg0[%c8][] : memref<512xi8, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xf32
    %0 = vector.transfer_read %view[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  // CHECK-LABEL: func.func @unaligned_affine_iter_arg_offset
  func.func @unaligned_affine_iter_arg_offset(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %res = scf.for %i = %c0 to %c2 step %c1 iter_args(%off = %c1) -> (index) {
      %0 = affine.apply affine_map<(d0) -> (d0)>(%off)
      %subview = memref.subview %arg0[0, %0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      // CHECK: ave.hir.vload <NORM> %{{[[:alnum:]_]+}}[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xf32
      %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
      vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
      scf.yield %off : index
    }
    return
  }
}
