// RUN: bishengir-opt %s --convert-vector-to-hivmave | FileCheck %s
 
// CHECK-LABEL: func.func {{.*}}@_attn_fwd_scope_2
module attributes {hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func private @_attn_fwd_scope_2(%arg0: i32 {hivm.constant_value = 0 : i64}, %arg1: i32 {hivm.constant_value = 64 : i64}, %arg2: i32 {hivm.constant_value = 1 : i64}, %arg3: memref<64x128xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64x128xf32, #hivm.address_space<ub>>, %arg6: f32, %arg7: memref<64xf32, #hivm.address_space<ub>>, %arg8: memref<1xf32, #hivm.address_space<ub>>, %arg9: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, noinline, outline = true, vector_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
 
    // CHECK: scf.for
    scf.for %arg10 = %arg0 to %arg1 step %arg2  : i32 {
      %3 = arith.index_cast %arg10 : i32 to index
      %subview = memref.subview %arg5[%3, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg5[%3, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %4 = vector.transfer_read %subview_1[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
      %5 = vector.broadcast %arg6 : f32 to vector<64xf32>
      %6 = arith.mulf %4, %5 : vector<64xf32>
      %subview_2 = memref.subview %arg3[%3, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      vector.transfer_write %6, %subview_3[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %7 = vector.transfer_read %subview_4[%c0], %cst {in_bounds = [true]} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
      %8 = arith.mulf %7, %5 : vector<64xf32>
      %subview_5 = memref.subview %arg3[%3, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %subview_5[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      vector.transfer_write %8, %subview_6[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %arg8[0] [1] [1] : memref<1xf32, #hivm.address_space<ub>> to memref<f32, #hivm.address_space<ub>>
      %9 = vector.transfer_read %subview_7[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
      %10 = vector.shape_cast %9 : vector<f32> to vector<1xf32>
      %11 = arith.maximumf %6, %8 : vector<64xf32>
      %12 = builtin.unrealized_conversion_cast %10 : vector<1xf32> to f32
      %13 = vector.reduction <maximumf>, %11, %12 : vector<64xf32> into f32
      %14 = builtin.unrealized_conversion_cast %13 : f32 to vector<1xf32>
      %subview_8 = memref.subview %arg4[%3] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_9 = memref.subview %subview_8[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
      
      %15 = vector.shape_cast %14 : vector<1xf32> to vector<f32>
      vector.transfer_write %15, %subview_9[] : vector<f32>, memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
 
      // CHECK:      %[[MASK:.*]] = ave.hir.pge <ALL> : vector<1xi1>
      // CHECK-NEXT: ave.hir.masked_store <ONEPT_B32> %{{.*}}[], %[[MASK]], %{{.*}} {{.*}}{{{.*}}hivm.is_continuous{{.*}}}
    }
    %0 = vector.transfer_read %arg7[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %arg4[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = arith.maximumf %0, %1 : vector<64xf32>
    vector.transfer_write %2, %arg9[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}