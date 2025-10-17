// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @scalar_brc(
// CHECK-SAME:                       %[[ARG0:.*]]: f32,
// CHECK-SAME:                       %[[ARG1:.*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    hivm.hir.vbrc ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : memref<224xf32>)
func.func @scalar_brc(%cst: f32, %arg: memref<32x7xf32>) {
  hivm.hir.vbrc ins(%cst: f32) outs(%arg: memref<32x7xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @scalar_brc(
// CHECK-SAME:                       %[[ARG0:.*]]: f32,
// CHECK-SAME:                       %[[ARG1:.*]]: memref<?x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    hivm.hir.vbrc ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : memref<?xf32>)
func.func @scalar_brc(%cst: f32, %arg: memref<?x7xf32>) {
  hivm.hir.vbrc ins(%cst: f32) outs(%arg: memref<?x7xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @strided_brc
func.func @strided_brc(%cst: f32, %arg: memref<16x16xf32, strided<[16, 2], offset: 0>>) {
  // CHECK-NOT: memref.collapse_shape
  hivm.hir.vbrc ins(%cst: f32) outs(%arg: memref<16x16xf32, strided<[16, 2], offset: 0>>)
  return
}

// -----

// CHECK-LABEL: func.func @strided_brc_collapse_continuous
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1], [2, 3]] : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>> into memref<8x?x8xf32, strided<[?, ?, 1]>>
func.func @strided_brc_collapse_continuous(%cst: f32, %arg: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>) {
  hivm.hir.vbrc ins(%cst: f32) outs(%arg: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>)
  return
}

// -----

// CHECK-LABEL: func.func @scalar_brc_cannot_collapse_continuous
// CHECK-NOT: memref.collapse_shape
func.func @scalar_brc_cannot_collapse_continuous(%cst: f32, %arg: memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>) {
  hivm.hir.vbrc ins(%cst: f32) outs(%arg: memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>)
  return
}

// -----


// CHECK-LABEL: func.func @scalar_brc(
// CHECK-SAME:                       %[[ARG0:.*]]: f32,
// CHECK-SAME:                       %[[ARG1:.*]]: memref<?x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    hivm.hir.vbrc ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : memref<?xf32>)
func.func @scalar_brc(%cst: f32, %arg: memref<?x7xf32>) -> memref<?x7xf32> {
  hivm.hir.vbrc ins(%cst: f32) outs(%arg: memref<?x7xf32>)
  return %arg : memref<?x7xf32>
}

// -----

// CHECK-LABEL: func.func @scalar_brc_with_scf(
// CHECK-SAME:                                %[[ARG0:.*]]: i1,
// CHECK-SAME:                                %[[ARG1:.*]]: memref<4x16xf32>,
// CHECK-SAME:                                %[[ARG2:.*]]: memref<4x16xf32>
// CHECK-NEXT:    %[[CST:.*]] = arith.constant
// CHECK-NEXT:    scf.if %[[ARG0]] {
// CHECK-NEXT:      %[[FLATTENED_BRC:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:      hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[FLATTENED_BRC]] : memref<64xf32>)
// CHECK-NEXT:    }
// CHECK-DAG:     %[[FLATTENED1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-DAG:     %[[FLATTENED2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK:         hivm.hir.vadd ins(%[[FLATTENED1:.*]], %[[FLATTENED2:.*]] : memref<64xf32>, memref<64xf32>) outs(%[[FLATTENED1:.*]] : memref<64xf32>)
func.func @scalar_brc_with_scf(%arg0: i1, %arg1: memref<4x16xf32>, %arg2: memref<4x16xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  scf.if %arg0 {
    hivm.hir.vbrc ins(%cst : f32) outs(%arg1 : memref<4x16xf32>)
  }
  hivm.hir.vadd ins(%arg1, %arg2 : memref<4x16xf32>, memref<4x16xf32>) outs(%arg1 : memref<4x16xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @scalar_brc_before_add(
// CHECK-SAME:                                  %[[ARG0:.*]]: i1,
// CHECK-SAME:                                  %[[ARG1:.*]]: memref<4x16xf32>,
// CHECK-SAME:                                  %[[ARG2:.*]]: memref<4x16xf32>
// CHECK-NEXT:    %[[CST:.*]] = arith.constant
// CHECK-NEXT:    %[[FLATTENED_BRC:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[FLATTENED_BRC]] : memref<64xf32>)
// CHECK-DAG:     %[[FLATTENED1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-DAG:     %[[FLATTENED2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK:         hivm.hir.vadd ins(%[[FLATTENED1:.*]], %[[FLATTENED2:.*]] : memref<64xf32>, memref<64xf32>) outs(%[[FLATTENED1:.*]] : memref<64xf32>)
func.func @scalar_brc_before_add(%arg0: i1, %arg1: memref<4x16xf32>, %arg2: memref<4x16xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  hivm.hir.vbrc ins(%cst : f32) outs(%arg1 : memref<4x16xf32>)
  hivm.hir.vadd ins(%arg1, %arg2 : memref<4x16xf32>, memref<4x16xf32>) outs(%arg1 : memref<4x16xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @hivm_vadd_with_scf(
// CHECK-SAME:                               %[[ARG0:.*]]: i1,
// CHECK-SAME:                               %[[ARG1:.*]]: memref<4x16xf32>,
// CHECK-SAME:                               %[[ARG2:.*]]: memref<4x16xf32>
// CHECK-NEXT:    scf.if %[[ARG0]] {
// CHECK-DAG:       %[[FLATTENED1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-DAG:       %[[FLATTENED2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK:           hivm.hir.vadd ins(%[[FLATTENED1:.*]], %[[FLATTENED2:.*]] : memref<64xf32>, memref<64xf32>) outs(%[[FLATTENED1:.*]] : memref<64xf32>)
// CHECK:         }
// CHECK-DAG:     %[[FLATTENED4:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-DAG:     %[[FLATTENED5:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK:         hivm.hir.vadd ins(%[[FLATTENED4:.*]], %[[FLATTENED5:.*]] : memref<64xf32>, memref<64xf32>) outs(%[[FLATTENED4:.*]] : memref<64xf32>)
func.func @hivm_vadd_with_scf(%arg0: i1, %arg1: memref<4x16xf32>, %arg2: memref<4x16xf32>) {
  scf.if %arg0 {
    hivm.hir.vadd ins(%arg1, %arg2 : memref<4x16xf32>, memref<4x16xf32>) outs(%arg1 : memref<4x16xf32>)
  }
  hivm.hir.vadd ins(%arg1, %arg2 : memref<4x16xf32>, memref<4x16xf32>) outs(%arg1 : memref<4x16xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @hivm_vadd
// CHECK: memref.collapse_shape
// CHECK-SAME: into memref<224xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: into memref<224xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: into memref<224xf32>
// CHECK: return
// CHECK-SAME: 32x7xf32
func.func @hivm_vadd(%arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) -> memref<32x7xf32> {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<32x7xf32>, memref<32x7xf32>)
                outs(%arg2 : memref<32x7xf32>)
  return %arg2 : memref<32x7xf32>
}

// -----

// CHECK-LABEL: func.func @hivm_vadd_strided_collapse_continuous_0
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1], [2, 3]] : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>> into memref<8x?x8xf32, strided<[?, ?, 1]>>
// CHECK: hivm.hir.vadd {{.*}} memref<8x?x8xf32, strided<[?, ?, 1]>>
func.func @hivm_vadd_strided_collapse_continuous_0(%arg0: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>, %arg1: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>) {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>, memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>)
              outs(%arg1 : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>)
  return
}

// -----

// CHECK-LABEL: func.func @hivm_vadd_strided_collapse_continuous_1
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1], [2], [3, 4]] : memref<4x2x?x4x2xf32, strided<[?, ?, ?, 2, 1]>> into memref<4x2x?x8xf32, strided<[?, ?, ?, 1]>>
// CHECK: hivm.hir.vadd {{.*}} memref<4x2x?x8xf32, strided<[?, ?, ?, 1]>>
func.func @hivm_vadd_strided_collapse_continuous_1(%arg0: memref<4x2x?x4x2xf32, strided<[?, ?, ?, 2, 1]>>, %arg1: memref<4x2x?x4x2xf32, strided<[?, ?, ?, 2, 1]>>) {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<4x2x?x4x2xf32, strided<[?, ?, ?, 2, 1]>>, memref<4x2x?x4x2xf32, strided<[?, ?, ?, 2, 1]>>)
              outs(%arg1 : memref<4x2x?x4x2xf32, strided<[?, ?, ?, 2, 1]>>)
  return
}

// -----

// CHECK-LABEL: func.func @hivm_vadd_cannot_collapse_continuous
// CHECK-NOT: memref.collapse_shape
func.func @hivm_vadd_cannot_collapse_continuous(%arg0: memref<?x?xf32, strided<[16, 1]>>, %arg1: memref<?x?xf32, strided<[16, 1]>>) {
  hivm.hir.vadd ins(%arg0, %arg1 :memref<?x?xf32, strided<[16, 1]>>, memref<?x?xf32, strided<[16, 1]>>)
              outs(%arg1 : memref<?x?xf32, strided<[16, 1]>>)
  return
}

// -----

// CHECK-LABEL: func.func @hivm_vadd_collapse_continuous_no_stride
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2, 3, 4]] : memref<4x2x?x4x2xf32> into memref<?xf32>
// CHECK: hivm.hir.vadd {{.*}} memref<?xf32>
func.func @hivm_vadd_collapse_continuous_no_stride(%arg0: memref<4x2x?x4x2xf32>, %arg1: memref<4x2x?x4x2xf32>) {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<4x2x?x4x2xf32>, memref<4x2x?x4x2xf32>)
              outs(%arg1 : memref<4x2x?x4x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @hivm_cast
// CHECK: memref.collapse_shape
// CHECK-SAME: into memref<224xf16>
// CHECK: memref.collapse_shape
// CHECK-SAME: into memref<224xf32>
// CHECK: return
// CHECK-SAME: 32x7xf32
func.func @hivm_cast(%arg0: memref<32x7xf16>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) -> memref<32x7xf32> {
  hivm.hir.vcast ins(%arg0 : memref<32x7xf16>)
                outs(%arg2 : memref<32x7xf32>)
  return %arg2 : memref<32x7xf32>
}

// -----

// CHECK-LABEL: func.func @hivm_transpose
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x1x8xf32> into memref<16x8xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<8x1x16xf32> into memref<8x16xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<1x8x16xf32> into memref<8x16xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<1x16x8xf32> into memref<16x8xf32>
// CHECK: return
// CHECK-SAME: memref<1x16x8xf32>
func.func @hivm_transpose(%arg0: memref<1x16x8xf32>) -> memref<1x16x8xf32> {
  %0 = memref.alloc() : memref<16x1x8xf32>
  %1 = memref.alloc() : memref<8x1x16xf32>
  %2 = memref.alloc() : memref<1x8x16xf32>
  %3 = memref.alloc() : memref<1x16x8xf32>
  hivm.hir.vtranspose ins(%arg0 : memref<1x16x8xf32>) outs(%0 : memref<16x1x8xf32>) permutation = [1, 0, 2]
  hivm.hir.vtranspose ins(%0 : memref<16x1x8xf32>) outs(%1 : memref<8x1x16xf32>) permutation = [2, 1, 0]
  hivm.hir.vtranspose ins(%1 : memref<8x1x16xf32>) outs(%2 : memref<1x8x16xf32>) permutation = [1, 0, 2] 
  hivm.hir.vtranspose ins(%2 : memref<1x8x16xf32>) outs(%3 : memref<1x16x8xf32>) permutation = [0, 2, 1]
  return %3 : memref<1x16x8xf32>
}

// -----

// CHECK-LABEL: func.func @hivm_transpose
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<2x16x8x4x3xf32> into memref<32x8x4x3xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<2x16x4x8x3xf32> into memref<32x4x8x3xf32>
// CHECK: return
// CHECK-SAME: memref<2x16x4x8x3xf32>
func.func @hivm_transpose(%arg0: memref<2x16x8x4x3xf32>) -> memref<2x16x4x8x3xf32> {
  %0 = memref.alloc() : memref<2x16x4x8x3xf32>
  hivm.hir.vtranspose ins(%arg0 : memref<2x16x8x4x3xf32>) outs(%0 : memref<2x16x4x8x3xf32>) permutation = [0, 1, 3, 2, 4]
  return %0 : memref<2x16x4x8x3xf32>
}

// -----
// CHECK-LABEL: func.func @test_reduce
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x?x?x?x8x?x8x32xf16> into memref<16x?xf16>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<1x?x?x?x8x?x8x32xf16> into memref<1x?xf16>
// CHECK: hivm.hir.vreduce
// CHECK-SAME: reduce_dims
// CHECK-SAME: {{\[}}0{{]}}
// CHECK: return
module {
  func.func @test_reduce(%src_5 : memref<16x?x?x?x8x?x8x32xf16>,
                         %dst_5 : memref<1x?x?x?x8x?x8x32xf16>) {
    hivm.hir.vreduce <max> ins(%src_5 : memref<16x?x?x?x8x?x8x32xf16>)
                           outs(%dst_5 : memref<1x?x?x?x8x?x8x32xf16>)
                           reduce_dims = [0]

    return
  }
}

// -----

// CHECK-LABEL: func.func @test_reduce
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x?x?x?x8x?x8x32xf16> into memref<?x8x?xf16>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x?x?x?x1x?x8x32xf16> into memref<?x1x?xf16>
// CHECK: hivm.hir.vreduce
// CHECK-SAME: reduce_dims
// CHECK-SAME: {{\[}}1{{]}}
// CHECK: return
module {
  func.func @test_reduce(%src_5 : memref<16x?x?x?x8x?x8x32xf16>,
                         %dst_5 : memref<16x?x?x?x1x?x8x32xf16>) {
    hivm.hir.vreduce <max> ins(%src_5 : memref<16x?x?x?x8x?x8x32xf16>)
                           outs(%dst_5 : memref<16x?x?x?x1x?x8x32xf16>)
                           reduce_dims = [4]

    return
  }
}


// -----

// CHECK-LABEL: func.func @test_reduce
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x?x?x?x8x?x8x32xf16> into memref<?x8x?x32xf16>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x?x?x?x1x?x8x1xf16> into memref<?x1x?x1xf16>
// CHECK: hivm.hir.vreduce
// CHECK-SAME: reduce_dims
// CHECK-SAME: {{\[}}1, 3{{]}}
// CHECK: return
module {
  func.func @test_reduce(%src_5 : memref<16x?x?x?x8x?x8x32xf16>,
                         %dst_5 : memref<16x?x?x?x1x?x8x1xf16>) {
    hivm.hir.vreduce <max> ins(%src_5 : memref<16x?x?x?x8x?x8x32xf16>)
                           outs(%dst_5 : memref<16x?x?x?x1x?x8x1xf16>)
                           reduce_dims = [4, 7]

    return
  }
}

// -----

// CHECK-LABEL: func.func @test_reduce_collapse_continuous
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2], [3], [4], [5], [6]] : memref<4x3x2x?x4x3x2xf16> into memref<24x?x4x3x2xf16>
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2], [3], [4], [5], [6]] : memref<4x3x2x1x4x1x2xf16> into memref<24x1x4x1x2xf16>
// CHECK: hivm.hir.vreduce {{.*}} reduce_dims = {{\[}}1, 3]
module {
  func.func @test_reduce_collapse_continuous(%src : memref<4x3x2x?x4x3x2xf16>,
                                             %dst : memref<4x3x2x1x4x1x2xf16>) {
    hivm.hir.vreduce <max> ins(%src : memref<4x3x2x?x4x3x2xf16>)
                           outs(%dst : memref<4x3x2x1x4x1x2xf16>)
                           reduce_dims = [3, 5]
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_reduce_collapse_continuous
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3]] : memref<?x?x?x?xf16> into memref<?x?x?xf16>
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3]] : memref<?x?x1x?xf16> into memref<?x1x?xf16>
// CHECK: hivm.hir.vreduce {{.*}} reduce_dims = {{\[}}1]
module {
  func.func @test_reduce_collapse_continuous_dynamic(%src : memref<?x?x?x?xf16>,
                                                     %dst : memref<?x?x1x?xf16>) {
    hivm.hir.vreduce <max> ins(%src : memref<?x?x?x?xf16>)
                           outs(%dst : memref<?x?x1x?xf16>)
                           reduce_dims = [2]
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_broadcast
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<1x1x32xi16> into memref<1x32xi16>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x8x32xi16> into memref<128x32xi16>
// CHECK: hivm.hir.vbrc
// CHECK-SAME: broadcast_dims
// CHECK-SAME: {{\[}}0{{]}}
// CHECK: return
module {
  func.func @test_broadcast(%src : memref<1x1x32xi16>, %dst : memref<16x8x32xi16>) {
    hivm.hir.vbrc ins(%src : memref<1x1x32xi16>) outs(%dst : memref<16x8x32xi16>) broadcast_dims = [0, 1]
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_broadcast_collapse_continuous
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1, 2], [3], [4, 5]] : memref<1x4x2x1x3x2xi16> into memref<1x8x1x6xi16>
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1, 2], [3], [4, 5]] : memref<16x4x2x16x3x2xi16> into memref<16x8x16x6xi16>
// CHECK: hivm.hir.vbrc {{.*}} broadcast_dims = {{\[}}0, 2]
module {
  func.func @test_broadcast_collapse_continuous(%src : memref<1x4x2x1x3x2xi16>, %dst : memref<16x4x2x16x3x2xi16>) {
    hivm.hir.vbrc ins(%src : memref<1x4x2x1x3x2xi16>) outs(%dst : memref<16x4x2x16x3x2xi16>) broadcast_dims = [0, 3]
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_broadcast_collapse_continuous_dynamic
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3]] : memref<?x?x1x?xf16> into memref<?x1x?xf16>
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3]] : memref<?x?x8x?xf16> into memref<?x8x?xf16>
// CHECK: hivm.hir.vbrc {{.*}} broadcast_dims = {{\[}}1]
module {
  func.func @test_broadcast_collapse_continuous_dynamic(%src : memref<?x?x1x?xf16>,
                                                        %dst : memref<?x?x8x?xf16>) {
    hivm.hir.vbrc ins(%src : memref<?x?x1x?xf16>) outs(%dst : memref<?x?x8x?xf16>) broadcast_dims = [2]
    return
  }
}

// -----

// CHECK-LABEL: func.func @brc_flatten_combine_reassociations
func.func @brc_flatten_combine_reassociations(%src: memref<1x1x1xi16>, 
                                              %dst1: memref<1024x1x1xi16>,
                                              %dst2: memref<1x1024x1xi16>,
                                              %dst3: memref<1x1x1024xi16>,
                                              %dst4: memref<1x16x1024xi16>,
                                              %dst5: memref<16x1x1024xi16>,
                                              %dst6: memref<16x1024x1xi16>,
                                              %dst7: memref<16x16x1024xi16>) {
  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1xi16>) outs({{.*}} memref<1024xi16>) broadcast_dims = {{\[}}0]
  hivm.hir.vbrc ins(%src: memref<1x1x1xi16>) outs(%dst1: memref<1024x1x1xi16>) broadcast_dims = [0]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1xi16>) outs({{.*}} memref<1024xi16>) broadcast_dims = {{\[}}0]
  hivm.hir.vbrc ins(%src: memref<1x1x1xi16>) outs(%dst2: memref<1x1024x1xi16>) broadcast_dims = [1]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1xi16>) outs({{.*}} memref<1024xi16>) broadcast_dims = {{\[}}0]
  hivm.hir.vbrc ins(%src: memref<1x1x1xi16>) outs(%dst3: memref<1x1x1024xi16>) broadcast_dims = [2]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1xi16>) outs({{.*}} memref<16384xi16>) broadcast_dims = {{\[}}0]
  hivm.hir.vbrc ins(%src: memref<1x1x1xi16>) outs(%dst4: memref<1x16x1024xi16>) broadcast_dims = [1, 2]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1xi16>) outs({{.*}} memref<16384xi16>) broadcast_dims = {{\[}}0]
  hivm.hir.vbrc ins(%src: memref<1x1x1xi16>) outs(%dst5: memref<16x1x1024xi16>) broadcast_dims = [0, 2]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1xi16>) outs({{.*}} memref<16384xi16>) broadcast_dims = {{\[}}0]
  hivm.hir.vbrc ins(%src: memref<1x1x1xi16>) outs(%dst6: memref<16x1024x1xi16>) broadcast_dims = [0, 1]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1xi16>) outs({{.*}} memref<262144xi16>) broadcast_dims = {{\[}}0]
  hivm.hir.vbrc ins(%src: memref<1x1x1xi16>) outs(%dst7: memref<16x16x1024xi16>) broadcast_dims = [0, 1, 2]
  return
}

// -----

// CHECK-LABEL: func.func @brc_reduce_flatten_combine_reassociations
func.func @brc_reduce_flatten_combine_reassociations(%src1: memref<1x1x1x2x4x1xi32>, 
                                                     %dst1: memref<1x1x8x2x4x8xi32>,
                                                     %src2: memref<1x1x1x2x4x2x1x1x1xi32>, 
                                                     %dst2: memref<1x1x?x2x4x2x8x8x1xi32>) {
  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1x8x1xi32>) outs({{.*}} memref<8x8x8xi32>) broadcast_dims = {{\[}}0, 2]
  hivm.hir.vbrc ins(%src1: memref<1x1x1x2x4x1xi32>) outs(%dst1: memref<1x1x8x2x4x8xi32>) broadcast_dims = [2, 5]

  // CHECK: hivm.hir.vreduce <max> ins({{.*}} memref<8x8x8xi32>) outs({{.*}} memref<1x8x1xi32>) reduce_dims = {{\[}}0, 2]
  hivm.hir.vreduce <max> ins(%dst1 : memref<1x1x8x2x4x8xi32>) outs(%src1 : memref<1x1x1x2x4x1xi32>) reduce_dims = [2, 5]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1x16x1xi32>) outs({{.*}} memref<?x16x64xi32>) broadcast_dims = {{\[}}0, 2]
  hivm.hir.vbrc ins(%src2: memref<1x1x1x2x4x2x1x1x1xi32>) outs(%dst2: memref<1x1x?x2x4x2x8x8x1xi32>) broadcast_dims = [2, 6, 7]

  // CHECK: hivm.hir.vreduce <sum> ins({{.*}} memref<?x16x64xi32>) outs({{.*}} memref<1x16x1xi32>) reduce_dims = {{\[}}0, 2]
  hivm.hir.vreduce <sum> ins(%dst2 : memref<1x1x?x2x4x2x8x8x1xi32>) outs(%src2 : memref<1x1x1x2x4x2x1x1x1xi32>) reduce_dims = [2, 6, 7]
  return
}

// -----

// CHECK-LABEL: func.func @otf_flatten_combine_reassociations
func.func @otf_flatten_combine_reassociations(%src1: memref<1x1x1x2x4xi32>,
                                              %dst1: memref<1x1x8x2x4xi32>,
                                              %src2: memref<1x1x1x1x4x2xi32>,
                                              %dst2: memref<1x1x?x2x4x?xi32>,
                                              %src3: memref<1x2x1x8x4x2xi32>,
                                              %dst3: memref<2x1x1x8x4x2xi32>,
                                              %src4: memref<4x2x8x1x8x4xi32>,
                                              %dst4: memref<8x4x2x1x8x4xi32>) {
  // CHECK: hivm.hir.vadd ins({{.*}} memref<1x8xi32>, memref<1x8xi32>) outs({{.*}} memref<8x8xi32>) broadcast = {{\[}}0]
  hivm.hir.vadd ins(%src1, %src1 : memref<1x1x1x2x4xi32>, memref<1x1x1x2x4xi32>)
                outs(%dst1 : memref<1x1x8x2x4xi32>)
                broadcast = [2]

  // CHECK: hivm.hir.vadd ins({{.*}} memref<1x8xi32>, memref<1x8xi32>) outs({{.*}} memref<?x?xi32>) broadcast = {{\[}}0]
  hivm.hir.vadd ins(%src2, %src2 : memref<1x1x1x1x4x2xi32>, memref<1x1x1x1x4x2xi32>)
                outs(%dst2 : memref<1x1x?x2x4x?xi32>)
                broadcast = [2, 3]

  // CHECK: hivm.hir.vadd ins({{.*}} memref<128xi32>, memref<128xi32>) outs({{.*}} memref<128xi32>) transpose = [0]
  hivm.hir.vadd ins(%src3, %src3 : memref<1x2x1x8x4x2xi32>, memref<1x2x1x8x4x2xi32>)
                outs(%dst3 : memref<2x1x1x8x4x2xi32>)
                transpose = [1, 0, 2, 3, 4, 5]

  // CHECK: hivm.hir.vadd ins({{.*}} memref<8x8x32xi32>, memref<8x8x32xi32>) outs({{.*}} memref<8x8x32xi32>) transpose = [1, 0, 2]
  hivm.hir.vadd ins(%src4, %src4 : memref<4x2x8x1x8x4xi32>, memref<4x2x8x1x8x4xi32>)
                outs(%dst4 : memref<8x4x2x1x8x4xi32>)
                transpose = [2, 0, 1, 3, 4, 5]
  return
}

// -----

// CHECK-LABEL: func.func @vtranspose_flatten_combine_reassociations
func.func @vtranspose_flatten_combine_reassociations(%src1: memref<1x2x1x8x4x2xi32>, 
                                                     %dst1: memref<2x1x1x8x4x2xi32>,
                                                     %src2: memref<4x?x8x1x8x4xi32>, 
                                                     %dst2: memref<?x2x8x1x4x8xi32>) {

  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1, 2], [3, 4, 5]]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1, 2], [3, 4, 5]]
  // CHECK: hivm.hir.vtranspose ins({{.*}} memref<1x2x64xi32>) outs({{.*}} memref<2x1x64xi32>) permutation = [1, 0, 2]
  hivm.hir.vtranspose ins(%src1 : memref<1x2x1x8x4x2xi32>) outs(%dst1 : memref<2x1x1x8x4x2xi32>) permutation = [1, 0, 2, 3, 4, 5]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2, 3], [4], [5]]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2, 3], [4], [5]]  
  // CHECK: hivm.hir.vtranspose ins({{.*}} memref<?x8x4xi32>) outs({{.*}} memref<?x4x8xi32>) permutation = [0, 2, 1]
  hivm.hir.vtranspose ins(%src2 : memref<4x?x8x1x8x4xi32>) outs(%dst2 : memref<?x2x8x1x4x8xi32>) permutation = [0, 1, 2, 3, 5, 4]
  return
}

// -----

// CHECK-LABEL: func.func @strided_combine_reassociations
func.func @strided_combine_reassociations(
                                          %src0: memref<1x1x2x1x1x1xi32>, %dst0: memref<1x8x2x1x8x1xi32>,
                                          %src1: memref<1x1x2x1x1x1xi32, strided<[?, ?, ?, ?, 1, 1]>>, 
                                          %dst1: memref<1x8x2x1x8x1xi32, strided<[?, ?, ?, ?, 1, 1]>>,
                                          %src2: memref<16x1x1x2x1x1x1xi32, strided<[?, ?, ?, ?, ?, 1, 1]>>, 
                                          %dst2: memref<16x1x8x2x1x8x1xi32, strided<[?, ?, ?, ?, ?, 1, 1]>>) {
  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1x2x1xi32>) outs({{.*}} memref<8x2x8xi32>) broadcast_dims = {{\[}}0, 2]
  hivm.hir.vbrc ins(%src0: memref<1x1x2x1x1x1xi32>) outs(%dst0: memref<1x8x2x1x8x1xi32>) broadcast_dims = [1, 4]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<1x2x1xi32, strided<[?, ?, 1]>>)
  // CHECK-SAME: outs({{.*}} memref<8x2x8xi32, strided<[?, ?, 1]>>) broadcast_dims = {{\[}}0, 2]
  hivm.hir.vbrc ins(%src1: memref<1x1x2x1x1x1xi32, strided<[?, ?, ?, ?, 1, 1]>>) 
                outs(%dst1: memref<1x8x2x1x8x1xi32, strided<[?, ?, ?, ?, 1, 1]>>) broadcast_dims = [1, 4]

  // CHECK: hivm.hir.vreduce <max> ins({{.*}} memref<8x2x8xi32, strided<[?, ?, 1]>>)
  // CHECK-SAME: outs({{.*}} memref<1x2x1xi32, strided<[?, ?, 1]>>) reduce_dims = {{\[}}0, 2]
  hivm.hir.vreduce <max> ins(%dst1 : memref<1x8x2x1x8x1xi32, strided<[?, ?, ?, ?, 1, 1]>>) 
                         outs(%src1 : memref<1x1x2x1x1x1xi32, strided<[?, ?, ?, ?, 1, 1]>>) reduce_dims = [1, 4]

  // CHECK: hivm.hir.vbrc ins({{.*}} memref<16x1x2x1xi32, strided<[?, ?, ?, 1]>>)
  // CHECK-SAME: outs({{.*}} memref<16x8x2x8xi32, strided<[?, ?, ?, 1]>>) broadcast_dims = {{\[}}1, 3]
  hivm.hir.vbrc ins(%src2: memref<16x1x1x2x1x1x1xi32, strided<[?, ?, ?, ?, ?, 1, 1]>>) 
                outs(%dst2: memref<16x1x8x2x1x8x1xi32, strided<[?, ?, ?, ?, ?, 1, 1]>>) broadcast_dims = [2, 5]

  // CHECK: hivm.hir.vreduce <max> ins({{.*}} memref<16x8x2x8xi32, strided<[?, ?, ?, 1]>>)
  // CHECK-SAME: outs({{.*}} memref<16x1x2x1xi32, strided<[?, ?, ?, 1]>>) reduce_dims = {{\[}}1, 3]
  hivm.hir.vreduce <max> ins(%dst2 : memref<16x1x8x2x1x8x1xi32, strided<[?, ?, ?, ?, ?, 1, 1]>>) 
                         outs(%src2 : memref<16x1x1x2x1x1x1xi32, strided<[?, ?, ?, ?, ?, 1, 1]>>) reduce_dims = [2, 5]
  return
}

// -----

// CHECK-LABEL: func.func @strided_combine_reassociations_otf
func.func @strided_combine_reassociations_otf(%src1: memref<16x1x1x1x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>, 
                                              %dst1: memref<16x1x8x4x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>) {
  // CHECK: hivm.hir.vadd ins({{.*}} memref<16x1x1x8xi32, strided<[?, ?, ?, 1]>>, memref<16x1x1x8xi32, strided<[?, ?, ?, 1]>>)
  // CHECK-SAME: outs({{.*}} memref<16x8x4x8xi32, strided<[?, ?, ?, 1]>>) broadcast = {{\[}}1, 2]
  hivm.hir.vadd ins(%src1, %src1 : memref<16x1x1x1x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>, 
                                   memref<16x1x1x1x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>)
                outs(%dst1 : memref<16x1x8x4x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>)
                broadcast = [2, 3]
  return  
}
 
// -----

// CHECK-LABEL: func.func @strided_combine_reassociations_transpose
func.func @strided_combine_reassociations_transpose(%src1: memref<16x1x8x4x1x8xi32>, 
                                                    %dst1: memref<16x1x4x8x1x8xi32>,
                                                    %src2: memref<16x1x8x4x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>, 
                                                    %dst2: memref<16x1x4x8x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>) {
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: hivm.hir.vadd ins({{.*}} memref<16x8x4x8xi32>, memref<16x8x4x8xi32>) 
  // CHECK-SAME: outs({{.*}} memref<16x4x8x8xi32>) transpose = {{\[}}0, 2, 1, 3]
  hivm.hir.vadd ins(%src1, %src1 : memref<16x1x8x4x1x8xi32>, 
                                   memref<16x1x8x4x1x8xi32>)
                outs(%dst1 : memref<16x1x4x8x1x8xi32>)
                transpose = [0, 1, 3, 2, 4, 5]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: hivm.hir.vadd ins({{.*}} memref<16x8x4x8xi32, strided<[?, ?, ?, 1]>>, memref<16x8x4x8xi32, strided<[?, ?, ?, 1]>>)
  // CHECK-SAME: outs({{.*}} memref<16x4x8x8xi32, strided<[?, ?, ?, 1]>>) transpose = {{\[}}0, 2, 1, 3]
  hivm.hir.vadd ins(%src2, %src2 : memref<16x1x8x4x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>,
                                   memref<16x1x8x4x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>)
                outs(%dst2 : memref<16x1x4x8x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>)
                transpose = [0, 1, 3, 2, 4, 5]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: hivm.hir.vtranspose ins({{.*}} memref<16x8x4x8xi32>) outs({{.*}} memref<16x4x8x8xi32>) permutation = {{\[}}0, 2, 1, 3]
  hivm.hir.vtranspose ins(%src1 : memref<16x1x8x4x1x8xi32>) outs(%dst1 : memref<16x1x4x8x1x8xi32>) permutation = [0, 1, 3, 2, 4, 5]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4], [5]]
  // CHECK: hivm.hir.vtranspose ins({{.*}} memref<16x8x4x8xi32, strided<[?, ?, ?, 1]>>)
  // CHECK-SAME: outs({{.*}} memref<16x4x8x8xi32, strided<[?, ?, ?, 1]>>) permutation = {{\[}}0, 2, 1, 3]
  hivm.hir.vtranspose ins(%src2 : memref<16x1x8x4x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>) 
                      outs(%dst2 : memref<16x1x4x8x1x8xi32, strided<[?, ?, ?, ?, 8, 1]>>) permutation = [0, 1, 3, 2, 4, 5]
  return  
}

// -----

// CHECK-LABEL: func.func @hivm_vadd
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<2x16x8x4x3xf32> into memref<32x8x4x3xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<2x16x8x4x3xf32> into memref<32x8x4x3xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<2x16x4x8x3xf32> into memref<32x4x8x3xf32>
// CHECK: hivm.hir.vadd
// CHECK-SAME: transpose
// CHECK-SAME: {{\[}}0, 2, 1, 3{{]}}
// CHECK: return
func.func @hivm_vadd(%arg0: memref<2x16x8x4x3xf32>, %arg1: memref<2x16x8x4x3xf32>, %arg2: memref<2x16x4x8x3xf32>) -> memref<2x16x4x8x3xf32> {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<2x16x8x4x3xf32>, memref<2x16x8x4x3xf32>)
                outs(%arg2 : memref<2x16x4x8x3xf32>)
                transpose = [0, 1, 3, 2, 4]
  return %arg2 : memref<2x16x4x8x3xf32>
}

// -----

// CHECK-LABEL: func.func @test_broadcastable_otf_static
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<1x1x16xf16> into memref<1x16xf16>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<32x16x16xf16> into memref<512x16xf16>
// CHECK: hivm.hir.vadd
// CHECK-SAME: broadcast
// CHECK-SAME: {{\[}}0{{]}}
// CHECK: return

func.func @test_broadcastable_otf_static(
  %src1 : memref<1x1x16xf16>, %dst : memref<32x16x16xf16>, %cst : f16) {
  hivm.hir.vadd ins(%src1, %src1: memref<1x1x16xf16>, memref<1x1x16xf16>)
                outs(%dst : memref<32x16x16xf16>)
                broadcast = [0, 1]
  return
}

// -----

// CHECK-LABEL: func.func @test_broadcastable_otf_static
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<32x1x1x16xf16> into memref<32x1x16xf16>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<32x16x4x16xf16> into memref<32x64x16xf16>
// CHECK: hivm.hir.vadd
// CHECK-SAME: broadcast
// CHECK-SAME: {{\[}}1{{]}}
// CHECK: return
func.func @test_broadcastable_otf_static(
  %src1 : memref<32x1x1x16xf16>, %dst : memref<32x16x4x16xf16>, %cst : f16) {
  hivm.hir.vadd ins(%src1, %src1: memref<32x1x1x16xf16>, memref<32x1x1x16xf16>)
                outs(%dst : memref<32x16x4x16xf16>)
                broadcast = [1, 2]
  return
}

// -----

// CHECK-LABEL: func.func @transpose_otf
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<2x16x8x4x3xf32> into memref<2x128x12xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<2x16x8x4x3xf32> into memref<2x128x12xf32>
// CHECK: memref.collapse_shape
// CHECK-SAME: memref<16x8x2x4x3xf32> into memref<128x2x12xf32>
// CHECK: hivm.hir.vadd
// CHECK-SAME: transpose
// CHECK-SAME: {{\[}}1, 0, 2{{]}}
// CHECK: return
func.func @transpose_otf(%arg0: memref<2x16x8x4x3xf32>, %arg1: memref<2x16x8x4x3xf32>, %arg2: memref<16x8x2x4x3xf32>) -> memref<16x8x2x4x3xf32> {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<2x16x8x4x3xf32>, memref<2x16x8x4x3xf32>)
                outs(%arg2 : memref<16x8x2x4x3xf32>)
                transpose = [1, 2, 0, 3, 4]
                // permuteDims = [1, 2, 0, 3, 4]
  return %arg2 : memref<16x8x2x4x3xf32>
}

// -----

// CHECK-LABEL: func.func @transpose_otf_collapse_continuous
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1, 2], [3, 4, 5]] : memref<4x3x2x?x3x2xf32> into memref<4x6x?xf32>
// CHECK: hivm.hir.vadd {{.*}} memref<4x6x?xf32>, memref<4x6x?xf32>
// CHECK-SAME: outs({{.*}} : memref<6x4x?xf32>
// CHECK-SAME: transpose = {{\[}}1, 0, 2]
func.func @transpose_otf_collapse_continuous(%arg0: memref<4x3x2x?x3x2xf32>, 
                                             %arg1: memref<4x3x2x?x3x2xf32>, 
                                             %arg2: memref<3x2x4x?x3x2xf32>) 
                                                 -> memref<3x2x4x?x3x2xf32> {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<4x3x2x?x3x2xf32>, memref<4x3x2x?x3x2xf32>)
                outs(%arg2 : memref<3x2x4x?x3x2xf32>)
                transpose = [1, 2, 0, 3, 4, 5]
  return %arg2 : memref<3x2x4x?x3x2xf32>
}

// -----

// CHECK-LABEL: func.func @transpose_otf_strided_collapse_continuous
// CHECK: memref.collapse_shape {{.*}} {{\[}}[0], [1], [2], [3], [4, 5]] : memref<4x3x2x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>> 
// CHECK-SAME: into memref<4x3x2x?x6xf32, strided<[?, ?, ?, ?, 1]>>
// CHECK: hivm.hir.vadd {{.*}} memref<4x3x2x?x6xf32, strided<[?, ?, ?, ?, 1]>>
// CHECK-SAME: outs({{.*}} : memref<4x2x3x?x6xf32, strided<[?, ?, ?, ?, 1]>>) 
// CHECK-SAME: transpose = {{\[}}0, 2, 1, 3, 4]
func.func @transpose_otf_strided_collapse_continuous(%arg0: memref<4x3x2x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>>, 
                                                     %arg1: memref<4x3x2x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>>, 
                                                     %arg2: memref<4x2x3x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>>) 
                                                         -> memref<4x2x3x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>> {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<4x3x2x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>>, 
                                   memref<4x3x2x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>>)
                outs(%arg2 : memref<4x2x3x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>>)
                transpose = [0, 2, 1, 3, 4, 5]
  return %arg2 : memref<4x2x3x?x3x2xf32, strided<[?, ?, ?, ?, 2, 1]>>
}

// -----

// CHECK-LABEL: hivm_copy
func.func @hivm_copy(%arg0: memref<8x1xf32>, %arg1: memref<8x1xf32>) {
  // CHECK-DAG: %[[SRC:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<8x1xf32> into memref<8xf32>
  // CHECK-DAG: %[[DST:.*]] = memref.collapse_shape %arg1 {{\[}}[0, 1]] : memref<8x1xf32> into memref<8xf32>
  // CHECK: hivm.hir.copy ins(%[[SRC]] : memref<8xf32>) outs(%[[DST]] : memref<8xf32>)
  hivm.hir.copy ins(%arg0 : memref<8x1xf32>)
                outs(%arg1 : memref<8x1xf32>)
  return
}

// -----

// CHECK-LABEL: hivm_copy_strided
func.func @hivm_copy_strided(%src0: memref<8x4x2x1xf32, strided<[8, 2, 1, 1]>>, 
                             %dst0: memref<8x4x2x1xf32, strided<[8, 2, 1, 1]>>,
                             %src1: memref<8x4x2x1xf32, strided<[16, 2, 1, 1]>>, 
                             %dst1: memref<8x4x2x1xf32, strided<[16, 2, 1, 1]>>) {
  // CHECK-DAG: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2, 3]] : memref<8x4x2x1xf32, strided<[8, 2, 1, 1]>> into memref<64xf32, strided<[1]>>
  // CHECK-DAG: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2, 3]] : memref<8x4x2x1xf32, strided<[8, 2, 1, 1]>> into memref<64xf32, strided<[1]>>
  // CHECK: hivm.hir.copy ins({{.*}} : memref<64xf32, strided<[1]>>) outs({{.*}} : memref<64xf32, strided<[1]>>)
  hivm.hir.copy ins(%src0 : memref<8x4x2x1xf32, strided<[8, 2, 1, 1]>>)
                outs(%dst0: memref<8x4x2x1xf32, strided<[8, 2, 1, 1]>>)
  // CHECK-DAG: memref.collapse_shape {{.*}} {{\[}}[0], [1, 2, 3]] : memref<8x4x2x1xf32, strided<[16, 2, 1, 1]>> into memref<8x8xf32, strided<[16, 1]>>
  // CHECK-DAG: memref.collapse_shape {{.*}} {{\[}}[0], [1, 2, 3]] : memref<8x4x2x1xf32, strided<[16, 2, 1, 1]>> into memref<8x8xf32, strided<[16, 1]>>
  // CHECK: hivm.hir.copy ins({{.*}} : memref<8x8xf32, strided<[16, 1]>>) outs({{.*}} : memref<8x8xf32, strided<[16, 1]>>)
  hivm.hir.copy ins(%src1 : memref<8x4x2x1xf32, strided<[16, 2, 1, 1]>>)
                outs(%dst1: memref<8x4x2x1xf32, strided<[16, 2, 1, 1]>>)
  return
}

// -----

// CHECK-LABEL: hivm_copy_notcollapsable
func.func @hivm_copy_notcollapsable(%arg0: memref<16x8x1xf32, strided<[128, 8, 1]>>, %arg1: memref<16x8x1xf32>) {
  // CHECK: memref.collapse_shape
  // CHECK: copy
  // CHECK-SAME: memref<16x8xf32, strided<[128, 8]>>
  hivm.hir.copy ins(%arg0 : memref<16x8x1xf32, strided<[128, 8, 1]>>)
                outs(%arg1 : memref<16x8x1xf32>)
  return
}

// -----

// CHECK-LABEL: hivm_copy_rank1
func.func @hivm_copy_rank1(%arg0: memref<8xf32>, %arg1: memref<8xf32>) {
  // CHECK-NOT: memref.collapse_shape
  hivm.hir.copy ins(%arg0 : memref<8xf32>)
                outs(%arg1 : memref<8xf32>)
  return
}

// ----- 

// CHECK-LABEL: func.func @partial_valid_collapsible(
// CHECK-NOT: memref.collapse_shape
func.func @partial_valid_collapsible(%arg0: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xi64, #hivm.address_space<gm>>, %arg3: memref<?xi32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi64, #hivm.address_space<gm>>, %arg8: memref<?xi32, #hivm.address_space<gm>>, %arg9: memref<?xi32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xf32, #hivm.address_space<gm>>, %arg12: i32, %arg13: i32) attributes {func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false]> : vector<14xi1>, global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c5280_i64 = arith.constant 5280 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c640_i64 = arith.constant 640 : i64
  %18 = hivm.hir.pointer_cast(%c4256_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview = memref.subview %18[0, 0, 0] [32, 4, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %19 = hivm.hir.pointer_cast(%c5280_i64) : memref<32x4xf32, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%19, %subview : memref<32x4xf32, #hivm.address_space<ub>>, memref<32x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview : memref<32x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  return
}

// ----- 

// Test if `isGuaranteedCollapsibleStrictly()` works as expected 
// CHECK-LABEL: func.func @strictly_collapse_with_dim_of_size1(
func.func @strictly_collapse_with_dim_of_size1(%arg0: memref<3x1xf32, strided<[2, 1]>>,
                                               %arg1: memref<1x1xf32, strided<[8, 1]>>,
                                               %arg2: memref<3x1x1x1xf32, strided<[3, 3, 3, 1]>>,
                                               %arg3: memref<3x1x1x1xf32, strided<[8, 4, 2, 1]>>) {
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1]] : memref<3x1xf32, strided<[2, 1]>> into memref<3xf32, strided<[2]>>
  hivm.hir.vadd ins(%arg0, %arg0 : memref<3x1xf32, strided<[2, 1]>>, 
                                   memref<3x1xf32, strided<[2, 1]>>) 
                outs(%arg0 : memref<3x1xf32, strided<[2, 1]>>)

  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1]] : memref<1x1xf32, strided<[8, 1]>> into memref<1xf32, strided<[8]>>
  hivm.hir.vadd ins(%arg1, %arg1 : memref<1x1xf32, strided<[8, 1]>>, 
                                   memref<1x1xf32, strided<[8, 1]>>) 
                outs(%arg1 : memref<1x1xf32, strided<[8, 1]>>)

  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2, 3]] : memref<3x1x1x1xf32, strided<[3, 3, 3, 1]>> into memref<3xf32, strided<[3]>>
  hivm.hir.vadd ins(%arg2, %arg2 : memref<3x1x1x1xf32, strided<[3, 3, 3, 1]>>,
                                  memref<3x1x1x1xf32, strided<[3, 3, 3, 1]>>) 
                outs(%arg2 : memref<3x1x1x1xf32, strided<[3, 3, 3, 1]>>)

  // this case would trigger assert of "invalid source layout map or collapsing non-contiguous dims" if generate collapse not strictly
  // CHECK: memref.collapse_shape {{.*}} {{\[}}[0, 1, 2, 3]] : memref<3x1x1x1xf32, strided<[8, 4, 2, 1]>>
  hivm.hir.vadd ins(%arg3, %arg3 : memref<3x1x1x1xf32, strided<[8, 4, 2, 1]>>, 
                                   memref<3x1x1x1xf32, strided<[8, 4, 2, 1]>>) 
                outs(%arg3 : memref<3x1x1x1xf32, strided<[8, 4, 2, 1]>>)
  return
}

// -----
// CHECK-LABEL: func.func @strictly_collapse_with_dim_of_size1_with_dyn_stride(
// CHECK: memref.collapse_shape
func.func @strictly_collapse_with_dim_of_size1_with_dyn_stride(
          %arg0: memref<1x1xf32, strided<[?, 1]>>,
          %arg1: memref<2x3x1x1x1xf32, strided<[?, ?, ?, ?, 2]>>) {
  hivm.hir.vadd ins(%arg0, %arg0 : memref<1x1xf32, strided<[?, 1]>>,
                                   memref<1x1xf32, strided<[?, 1]>>)
                outs(%arg0 : memref<1x1xf32, strided<[?, 1]>>)

  hivm.hir.vadd ins(%arg1, %arg1 : memref<2x3x1x1x1xf32, strided<[?, ?, ?, ?, 2]>>,
                                   memref<2x3x1x1x1xf32, strided<[?, ?, ?, ?, 2]>>)
                outs(%arg1 : memref<2x3x1x1x1xf32, strided<[?, ?, ?, ?, 2]>>)
  return
}

// -----
// CHECK-LABEL: func.func @test_interleaveop(
func.func @test_interleaveop(%arg0: memref<2x64xf16>, %arg1: memref<2x128xf16>) {
  // CHECK: hivm.hir.vinterleave ins({{.*}}, {{.*}} : memref<128xf16>, memref<128xf16>) outs({{.*}} : memref<256xf16>)
  hivm.hir.vinterleave ins(%arg0, %arg0 : memref<2x64xf16>, memref<2x64xf16>) outs(%arg1 : memref<2x128xf16>) interleave_channel_nums = 2
  return
}

// -----

// CHECK-LABEL: func.func @inline_transpose_op
// CHECK: %[[r0:.*]] = memref.collapse_shape %arg[[a0:.*]] {{\[}}{{\[}}0, 1], [2], {{\[}}3]] : memref<4x1x8x?xi32> into memref<4x8x?xi32>
// CHECK: %[[r1:.*]] = memref.collapse_shape %arg[[a1:.*]] {{\[}}{{\[}}0, 1], [2], {{\[}}3]] : memref<4x1x8x?xi32> into memref<4x8x?xi32>
// CHECK: %[[r2:.*]] = memref.collapse_shape %arg[[a2:.*]] {{\[}}{{\[}}0, 1], {{\[}}2], {{\[}}3]] : memref<1x8x4x?xi32> into memref<8x4x?xi32>
// CHECK: hivm.hir.vadd ins(%[[r0:.*]], %[[r1:.*]] : memref<4x8x?xi32>, memref<4x8x?xi32>) outs(%[[r2:.*]] : memref<8x4x?xi32>) transpose = {{\[}}1, 0, 2]
func.func @inline_transpose_op(
  %arg0: memref<4x1x8x?xi32>, 
  %arg1: memref<4x1x8x?xi32>, 
  %arg2: memref<1x8x4x?xi32>) {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<4x1x8x?xi32>, memref<4x1x8x?xi32>) outs(%arg2 : memref<1x8x4x?xi32>) transpose = [1, 2, 0, 3]
  return
}

// -----

// CHECK-LABEL: func.func @non_contiguous_stride(
// CHECK-NOT: memref.collapse_shape
// CHECK: hivm.hir.vreduce 
// CHECK-NEXT: return
func.func @non_contiguous_stride(%arg0: memref<7x17x15xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>, %arg1: memref<1x17x15xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<1x17x15xi32, strided<[272, 16, 1]>, #hivm.address_space<ub>>) {
  hivm.hir.vreduce <max_with_index_left> ins(%arg0 : memref<7x17x15xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>) outs(%arg1, %arg2 : memref<1x17x15xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>, memref<1x17x15xi32, strided<[272, 16, 1]>, #hivm.address_space<ub>>) reduce_dims = [0]
  return
}

// -----

// CHECK-LABEL: func.func @triton_argmax_3d(
// CHECK: memref.collapse_shape
// CHECK: memref.collapse_shape
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.vreduce 
// CHECK-NEXT: return
func.func @triton_argmax_3d(%arg0: memref<7x17x15x1xf16, strided<[272, 16, 1, 1]>, #hivm.address_space<ub>>, %arg1: memref<1x17x15x1xf16, strided<[272, 16, 1, 1]>, #hivm.address_space<ub>>, %arg2: memref<1x17x15x1xi32, strided<[272, 16, 1, 1]>, #hivm.address_space<ub>>) {
  hivm.hir.vreduce <max_with_index_left> ins(%arg0 : memref<7x17x15x1xf16, strided<[272, 16, 1, 1]>, #hivm.address_space<ub>>) outs(%arg1, %arg2 : memref<1x17x15x1xf16, strided<[272, 16, 1, 1]>, #hivm.address_space<ub>>, memref<1x17x15x1xi32, strided<[272, 16, 1, 1]>, #hivm.address_space<ub>>) reduce_dims = [0]
  return
}

// -----

// CHECK-LABEL: transpose_with_unit(
// CHECK: vtranspose
// CHECK-SAME: memref<27x22xf16
// CHECK-SAME: memref<22x27xf16
func.func @transpose_with_unit(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.shape_0 = 0 : i32, tt.shape_1 = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
  hivm.hir.set_mask_norm
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [27, 22, 1], strides: [22, 1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<27x22x1xf16, strided<[22, 1, 1]>, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<27x32x1x1xf16, #hivm.address_space<ub>>
  %subview = memref.subview %alloc[0, 0, 0, 0] [27, 22, 1, 1] [1, 1, 1, 1] : memref<27x32x1x1xf16, #hivm.address_space<ub>> to memref<27x22x1xf16, strided<[32, 1, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast : memref<27x22x1xf16, strided<[22, 1, 1]>, #hivm.address_space<gm>>) outs(%subview : memref<27x22x1xf16, strided<[32, 1, 1]>, #hivm.address_space<ub>>)
  %alloc_0 = memref.alloc() : memref<22x32x1x1xf16, #hivm.address_space<ub>>
  %subview_1 = memref.subview %alloc_0[0, 0, 0, 0] [22, 27, 1, 1] [1, 1, 1, 1] : memref<22x32x1x1xf16, #hivm.address_space<ub>> to memref<22x27x1xf16, strided<[32, 1, 1]>, #hivm.address_space<ub>>
  hivm.hir.vtranspose ins(%subview : memref<27x22x1xf16, strided<[32, 1, 1]>, #hivm.address_space<ub>>) outs(%subview_1 : memref<22x27x1xf16, strided<[32, 1, 1]>, #hivm.address_space<ub>>) permutation = [1, 0, 2]
  %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [22, 27, 1], strides: [27, 1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<22x27x1xf16, strided<[27, 1, 1]>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_1 : memref<22x27x1xf16, strided<[32, 1, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<22x27x1xf16, strided<[27, 1, 1]>, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func.func @triton_cumsum_2
// CHECK: vcumsum
// CHECK-SAME: memref<4xf16
// CHECK-SAME: memref<4xf16
// CHECK-NEXT: return
func.func @triton_cumsum_2(%arg0: memref<1x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>, %arg1: memref<1x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>) {
  hivm.hir.vcumsum ins(%arg0 : memref<1x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>) outs(%arg1: memref<1x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>) cum_dims = [2] reverse = false
  return
}

// -----

// CHECK-LABEL: func.func @triton_cumprod_0
// CHECK: vcumsum
// CHECK-SAME: memref<3x4xf16
// CHECK-SAME: memref<3x4xf16
// CHECK-NEXT: return
func.func @triton_cumprod_0(%arg0: memref<3x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>, %arg1: memref<3x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>) {
  hivm.hir.vcumsum ins(%arg0 : memref<3x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>) outs(%arg1: memref<3x1x4x1xf16, strided<[4, 4, 1, 1]>, #hivm.address_space<ub>>) cum_dims = [0] reverse = false
  return
}

// -----
// CHECK-LABEL: test_transposable_otf
// CHECK: hivm.hir.vadd
// CHECK-SAME: transpose = [1, 0, 2]
func.func @test_transposable_otf(
  %arg0: memref<1x4x1x8x1xf32>,
  %arg1: memref<1x4x1x8x1xf32>,
  %arg2: memref<8x4x1x1x1xf32>) {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<1x4x1x8x1xf32>, memref<1x4x1x8x1xf32>) outs(%arg2 : memref<8x4x1x1x1xf32>) transpose = [3, 1, 0, 2, 4]
  return
}

// -----
// CHECK-LABEL: test_transposable_otf
// CHECK: hivm.hir.vadd
// CHECK-SAME: ins
// CHECK-SAME: memref<4x8x2xf32>
// CHECK-SAME: memref<4x8x2xf32>
// CHECK-SAME: outs
// CHECK-SAME: memref<8x4x2xf32>
// CHECK-SAME: transpose = [1, 0, 2]
func.func @test_transposable_otf_6d(
  %arg0: memref<1x4x1x8x2x1xf32>,
  %arg1: memref<1x4x1x8x2x1xf32>,
  %arg2: memref<8x4x1x1x2x1xf32>) {
  hivm.hir.vadd ins(%arg0, %arg1 : memref<1x4x1x8x2x1xf32>, memref<1x4x1x8x2x1xf32>) outs(%arg2 : memref<8x4x1x1x2x1xf32>) transpose = [3, 1, 0, 2, 4, 5]
  return
}

// -----

// CHECK-LABEL: @test_vflip_op_memref(
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x2x1xf16>
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x1x2x1xf16>
// CHECK: %[[COLLAPSED_0:.*]] = memref.collapse_shape %[[ALLOC_0]] [[_:.*]] into memref<2x1xf16>
// CHECK: %[[COLLAPSED_1:.*]] = memref.collapse_shape %[[ALLOC_1]] [[_:.*]] into memref<2x1xf16>
// CHECK: hivm.hir.vflip ins(%[[COLLAPSED_0]] : memref<2x1xf16>) outs(%[[COLLAPSED_1]] : memref<2x1xf16>)
func.func @test_vflip_op_memref() {
  %src = memref.alloc() : memref<1x1x2x1xf16>
  %dst = memref.alloc() : memref<1x1x2x1xf16>
  hivm.hir.vflip ins(%src: memref<1x1x2x1xf16>)
                outs(%dst : memref<1x1x2x1xf16>)
                flip_axis = 3 
  return
}

// -----

// CHECK: @test_vpad_op_memref(%[[ARG:.*]]: f16
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x3x1x1xf16>
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<2x2x4x2x1xf16>
// CHECK: %[[COLLAPSED_0:.*]] = memref.collapse_shape %[[ALLOC_0]] [[_:.*]] into memref<2x1x3x1xf16>
// CHECK: %[[COLLAPSED_1:.*]] = memref.collapse_shape %[[ALLOC_1]] [[_:.*]] into memref<2x2x4x2xf16>
// CHECK: hivm.hir.vpad ins(%[[COLLAPSED_0]] : memref<2x1x3x1xf16>) outs(%[[COLLAPSED_1]] : memref<2x2x4x2xf16>) low[0, 1, 1, 0] high[0, 0, 0, 1] pad_value %[[ARG]] : f16
func.func @test_vpad_op_memref(
  %pad_value : f16
) {
  %src = memref.alloc() : memref<2x1x3x1x1xf16>
  %dst = memref.alloc() : memref<2x2x4x2x1xf16>
  hivm.hir.vpad ins(%src : memref<2x1x3x1x1xf16>)
                outs(%dst : memref<2x2x4x2x1xf16>)
                low[0, 1, 1, 0, 0]
                high[0, 0, 0, 1, 0]
                pad_value %pad_value : f16
  return
}

// -----

// CHECK-LABEL: @test_vconcat_op_memref(
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x5x1xf16>
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x3x5x1xf16>
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<1x4x5x1xf16>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<2x4x5x1xf16>
// CHECK: %[[COLLAPSED_0:.*]] = memref.collapse_shape %[[ALLOC_0]] [[_:.*]] into memref<1x5xf16>
// CHECK: %[[COLLAPSED_1:.*]] = memref.collapse_shape %[[ALLOC_1]] [[_:.*]] into memref<3x5xf16>
// CHECK: %[[COLLAPSED_2:.*]] = memref.collapse_shape %[[ALLOC_2]] [[_:.*]] into memref<4x5xf16>
// CHECK: hivm.hir.vconcat dim(0) ins(%[[COLLAPSED_0]], %[[COLLAPSED_1]] : memref<1x5xf16>, memref<3x5xf16>) outs(%[[COLLAPSED_2]] : memref<4x5xf16>)
// CHECK: %[[COLLAPSED_3:.*]] = memref.collapse_shape %[[ALLOC_2]] [[_:.*]] into memref<1x20xf16>
// CHECK: %[[COLLAPSED_4:.*]] = memref.collapse_shape %[[ALLOC_3]] [[_:.*]] into memref<2x20xf16>
// CHECK: hivm.hir.vconcat dim(0) ins(%[[COLLAPSED_3]], %[[COLLAPSED_3]] : memref<1x20xf16>, memref<1x20xf16>) outs(%[[COLLAPSED_4]] : memref<2x20xf16>)
func.func @test_vconcat_op_memref() {
  %a_f16 = memref.alloc() : memref<1x1x5x1xf16>
  %b_f16 = memref.alloc() : memref<1x3x5x1xf16>
  %c_f16 = memref.alloc() : memref<1x4x5x1xf16>
  %d_f16 = memref.alloc() : memref<2x4x5x1xf16>
  hivm.hir.vconcat dim(1) ins(%a_f16, %b_f16: memref<1x1x5x1xf16>, memref<1x3x5x1xf16>)
                   outs(%c_f16 : memref<1x4x5x1xf16>)
  hivm.hir.vconcat dim(0) ins(%c_f16, %c_f16: memref<1x4x5x1xf16>, memref<1x4x5x1xf16>)
                   outs(%d_f16 : memref<2x4x5x1xf16>)

  return
}

// -----

// CHECK-LABEL: @test_gather(
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x2x1xf32>
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x1x2x1xi32>
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<1x1x2x1xf32>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<1x1x2x1xi32>
// CHECK: %[[COLLAPSED_0:.*]] = memref.collapse_shape %[[ALLOC_0]] [[_:.*]] into memref<2x1xf32>
// CHECK: %[[COLLAPSED_1:.*]] = memref.collapse_shape %[[ALLOC_1]] [[_:.*]] into memref<2x1xi32>
// CHECK: %[[COLLAPSED_2:.*]] = memref.collapse_shape %[[ALLOC_2]] [[_:.*]] into memref<2x1xf32>
// CHECK: %[[COLLAPSED_3:.*]] = memref.collapse_shape %[[ALLOC_3]] [[_:.*]] into memref<2x1xi32>
// CHECK: hivm.hir.vgather ins(%[[COLLAPSED_0]] : memref<2x1xf32>) indices(%[[COLLAPSED_1]] : memref<2x1xi32>) outs(%[[COLLAPSED_2]] : memref<2x1xf32>) temp_buffer(%[[COLLAPSED_3]] : memref<2x1xi32>)
func.func @test_gather() {
  %src = memref.alloc() : memref<1x1x2x1xf32>
  %idx = memref.alloc() : memref<1x1x2x1xi32>
  %dst = memref.alloc() : memref<1x1x2x1xf32>
  %tmp = memref.alloc() : memref<1x1x2x1xi32>
  hivm.hir.vgather ins(%src : memref<1x1x2x1xf32>) 
                    indices(%idx : memref<1x1x2x1xi32>) 
                    outs(%dst : memref<1x1x2x1xf32>) 
                    temp_buffer(%tmp : memref<1x1x2x1xi32>) 
  return
}

// -----

// CHECK-LABEL: @test_arange_2d_remains(
// CHECK-NOT: memref.collapse_shape
func.func @test_arange_2d_remains() {
  %ub = memref.alloc() : memref<16x16xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  hivm.hir.varange offset[%c0] strides[%c1, %c2] outs(%ub : memref<16x16xi32>)
  return
}

// -----

// CHECK-LABEL: @test_vmulextended_op(
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x32x2xi16>
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x32x2xi16>
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<1x32x2xi16>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<1x32x2xi16>
// CHECK: %[[ALLOC_4:.*]] = memref.alloc() : memref<1x64x2xi32>
// CHECK: %[[COLLAPSED_0:.*]] = memref.collapse_shape %[[ALLOC_0]] [[_:.*]] into memref<64xi16>
// CHECK: %[[COLLAPSED_1:.*]] = memref.collapse_shape %[[ALLOC_1]] [[_:.*]] into memref<64xi16>
// CHECK: %[[COLLAPSED_2:.*]] = memref.collapse_shape %[[ALLOC_2]] [[_:.*]] into memref<64xi16>
// CHECK: %[[COLLAPSED_3:.*]] = memref.collapse_shape %[[ALLOC_3]] [[_:.*]] into memref<64xi16>
// CHECK: %[[COLLAPSED_4:.*]] = memref.collapse_shape %[[ALLOC_4]] [[_:.*]] into memref<128xi32>
// CHECK: hivm.hir.vmulextended ins(%[[COLLAPSED_0]], %[[COLLAPSED_1]] : memref<64xi16>, memref<64xi16>) outs(%[[COLLAPSED_2]], %[[COLLAPSED_3]] : memref<64xi16>, memref<64xi16>) temp_buffer(%[[COLLAPSED_4]] : memref<128xi32>)
func.func @test_vmulextended_op() {
  %input_0 = memref.alloc() : memref<1x32x2xi16>
  %input_1 = memref.alloc() : memref<1x32x2xi16>
  %output_0 = memref.alloc() : memref<1x32x2xi16>
  %output_1 = memref.alloc() : memref<1x32x2xi16>
  %alloc = memref.alloc() : memref<1x64x2xi32>
  hivm.hir.vmulextended ins(%input_0, %input_1 : memref<1x32x2xi16>, memref<1x32x2xi16>)
                        outs(%output_0, %output_1 : memref<1x32x2xi16>, memref<1x32x2xi16>)
                        temp_buffer(%alloc : memref<1x64x2xi32>)
  return
}

// -----
// CHECK-LABEL: test_broadcastable_otf_different_operand
// CHECK: hivm.hir.vadd
// CHECK-SAME: memref<1x16x16xf16>, memref<32x1x16xf16>
func.func @test_broadcastable_otf_different_operand(
  %src1 : memref<1x16x16xf16>, %src2 : memref<32x1x16xf16>, %dst : memref<32x16x16xf16>, %cst : f16) {
  hivm.hir.vadd ins(%src1, %src2: memref<1x16x16xf16>, memref<32x1x16xf16>)
                outs(%dst : memref<32x16x16xf16>)
                broadcast = [0, 1]
  return
}

// -----
// CHECK-LABEL: @test_arange(
// CHECK: hivm.hir.varange
// CHECK-SAME: memref<16x16x16xi32>
// CHECK: hivm.hir.varange
// CHECK-SAME: memref<16x16x16xi32>
// CHECK: return
func.func @test_arange(%in : memref<16x16x16xi32>, %offset : index, %s0:index, %s1:index, %s2:index) {
  hivm.hir.varange offset[%offset] strides[%s0,%s1,%s2] outs(%in: memref<16x16x16xi32>)
  hivm.hir.varange strides[%s0,%s1,%s2] outs(%in: memref<16x16x16xi32>)
  return
}
