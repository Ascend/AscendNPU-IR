// REQUIRES: execution-engine
// RUN: bishengir-opt %s --hivm-vectorize --canonicalize --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" --convert-linalg-to-loops --lower-vector-mask --convert-vector-to-scf --lower-affine --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-index-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=i32 -shared-libs=%mlir_c_runner_utils | FileCheck %s
// RUN: bishengir-opt %s --execution-engine-convert-hivm-to-upstream --canonicalize --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" --convert-linalg-to-loops --lower-vector-mask --convert-vector-to-scf --lower-affine --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-index-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=i32 -shared-libs=%mlir_c_runner_utils | FileCheck %s
// CHECK: 0

module {
  func.func @compute(%a: tensor<67xf32>, %b: tensor<67xf32>,
                     %init: tensor<67xf32>) -> tensor<67xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<67xf32>, tensor<67xf32>)
        outs(%init : tensor<67xf32>) -> tensor<67xf32>
    %product = hivm.hir.vmul ins(%sum, %b : tensor<67xf32>, tensor<67xf32>)
        outs(%init : tensor<67xf32>) -> tensor<67xf32>
    return %product : tensor<67xf32>
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c67 = arith.constant 67 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2.0 : f32
    %empty = tensor.empty() : tensor<67xf32>
    %ramp = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>],
                           iterator_types = ["parallel"]}
        outs(%empty : tensor<67xf32>) {
      ^bb0(%unused: f32):
        %index = linalg.index 0 : index
        %integer = arith.index_cast %index : index to i32
        %value = arith.sitofp %integer : i32 to f32
        linalg.yield %value : f32
    } -> tensor<67xf32>
    %rhs = arith.constant dense<2.0> : tensor<67xf32>
    %actual = func.call @compute(%ramp, %rhs, %empty)
        : (tensor<67xf32>, tensor<67xf32>, tensor<67xf32>) -> tensor<67xf32>
    %errors = scf.for %i = %c0 to %c67 step %c1 iter_args(%count = %zero) -> i32 {
      %integer = arith.index_cast %i : index to i32
      %value = arith.sitofp %integer : i32 to f32
      %sum = arith.addf %value, %two : f32
      %expected = arith.mulf %sum, %two : f32
      %observed = tensor.extract %actual[%i] : tensor<67xf32>
      %wrong = arith.cmpf une, %expected, %observed : f32
      %delta = arith.select %wrong, %one, %zero : i32
      %next = arith.addi %count, %delta : i32
      scf.yield %next : i32
    }
    return %errors : i32
  }
}
