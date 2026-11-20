// REQUIRES: execution-engine
// Allow loop yields to use new buffers when bufferization cannot reuse iter_args.
// DEFINE: %{execute} = --convert-elementwise-to-linalg --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" --convert-linalg-to-loops --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_c_runner_utils | FileCheck %s
// RUN: bishengir-opt %s %{execute}
// RUN: bishengir-opt %s --fuse-reduction-into-loop="mode=zero-only" %{execute}
// RUN: bishengir-opt %s --fuse-reduction-into-loop="mode=late" %{execute}

// Compare the original loop and both fusion modes with exact small integers.
// The two states start with a nonzero seed and a zero fill. Each output includes
// [10,20] once. Check zero, one and three trips, then a unit and an empty extent.
// CHECK: 14
// CHECK-NEXT: 26
// CHECK-NEXT: 10
// CHECK-NEXT: 20
// CHECK-NEXT: 20
// CHECK-NEXT: 34
// CHECK-NEXT: 16
// CHECK-NEXT: 28
// CHECK-NEXT: 32
// CHECK-NEXT: 50
// CHECK-NEXT: 28
// CHECK-NEXT: 44
// CHECK-NEXT: 14
// CHECK-NEXT: 28
// CHECK-NEXT: 10
// CHECK-NEXT: 20
// CHECK-NEXT: 10
// CHECK-NEXT: 20

func.func @mixed(%seed: tensor<?x2xf32>, %tile: tensor<?x2xf32>,
                 %init: tensor<2xf32>, %n: index)
    -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %rows = tensor.dim %seed, %c0 : tensor<?x2xf32>
  %empty = tensor.empty(%rows) : tensor<?x2xf32>
  %zero = linalg.fill ins(%f0 : f32) outs(%empty : tensor<?x2xf32>) -> tensor<?x2xf32>
  %loop:2 = scf.for %iv = %c0 to %n step %c1
      iter_args(%a = %seed, %b = %zero)
      -> (tensor<?x2xf32>, tensor<?x2xf32>) {
    %next_a = arith.addf %a, %tile : tensor<?x2xf32>
    %next_b = arith.addf %b, %tile : tensor<?x2xf32>
    scf.yield %next_a, %next_b : tensor<?x2xf32>, tensor<?x2xf32>
  }
  %a = linalg.reduce ins(%loop#0 : tensor<?x2xf32>)
      outs(%init : tensor<2xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
    %sum = arith.addf %in, %acc : f32
    linalg.yield %sum : f32
  }
  %b = linalg.reduce ins(%loop#1 : tensor<?x2xf32>)
      outs(%init : tensor<2xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
    %sum = arith.addf %in, %acc : f32
    linalg.yield %sum : f32
  }
  return %a, %b : tensor<2xf32>, tensor<2xf32>
}

// Reducing a unit innermost dimension must still include its initializer.
func.func @unit_axis(%init: tensor<2xf32>, %n: index) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %seed = arith.constant dense<0.0> : tensor<2x1xf32>
  %tile = arith.constant dense<[[2.0], [4.0]]> : tensor<2x1xf32>
  %loop = scf.for %iv = %c0 to %n step %c1
      iter_args(%acc = %seed) -> tensor<2x1xf32> {
    %next = arith.addf %acc, %tile : tensor<2x1xf32>
    scf.yield %next : tensor<2x1xf32>
  }
  %result = linalg.reduce ins(%loop : tensor<2x1xf32>)
      outs(%init : tensor<2xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
    %sum = arith.addf %in, %acc : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<2xf32>
}

func.func @print_pair(%value: tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a = tensor.extract %value[%c0] : tensor<2xf32>
  %b = tensor.extract %value[%c1] : tensor<2xf32>
  call @printF32(%a) : (f32) -> ()
  call @printNewline() : () -> ()
  call @printF32(%b) : (f32) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %seed_value = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %tile_value = arith.constant dense<[[2.0, 3.0], [4.0, 5.0]]> : tensor<2x2xf32>
  %seed = tensor.cast %seed_value : tensor<2x2xf32> to tensor<?x2xf32>
  %tile = tensor.cast %tile_value : tensor<2x2xf32> to tensor<?x2xf32>
  %init = arith.constant dense<[10.0, 20.0]> : tensor<2xf32>
  %zero:2 = call @mixed(%seed, %tile, %init, %c0)
      : (tensor<?x2xf32>, tensor<?x2xf32>, tensor<2xf32>, index) -> (tensor<2xf32>, tensor<2xf32>)
  call @print_pair(%zero#0) : (tensor<2xf32>) -> ()
  call @print_pair(%zero#1) : (tensor<2xf32>) -> ()
  %one:2 = call @mixed(%seed, %tile, %init, %c1)
      : (tensor<?x2xf32>, tensor<?x2xf32>, tensor<2xf32>, index) -> (tensor<2xf32>, tensor<2xf32>)
  call @print_pair(%one#0) : (tensor<2xf32>) -> ()
  call @print_pair(%one#1) : (tensor<2xf32>) -> ()
  %three:2 = call @mixed(%seed, %tile, %init, %c3)
      : (tensor<?x2xf32>, tensor<?x2xf32>, tensor<2xf32>, index) -> (tensor<2xf32>, tensor<2xf32>)
  call @print_pair(%three#0) : (tensor<2xf32>) -> ()
  call @print_pair(%three#1) : (tensor<2xf32>) -> ()
  %unit = call @unit_axis(%init, %c2)
      : (tensor<2xf32>, index) -> tensor<2xf32>
  call @print_pair(%unit) : (tensor<2xf32>) -> ()
  %empty_value = arith.constant dense<> : tensor<0x2xf32>
  %empty = tensor.cast %empty_value : tensor<0x2xf32> to tensor<?x2xf32>
  %no_rows:2 = call @mixed(%empty, %empty, %init, %c3)
      : (tensor<?x2xf32>, tensor<?x2xf32>, tensor<2xf32>, index) -> (tensor<2xf32>, tensor<2xf32>)
  call @print_pair(%no_rows#0) : (tensor<2xf32>) -> ()
  call @print_pair(%no_rows#1) : (tensor<2xf32>) -> ()
  return
}

func.func private @printF32(f32)
func.func private @printNewline()
