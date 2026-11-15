// REQUIRES: execution-engine
// RUN: bishengir-opt %s --hivm-vectorize-buffers --canonicalize --lower-vector-mask --convert-vector-to-scf --lower-affine --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-index-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=i32 -shared-libs=%mlir_c_runner_utils | FileCheck %s
// RUN: bishengir-opt %s --hivm-vectorize-buffers | FileCheck %s --check-prefix=IR --implicit-check-not=hivm.hir.vbrc --implicit-check-not=hivm.hir.vinterleave --implicit-check-not=hivm.hir.vdeinterleave --implicit-check-not=hivm.hir.vtranspose
// CHECK: 0
// IR: vector.transfer_write

module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c67 = arith.constant 67 : index
    %c70 = arith.constant 70 : index
    %c73 = arith.constant 73 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %hundred = arith.constant 100 : i32
    %sentinel = arith.constant -777.0 : f32
    %fill = arith.constant 13.0 : f32
    %left = memref.alloc() : memref<73xf32>
    %right = memref.alloc() : memref<73xf32>
    %filled = memref.alloc() : memref<73xf32>
    %joined = memref.alloc() : memref<134xf32>
    %scratch = memref.alloc() : memref<300xf32>
    %even = memref.alloc() : memref<67xf32>
    %odd = memref.alloc() : memref<67xf32>
    %onlyOdd = memref.alloc() : memref<67xf32>
    scf.for %i = %c0 to %c73 step %c1 {
      %integer = arith.index_cast %i : index to i32
      %lhs = arith.sitofp %integer : i32 to f32
      %negative = arith.subi %hundred, %integer : i32
      %rhs = arith.sitofp %negative : i32 to f32
      memref.store %lhs, %left[%i] : memref<73xf32>
      memref.store %rhs, %right[%i] : memref<73xf32>
      memref.store %sentinel, %filled[%i] : memref<73xf32>
    }
    %leftView = memref.subview %left[3] [67] [1]
        : memref<73xf32> to memref<67xf32, strided<[1], offset: 3>>
    %rightView = memref.subview %right[3] [67] [1]
        : memref<73xf32> to memref<67xf32, strided<[1], offset: 3>>
    %fillView = memref.subview %filled[3] [67] [1]
        : memref<73xf32> to memref<67xf32, strided<[1], offset: 3>>
    hivm.hir.vbrc ins(%fill : f32)
        outs(%fillView : memref<67xf32, strided<[1], offset: 3>>)
        temp_buffer(%scratch : memref<300xf32>)
    hivm.hir.vinterleave
        ins(%leftView, %rightView : memref<67xf32, strided<[1], offset: 3>>,
                                   memref<67xf32, strided<[1], offset: 3>>)
        outs(%joined : memref<134xf32>) interleave_channel_nums = 2
        temp_buffer(%scratch : memref<300xf32>)
    hivm.hir.vdeinterleave ins(%joined : memref<134xf32>)
        outs(%even, %odd : memref<67xf32>, memref<67xf32>)
    hivm.hir.vdeinterleave ins(%joined : memref<134xf32>)
        outs(%onlyOdd : memref<67xf32>) index_mode = <CHANNEL_1>
    %channelErrors = scf.for %i = %c0 to %c67 step %c1
        iter_args(%errors = %zero) -> i32 {
      %offset = arith.addi %i, %c3 : index
      %integer = arith.index_cast %offset : index to i32
      %expectedLhs = arith.sitofp %integer : i32 to f32
      %negative = arith.subi %hundred, %integer : i32
      %expectedRhs = arith.sitofp %negative : i32 to f32
      %evenIndex = arith.muli %i, %c2 : index
      %oddIndex = arith.addi %evenIndex, %c1 : index
      %joinedLhs = memref.load %joined[%evenIndex] : memref<134xf32>
      %joinedRhs = memref.load %joined[%oddIndex] : memref<134xf32>
      %actualLhs = memref.load %even[%i] : memref<67xf32>
      %actualRhs = memref.load %odd[%i] : memref<67xf32>
      %selectedRhs = memref.load %onlyOdd[%i] : memref<67xf32>
      %w0 = arith.cmpf une, %expectedLhs, %joinedLhs : f32
      %w1 = arith.cmpf une, %expectedRhs, %joinedRhs : f32
      %w2 = arith.cmpf une, %expectedLhs, %actualLhs : f32
      %w3 = arith.cmpf une, %expectedRhs, %actualRhs : f32
      %w4 = arith.cmpf une, %expectedRhs, %selectedRhs : f32
      %w01 = arith.ori %w0, %w1 : i1
      %w23 = arith.ori %w2, %w3 : i1
      %w03 = arith.ori %w01, %w23 : i1
      %wrong = arith.ori %w03, %w4 : i1
      %delta = arith.select %wrong, %one, %zero : i32
      %next = arith.addi %errors, %delta : i32
      scf.yield %next : i32
    }
    %fillErrors = scf.for %i = %c0 to %c73 step %c1
        iter_args(%errors = %channelErrors) -> i32 {
      %lower = arith.cmpi uge, %i, %c3 : index
      %upper = arith.cmpi ult, %i, %c70 : index
      %inside = arith.andi %lower, %upper : i1
      %expected = arith.select %inside, %fill, %sentinel : f32
      %actual = memref.load %filled[%i] : memref<73xf32>
      %wrong = arith.cmpf une, %expected, %actual : f32
      %delta = arith.select %wrong, %one, %zero : i32
      %next = arith.addi %errors, %delta : i32
      scf.yield %next : i32
    }
    %matrix = memref.alloc() : memref<3x5xf32>
    %transposed = memref.alloc() : memref<5x3xf32>
    %c5 = arith.constant 5 : index
    scf.for %i = %c0 to %c3 step %c1 {
      scf.for %j = %c0 to %c5 step %c1 {
        %row = arith.muli %i, %c5 : index
        %index = arith.addi %row, %j : index
        %integer = arith.index_cast %index : index to i32
        %value = arith.sitofp %integer : i32 to f32
        memref.store %value, %matrix[%i, %j] : memref<3x5xf32>
      }
    }
    hivm.hir.vtranspose ins(%matrix : memref<3x5xf32>)
        outs(%transposed : memref<5x3xf32>)
        temp_buffer(%scratch : memref<300xf32>) permutation = [1, 0]
    %errors = scf.for %i = %c0 to %c3 step %c1
        iter_args(%total = %fillErrors) -> i32 {
      %nextRow = scf.for %j = %c0 to %c5 step %c1
          iter_args(%count = %total) -> i32 {
        %row = arith.muli %i, %c5 : index
        %index = arith.addi %row, %j : index
        %integer = arith.index_cast %index : index to i32
        %expected = arith.sitofp %integer : i32 to f32
        %actual = memref.load %transposed[%j, %i] : memref<5x3xf32>
        %wrong = arith.cmpf une, %expected, %actual : f32
        %delta = arith.select %wrong, %one, %zero : i32
        %next = arith.addi %count, %delta : i32
        scf.yield %next : i32
      }
      scf.yield %nextRow : i32
    }
    memref.dealloc %matrix : memref<3x5xf32>
    memref.dealloc %transposed : memref<5x3xf32>
    memref.dealloc %left : memref<73xf32>
    memref.dealloc %right : memref<73xf32>
    memref.dealloc %filled : memref<73xf32>
    memref.dealloc %joined : memref<134xf32>
    memref.dealloc %scratch : memref<300xf32>
    memref.dealloc %even : memref<67xf32>
    memref.dealloc %odd : memref<67xf32>
    memref.dealloc %onlyOdd : memref<67xf32>
    return %errors : i32
  }
}
