// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-clone-tensor-empty --split-input-file %s | FileCheck %s

// -----
module {
  func.func @triton_vf_func_0(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<-1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c1024 step %c64 iter_args(%arg3 = %arg1) -> (tensor<1024xbf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_2 = tensor.extract_slice %arg3[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %1 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<64xbf16>, vector<64xbf16>
      %2 = arith.extf %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xbf16> to vector<64xf32>
      %3 = arith.mulf %2, %cst_0 : vector<64xf32>
      %4 = math.exp %3 : vector<64xf32>
      %5 = arith.addf %4, %cst : vector<64xf32>
      %6 = arith.divf %2, %5 : vector<64xf32>
      %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xbf16>
      %8 = vector.transfer_write %7, %extracted_slice_2[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %inserted_slice = tensor.insert_slice %8 into %arg3[%arg2] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      scf.yield %inserted_slice : tensor<1024xbf16>
    }
    return %0 : tensor<1024xbf16>
  }
  // CHECK-LABEL: test_clone_tensor_empty_with_vf_function_0
  func.func @test_clone_tensor_empty_with_vf_function_0(%arg0: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tensor.empty() : tensor<1024xbf16>
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %1 = arith.muli %arg1, %c1024_i32 : i32
      %2 = arith.addi %c0_i32, %1 : i32
      %3 = arith.index_cast %2 : i32 to index
      %alloc = memref.alloc() : memref<1024xbf16>
      // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
      %4 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
      %5 = func.call @triton_vf_func_0(%4, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %5[0] [%3] [1] : tensor<1024xbf16> to tensor<?xbf16>
      %subview = memref.subview %reinterpret_cast[0] [%3] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?xbf16>) outs(%subview : memref<?xbf16, strided<[1], offset: ?>>)
    }
    return
  }
}

// -----
module {
  func.func @triton_vf_func_0(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<-1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c1024 step %c64 iter_args(%arg3 = %arg1) -> (tensor<1024xbf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_2 = tensor.extract_slice %arg3[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %1 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<64xbf16>, vector<64xbf16>
      %2 = arith.extf %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xbf16> to vector<64xf32>
      %3 = arith.mulf %2, %cst_0 : vector<64xf32>
      %4 = math.exp %3 : vector<64xf32>
      %5 = arith.addf %4, %cst : vector<64xf32>
      %6 = arith.divf %2, %5 : vector<64xf32>
      %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xbf16>
      %8 = vector.transfer_write %7, %extracted_slice_2[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %inserted_slice = tensor.insert_slice %8 into %arg3[%arg2] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      scf.yield %inserted_slice : tensor<1024xbf16>
    }
    return %0 : tensor<1024xbf16>
  }
  // CHECK-LABEL: test_clone_tensor_empty_with_vf_function_1
  func.func @test_clone_tensor_empty_with_vf_function_1(%arg0: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tensor.empty() : tensor<1024xbf16>
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %1 = arith.muli %arg1, %c1024_i32 : i32
      %2 = arith.addi %c0_i32, %1 : i32
      %3 = arith.index_cast %2 : i32 to index
      %alloc = memref.alloc() : memref<1024xbf16>
      %4 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
      // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
      %5 = func.call @triton_vf_func_0(%4, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %5[0] [%3] [1] : tensor<1024xbf16> to tensor<?xbf16>
      %subview = memref.subview %reinterpret_cast[0] [%3] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?xbf16>) outs(%subview : memref<?xbf16, strided<[1], offset: ?>>)
    }
    scf.for %arg3 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %6 = arith.muli %arg3, %c1024_i32 : i32
      %7 = arith.addi %c0_i32, %6 : i32
      %8 = arith.index_cast %7 : i32 to index
      %alloc_1 = memref.alloc() : memref<1024xbf16>
      %9 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xbf16>
      %10 = func.call @triton_vf_func_0(%9, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
      %extracted_slice_1 = tensor.extract_slice %10[0] [%8] [1] : tensor<1024xbf16> to tensor<?xbf16>
      %subview_1 = memref.subview %reinterpret_cast_1[0] [%8] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      hivm.hir.store ins(%extracted_slice_1 : tensor<?xbf16>) outs(%subview_1 : memref<?xbf16, strided<[1], offset: ?>>)
    }
    return
  }
}

// -----
module {
  func.func @triton_vf_func_0(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<-1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c1024 step %c64 iter_args(%arg3 = %arg1) -> (tensor<1024xbf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_2 = tensor.extract_slice %arg3[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %1 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<64xbf16>, vector<64xbf16>
      %2 = arith.extf %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xbf16> to vector<64xf32>
      %3 = arith.mulf %2, %cst_0 : vector<64xf32>
      %4 = math.exp %3 : vector<64xf32>
      %5 = arith.addf %4, %cst : vector<64xf32>
      %6 = arith.divf %2, %5 : vector<64xf32>
      %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xbf16>
      %8 = vector.transfer_write %7, %extracted_slice_2[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %inserted_slice = tensor.insert_slice %8 into %arg3[%arg2] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      scf.yield %inserted_slice : tensor<1024xbf16>
    }
    return %0 : tensor<1024xbf16>
  }
  // CHECK-LABEL: test_clone_tensor_empty_with_vf_function_2
  func.func @test_clone_tensor_empty_with_vf_function_2(%arg0: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tensor.empty() : tensor<1024xbf16>
		scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
			%1 = arith.muli %arg1, %c1024_i32 : i32
			%2 = arith.addi %c0_i32, %1 : i32
			%3 = arith.index_cast %2 : i32 to index
			%alloc = memref.alloc() : memref<1024xbf16>
			%4 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
      // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
			%5 = func.call @triton_vf_func_0(%4, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
			%reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
			%extracted_slice = tensor.extract_slice %5[0] [%3] [1] : tensor<1024xbf16> to tensor<?xbf16>
			%subview = memref.subview %reinterpret_cast[0] [%3] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
			hivm.hir.store ins(%extracted_slice : tensor<?xbf16>) outs(%subview : memref<?xbf16, strided<[1], offset: ?>>)
			scf.for %arg2 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
				scf.for %arg3 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
					%6 = arith.muli %arg3, %c1024_i32 : i32
					%7 = arith.addi %c0_i32, %6 : i32
					%8 = arith.index_cast %7 : i32 to index
					%alloc_1 = memref.alloc() : memref<1024xbf16>
					%9 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xbf16>
					%10 = func.call @triton_vf_func_0(%9, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
					%reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
					%extracted_slice_1 = tensor.extract_slice %10[0] [%8] [1] : tensor<1024xbf16> to tensor<?xbf16>
					%subview_1 = memref.subview %reinterpret_cast_1[0] [%8] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
					hivm.hir.store ins(%extracted_slice_1 : tensor<?xbf16>) outs(%subview_1 : memref<?xbf16, strided<[1], offset: ?>>)
				}
			}
			scf.for %arg4 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
				%11 = arith.muli %arg4, %c1024_i32 : i32
				%12 = arith.addi %c0_i32, %11 : i32
				%13 = arith.index_cast %12 : i32 to index
				%alloc_2 = memref.alloc() : memref<1024xbf16>
				%14 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xbf16>
				%15 = func.call @triton_vf_func_0(%14, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
				%reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
				%extracted_slice_2 = tensor.extract_slice %15[0] [%13] [1] : tensor<1024xbf16> to tensor<?xbf16>
				%subview_2 = memref.subview %reinterpret_cast_2[0] [%13] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
				hivm.hir.store ins(%extracted_slice_2 : tensor<?xbf16>) outs(%subview_2 : memref<?xbf16, strided<[1], offset: ?>>)
			}
		}
    return
  }
}


// -----
module {
  func.func @triton_vf_func_0(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<-1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c1024 step %c64 iter_args(%arg3 = %arg1) -> (tensor<1024xbf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_2 = tensor.extract_slice %arg3[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %1 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<64xbf16>, vector<64xbf16>
      %2 = arith.extf %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xbf16> to vector<64xf32>
      %3 = arith.mulf %2, %cst_0 : vector<64xf32>
      %4 = math.exp %3 : vector<64xf32>
      %5 = arith.addf %4, %cst : vector<64xf32>
      %6 = arith.divf %2, %5 : vector<64xf32>
      %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xbf16>
      %8 = vector.transfer_write %7, %extracted_slice_2[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %inserted_slice = tensor.insert_slice %8 into %arg3[%arg2] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      scf.yield %inserted_slice : tensor<1024xbf16>
    }
    return %0 : tensor<1024xbf16>
  }
  // CHECK-LABEL: test_clone_tensor_empty_with_vf_function_2
  func.func @test_clone_tensor_empty_with_vf_function_2(%arg0: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tensor.empty() : tensor<1024xbf16>
		scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
			%1 = arith.muli %arg1, %c1024_i32 : i32
			%2 = arith.addi %c0_i32, %1 : i32
			%3 = arith.index_cast %2 : i32 to index
			%alloc = memref.alloc() : memref<1024xbf16>
			%4 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
      // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
			%5 = func.call @triton_vf_func_0(%4, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
			%reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
			%extracted_slice = tensor.extract_slice %5[0] [%3] [1] : tensor<1024xbf16> to tensor<?xbf16>
			%subview = memref.subview %reinterpret_cast[0] [%3] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
			hivm.hir.store ins(%extracted_slice : tensor<?xbf16>) outs(%subview : memref<?xbf16, strided<[1], offset: ?>>)
			scf.for %arg2 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
				scf.for %arg3 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
					%6 = arith.muli %arg3, %c1024_i32 : i32
					%7 = arith.addi %c0_i32, %6 : i32
					%8 = arith.index_cast %7 : i32 to index
					%alloc_1 = memref.alloc() : memref<1024xbf16>
					%9 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xbf16>
          // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
          // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
					%10 = func.call @triton_vf_func_0(%9, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
					%reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
					%extracted_slice_1 = tensor.extract_slice %10[0] [%8] [1] : tensor<1024xbf16> to tensor<?xbf16>
					%subview_1 = memref.subview %reinterpret_cast_1[0] [%8] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
					hivm.hir.store ins(%extracted_slice_1 : tensor<?xbf16>) outs(%subview_1 : memref<?xbf16, strided<[1], offset: ?>>)
				}
			}
			scf.for %arg4 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
				%11 = arith.muli %arg4, %c1024_i32 : i32
				%12 = arith.addi %c0_i32, %11 : i32
				%13 = arith.index_cast %12 : i32 to index
				%alloc_2 = memref.alloc() : memref<1024xbf16>
        // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
        // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
				%14 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xbf16>
				%15 = func.call @triton_vf_func_0(%14, %0) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
				%reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
				%extracted_slice_2 = tensor.extract_slice %15[0] [%13] [1] : tensor<1024xbf16> to tensor<?xbf16>
				%subview_2 = memref.subview %reinterpret_cast_2[0] [%13] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
				hivm.hir.store ins(%extracted_slice_2 : tensor<?xbf16>) outs(%subview_2 : memref<?xbf16, strided<[1], offset: ?>>)
			}
		}
    return
  }
}


// -----
module {
  func.func @triton_vf_func_0(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>, %arg2: tensor<1024xbf16>, %arg3: tensor<1024xbf16>) -> (tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<-1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0, %1, %2 = scf.for %arg4 = %c0 to %c1024 step %c64 iter_args(%arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3) -> (tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_2 = tensor.extract_slice %arg5[%arg4] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_3 = tensor.extract_slice %arg6[%arg4] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_4 = tensor.extract_slice %arg7[%arg4] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %3 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<64xbf16>, vector<64xbf16>
      %4 = arith.extf %3 {round_mode = #hfusion.round_mode<rint>} : vector<64xbf16> to vector<64xf32>
      %5 = arith.mulf %4, %cst_0 : vector<64xf32>
      %6 = math.exp %5 : vector<64xf32>
      %7 = arith.addf %6, %cst : vector<64xf32>
      %8 = arith.divf %6, %7 : vector<64xf32>
      %9 = arith.truncf %8 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xbf16>
      %10 = vector.transfer_write %9, %extracted_slice_2[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %11 = vector.transfer_write %9, %extracted_slice_3[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %12 = vector.transfer_write %9, %extracted_slice_4[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %inserted_slice = tensor.insert_slice %10 into %arg5[%arg4] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      %inserted_slice_1 = tensor.insert_slice %11 into %arg6[%arg4] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      %inserted_slice_2 = tensor.insert_slice %12 into %arg7[%arg4] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      scf.yield %inserted_slice, %inserted_slice_1, %inserted_slice_2 : tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>
    }
    return %0, %1, %2 : tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>
  }
  // CHECK-LABEL: test_clone_tensor_empty_with_vf_function_0
  func.func @test_clone_tensor_empty_with_vf_function_0(%arg0: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tensor.empty() : tensor<1024xbf16>
    %1 = tensor.empty() : tensor<1024xbf16>
    %2 = tensor.empty() : tensor<1024xbf16>
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %3 = arith.muli %arg1, %c1024_i32 : i32
      %4 = arith.addi %c0_i32, %3 : i32
      %5 = arith.index_cast %4 : i32 to index
      %alloc = memref.alloc() : memref<1024xbf16>
      // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
      %6 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
      %7, %8, %9 = func.call @triton_vf_func_0(%6, %0, %1, %2) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>) -> (tensor<1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>)
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %7[0] [%5] [1] : tensor<1024xbf16> to tensor<?xbf16>
      %subview = memref.subview %reinterpret_cast[0] [%5] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?xbf16>) outs(%subview : memref<?xbf16, strided<[1], offset: ?>>)
    }
    return
  }
}


// -----
module {
  func.func @triton_vf_func_0(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<-1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c1024 step %c64 iter_args(%arg3 = %arg1) -> (tensor<1024xbf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_2 = tensor.extract_slice %arg3[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %1 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<64xbf16>, vector<64xbf16>
      %2 = arith.extf %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xbf16> to vector<64xf32>
      %3 = arith.mulf %2, %cst_0 : vector<64xf32>
      %4 = math.exp %3 : vector<64xf32>
      %5 = arith.addf %4, %cst : vector<64xf32>
      %6 = arith.divf %2, %5 : vector<64xf32>
      %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xbf16>
      %8 = vector.transfer_write %7, %extracted_slice_2[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %inserted_slice = tensor.insert_slice %8 into %arg3[%arg2] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      scf.yield %inserted_slice : tensor<1024xbf16>
    }
    return %0 : tensor<1024xbf16>
  }
  func.func @triton_vf_func_1(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<-1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c1024 step %c64 iter_args(%arg3 = %arg1) -> (tensor<1024xbf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %extracted_slice_2 = tensor.extract_slice %arg3[%arg2] [64] [1] : tensor<1024xbf16> to tensor<64xbf16>
      %1 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<64xbf16>, vector<64xbf16>
      %2 = arith.extf %1 {round_mode = #hfusion.round_mode<rint>} : vector<64xbf16> to vector<64xf32>
      %3 = arith.mulf %2, %cst_0 : vector<64xf32>
      %4 = math.exp %3 : vector<64xf32>
      %5 = arith.addf %4, %cst : vector<64xf32>
      %6 = arith.divf %2, %5 : vector<64xf32>
      %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xbf16>
      %8 = vector.transfer_write %7, %extracted_slice_2[%c0] {in_bounds = [true]} : vector<64xbf16>, tensor<64xbf16>
      %inserted_slice = tensor.insert_slice %8 into %arg3[%arg2] [64] [1] : tensor<64xbf16> into tensor<1024xbf16>
      scf.yield %inserted_slice : tensor<1024xbf16>
    }
    return %0 : tensor<1024xbf16>
  }
  // CHECK-LABEL: test_clone_tensor_empty_with_vf_function_0
  func.func @test_clone_tensor_empty_with_vf_function_0(%arg0: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1024_i32 = arith.constant 1024 : i32
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %0 = arith.muli %arg1, %c1024_i32 : i32
      %1 = arith.addi %c0_i32, %0 : i32
      %2 = arith.index_cast %1 : i32 to index
      %alloc = memref.alloc() : memref<1024xbf16>
      // CHECK: %{{.*}} = bufferization.to_tensor %{{.*}} restrict writable : memref
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
      // CHECK-NEXT: %{{.*}} = func.call @triton_vf_func_0
      // CHECK-NEXT: %{{.*}} = tensor.empty() : tensor<1024xbf16>
      // CHECK-NEXT: %{{.*}} = func.call @triton_vf_func_1
      %3 = tensor.empty() : tensor<1024xbf16>
      %4 = bufferization.to_tensor %alloc restrict writable : memref<1024xbf16>
      %5 = func.call @triton_vf_func_0(%4, %3) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
      %6 = func.call @triton_vf_func_1(%4, %3) {hivm.vector_function} : (tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<1024xbf16>
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %5[0] [%2] [1] : tensor<1024xbf16> to tensor<?xbf16>
      %subview = memref.subview %reinterpret_cast[0] [%2] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      %extracted_slice_1 = tensor.extract_slice %6[0] [%2] [1] : tensor<1024xbf16> to tensor<?xbf16>
      %subview_1 = memref.subview %reinterpret_cast[20480] [%2] [1] : memref<1024xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?xbf16>) outs(%subview : memref<?xbf16, strided<[1], offset: ?>>)
      hivm.hir.store ins(%extracted_slice_1 : tensor<?xbf16>) outs(%subview_1 : memref<?xbf16, strided<[1], offset: ?>>)
    }
    return
  }
}