// RUN: bishengir-compile --enable-lir-compile=false --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true %s | FileCheck %s

// CHECK-DAG: @_debug_prefix_0
// CHECK-DAG: @_debug_prefix_1
// CHECK-DAG: @_debug_prefix_2
// CHECK-DAG: @_debug_prefix_3
// CHECK-DAG: @_debug_prefix_4
// CHECK-DAG: @_debug_prefix_5
// CHECK-DAG: @_debug_prefix_6
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_init_debug
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_assert_scalar_bool_gm
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_assert_1d_int8_t_ubuf
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_finish_debug
#map = affine_map<(d0) -> (d0)>
module {
  func.func private @triton_assert_0(i1) attributes {msg = "int32 overflow detected for operation mul"}
  func.func private @triton_assert_1(i1) attributes {msg = "int32 overflow detected for operation mul"}
  func.func private @triton_assert_2(i1) attributes {msg = "int32 overflow detected for operation add"}
  func.func private @triton_assert_3(tensor<1024xi1>) attributes {msg = "int32 overflow detected for operation add"}
  func.func private @triton_assert_4(i1) attributes {msg = "int32 overflow detected for operation mul"}
  func.func private @triton_assert_5(i1) attributes {msg = "int32 overflow detected for operation add"}
  func.func private @triton_assert_6(tensor<1024xi1>) attributes {msg = "int32 overflow detected for operation add"}
  func.func @triton_add(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c32768_i32 = arith.constant 32768 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %0 = tensor.empty() : tensor<1024xi64>
    %1 = linalg.fill ins(%c-2147483648_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
    %2 = linalg.fill ins(%c2147483647_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
    %3 = arith.extsi %arg8 : i32 to i64
    %4 = arith.muli %3, %c32768_i64 : i64
    %5 = arith.cmpi sle, %4, %c2147483647_i64 : i64
    %6 = arith.cmpi sge, %4, %c-2147483648_i64 : i64
    %7 = arith.andi %5, %6 : i1
    call @triton_assert_0(%7) : (i1) -> ()
    %8 = arith.muli %arg8, %c32768_i32 : i32
    %9 = tensor.empty() : tensor<1024xi32>
    %10 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%9 : tensor<1024xi32>) {
    ^bb0(%out: i32):
      %13 = linalg.index 0 : index
      %14 = arith.index_cast %13 : index to i32
      linalg.yield %14 : i32
    } -> tensor<1024xi32>
    %11 = arith.extsi %8 : i32 to i64
    %12 = arith.extsi %10 : tensor<1024xi32> to tensor<1024xi64>
    scf.for %arg11 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %13 = arith.extsi %arg11 : i32 to i64
      %14 = arith.muli %13, %c1024_i64 : i64
      %15 = arith.cmpi sle, %14, %c2147483647_i64 : i64
      %16 = arith.cmpi sge, %14, %c-2147483648_i64 : i64
      %17 = arith.andi %15, %16 : i1
      func.call @triton_assert_1(%17) : (i1) -> ()
      %18 = arith.muli %arg11, %c1024_i32 : i32
      %19 = arith.extsi %18 : i32 to i64
      %20 = arith.addi %11, %19 : i64
      %21 = arith.cmpi sle, %20, %c2147483647_i64 : i64
      %22 = arith.cmpi sge, %20, %c-2147483648_i64 : i64
      %23 = arith.andi %21, %22 : i1
      func.call @triton_assert_2(%23) : (i1) -> ()
      %24 = arith.addi %8, %18 : i32
      %25 = arith.extsi %24 : i32 to i64
      %26 = linalg.fill ins(%25 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
      %27 = arith.addi %26, %12 : tensor<1024xi64>
      %28 = arith.cmpi sle, %27, %2 : tensor<1024xi64>
      %29 = arith.cmpi sge, %27, %1 : tensor<1024xi64>
      %30 = arith.andi %28, %29 : tensor<1024xi1>
      func.call @triton_assert_3(%30) : (tensor<1024xi1>) -> ()
      func.call @triton_assert_4(%17) : (i1) -> ()
      func.call @triton_assert_5(%23) : (i1) -> ()
      func.call @triton_assert_6(%30) : (tensor<1024xi1>) -> ()
      %31 = arith.index_cast %24 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<1024xf32>
      memref.copy %reinterpret_cast, %alloc : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
      %32 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc_1 = memref.alloc() : memref<1024xf32>
      memref.copy %reinterpret_cast_0, %alloc_1 : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
      %33 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xf32>
      %34 = arith.addf %32, %33 : tensor<1024xf32>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %34 in writable %reinterpret_cast_2 : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}