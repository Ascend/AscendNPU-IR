// RUN: bishengir-compile --enable-lir-compile=false --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-sanitizer=true %s | FileCheck %s

// CHECK-DAG: call void @llvm.hivm.SET.FLAG.IMM({{.*}}, {{.*}}, {{.*}}), !dbg {{.*}}, !asan.cce.api.name ![[S_API:.*]], !asan.stub.mangling.name ![[S_STUB:.*]]
// CHECK-DAG: call void @llvm.hivm.WAIT.FLAG.IMM({{.*}}, {{.*}}, {{.*}}), !dbg {{.*}}, !asan.cce.api.name ![[W_API:.*]], !asan.stub.mangling.name ![[W_STUB:.*]]
// CHECK-DAG: ![[S_API]] = !{!"set_flag"}
// CHECK-DAG: ![[S_STUB]] = !{!"_Z27__sanitizer_report_set_flagPU3AS1hmmljjj"}
// CHECK-DAG: ![[W_API]] = !{!"wait_flag"}
// CHECK-DAG: ![[W_STUB]] = !{!"_Z28__sanitizer_report_wait_flagPU3AS1hmmljjj"}

#loc = loc("/home/dingshuo/workspace/test/san/test_add.py":14:0)
module {
  func.func @triton_add(%arg0: memref<?xi8> loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg1: memref<?xi8> {tt.divisibility = 16 : i32} loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg2: memref<?xi8> {tt.divisibility = 16 : i32} loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.shape_0 = 0 : i32, tt.shape_1 = 0 : i32, tt.shape_2 = 0 : i32} loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg4: i32 loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg5: i32 loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg6: i32 loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg7: i32 loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg8: i32 loc("/home/dingshuo/workspace/test/san/test_add.py":14:0), %arg9: i32 loc("/home/dingshuo/workspace/test/san/test_add.py":14:0)) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %c32768_i32 = arith.constant 32768 : i32 loc(#loc3)
    %0 = arith.muli %arg7, %c32768_i32 : i32 loc(#loc3)
    scf.for %arg10 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %1 = arith.muli %arg10, %c1024_i32 : i32 loc(#loc2)
      %2 = arith.addi %0, %1 : i32 loc(#loc4)
      %3 = arith.index_cast %2 : i32 to index loc(#loc4)
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>> loc(#loc5)
      %alloc = memref.alloc() : memref<1024xi8> loc(#loc6)
      memref.copy %reinterpret_cast, %alloc : memref<1024xi8, strided<[1], offset: ?>> to memref<1024xi8> loc(#loc6)
      %4 = bufferization.to_tensor %alloc restrict writable : memref<1024xi8> loc(#loc6)
      %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>> loc(#loc7)
      %alloc_1 = memref.alloc() : memref<1024xi8> loc(#loc8)
      memref.copy %reinterpret_cast_0, %alloc_1 : memref<1024xi8, strided<[1], offset: ?>> to memref<1024xi8> loc(#loc8)
      %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xi8> loc(#loc8)
      %6 = arith.addi %4, %5 : tensor<1024xi8> loc(#loc9)
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>> loc(#loc10)
      bufferization.materialize_in_destination %6 in writable %reinterpret_cast_2 : (tensor<1024xi8>, memref<1024xi8, strided<[1], offset: ?>>) -> () loc(#loc11)
    } loc(#loc1)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/dingshuo/workspace/test/san/test_add.py":18:23)
#loc2 = loc("/home/dingshuo/workspace/test/san/test_add.py":19:37)
#loc3 = loc("/home/dingshuo/workspace/test/san/test_add.py":15:32)
#loc4 = loc("/home/dingshuo/workspace/test/san/test_add.py":19:29)
#loc5 = loc("/home/dingshuo/workspace/test/san/test_add.py":21:34)
#loc6 = loc("/home/dingshuo/workspace/test/san/test_add.py":21:39)
#loc7 = loc("/home/dingshuo/workspace/test/san/test_add.py":22:34)
#loc8 = loc("/home/dingshuo/workspace/test/san/test_add.py":22:39)
#loc9 = loc("/home/dingshuo/workspace/test/san/test_add.py":23:22)
#loc10 = loc("/home/dingshuo/workspace/test/san/test_add.py":24:29)
#loc11 = loc("/home/dingshuo/workspace/test/san/test_add.py":24:40)
