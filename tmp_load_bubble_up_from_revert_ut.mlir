#map = affine_map<()[s0] -> (s0 * 72)>
module {
  func.func @load_bubble_up(%arg0: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c144 = arith.constant 144 : index
    scf.for %arg5 = %c0 to %c2 step %c1 {
      %tile_offset = affine.apply #map()[%arg5]
      %0 = arith.muli %arg2, %arg3 : i32
      %1 = arith.muli %0, %arg4 : i32
      %2 = arith.index_cast %1 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [144, 64], strides: [256, 1] : memref<?xf32> to memref<144x64xf32, strided<[256, 1], offset: ?>>
      %alloc = memref.alloc() : memref<144x64xf32>
      %3 = arith.addi %2, %c144 : index
      %4 = arith.minsi %3, %2 : index
      %5 = arith.subi %4, %2 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%5, 64] [1, 1] : memref<144x64xf32, strided<[256, 1], offset: ?>> to memref<?x64xf32, strided<[256, 1], offset: ?>>
      %subview_0 = memref.subview %alloc[0, 0] [%5, 64] [1, 1] : memref<144x64xf32> to memref<?x64xf32, strided<[64, 1]>>
      hivm.hir.load ins(%subview : memref<?x64xf32, strided<[256, 1], offset: ?>>) outs(%subview_0 : memref<?x64xf32, strided<[64, 1]>>)
      %to_tensor = bufferization.to_tensor %alloc restrict writable : memref<144x64xf32>
      %slice = tensor.extract_slice %to_tensor[%tile_offset, 0] [72, 64] [1, 1] {to_be_bubbled_slice} : tensor<144x64xf32> to tensor<72x64xf32>
      annotation.mark %slice : tensor<72x64xf32>
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}
