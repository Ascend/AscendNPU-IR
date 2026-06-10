module {
  func.func @load_bubble_up(%arg5: memref<?xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %arg10 = %c0 to %c2 step %c1 {
      %0 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg10]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
      %alloc = memref.alloc() : memref<16x16xf32>
      hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc : memref<16x16xf32>)
      %to_tensor = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
      %slice = tensor.extract_slice %to_tensor[%0, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
      annotation.mark %slice : tensor<8x16xf32>
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}
