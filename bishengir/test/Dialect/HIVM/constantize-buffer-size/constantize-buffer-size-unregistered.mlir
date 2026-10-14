// Off-registry kernels keep the historical bits-vs-byte-attr compare, so an
// annotation of 16384 bits still constantizes a 2048-byte buffer.
// RUN: bishengir-opt --hivm-constantize-buffer-size -allow-unregistered-dialect %s | FileCheck %s

#map = affine_map<(d0) -> (d0, 32)>
func.func @test_unregistered_bits_compare(%arg0 : index) {
  %size = affine.min #map(%arg0)
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2048xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]]
  // CHECK: "some_use"(%[[VIEW]])
  %alloc = memref.alloc(%size) : memref<?x16xi32>
  annotation.mark %alloc {buffer_size_in_byte = 16384 : i64} : memref<?x16xi32>
  "some_use"(%alloc) : (memref<?x16xi32>) -> ()
}
