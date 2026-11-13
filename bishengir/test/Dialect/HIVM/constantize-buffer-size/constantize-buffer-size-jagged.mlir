// Registered jagged-DK kernel enables the 3160 byte-vs-byte compare without
// --bypass-shape-registry. Annotation 2048 is bytes, matching the 2048-byte
// constantized alloc (16*32*i32).
// RUN: bishengir-opt --hivm-constantize-buffer-size="enable-preload=true set-workspace-multibuffer=2" -allow-unregistered-dialect %s | FileCheck %s

#map = affine_map<(d0) -> (d0, 32)>
func.func @_jagged_dense_flash_attention_bwd_dk_kernel(%arg0 : index) {
  %size = affine.min #map(%arg0)
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2048xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]]
  // CHECK: "some_use"(%[[VIEW]])
  %alloc = memref.alloc(%size) : memref<?x16xi32>
  annotation.mark %alloc {buffer_size_in_byte = 2048 : i64} : memref<?x16xi32>
  "some_use"(%alloc) : (memref<?x16xi32>) -> ()
}
