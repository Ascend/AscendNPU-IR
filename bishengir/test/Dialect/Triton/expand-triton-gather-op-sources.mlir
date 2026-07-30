// RUN: bishengir-opt -split-input-file -expand-gather-op-sources %s | FileCheck %s

// CHECK-LABEL: @small
// CHECK-NOT: tt.expand_dims
// CHECK-NOT: tt.broadcast
// CHECK-NOT: tt.reshape
// CHECK: tt.gather %{{.*}}[%{{.*}}] {axis = 1 : i32} : (tensor<4x1xf16>, tensor<4x2xi32>) -> tensor<4x2xf16>
tt.func @small(%src: tensor<4x1xf16>, %ind: tensor<4x2xi32>) -> tensor<4x2xf16> {
  %res = tt.gather %src[%ind] {axis = 1 : i32} : (tensor<4x1xf16>, tensor<4x2xi32>) -> tensor<4x2xf16>
  tt.return %res : tensor<4x2xf16>
}

// -----

// CHECK-LABEL: @sizeBoundary
// CHECK-NOT: tt.expand_dims
// CHECK-NOT: tt.broadcast
// CHECK-NOT: tt.reshape
// CHECK: tt.gather %{{.*}}[%{{.*}}] {axis = 1 : i32} : (tensor<4x1xf16>, tensor<4x8xi32>) -> tensor<4x8xf16>
tt.func @sizeBoundary(%src: tensor<4x1xf16>, %ind: tensor<4x8xi32>) -> tensor<4x8xf16> {
  %res = tt.gather %src[%ind] {axis = 1 : i32} : (tensor<4x1xf16>, tensor<4x8xi32>) -> tensor<4x8xf16>
  tt.return %res : tensor<4x8xf16>
}

// -----

// CHECK-LABEL: @sameSize
// CHECK-NOT: tt.expand_dims
// CHECK-NOT: tt.broadcast
// CHECK-NOT: tt.reshape
// CHECK: tt.gather %{{.*}}[%{{.*}}] {axis = 0 : i32} : (tensor<16x4x2xf32>, tensor<16x4x2xi32>) -> tensor<16x4x2xf32>
tt.func @sameSize(%src: tensor<16x4x2xf32>, %ind: tensor<16x4x2xi32>) -> tensor<16x4x2xf32> {
  %res = tt.gather %src[%ind] {axis = 0 : i32} : (tensor<16x4x2xf32>, tensor<16x4x2xi32>) -> tensor<16x4x2xf32>
  tt.return %res : tensor<16x4x2xf32>
}

// -----

// CHECK-LABEL: @halfSizeAxis0
// CHECK-SAME: (%[[SRC:.*]]: tensor<2x2x8xbf16>, %[[IND:.*]]: tensor<4x2x8xi32>)
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SRC]] {axis = 0 : i32}
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPANDED]] : tensor<1x2x2x8xbf16> -> tensor<2x2x2x8xbf16>
// CHECK: %[[RESHAPE:.*]] = tt.reshape %[[BROADCAST]] : tensor<2x2x2x8xbf16> -> tensor<4x2x8xbf16>
// CHECK: tt.gather %[[RESHAPE]][%[[IND]]] {axis = 0 : i32} : (tensor<4x2x8xbf16>, tensor<4x2x8xi32>) -> tensor<4x2x8xbf16>
tt.func @halfSizeAxis0(%src: tensor<2x2x8xbf16>, %ind: tensor<4x2x8xi32>) -> tensor<4x2x8xbf16> {
  %res = tt.gather %src[%ind] {axis = 0 : i32} : (tensor<2x2x8xbf16>, tensor<4x2x8xi32>) -> tensor<4x2x8xbf16>
  tt.return %res : tensor<4x2x8xbf16>
}

// -----

// CHECK-LABEL: @halfSizeAxis1
// CHECK-SAME: (%[[SRC:.*]]: tensor<4x2x4xbf16>, %[[IND:.*]]: tensor<4x4x4xi32>)
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SRC]] {axis = 1 : i32}
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPANDED]] : tensor<4x1x2x4xbf16> -> tensor<4x2x2x4xbf16>
// CHECK: %[[RESHAPE:.*]] = tt.reshape %[[BROADCAST]] : tensor<4x2x2x4xbf16> -> tensor<4x4x4xbf16>
// CHECK: tt.gather %[[RESHAPE]][%[[IND]]] {axis = 1 : i32} : (tensor<4x4x4xbf16>, tensor<4x4x4xi32>) -> tensor<4x4x4xbf16>
tt.func @halfSizeAxis1(%src: tensor<4x2x4xbf16>, %ind: tensor<4x4x4xi32>) -> tensor<4x4x4xbf16> {
  %res = tt.gather %src[%ind] {axis = 1 : i32} : (tensor<4x2x4xbf16>, tensor<4x4x4xi32>) -> tensor<4x4x4xbf16>
  tt.return %res : tensor<4x4x4xbf16>
}


// -----

// CHECK-LABEL: @halfSizeDimSize1
// CHECK-SAME: (%[[SRC:.*]]: tensor<16x4x1xf16>, %[[IND:.*]]: tensor<16x4x2xi32>)
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[SRC]] : tensor<16x4x1xf16> -> tensor<16x4x2xf16>
// CHECK: tt.gather %[[BROADCAST]][%[[IND]]] {axis = 2 : i32} : (tensor<16x4x2xf16>, tensor<16x4x2xi32>) -> tensor<16x4x2xf16>
tt.func @halfSizeDimSize1(%src: tensor<16x4x1xf16>, %ind: tensor<16x4x2xi32>) -> tensor<16x4x2xf16> {
  %res = tt.gather %src[%ind] {axis = 2 : i32} : (tensor<16x4x1xf16>, tensor<16x4x2xi32>) -> tensor<16x4x2xf16>
  tt.return %res : tensor<16x4x2xf16>
}

// -----

// CHECK-LABEL: @quarterSizeAxis0
// CHECK-SAME: (%[[SRC:.*]]: tensor<16xbf16>, %[[IND:.*]]: tensor<64xi32>) -> tensor<64xbf16>
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SRC]] {axis = 0 : i32}
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPANDED]] : tensor<1x16xbf16> -> tensor<4x16xbf16>
// CHECK: %[[RESHAPE:.*]] = tt.reshape %[[BROADCAST]] : tensor<4x16xbf16> -> tensor<64xbf16>
// CHECK: tt.gather %[[RESHAPE]][%[[IND]]] {axis = 0 : i32} : (tensor<64xbf16>, tensor<64xi32>) -> tensor<64xbf16>
tt.func @quarterSizeAxis0(%src: tensor<16xbf16>, %ind: tensor<64xi32>) -> tensor<64xbf16> {
  %res = tt.gather %src[%ind] {axis = 0 : i32} : (tensor<16xbf16>, tensor<64xi32>) -> tensor<64xbf16>
  tt.return %res : tensor<64xbf16>
}

// -----

// CHECK-LABEL: @quarterSizeAxis1
// CHECK-SAME: (%[[SRC:.*]]: tensor<2x8xbf16>, %[[IND:.*]]: tensor<2x32xi32>)
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[SRC]] {axis = 1 : i32}
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPANDED]] : tensor<2x1x8xbf16> -> tensor<2x4x8xbf16>
// CHECK: %[[RESHAPE:.*]] = tt.reshape %[[BROADCAST]] : tensor<2x4x8xbf16> -> tensor<2x32xbf16>
// CHECK: tt.gather %[[RESHAPE]][%[[IND]]] {axis = 1 : i32} : (tensor<2x32xbf16>, tensor<2x32xi32>) -> tensor<2x32xbf16>
tt.func @quarterSizeAxis1(%src: tensor<2x8xbf16>, %ind: tensor<2x32xi32>) -> tensor<2x32xbf16> {
  %res = tt.gather %src[%ind] {axis = 1 : i32} : (tensor<2x8xbf16>, tensor<2x32xi32>) -> tensor<2x32xbf16>
  tt.return %res : tensor<2x32xbf16>
}

// -----

// CHECK-LABEL: @quarterSizeDimSize1
// CHECK-SAME: (%[[SRC:.*]]: tensor<4x4x1xf16>, %[[IND:.*]]: tensor<4x4x4xi32>)
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[SRC]] : tensor<4x4x1xf16> -> tensor<4x4x4xf16>
// CHECK: tt.gather %[[BROADCAST]][%[[IND]]] {axis = 2 : i32} : (tensor<4x4x4xf16>, tensor<4x4x4xi32>) -> tensor<4x4x4xf16>
tt.func @quarterSizeDimSize1(%src: tensor<4x4x1xf16>, %ind: tensor<4x4x4xi32>) -> tensor<4x4x4xf16> {
  %res = tt.gather %src[%ind] {axis = 2 : i32} : (tensor<4x4x1xf16>, tensor<4x4x4xi32>) -> tensor<4x4x4xf16>
  tt.return %res : tensor<4x4x4xf16>
}