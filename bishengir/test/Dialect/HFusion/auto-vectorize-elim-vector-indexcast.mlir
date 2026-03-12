// RUN: bishengir-opt %s -normalize-arith -verify-each | FileCheck %s

module {
  func.func @gather_i32_index(%tbl: tensor<320xi32>,%idx_high: vector<64xi32>,%idx_low:  vector<64xi32>) -> (vector<64xi32>, vector<64xi32>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %mask = arith.constant dense<true> : vector<64xi1>
    %passthru = arith.constant dense<0> : vector<64xi32>
    %idx_high_index = arith.index_castui %idx_high : vector<64xi32> to vector<64xindex>
    %idx_low_index  = arith.index_cast %idx_low  : vector<64xi32> to vector<64xindex>
    %vhigh = vector.mask %mask {
      vector.gather %tbl[%c0] [%idx_high_index], %mask, %passthru : tensor<320xi32>, vector<64xindex>, vector<64xi1>, vector<64xi32> into vector<64xi32>
    } : vector<64xi1> -> vector<64xi32>
    %vlow = vector.mask %mask {
      vector.gather %tbl[%c0] [%idx_low_index],%mask, %passthru
        : tensor<320xi32>, vector<64xindex>, vector<64xi1>, vector<64xi32> into vector<64xi32>
    } : vector<64xi1> -> vector<64xi32>
    return %vhigh, %vlow : vector<64xi32>, vector<64xi32>
  }

// CHECK-LABEL: func.func @gather_i32_index
// CHECK-NOT: arith.index_cast
// CHECK: %[[G0:.*]] = vector.gather %arg0[%c0] [%arg1], %cst, %cst_0 : tensor<320xi32>, vector<64xi32>, vector<64xi1>, vector<64xi32> into vector<64xi32>
// CHECK: %[[G1:.*]] = vector.gather %arg0[%c0] [%arg2], %cst, %cst_0 : tensor<320xi32>, vector<64xi32>, vector<64xi1>, vector<64xi32> into vector<64xi32>
// CHECK: return %[[G0]], %[[G1]] : vector<64xi32>, vector<64xi32>
  func.func @scalar_index_cast_kept(%tbl: tensor<320xi32>, %i: i32) -> i32 attributes {hivm.vector_function} {
    %idx = arith.index_castui %i : i32 to index
    %v = tensor.extract %tbl[%idx] : tensor<320xi32>
    return %v : i32
  }

// CHECK-LABEL: func.func @scalar_index_cast_kept
// CHECK: %[[IDX:.*]] = arith.index_castui %arg1 : i32 to index
// CHECK: %[[EX:.*]] = tensor.extract %arg0[%[[IDX]]] : tensor<320xi32>
// CHECK: return %[[EX]] : i32
}

