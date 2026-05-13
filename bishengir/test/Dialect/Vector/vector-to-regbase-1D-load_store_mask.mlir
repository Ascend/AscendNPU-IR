// RUN: bishengir-opt -split-input-file -convert-vector-to-hivmave %s | FileCheck %s

func.func @vector_to_regbase_load_16xf32(%base: memref<16xf32>) -> vector<16xf32> {
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.vload <NORM> %{{.*}} : memref<16xf32> into vector<16xf32>
  %1 = vector.load %base[%0]: memref<16xf32>, vector<16xf32>
  return %1 : vector<16xf32>
}

func.func @vector_to_regbase_load_32xf32(%base: memref<32xf32>) -> vector<32xf32> {
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.vload <NORM> %{{.*}} : memref<32xf32> into vector<32xf32>
  %1 = vector.load %base[%0]: memref<32xf32>, vector<32xf32>
  return %1 : vector<32xf32>
}

func.func @vector_to_regbase_load_64xf32(%base: memref<64xf32>) -> vector<64xf32> {
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.vload <NORM> %{{.*}} : memref<64xf32> into vector<64xf32>
  %1 = vector.load %base[%0]: memref<64xf32>, vector<64xf32>
  return %1 : vector<64xf32>
}

func.func @vector_to_regbase_store_16xf32(%valueToStore: vector<16xf32>) {
  %base = arith.constant dense<1.000000e+00> : memref<16xf32>
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.pge <ALL> : vector<16xi1>
  // CHECK: ave.hir.masked_store <NORM_B32> %{{.*}}, %{{.*}}, %{{.*}} : memref<16xf32>, vector<16xi1>, vector<16xf32>
  vector.store %valueToStore, %base[%0] : memref<16xf32>, vector<16xf32>
  return
}

func.func @vector_to_regbase_store_32xf32(%valueToStore: vector<32xf32>) {
  %base = arith.constant dense<1.000000e+00> : memref<32xf32>
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.pge <ALL> : vector<32xi1>
  // CHECK: ave.hir.masked_store <NORM_B32> %{{.*}}, %{{.*}}, %{{.*}} : memref<32xf32>, vector<32xi1>, vector<32xf32>
  vector.store %valueToStore, %base[%0] : memref<32xf32>, vector<32xf32>
  return
}

func.func @vector_to_regbase_store_64xf32(%valueToStore: vector<64xf32>) {
  %base = arith.constant dense<1.000000e+00> : memref<64xf32>
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: ave.hir.masked_store <NORM_B32> %{{.*}}, %{{.*}}, %{{.*}} : memref<64xf32>, vector<64xi1>, vector<64xf32>
  vector.store %valueToStore, %base[%0] : memref<64xf32>, vector<64xf32>
  return
}

func.func @vector_to_regbase_maskedload_16xf32(%base: memref<16xf32>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.vload <NORM> %{{.*}} : memref<16xf32> into vector<16xf32>
  // CHECK: %{{.*}} = ave.hir.vsel %{{.*}}, %{{.*}}, %{{.*}} : vector<16xi1>, vector<16xf32>
  %1 = vector.maskedload %base[%0], %mask, %pass_thru : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %1 : vector<16xf32>
}

func.func @vector_to_regbase_maskedload_32xf32(%base: memref<32xf32>, %mask: vector<32xi1>, %pass_thru: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.vload <NORM> %{{.*}} : memref<32xf32> into vector<32xf32>
  // CHECK: %{{.*}} = ave.hir.vsel %{{.*}}, %{{.*}}, %{{.*}} : vector<32xi1>, vector<32xf32>
  %1 = vector.maskedload %base[%0], %mask, %pass_thru : memref<32xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
  return %1 : vector<32xf32>
}

func.func @vector_to_regbase_maskedload_64xf32(%base: memref<64xf32>, %mask: vector<64xi1>, %pass_thru: vector<64xf32>) -> vector<64xf32> {
  %0 = arith.constant 0 : index
  // CHECK: %{{.*}} = ave.hir.vload <NORM> %{{.*}} : memref<64xf32> into vector<64xf32>
  // CHECK: %{{.*}} = ave.hir.vsel %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi1>, vector<64xf32>
  %1 = vector.maskedload %base[%0], %mask, %pass_thru : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  return %1 : vector<64xf32>
}

func.func @vector_to_regbase_maskedStore_16xf32(%base: memref<16xf32>, %mask: vector<16xi1>, %valueToStore: vector<16xf32>) {
  %0 = arith.constant 0 : index
  // CHECK: ave.hir.masked_store <NORM_B32> %{{.*}}, %{{.*}}, %{{.*}} : memref<16xf32>, vector<16xi1>, vector<16xf32>
  vector.maskedstore %base[%0], %mask, %valueToStore : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

func.func @vector_to_regbase_maskedStore_32xf32(%base: memref<32xf32>, %mask: vector<32xi1>, %valueToStore: vector<32xf32>) {
  %0 = arith.constant 0 : index
  // CHECK: ave.hir.masked_store <NORM_B32> %{{.*}}, %{{.*}}, %{{.*}} : memref<32xf32>, vector<32xi1>, vector<32xf32>
  vector.maskedstore %base[%0], %mask, %valueToStore : memref<32xf32>, vector<32xi1>, vector<32xf32>
  return
}

func.func @vector_to_regbase_maskedStore_64xf32(%base: memref<64xf32>, %mask: vector<64xi1>, %valueToStore: vector<64xf32>) {
  %0 = arith.constant 0 : index
  // CHECK: ave.hir.masked_store <NORM_B32> %{{.*}}, %{{.*}}, %{{.*}} : memref<64xf32>, vector<64xi1>, vector<64xf32>
  vector.maskedstore %base[%0], %mask, %valueToStore : memref<64xf32>, vector<64xi1>, vector<64xf32>
  return
}

func.func @vector_to_regbase_createmask() -> (vector<64xi1>, vector<128xi1>, vector<256xi1>) {
  %0 = arith.constant 8 : index
  // CHECK: %{{.*}}, %{{.*}} = ave.hir.plt %{{.*}} : vector<64xi1>, index
  %mask0 = vector.create_mask %0 : vector<64xi1>
  // CHECK: %{{.*}}, %{{.*}} = ave.hir.plt %{{.*}} : vector<128xi1>, index
  %mask1 = vector.create_mask %0 : vector<128xi1>
  // CHECK: %{{.*}}, %{{.*}} = ave.hir.plt %{{.*}} : vector<256xi1>, index
  %mask2 = vector.create_mask %0 : vector<256xi1>
  return %mask0, %mask1, %mask2 : vector<64xi1>, vector<128xi1>, vector<256xi1>
}

func.func @vector_to_regbase_constantmask() -> (
    vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>,
    vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>,
    vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>) {
  // CHECK: %{{.*}} = ave.hir.pge <ALL> : vector<64xi1>
  %mask64_ALL = vector.constant_mask [64] : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL32> : vector<64xi1>
  %mask64_VL32 = vector.constant_mask [32] : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL16> : vector<64xi1>
  %mask64_VL16 = vector.constant_mask [16] : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL8> : vector<64xi1>
  %mask64_VL8 = vector.constant_mask [8] : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL4> : vector<64xi1>
  %mask64_VL4 = vector.constant_mask [4] : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL3> : vector<64xi1>
  %mask64_VL3 = vector.constant_mask [3] : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL2> : vector<64xi1>
  %mask64_VL2 = vector.constant_mask [2] : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL1> : vector<64xi1>
  %mask64_VL1 = vector.constant_mask [1] : vector<64xi1>

  // CHECK: %{{.*}} = ave.hir.pge <ALL> : vector<128xi1>
  %mask128_ALL = vector.constant_mask [128] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL64> : vector<128xi1>
  %mask128_VL64 = vector.constant_mask [64] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL32> : vector<128xi1>
  %mask128_VL32 = vector.constant_mask [32] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL16> : vector<128xi1>
  %mask128_VL16 = vector.constant_mask [16] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL8> : vector<128xi1>
  %mask128_VL8 = vector.constant_mask [8] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL4> : vector<128xi1>
  %mask128_VL4 = vector.constant_mask [4] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL3> : vector<128xi1>
  %mask128_VL3 = vector.constant_mask [3] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL2> : vector<128xi1>
  %mask128_VL2 = vector.constant_mask [2] : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL1> : vector<128xi1>
  %mask128_VL1 = vector.constant_mask [1] : vector<128xi1>

  // CHECK: %{{.*}} = ave.hir.pge <ALL> : vector<256xi1>
  %mask256_ALL = vector.constant_mask [256] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL128> : vector<256xi1>
  %mask256_VL128 = vector.constant_mask [128] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL64> : vector<256xi1>
  %mask256_VL64 = vector.constant_mask [64] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL32> : vector<256xi1>
  %mask256_VL32 = vector.constant_mask [32] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL16> : vector<256xi1>
  %mask256_VL16 = vector.constant_mask [16] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL8> : vector<256xi1>
  %mask256_VL8 = vector.constant_mask [8] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL4> : vector<256xi1>
  %mask256_VL4 = vector.constant_mask [4] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL3> : vector<256xi1>
  %mask256_VL3 = vector.constant_mask [3] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL2> : vector<256xi1>
  %mask256_VL2 = vector.constant_mask [2] : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.pge <VL1> : vector<256xi1>
  %mask256_VL1 = vector.constant_mask [1] : vector<256xi1>
  return %mask64_ALL, %mask64_VL32, %mask64_VL16, %mask64_VL8, %mask64_VL4, %mask64_VL3, %mask64_VL2, %mask64_VL1,
         %mask128_ALL, %mask128_VL64, %mask128_VL32, %mask128_VL16, %mask128_VL8, %mask128_VL4, %mask128_VL3, %mask128_VL2, %mask128_VL1,
         %mask256_ALL, %mask256_VL128, %mask256_VL64, %mask256_VL32, %mask256_VL16, %mask256_VL8, %mask256_VL4, %mask256_VL3, %mask256_VL2, %mask256_VL1:
         vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>,
         vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>, vector<128xi1>,
         vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>
}
