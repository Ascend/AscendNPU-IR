// RUN: bishengir-opt -simt-fast-div %s | FileCheck %s --implicit-check-not=_mlir_ciface_simt_div_magic --implicit-check-not=tt.call_scalar --implicit-check-not=ttg.local_

// Constant division and remainder stay available for backend constant lowering,
// without runtime helper declarations or shared-memory allocations and loads.
// Preserve signedness even when the divisor's high bit is set.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @scalar_constants
  // CHECK: [[C2:%[^ ]+]] = arith.constant 2 : i32
  // CHECK: [[C30:%[^ ]+]] = arith.constant 30 : i32
  // CHECK: [[C128:%[^ ]+]] = arith.constant 128 : i32
  // CHECK: [[MIN:%[^ ]+]] = arith.constant -2147483648 : i32
  // CHECK: [[QS:%[^ ]+]] = arith.divsi {{%[^ ]+}}, [[C2]] : i32
  // CHECK: [[QU:%[^ ]+]] = arith.divui {{%[^ ]+}}, [[C30]] : i32
  // CHECK: [[RS:%[^ ]+]] = arith.remsi {{%[^ ]+}}, [[MIN]] : i32
  // CHECK: [[RU:%[^ ]+]] = arith.remui {{%[^ ]+}}, [[C128]] : i32
  // CHECK: [[QM:%[^ ]+]] = arith.divsi {{%[^ ]+}}, [[MIN]] : i32
  tt.func public @scalar_constants(%a: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %c30 = arith.constant 30 : i32
    %c128 = arith.constant 128 : i32
    %min = arith.constant -2147483648 : i32
    %qs = arith.divsi %a, %c2 : i32
    %qu = arith.divui %a, %c30 : i32
    %rs = arith.remsi %a, %min : i32
    %ru = arith.remui %a, %c128 : i32
    %qm = arith.divsi %a, %min : i32
    %sum0 = arith.addi %qs, %qu : i32
    %sum1 = arith.addi %rs, %ru : i32
    %sum2 = arith.addi %sum0, %sum1 : i32
    %sum3 = arith.addi %sum2, %qm : i32
    tt.return %sum3 : i32
  }

  // CHECK-LABEL: @splat_constants
  // CHECK: [[C2:%[^ ]+]] = arith.constant 2 : i32
  // CHECK: [[C30:%[^ ]+]] = arith.constant 30 : i32
  // CHECK: [[C128:%[^ ]+]] = arith.constant 128 : i32
  // CHECK: [[MIN:%[^ ]+]] = arith.constant -2147483648 : i32
  // CHECK: [[D2:%[^ ]+]] = tt.splat [[C2]]
  // CHECK: [[D30:%[^ ]+]] = tt.splat [[C30]]
  // CHECK: [[D128:%[^ ]+]] = tt.splat [[C128]]
  // CHECK: [[DMIN:%[^ ]+]] = tt.splat [[MIN]]
  // CHECK: arith.divsi {{%[^ ]+}}, [[D2]] : tensor<32xi32,
  // CHECK: arith.divui {{%[^ ]+}}, [[D30]] : tensor<32xi32,
  // CHECK: arith.remsi {{%[^ ]+}}, [[DMIN]] : tensor<32xi32,
  // CHECK: arith.remui {{%[^ ]+}}, [[D128]] : tensor<32xi32,
  tt.func public @splat_constants(%a: tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    %c2 = arith.constant 2 : i32
    %c30 = arith.constant 30 : i32
    %c128 = arith.constant 128 : i32
    %min = arith.constant -2147483648 : i32
    %d2 = tt.splat %c2 : i32 -> tensor<32xi32, #blocked>
    %d30 = tt.splat %c30 : i32 -> tensor<32xi32, #blocked>
    %d128 = tt.splat %c128 : i32 -> tensor<32xi32, #blocked>
    %dmin = tt.splat %min : i32 -> tensor<32xi32, #blocked>
    %qs = arith.divsi %a, %d2 : tensor<32xi32, #blocked>
    %qu = arith.divui %a, %d30 : tensor<32xi32, #blocked>
    %rs = arith.remsi %a, %dmin : tensor<32xi32, #blocked>
    %ru = arith.remui %a, %d128 : tensor<32xi32, #blocked>
    %sum0 = arith.addi %qs, %qu : tensor<32xi32, #blocked>
    %sum1 = arith.addi %rs, %ru : tensor<32xi32, #blocked>
    %sum2 = arith.addi %sum0, %sum1 : tensor<32xi32, #blocked>
    tt.return %sum2 : tensor<32xi32, #blocked>
  }

  // CHECK-LABEL: @dense_constants
  // CHECK: [[D2:%[^ ]+]] = arith.constant dense<2> : tensor<32xi32,
  // CHECK: [[D30:%[^ ]+]] = arith.constant dense<30> : tensor<32xi32,
  // CHECK: [[D128:%[^ ]+]] = arith.constant dense<128> : tensor<32xi32,
  // CHECK: [[DMIN:%[^ ]+]] = arith.constant dense<-2147483648> : tensor<32xi32,
  // CHECK: arith.divsi {{%[^ ]+}}, [[D2]] : tensor<32xi32,
  // CHECK: arith.divui {{%[^ ]+}}, [[D30]] : tensor<32xi32,
  // CHECK: arith.remsi {{%[^ ]+}}, [[DMIN]] : tensor<32xi32,
  // CHECK: arith.remui {{%[^ ]+}}, [[D128]] : tensor<32xi32,
  tt.func public @dense_constants(%a: tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    %d2 = arith.constant dense<2> : tensor<32xi32, #blocked>
    %d30 = arith.constant dense<30> : tensor<32xi32, #blocked>
    %d128 = arith.constant dense<128> : tensor<32xi32, #blocked>
    %dmin = arith.constant dense<-2147483648> : tensor<32xi32, #blocked>
    %qs = arith.divsi %a, %d2 : tensor<32xi32, #blocked>
    %qu = arith.divui %a, %d30 : tensor<32xi32, #blocked>
    %rs = arith.remsi %a, %dmin : tensor<32xi32, #blocked>
    %ru = arith.remui %a, %d128 : tensor<32xi32, #blocked>
    %sum0 = arith.addi %qs, %qu : tensor<32xi32, #blocked>
    %sum1 = arith.addi %rs, %ru : tensor<32xi32, #blocked>
    %sum2 = arith.addi %sum0, %sum1 : tensor<32xi32, #blocked>
    tt.return %sum2 : tensor<32xi32, #blocked>
  }
}
