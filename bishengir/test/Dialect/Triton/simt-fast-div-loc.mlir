// RUN: bishengir-opt -split-input-file -simt-fast-div --mlir-print-debuginfo %s | FileCheck %s

// The magic-multiply expansion must inherit the source location of the
// replaced division instead of falling back to the function definition line.
// Intermediate operations only keep the source line, while the operation that
// produces the quotient also keeps the variable name of the original op.

// CHECK-LABEL: tt.func public @scalar_divsi
// CHECK:       ttg.local_load {{.*}} loc(#[[INNER:loc[0-9]+]])
// CHECK:       ttg.local_load {{.*}} loc(#[[INNER]])
// CHECK:       ascend_dpx.umulhi {{.*}} loc(#[[INNER]])
// CHECK:       arith.addi {{.*}} loc(#[[INNER]])
// CHECK:       arith.shrui {{.*}} loc(#[[FINAL:loc[0-9]+]])
// CHECK-DAG:   #[[FINAL]] = loc("pid_x"(#[[INNER]]))
// CHECK-DAG:   #[[INNER]] = loc("div_source.py":12:27)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_divsi(%dividend: i32, %divisor: i32) -> i32 {
    %0 = arith.divsi %dividend, %divisor : i32 loc("pid_x"("div_source.py":12:27))
    tt.return %0 : i32
  }
}

// -----

// Same for the tensorized path: the broadcasts and the multiply keep the
// source line, the final shift keeps the variable name as well.

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: tt.func public @tensor_divsi
// CHECK:       ttg.local_load {{.*}} loc(#[[INNER:loc[0-9]+]])
// CHECK:       tt.broadcast {{.*}} loc(#[[INNER]])
// CHECK:       tt.mulhiui {{.*}} loc(#[[INNER]])
// CHECK:       arith.addi {{.*}} loc(#[[INNER]])
// CHECK:       arith.shrui {{.*}} loc(#[[FINAL:loc[0-9]+]])
// CHECK-DAG:   #[[FINAL]] = loc("pid_x"(#[[INNER]]))
// CHECK-DAG:   #[[INNER]] = loc("div_source.py":12:27)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_divsi(%dividend: tensor<32xi32, #blocked1>, %divisor: i32) -> tensor<32xi32, #blocked1> {
    %splat = tt.splat %divisor : i32 -> tensor<32xi32, #blocked1>
    %0 = arith.divsi %dividend, %splat : tensor<32xi32, #blocked1> loc("pid_x"("div_source.py":12:27))
    tt.return %0 : tensor<32xi32, #blocked1>
  }
}
