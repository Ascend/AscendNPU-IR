
// RUN: bishengir-opt %s --split-input-file -convert-torch-to-symbol -allow-unregistered-dialect -cse | FileCheck %s

module {
  func.func @symbolic_int() -> () {
    // CHECK: %[[SYMBOL:.*]] = symbol.symbolic_int @s0 {max_val = 1024 : i64, min_val = 128 : i64} : index
    // CHECK: %[[I64:.*]] = arith.index_cast %[[SYMBOL]] : index to i64
    // CHECK: torch_c.from_i64 %[[I64]]
    %0 = torch.symbolic_int "s0" {min_val = 128, max_val = 1024} : !torch.int
    "some_use"(%0) : (!torch.int) -> ()
  }
}

// -----

module {
  // CHECK: bind_symbolc_shape(%[[ARG0:.*]]: !torch.vtensor<[?,640],f16>
  func.func @bind_symbolc_shape(%arg0: !torch.vtensor<[?,640],f16>) -> () {
    // CHECK: %[[SYMBOL_0:.*]] = symbol.symbolic_int @s0 {max_val = 1024 : i64, min_val = 128 : i64} : index
    // CHECK: %[[SYMBOL_0_I64:.*]] = arith.index_cast %[[SYMBOL_0]] : index to i64
    // CHECK: %[[SYMBOL_0_TORCH_INT:.*]] = torch_c.from_i64 %[[SYMBOL_0_I64]]
    %1 = torch.symbolic_int "s0" {min_val = 128, max_val = 1024} : !torch.int
    // CHECK: %[[SYMBOL_1:.*]] = symbol.symbolic_int @s1 {max_val = 1024 : i64, min_val = 128 : i64} : index
    // CHECK: %[[SYMBOL_1_I64:.*]] = arith.index_cast %[[SYMBOL_1]] : index to i64
    // CHECK: %[[SYMBOL_1_TORCH_INT:.*]] = torch_c.from_i64 %[[SYMBOL_1_I64]]
    %2 = torch.symbolic_int "s1" {min_val = 128, max_val = 1024} : !torch.int
    // CHECK: %[[SYMBOL_0_I64_1:.*]] = torch_c.to_i64 %[[SYMBOL_0_TORCH_INT]]
    // CHECK: %[[SYMBOL_0_INDEX:.*]] = arith.index_cast %[[SYMBOL_0_I64_1]] : i64 to index
    // CHECK: %[[SYMBOL_1_I64_1:.*]] = torch_c.to_i64 %[[SYMBOL_1_TORCH_INT]]
    // CHECK: %[[SYMBOL_1_INDEX:.*]] = arith.index_cast %[[SYMBOL_1_I64_1]] : i64 to index
    // CHECK: %[[ARG0_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,640],f16> -> tensor<?x640xf16>
    // CHECK: symbol.bind_symbolic_shape %[[ARG0_TENSOR]], [%[[SYMBOL_0_INDEX]], %[[SYMBOL_1_INDEX]]], affine_map<()[s0, s1] -> (s0 * s1, 640)> : tensor<?x640xf16>
    torch.bind_symbolic_shape %arg0, [%1, %2], affine_map<()[s0, s1] -> (s0 * s1, 640)> : !torch.vtensor<[?,640],f16>
    return
  }
}
