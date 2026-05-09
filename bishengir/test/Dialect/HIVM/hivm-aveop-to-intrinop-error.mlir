// RUN: bishengir-opt %s -convert-hivmave-to-ave-intrin --split-input-file 2>&1 | FileCheck %s

// CHECK: error: plt has multiple users of various elementAlignmentBitWidth
func.func @test_plt_mixed_users_error(%arg0: index) attributes {element_alignment_bit_width = 8 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %res, %new_true_shape = ave.hir.plt %arg0 : vector<64xi1>, index
  "test.test"(%res) {element_alignment_bit_width = 32 : i32} : (vector<64xi1>) -> ()
  "test.test"(%res) {element_alignment_bit_width = 16 : i32} : (vector<64xi1>) -> ()
  return
}
