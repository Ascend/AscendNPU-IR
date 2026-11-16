// RUN: bishengir-opt %s -convert-arith-to-hivmave -split-input-file | FileCheck %s
// VCP-only candidates must fall through to segmented VCI, not abort.

// CHECK-LABEL: @vcp_4_8_i16_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_8_i16_plain() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_8_i16_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_8_i16_scaled() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 23, 25, 27, 29, 39, 41, 43, 45, 55, 57, 59, 61, 71, 73, 75, 77, 87, 89, 91, 93, 103, 105, 107, 109, 119, 121, 123, 125]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_8_i16_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_8_i16_decreasing() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 192, 191, 190, 189, 184, 183, 182, 181, 176, 175, 174, 173, 168, 167, 166, 165, 160, 159, 158, 157, 152, 151, 150, 149, 144, 143, 142, 141]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_16_i16_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_16_i16_plain() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_16_i16_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_16_i16_scaled() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 39, 41, 43, 45, 71, 73, 75, 77, 103, 105, 107, 109, 135, 137, 139, 141, 167, 169, 171, 173, 199, 201, 203, 205, 231, 233, 235, 237]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_16_i16_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_16_i16_decreasing() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 184, 183, 182, 181, 168, 167, 166, 165, 152, 151, 150, 149, 136, 135, 134, 133, 120, 119, 118, 117, 104, 103, 102, 101, 88, 87, 86, 85]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_32_i16_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_32_i16_plain() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99, 128, 129, 130, 131, 160, 161, 162, 163, 192, 193, 194, 195, 224, 225, 226, 227]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_32_i16_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_32_i16_scaled() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 71, 73, 75, 77, 135, 137, 139, 141, 199, 201, 203, 205, 263, 265, 267, 269, 327, 329, 331, 333, 391, 393, 395, 397, 455, 457, 459, 461]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_32_i16_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_32_i16_decreasing() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 168, 167, 166, 165, 136, 135, 134, 133, 104, 103, 102, 101, 72, 71, 70, 69, 40, 39, 38, 37, 8, 7, 6, 5, -24, -25, -26, -27]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_8_16_i16_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_16_i16_plain() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_8_16_i16_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_16_i16_scaled() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 39, 41, 43, 45, 47, 49, 51, 53, 71, 73, 75, 77, 79, 81, 83, 85, 103, 105, 107, 109, 111, 113, 115, 117]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_8_16_i16_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_16_i16_decreasing() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 196, 195, 194, 193, 184, 183, 182, 181, 180, 179, 178, 177, 168, 167, 166, 165, 164, 163, 162, 161, 152, 151, 150, 149, 148, 147, 146, 145]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_8_32_i16_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_32_i16_plain() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39, 64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 98, 99, 100, 101, 102, 103]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_8_32_i16_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_32_i16_scaled() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 71, 73, 75, 77, 79, 81, 83, 85, 135, 137, 139, 141, 143, 145, 147, 149, 199, 201, 203, 205, 207, 209, 211, 213]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_8_32_i16_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_32_i16_decreasing() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 196, 195, 194, 193, 168, 167, 166, 165, 164, 163, 162, 161, 136, 135, 134, 133, 132, 131, 130, 129, 104, 103, 102, 101, 100, 99, 98, 97]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vcp_4_8_i32_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_8_i32_plain() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_8_i32_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_8_i32_scaled() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 23, 25, 27, 29, 39, 41, 43, 45, 55, 57, 59, 61, 71, 73, 75, 77, 87, 89, 91, 93, 103, 105, 107, 109, 119, 121, 123, 125]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_8_i32_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_8_i32_decreasing() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 192, 191, 190, 189, 184, 183, 182, 181, 176, 175, 174, 173, 168, 167, 166, 165, 160, 159, 158, 157, 152, 151, 150, 149, 144, 143, 142, 141]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_16_i32_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_16_i32_plain() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_16_i32_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_16_i32_scaled() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 39, 41, 43, 45, 71, 73, 75, 77, 103, 105, 107, 109, 135, 137, 139, 141, 167, 169, 171, 173, 199, 201, 203, 205, 231, 233, 235, 237]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_16_i32_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_16_i32_decreasing() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 184, 183, 182, 181, 168, 167, 166, 165, 152, 151, 150, 149, 136, 135, 134, 133, 120, 119, 118, 117, 104, 103, 102, 101, 88, 87, 86, 85]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_32_i32_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_32_i32_plain() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99, 128, 129, 130, 131, 160, 161, 162, 163, 192, 193, 194, 195, 224, 225, 226, 227]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_32_i32_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_32_i32_scaled() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 71, 73, 75, 77, 135, 137, 139, 141, 199, 201, 203, 205, 263, 265, 267, 269, 327, 329, 331, 333, 391, 393, 395, 397, 455, 457, 459, 461]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_4_32_i32_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_4_32_i32_decreasing() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 168, 167, 166, 165, 136, 135, 134, 133, 104, 103, 102, 101, 72, 71, 70, 69, 40, 39, 38, 37, 8, 7, 6, 5, -24, -25, -26, -27]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_8_16_i32_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_16_i32_plain() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_8_16_i32_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_16_i32_scaled() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 39, 41, 43, 45, 47, 49, 51, 53, 71, 73, 75, 77, 79, 81, 83, 85, 103, 105, 107, 109, 111, 113, 115, 117]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_8_16_i32_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_16_i32_decreasing() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 196, 195, 194, 193, 184, 183, 182, 181, 180, 179, 178, 177, 168, 167, 166, 165, 164, 163, 162, 161, 152, 151, 150, 149, 148, 147, 146, 145]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_8_32_i32_plain(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_32_i32_plain() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39, 64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 98, 99, 100, 101, 102, 103]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_8_32_i32_scaled(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_32_i32_scaled() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 71, 73, 75, 77, 79, 81, 83, 85, 135, 137, 139, 141, 143, 145, 147, 149, 199, 201, 203, 205, 207, 209, 211, 213]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_8_32_i32_decreasing(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: return
func.func @vcp_8_32_i32_decreasing() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[200, 199, 198, 197, 196, 195, 194, 193, 168, 167, 166, 165, 164, 163, 162, 161, 136, 135, 134, 133, 132, 131, 130, 129, 104, 103, 102, 101, 100, 99, 98, 97]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_tail_16(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: ave.hir.vsel
// CHECK: return
func.func @vcp_tail_16() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vcp_tail_24(
// CHECK: ave.hir.vci
// CHECK: ave.hir.vdiv
// CHECK: ave.hir.vadd
// CHECK: ave.hir.vsel
// CHECK: return
func.func @vcp_tail_24() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vci_i8_plain(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i8_plain() -> vector<32xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : vector<32xi8>
  return %c : vector<32xi8>
}

// -----

// CHECK-LABEL: @vci_i8_scaled(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i8_scaled() -> vector<32xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69]> : vector<32xi8>
  return %c : vector<32xi8>
}

// -----

// CHECK-LABEL: @vci_i8_decreasing(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i8_decreasing() -> vector<32xi8> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69]> : vector<32xi8>
  return %c : vector<32xi8>
}

// -----

// CHECK-LABEL: @vci_i16_plain(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i16_plain() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vci_i16_scaled(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i16_scaled() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vci_i16_decreasing(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i16_decreasing() -> vector<32xi16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69]> : vector<32xi16>
  return %c : vector<32xi16>
}

// -----

// CHECK-LABEL: @vci_i32_plain(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i32_plain() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vci_i32_scaled(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i32_scaled() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vci_i32_decreasing(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i32_decreasing() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69]> : vector<32xi32>
  return %c : vector<32xi32>
}

// -----

// CHECK-LABEL: @vci_i64_plain(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i64_plain() -> vector<32xi64> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : vector<32xi64>
  return %c : vector<32xi64>
}

// -----

// CHECK-LABEL: @vci_i64_scaled(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i64_scaled() -> vector<32xi64> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69]> : vector<32xi64>
  return %c : vector<32xi64>
}

// -----

// CHECK-LABEL: @vci_i64_decreasing(
// CHECK: ave.hir.vci
// CHECK-NOT: ave.hir.vdiv
// CHECK: return
func.func @vci_i64_decreasing() -> vector<32xi64> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69]> : vector<32xi64>
  return %c : vector<32xi64>
}

// -----
