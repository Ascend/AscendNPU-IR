// RUN: bishengir-opt %s --hivm-vectorize | FileCheck %s

module {
  func.func @empty() {
    return
  }
}

// CHECK-LABEL: func.func @empty()
// CHECK-NEXT: return
