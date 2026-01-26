// RUN: bishengir-opt --inline %s | FileCheck %s

module {
  func.func @annotation(%arg0 : i32) {
    annotation.mark %arg0: i32
    func.return
  }
  func.func @foo(%arg0: i32) {
    // CHECK-NOT: call
    func.call @annotation(%arg0) : (i32) -> ()
    func.return
  }
}