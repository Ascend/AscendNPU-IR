// Module used as inject file for inject-ir test.
// Functions here replace the bodies of same-named functions in the main module.
module {
  func.func @foo() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
  // Function with parameters: uses block arguments in the body.
  func.func @bar(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
}
