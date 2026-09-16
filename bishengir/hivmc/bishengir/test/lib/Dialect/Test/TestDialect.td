//===-- TestDialect.td - BishengIR Test dialect definition -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TESTDIALECT
#define TEST_TESTDIALECT

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "bishengir_test";
  let cppNamespace = "::bishengir_test";
}

#endif // TEST_TESTDIALECT
