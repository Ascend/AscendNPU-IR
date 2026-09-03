//===- InitTestDialect.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#ifndef TEST_INITTESTDIALECT_H
#define TEST_INITTESTDIALECT_H

#include "mlir/IR/DialectRegistry.h"
namespace bishengir_test {
void registerTestDialect(::mlir::DialectRegistry &registry);
} // namespace bishengir_test

#endif // TEST_INITTESTDIALECT_H
