//===- HIVMDialectExtension.cpp - HIVM Dialect Extension ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMDialectExtension.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/HIVM/Interfaces/LibraryFunctionOpInterface.h"

using namespace mlir;

void bishengir::hivm::registerHIVMDialectExtension(DialectRegistry &registry) {
  bishengir::hivm::detail::registerLibraryFunctionOpInterfaceExtension(
      registry);
}