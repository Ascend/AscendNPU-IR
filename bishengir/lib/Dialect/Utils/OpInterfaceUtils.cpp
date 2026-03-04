//===- OpInterfaceUtils.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for overriding upstream's interface
// implementation.
// Original implementation is from:
// https://github.com/bytedance/byteir/blob/0e83d42baff5842ddd433b8f1a04e0d783683536/compiler/lib/Utils/OpInterfaceUtils.cpp
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace llvm;

namespace {
struct ExtensionRegistry {
  struct Extension final : public DialectExtensionBase {
    using ImplT = std::function<void(MLIRContext *ctx)>;
    using CtorParam = std::pair<ImplT, StringRef>;

    explicit Extension(const CtorParam &param)
        : DialectExtensionBase(ArrayRef<StringRef>{param.second}),
          impl(param.first) {}

    void apply(MLIRContext *context,
               MutableArrayRef<Dialect *> /* dialects */) const override {
      if (enableOpInterfaceExtensions) {
        impl(context);
      }
    }

    std::unique_ptr<DialectExtensionBase> clone() const override {
      return std::make_unique<Extension>(*this);
    }

    ImplT impl;
  };

  void insert(Extension::ImplT extensionFn, StringRef dialectName) {
    ctorParams.push_back({std::move(extensionFn), dialectName});
  }

  void apply(DialectRegistry &registry) {
    for (auto &&param : ctorParams) {
      registry.addExtension(std::make_unique<Extension>(param));
    }
  }

  static ExtensionRegistry &inst();

private:
  static llvm::cl::opt<bool> enableOpInterfaceExtensions;

  SmallVector<Extension::CtorParam> ctorParams;
};

ExtensionRegistry &ExtensionRegistry::inst() {
  static ExtensionRegistry inst;
  return inst;
}

llvm::cl::opt<bool> ExtensionRegistry::enableOpInterfaceExtensions(
    "enable-op-interface-extensions",
    llvm::cl::desc("Enable op interface extensions, this would override "
                   "some implementations of op interface"),
    llvm::cl::init(true));
} // namespace

void mlir::detail::addOpInterfaceExtension(
    std::function<void(MLIRContext *ctx)> extensionFn,
    llvm::StringRef dialectName) {
  ExtensionRegistry::inst().insert(std::move(extensionFn), dialectName);
}

void mlir::registerOpInterfaceExtensions(DialectRegistry &registry) {
  ExtensionRegistry::inst().apply(registry);
}