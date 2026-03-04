//===- OpInterfaceUtils.h ---------------------------------------*- C++ -*-===//
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
// https://github.com/bytedance/byteir/blob/0e83d42baff5842ddd433b8f1a04e0d783683536/compiler/include/byteir/Utils/OpInterfaceUtils.h
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_UTILS_OPINTERFACEUTILS_H
#define BISHENGIR_DIALECT_UTILS_OPINTERFACEUTILS_H

#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace detail {
template <typename Op, typename Interface, auto method>
struct OpInterfaceOverrider {
  using Concept = typename Interface::Concept;
  template <typename MethodType> struct WrapperT;

  template <typename Ret, typename... Args>
  struct WrapperT<Ret (*Concept::*)(const Concept *, Args...)> {
    using ImplType = std::function<Ret(Args...)>;
    static Ret call(const Concept *, Args... args) { return sm_impl(args...); }
  };

#if !defined(_MSC_VER)
  // MSVC will report C2752, more than one partial specialization matches the
  // template argument list, for the following.
  template <typename Ret, typename... Args>
  struct WrapperT<Ret (*Concept::*)(Args...)> {
    using ImplType = std::function<Ret(Args...)>;
    static Ret call(Args... args) { return sm_impl(args...); }
  };
#endif

  using Wrapper = WrapperT<decltype(method)>;
  using Impl = typename Wrapper::ImplType;
  static inline Impl sm_impl = nullptr;

  struct ExternalInterfaceImpl
      : public Interface::template ExternalModel<ExternalInterfaceImpl, Op> {};

  static void apply(const Impl &impl);
};

void addOpInterfaceExtension(std::function<void(MLIRContext *ctx)> extensionFn,
                             llvm::StringRef dialectName);

template <typename Op, typename Interface, auto method>
void OpInterfaceOverrider<Op, Interface, method>::apply(const Impl &impl) {
  if (sm_impl == nullptr) {
    sm_impl = impl;
  } else {
    llvm::report_fatal_error("Override interface of " + Op::getOperationName() +
                             " more than once");
  }
  addOpInterfaceExtension(
      +[](MLIRContext *ctx) {
        auto info =
            RegisteredOperationName::lookup(Op::getOperationName(), ctx);
        if (info) {
          if (!info->template hasInterface<Interface>()) {
            if constexpr (!Op::template hasTrait<Interface::template Trait>()) {
              info->template attachInterface<ExternalInterfaceImpl>();
            }
          }
          if (auto concept = info->template getInterface<Interface>()) {
            concept->*method = &Wrapper::call;
          } else {
            llvm::report_fatal_error(
                "Cannot find registered interface model of op " +
                Op::getOperationName());
          }
        } else {
          llvm::report_fatal_error("Unregistered op " + Op::getOperationName());
        }
      },
      Op::getOperationName().split('.').first);
}
} // namespace detail

void registerOpInterfaceExtensions(DialectRegistry &registry);

} // namespace mlir

#define RegisterOpInterfaceOverrideImpl(op, interface, method, impl, N)        \
  template struct ::mlir::detail::OpInterfaceOverrider<                        \
      op, interface, &interface::Concept::method>;                             \
  [[maybe_unused]] static bool __override_op_interface##N = [] {               \
    ::mlir::detail::OpInterfaceOverrider<                                      \
        op, interface, &interface::Concept::method>::apply(impl);              \
    return false;                                                              \
  }()

#define RegisterOpInterfaceOverrideCounter(op, interface, method, impl, N)     \
  RegisterOpInterfaceOverrideImpl(op, interface, method, impl, N)

#define RegisterOpInterfaceOverride(op, interface, method, impl)               \
  RegisterOpInterfaceOverrideCounter(op, interface, method, impl, __COUNTER__)

#endif // BISHENGIR_DIALECT_UTILS_OPINTERFACEUTILS_H