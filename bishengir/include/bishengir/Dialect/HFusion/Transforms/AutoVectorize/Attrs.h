//===- Attrs.h - Auto vectorization transform attributes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ATTRS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ATTRS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace mlir {
namespace hfusion {
namespace auto_vectorize {

inline constexpr llvm::StringLiteral kPayloadRootTagPrefix =
    "auto_vectorize_v2.payload.";
inline constexpr llvm::StringLiteral kTransformRootTagPrefix =
    "auto_vectorize_v2.transform.";

inline std::string getPayloadRootTag(llvm::StringRef funcName) {
  return (llvm::Twine(kPayloadRootTagPrefix) + funcName).str();
}

inline std::string getTransformRootTag(llvm::StringRef funcName) {
  return (llvm::Twine(kTransformRootTagPrefix) + funcName).str();
}

} // namespace auto_vectorize
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ATTRS_H
