//===- MultiBufferMode.h - Per-level multi-buffer counts ---------*- C++-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Parses and formats --multibuffer-mode="[(gm, N), (l1, N), (l0c, N), (ub,
// N)]". Count > 1 enables that memory space; count == 1 disables it; count == 0
// is illegal. Typical MixedCV defaults match the historical limit-* flags:
// A3/membase
// [(gm, 4), (l1, 2), (l0c, 1), (ub, 1)], A5/regbase
// [(gm, 2), (l1, 2), (l0c, 1), (ub, 2)].
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_UTILS_MULTIBUFFER_MODE_H
#define BISHENGIR_DIALECT_HIVM_UTILS_MULTIBUFFER_MODE_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace mlir {
namespace hivm {

struct MultiBufferMode {
  unsigned gm = 1;
  unsigned l1 = 1;
  unsigned l0c = 1;
  unsigned ub = 1;

  bool isEnabledGM() const { return gm > 1; }
  bool isEnabledL1() const { return l1 > 1; }
  bool isEnabledL0C() const { return l0c > 1; }
  bool isEnabledUB() const { return ub > 1; }
};

/// Arch-dependent compile defaults that match the legacy limit-* flags.
inline MultiBufferMode defaultMultiBufferMode(bool isRegBased) {
  if (isRegBased)
    return MultiBufferMode{/*gm=*/2, /*l1=*/2, /*l0c=*/1, /*ub=*/2};
  return MultiBufferMode{/*gm=*/4, /*l1=*/2, /*l0c=*/1, /*ub=*/1};
}

inline std::string formatMultiBufferMode(const MultiBufferMode &mode) {
  return llvm::formatv("[(gm, {0}), (l1, {1}), (l0c, {2}), (ub, {3})]", mode.gm,
                       mode.l1, mode.l0c, mode.ub)
      .str();
}

/// Parse "[(gm, N), (l1, N), (l0c, N), (ub, N)]". Names are case-insensitive,
/// whitespace is optional. All four levels are required. Count 0 is an error.
inline LogicalResult parseMultiBufferMode(llvm::StringRef spec,
                                          MultiBufferMode &mode,
                                          std::string &error) {
  auto fail = [&](const llvm::Twine &msg) -> LogicalResult {
    error = msg.str();
    return failure();
  };

  llvm::StringRef rest = spec.trim();
  if (!rest.consume_front("[") || !rest.consume_back("]"))
    return fail("expected [(gm, N), (l1, N), (l0c, N), (ub, N)]");

  bool seenGM = false, seenL1 = false, seenL0C = false, seenUB = false;
  bool any = false;
  rest = rest.trim();
  while (!rest.empty()) {
    rest = rest.ltrim();
    if (!rest.consume_front("("))
      return fail("expected '(' in --multibuffer-mode");
    rest = rest.ltrim();

    size_t comma = rest.find(',');
    if (comma == llvm::StringRef::npos)
      return fail("expected ',' between level name and count");
    llvm::StringRef name = rest.take_front(comma).trim();
    rest = rest.drop_front(comma + 1).ltrim();

    size_t close = rest.find(')');
    if (close == llvm::StringRef::npos)
      return fail("expected ')' after multi-buffer count");
    llvm::StringRef countStr = rest.take_front(close).trim();
    rest = rest.drop_front(close + 1).ltrim();

    unsigned count = 0;
    if (countStr.empty() || countStr.getAsInteger(10, count))
      return fail("invalid multi-buffer count '" + countStr + "'");
    if (count == 0)
      return fail("multi-buffer count for '" + name +
                  "' must be >= 1; 0 is illegal");
    if (count > static_cast<unsigned>(std::numeric_limits<int32_t>::max()))
      return fail("multi-buffer count for '" + name +
                  "' must fit in a signed 32-bit annotation");

    if (name.equals_insensitive("gm")) {
      if (seenGM)
        return fail("duplicate level 'gm'");
      seenGM = true;
      mode.gm = count;
    } else if (name.equals_insensitive("l1")) {
      if (seenL1)
        return fail("duplicate level 'l1'");
      seenL1 = true;
      mode.l1 = count;
    } else if (name.equals_insensitive("l0c")) {
      if (seenL0C)
        return fail("duplicate level 'l0c'");
      seenL0C = true;
      mode.l0c = count;
    } else if (name.equals_insensitive("ub")) {
      if (seenUB)
        return fail("duplicate level 'ub'");
      seenUB = true;
      mode.ub = count;
    } else {
      return fail("unknown memory level '" + name +
                  "'; expected gm, l1, l0c, or ub");
    }
    any = true;

    if (rest.consume_front(","))
      continue;
    rest = rest.ltrim();
    if (!rest.empty())
      return fail("unexpected trailing text '" + rest + "'");
  }

  if (!any || !seenGM || !seenL1 || !seenL0C || !seenUB)
    return fail("must specify all of gm, l1, l0c, and ub");
  return success();
}

/// Translate the historical limit-* / workspace flags into a mode. Used for
/// deprecation diagnostics and as the starting point when --multibuffer-mode
/// is not set.
inline MultiBufferMode
legacyMultiBufferMode(bool isRegBased, bool enableAuto, bool onlyForLocal,
                      bool ofLocalIsNoL0C, bool mixOnlyCube, bool mixOnlyVector,
                      unsigned workspaceNum) {
  if (!enableAuto)
    return MultiBufferMode{1, 1, 1, 1};

  MultiBufferMode mode = defaultMultiBufferMode(isRegBased);
  if (workspaceNum == 0)
    mode.gm = 1;
  else
    mode.gm = workspaceNum;
  if (onlyForLocal)
    mode.gm = 1;
  mode.l0c = ofLocalIsNoL0C ? 1u : std::max(mode.l0c, 2u);
  if (mixOnlyCube)
    mode.ub = 1;
  else if (mixOnlyVector) {
    mode.l1 = 1;
    mode.l0c = 1;
  } else {
    mode.l1 = std::max(mode.l1, 2u);
    mode.ub = std::max(mode.ub, 2u);
  }
  return mode;
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_UTILS_MULTIBUFFER_MODE_H
