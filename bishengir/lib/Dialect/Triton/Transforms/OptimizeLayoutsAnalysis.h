//===------------------OptimizeLayoutsAnalysis.h----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt  for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
#ifndef OPTIMIZE_LAYOUT_ANALYSIS
#define OPTIMIZE_LAYOUT_ANALYSIS

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
namespace triton {

// Layout action types
enum class LayoutAction { None, DontRemove, PropagateDown, PropagateUp };

// Convert layout action attribute names
extern const char *kLayoutActionAttr;
extern const char *kLayoutPriorityAttr;
extern const char *kLayoutCostAttr;

// Convert action to string
inline llvm::StringRef actionToString(LayoutAction a) {
  switch (a) {
  case LayoutAction::None:
    return "none";
  case LayoutAction::DontRemove:
    return "dontremove";
  case LayoutAction::PropagateDown:
    return "propagate_down";
  case LayoutAction::PropagateUp:
    return "propagate_up";
  }
  return "none";
}

// Layout propagation analysis for convert_layout operations
//
// This analysis implements a 3-phase approach:
// 1. Phase 1 (runAnalysis): Identify converts to keep (DontRemove) vs
//    propagate down using cost model
// 2. Apply propagate-down transforms externally
// 3. Phase 2 (refineAfterPropDown): Propagate remaining converts upward when
// safe
class OptimizeLayoutsAnalysis {
public:
  // Constructor - root operation should be a ModuleOp or FunctionOp
  explicit OptimizeLayoutsAnalysis(Operation *root);

  // Destructor must be defined out-of-line where Impl is complete (PIMPL)
  ~OptimizeLayoutsAnalysis();

  // non-copyable
  OptimizeLayoutsAnalysis(const OptimizeLayoutsAnalysis &) = delete;
  OptimizeLayoutsAnalysis &operator=(const OptimizeLayoutsAnalysis &) = delete;

  // movable
  OptimizeLayoutsAnalysis(OptimizeLayoutsAnalysis &&) noexcept = default;
  OptimizeLayoutsAnalysis &
  operator=(OptimizeLayoutsAnalysis &&) noexcept = default;

  //  phase 1 analysis to identify converts for propagate-down
  void runAnalysis();

  //  phase 2 analysis after propagate-down transforms have been applied
  void refineAfterPropDown();

  //  the ordered list of operations for propagate-down transformation
  llvm::SmallVector<Operation *, 64> getPropagateDownOrder() const;

  //  the ordered list of operations for propagate-up transformation
  llvm::SmallVector<Operation *, 64> getPropagateUpOrder() const;

  static bool encodingNotRegular(Attribute enc);

private:
  struct ConvertInfo;
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace triton
} // namespace mlir

#endif // OPTIMIZE_LAYOUT_ANALYSIS
