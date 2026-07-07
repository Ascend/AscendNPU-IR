//===- InjectIRInstrumentation.h - BiShengIR Pass Instrumentation --*-
// C++-*-===//

//===------------------------------------------------------------------------===//

#ifndef BISHENGIR_TRANSFORMS_PASSINSTRUMENTATION_H
#define BISHENGIR_TRANSFORMS_PASSINSTRUMENTATION_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassInstrumentation.h"

namespace bishengir {

/// Returns a stable name for the operation (module -> "module", func -> sym
/// name).
std::string getOpNameForPassId(mlir::Operation *op);

/// Pass execution ID: pass_name/op_name/call_index (0-based). When update is
/// true, assigns the next index and returns the new ID; when false, returns
/// the ID for the current execution without updating.
std::string getPassExecutionId(mlir::Pass *pass, mlir::Operation *op,
                               bool update);

/// Injects IR from file at matched pass (before/after per option). Format:
/// pass_id@file_path.
class InjectIRInstrumentation : public mlir::PassInstrumentation {
public:
  explicit InjectIRInstrumentation(bool printPassId,
                                   const std::string &injectIrBefore,
                                   const std::string &injectIrAfter)
      : printPassId(printPassId), injectIrBefore(injectIrBefore),
        injectIrAfter(injectIrAfter) {}

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  bool printPassId;
  std::string injectIrBefore;
  std::string injectIrAfter;
};

} // namespace bishengir

#endif // BISHENGIR_TRANSFORMS_PASSINSTRUMENTATION_H