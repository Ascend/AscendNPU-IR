#include "bishengir-c/Dialect/HIVMAVE.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

static void populateDialectHIVMAVESubmodule(const py::module &m) {}

PYBIND11_MODULE(_bishengirDialectsHIVMAVE, m) {
  m.doc() = "bishengir HIVMAVE dialect.";
  populateDialectHIVMAVESubmodule(m);
}
