#include "bishengir-c/Dialect/HIVM.h"
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

static void populateDialectHIVMSubmodule(const py::module &m) {

  auto addressSpaceEnum =
      py::enum_<MLIRHIVMAddressSpace>(m, "addressSpace")
          .value("Zero", MLIRHIVMAddressSpace::HIVM_AddressSpace_Default)
          .value("GM", MLIRHIVMAddressSpace::HIVM_AddressSpace_GM)
          .value("L1", MLIRHIVMAddressSpace::HIVM_AddressSpace_L1)
          .value("L0A", MLIRHIVMAddressSpace::HIVM_AddressSpace_L0A)
          .value("L0B", MLIRHIVMAddressSpace::HIVM_AddressSpace_L0B)
          .value("L0C", MLIRHIVMAddressSpace::HIVM_AddressSpace_L0C)
          .value("UB", MLIRHIVMAddressSpace::HIVM_AddressSpace_UB)
          .export_values();

  auto hivmAddressSpaceAttr = mlir_attribute_subclass(
      m, "addressSpaceAttr", mlirAttrIsAHivmAddressSpaceAttr);
  hivmAddressSpaceAttr
      .def_classmethod(
          "get",
          [](py::object cls, MLIRHIVMAddressSpace addrSpace, MlirContext ctx) {
            return cls(mlirHivmAddressSpaceAttrGet(ctx, addrSpace));
          },
          py::arg("cls"), py::arg("addressSpace"), py::arg("ctx") = py::none(),
          "Get an AddressSpaceAttr for the given space.")

      .def_property_readonly("address_space", [](MlirAttribute attr) {
        return mlirHivmAddressSpaceAttrGetAddrSpace(attr);
      });

  auto pipeEnum = py::enum_<MLIRHIVMPipe>(m, "pipe")
                      .value("s", MLIRHIVMPipe::HIVM_PIPE_S)
                      .value("v", MLIRHIVMPipe::HIVM_PIPE_V)
                      .value("m", MLIRHIVMPipe::HIVM_PIPE_M)
                      .value("mte1", MLIRHIVMPipe::HIVM_PIPE_MTE1)
                      .value("mte2", MLIRHIVMPipe::HIVM_PIPE_MTE2)
                      .value("mte3", MLIRHIVMPipe::HIVM_PIPE_MTE3)
                      .value("all", MLIRHIVMPipe::HIVM_PIPE_ALL)
                      .value("fix", MLIRHIVMPipe::HIVM_PIPE_FIX)
                      .export_values();

  auto hivmPipeAttr =
      mlir_attribute_subclass(m, "pipeAttr", mlirAttrIsAHivmPipeAttr);

  hivmPipeAttr
      .def_classmethod(
          "get",
          [](py::object cls, MLIRHIVMPipe pipe, MlirContext ctx) {
            return cls(mlirHivmPipeAttrGet(ctx, pipe));
          },
          py::arg("cls"), py::arg("pipe"), py::arg("ctx") = py::none(),
          "Get a PipeAttr for the given pipe.")

      .def_property_readonly("pipe", [](MlirAttribute attr) {
        return mlirHivmPipeAttrGetPipe(attr);
      });

  auto eventEnum = py::enum_<MLIRHIVMEvent>(m, "event")
                       .value("id_0", MLIRHIVMEvent::HIVM_EventID0)
                       .value("id_1", MLIRHIVMEvent::HIVM_EventID1)
                       .value("id_2", MLIRHIVMEvent::HIVM_EventID2)
                       .value("id_3", MLIRHIVMEvent::HIVM_EventID3)
                       .value("id_4", MLIRHIVMEvent::HIVM_EventID4)
                       .value("id_5", MLIRHIVMEvent::HIVM_EventID5)
                       .value("id_6", MLIRHIVMEvent::HIVM_EventID6)
                       .value("id_7", MLIRHIVMEvent::HIVM_EventID7)
                       .export_values();

  auto hivmEventAttr =
      mlir_attribute_subclass(m, "eventAttr", mlirAttrIsAHivmEventAttr);

  hivmEventAttr
      .def_classmethod(
          "get",
          [](py::object cls, MLIRHIVMEvent event, MlirContext ctx) {
            return cls(mlirHivmEventAttrGet(ctx, event));
          },
          py::arg("cls"), py::arg("id"), py::arg("ctx") = py::none(),
          "Get a EventAttr for the given event id.")

      .def_property_readonly("event", [](MlirAttribute attr) {
        return mlirHivmEventAttrGetEvent(attr);
      });
}

PYBIND11_MODULE(_bishengirDialectsHIVM, m) {
  m.doc() = "bishengir HIVM dialect.";
  populateDialectHIVMSubmodule(m);
}