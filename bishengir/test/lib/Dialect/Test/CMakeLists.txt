set(LLVM_OPTIONAL_SOURCES
   HIVMDialect.cpp
)

set(LLVM_TARGET_DEFINITIONS TestOps.td)
mlir_tablegen(TestOpsDialect.h.inc -gen-dialect-decls -dialect=bishengir_test)
mlir_tablegen(TestOpsDialect.cpp.inc -gen-dialect-defs -dialect=bishengir_test)
mlir_tablegen(TestOps.h.inc -gen-op-decls)
mlir_tablegen(TestOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BishengIRTestOpsIncGen)

add_bishengir_library(BishengIRTestDialect
  TestDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  BishengIRTestOpsIncGen

  LINK_LIBS PUBLIC
  MLIRHIVMDialect
  MLIRDestinationStyleOpInterface
)