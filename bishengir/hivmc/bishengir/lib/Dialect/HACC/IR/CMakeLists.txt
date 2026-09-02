add_bishengir_dialect_library(MLIRHACCDialect
  HACCDialect.cpp
  HACCInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HACC

  DEPENDS
  BiShengIRHACCEnumsIncGen
  BiShengIRHACCAttributesIncGen
  BiShengIRHACCInterfacesIncGen
  BiShengIRNPUTargetSpecIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRFunctionInterfaces
  MLIRSideEffectInterfaces
  MLIRDestinationStyleOpInterface
)
