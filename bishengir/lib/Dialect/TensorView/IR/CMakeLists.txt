add_bishengir_dialect_library(BiShengIRTensorViewDialect
  TensorViewDialect.cpp
  TensorViewTypes.cpp
  TensorViewAttrs.cpp
  TensorViewOps.cpp

  DEPENDS
  BiShengIRTensorViewTableGen
  BiShengIRTensorViewTypesIncGen
  BiShengIRTensorViewAttrsIncGen
  BiShengIRTensorViewInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialectUtils
  MLIRDestinationStyleOpInterface
  MLIRSideEffectInterfaces
)
