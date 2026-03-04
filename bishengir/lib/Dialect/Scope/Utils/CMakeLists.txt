add_bishengir_dialect_library(BiShengIRScopeUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRScopeDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
