get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(triton_libs)
set(triton_deps)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND triton_libs TritonAggregateLib)
  # Explicitly list all proton IncGen targets whose generated .h.inc files are
  # pulled in (via InitAllDialects.h -> RegisterTritonDialects.h) when
  # compiling bishengir-compile.cpp. Without these, ninja may schedule the
  # .cpp.o build before tablegen runs under RelWithDebInfo builds.
  list(APPEND triton_deps
    ProtonTableGen
    ProtonAttrDefsIncGen
    ProtonGPUTableGen
    ProtonGPUAttrDefsIncGen
    ProtonGPUTypesIncGen
    ProtonToProtonGPUIncGen
    ProtonGPUConversionPassIncGen
    ProtonGPUTransformsIncGen
  )
endif()

add_bishengir_library(BiShengIRRegBaseCompileLib
  PassPipeline.cpp
  BiShengIRCompileMain.cpp
  Driver.cpp
  Utility.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Tools/bishengir-compile/regbase

  DEPENDS
  ${triton_deps}

  LINK_LIBS PUBLIC
  ${translation_libs}
  ${conversion_libs}
  ${dialect_libs}
  ${extension_libs}
  BiShengIRCompileConfigLib
  BiShengIRHACCPipelines
  BiShengIRHFusionDialect
  BiShengIRHFusionPipelines
  BiShengIRHIVMPipelines
  BiShengIRPass
  BiShengIRScopeDialect
  BiShengIRTransform
  BiShengIRToolUtils
  MLIRHIVMAVEPipelines
  ${triton_libs}
)
