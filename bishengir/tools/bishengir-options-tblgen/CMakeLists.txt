set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(bishengir-options-tblgen BISHENGIR_OPTIONS
  bishengir-options-tblgen.cpp
  CompileOptionsGen.cpp
  CompileConfigGen.cpp
  ConfigsToOptionsConversionGen.cpp
  PassOptionsGen.cpp
)

target_link_libraries(bishengir-options-tblgen
  PRIVATE
  MLIRTblgenLib
  BiShengIRConfigOptions
)

mlir_check_all_link_libraries(bishengir-options-tblgen)
