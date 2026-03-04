# Declare the bishengir library associated with a dialect.
function(add_bishengir_library name)
  cmake_parse_arguments(ARG "IS_OPEN_SOURCE" "" "" ${ARGN})
  if(ARG_IS_OPEN_SOURCE)
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_OPEN_SOURCE_LIBS ${name})
    list(REMOVE_ITEM ARGV "IS_OPEN_SOURCE")
    add_mlir_library(${ARGV} DEPENDS mlir-headers)
  else()
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_LIBS ${name})
    add_mlir_library(${ARGV} DEPENDS mlir-headers)
  endif()
endfunction(add_bishengir_library)

# Declare the bishengir library associated with a dialect.
function(add_bishengir_dialect_library name)
  cmake_parse_arguments(ARG "IS_OPEN_SOURCE" "" "" ${ARGN})
  if(ARG_IS_OPEN_SOURCE)
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_DIALECT_OPEN_SOURCE_LIBS
                                        ${name})
    list(REMOVE_ITEM ARGV "IS_OPEN_SOURCE")
    add_mlir_dialect_library(${ARGV})
  else()
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_DIALECT_LIBS ${name})
    add_mlir_dialect_library(${ARGV})
  endif()
endfunction(add_bishengir_dialect_library)

# Declare the bishengir library associated with a conversion.
function(add_bishengir_conversion_library name)
  cmake_parse_arguments(ARG "IS_OPEN_SOURCE" "" "" ${ARGN})
  if(ARG_IS_OPEN_SOURCE)
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_CONVERSION_OPEN_SOURCE_LIBS
                                        ${name})
    list(REMOVE_ITEM ARGV "IS_OPEN_SOURCE")
    add_mlir_conversion_library(${ARGV})
  else()
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_CONVERSION_LIBS ${name})
    add_mlir_conversion_library(${ARGV})
  endif()
endfunction(add_bishengir_conversion_library)

# Declare the bishengir library associated with a translation.
function(add_bishengir_translation_library name)
  cmake_parse_arguments(ARG "IS_OPEN_SOURCE" "" "" ${ARGN})
  if(ARG_IS_OPEN_SOURCE)
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_TRANSLATION_OPEN_SOURCE_LIBS
                                        ${name})
    list(REMOVE_ITEM ARGV "IS_OPEN_SOURCE")
    add_mlir_translation_library(${ARGV})
  else()
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_TRANSLATION_LIBS ${name})
    add_mlir_translation_library(${ARGV})
  endif()
endfunction(add_bishengir_translation_library)

# Declare the bishengir library associated with an extension.
function(add_bishengir_extension_library name)
  cmake_parse_arguments(ARG "IS_OPEN_SOURCE" "" "" ${ARGN})
  if(ARG_IS_OPEN_SOURCE)
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_EXTENSION_OPEN_SOURCE_LIBS
                                        ${name})
    list(REMOVE_ITEM ARGV "IS_OPEN_SOURCE")
    add_mlir_extension_library(${ARGV})
  else()
    set_property(GLOBAL APPEND PROPERTY BISHENGIR_EXTENSION_LIBS ${name})
    add_mlir_extension_library(${ARGV})
  endif()
endfunction(add_bishengir_extension_library)

# Declare the bishengir target spec tablegen target.
function(bishengir_target_tablegen ofn)
  tablegen(BISHENGIR_TARGET_SPEC ${ARGV})
  set(TABLEGEN_OUTPUT
      ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)

  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if(IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
        ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  # Append the includes used for this file to the tablegen_compile_commands
  # file.
  file(
    APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
    "--- !FileInfo:\n" "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n")
endfunction()

# Generate Documentation
function(add_bishengir_doc doc_filename output_file output_directory command)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  # The MLIR docs use Hugo, so we allow Hugo specific features here.
  tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
  set(GEN_DOC_FILE ${BISHENGIR_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  set_target_properties(${output_file}DocGen PROPERTIES FOLDER "BiShengIR/Tablegenning/Docs")
  add_dependencies(bishengir-doc ${output_file}DocGen)
endfunction()