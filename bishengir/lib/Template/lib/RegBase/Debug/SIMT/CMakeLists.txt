set(BISHENGIR_SIMT_DEBUG_LEGACY_MAX_REGISTERS "" CACHE STRING
    "Callable register limit for a legacy CCEC consumer (empty disables rewriting)")
if (NOT BISHENGIR_SIMT_DEBUG_LEGACY_MAX_REGISTERS STREQUAL "")
    set(BISHENGIR_SIMT_DEBUG_LEGACY_LLVM_LINK "${LLVM_LINK_TOOL}" CACHE FILEPATH
        "Vendor llvm-link with textual IR support for the SIMT debug template")
    if (NOT EXISTS "${BISHENGIR_SIMT_DEBUG_LEGACY_LLVM_LINK}")
        message(FATAL_ERROR "Set BISHENGIR_SIMT_DEBUG_LEGACY_LLVM_LINK to a vendor llvm-link")
    endif ()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(SIMT_DEBUG_LEGACY_REWRITE "${CMAKE_CURRENT_SOURCE_DIR}/add_legacy_callable_limits.py")
endif ()

set(EXTRA_BISHENG_DEFINES -DCCE_PRINT_SIMT=1)
compile_cpp_to_bc("aiv" "Ascend910D")
