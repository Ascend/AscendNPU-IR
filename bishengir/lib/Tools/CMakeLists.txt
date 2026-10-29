# Vendor in-process compiler service header (bisheng-llc.h).
# The SO is loaded at runtime via dlmopen; only the header is needed at build.
# Defined here (before any add_subdirectory) so every downstream target
# (bishengir-compile, regbase, hivmc/A3, hivmc/A5, ...) sees the same value.
# When BISHENG_COMPILER_PATH is given (pointing at the bisheng/ccec bin/ dir),
# default BISHENG_LLC_INCLUDE_DIR to ${BISHENG_COMPILER_PATH}/../include (the
# standard install layout: bin/ sibling of include/).  Otherwise require the
# user to pass -DBISHENG_LLC_INCLUDE_DIR=... explicitly.
if(BISHENG_COMPILER_PATH)
  set(BISHENG_LLC_INCLUDE_DIR_DEFAULT "${BISHENG_COMPILER_PATH}/../include")
else()
  set(BISHENG_LLC_INCLUDE_DIR_DEFAULT "")
endif()
set(BISHENG_LLC_INCLUDE_DIR "${BISHENG_LLC_INCLUDE_DIR_DEFAULT}" CACHE PATH
    "Path to the installed bisheng-llc.h vendor header directory")
if(NOT BISHENG_LLC_INCLUDE_DIR AND NOT BISHENG_COMPILER_PATH)
  message(WARNING "BISHENG_LLC_INCLUDE_DIR is empty and BISHENG_COMPILER_PATH is "
    "not set; the Native in-process backend (bisheng-llc.h) will not be found. "
    "Pass -DBISHENG_COMPILER_PATH=<bisheng-bin-dir> or "
    "-DBISHENG_LLC_INCLUDE_DIR=<bisheng-llc.h-dir>.")
endif()

add_subdirectory(bishengir-compile)
add_subdirectory(bishengir-compile/regbase)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  add_subdirectory(bishengir-triton-libs)
endif()
add_subdirectory(ConfigOptions)
add_subdirectory(RetriablePassManager)
add_subdirectory(Utils)
add_subdirectory(Version)
if(BISHENGIR_BUILD_HIVMC)
  add_subdirectory(hivmc)
endif()
