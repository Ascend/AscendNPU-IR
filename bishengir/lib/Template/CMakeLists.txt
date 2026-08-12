option(ENABLE_CPU_TRACE_INTRINSIC "If enable debug, use cpu compile." OFF)
message("-- ENABLE_CPU_TRACE_INTRINSIC Value: ${ENABLE_CPU_TRACE_INTRINSIC}")

if(CMAKE_SYSROOT)
    message("-- Using custom sysroot: ${CMAKE_SYSROOT}")
else()
    message("-- No CMAKE_SYSROOT specified. Omitting sysroot flag.")
endif()

# Track headers used by the template sources. ccec is invoked as a custom
# command rather than through CMake's native C++ rules, so CMake cannot infer
# header dependencies from compiler-generated depfiles here.
file(GLOB_RECURSE BISHENGIR_TEMPLATE_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx"
    "${PROJECT_SOURCE_DIR}/third-party/shmem/*.h"
    "${PROJECT_SOURCE_DIR}/third-party/shmem/*.hh"
    "${PROJECT_SOURCE_DIR}/third-party/shmem/*.hpp"
    "${PROJECT_SOURCE_DIR}/third-party/shmem/*.hxx")

# Create an ALL custom target in the current (subdir) scope that depends on the
# given .bc outputs. CMake only emits an add_custom_command(OUTPUT ...) rule
# into the build when its output is consumed by some target *in the same
# directory*. Without a same-dir consumer the per-file rules registered by
# compile_*_to_bc (which run in subdir scope) are not emitted, and the parent
# link targets would fail with "<file>.bc missing, no known rule to make it".
# This target provides that same-dir consumer; its name is also recorded
# globally so link targets can add_dependencies on it (target->target, which is
# resolved across directories).
function(bishengir_register_bcgen_target outputs)
    if(NOT outputs)
        return()
    endif()
    string(MD5 call_id "${outputs}")
    set(bcgen_target "bishengir_bcgen_${call_id}")
    if(NOT TARGET ${bcgen_target})
        add_custom_target(${bcgen_target} ALL DEPENDS ${outputs})
    endif()
    set_property(GLOBAL APPEND PROPERTY bishengir_bcgen_targets "${bcgen_target}")
endfunction()

#[[Compile single xx.cpp to xx.bc
 Core_type: "aic" "aiv" "host"
 Platform_type: "Ascend910B"]]
function(compile_single_cpp_to_bc cpp_with_path core_type platform_type)
    # Get name of xx.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})

    set(bisheng_args "-O2")
    list(APPEND bisheng_args -x cce)
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args --cce-auto-sync=off)
    list(APPEND bisheng_args --cce-aicore-only)
    list(APPEND bisheng_args --cce-generic-addrspace=off)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if ((${core_type} STREQUAL "mix_aic") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args -cce-enable-mix)
        list(APPEND bisheng_args -mllvm -enable-mix=true)
    endif ()

    if ((${core_type} STREQUAL "aic") OR (${core_type} STREQUAL "mix_aic"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-cube)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-cube)
            list(APPEND bisheng_args --cce-enable-sanitizer)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aic")
            string(REGEX REPLACE ".cpp" ".aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif ((${core_type} STREQUAL "aiv") OR (${core_type} STREQUAL "mix_aiv"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-vec)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-vec)
            list(APPEND bisheng_args --cce-enable-sanitizer)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aiv")
            string(REGEX REPLACE ".cpp" ".aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
    else ()
        message(FATAL_ERROR "Core type must be aic/mix_aic/aiv/mix_aiv or host.")
    endif ()

    if (${core_type} STREQUAL "host")
        message(STATUS "Running on host platform.")
    else ()
        if (NOT ${platform_type} STREQUAL "Ascend910B" AND
            NOT ${platform_type} STREQUAL "Ascend910D")
            message(FATAL_ERROR
                    "Unsupported platform! Currently support Ascend910B and Ascend910D.")
        endif ()
    endif ()

    list(APPEND bisheng_args -std=c++17)
    if ("${platform_type}" STREQUAL "Ascend910D")
        # RegBase sources carry private C310 variants of shared headers here.
        # Keeping this path ahead of the global include path prevents the A5
        # ABI from replacing Ascend910B declarations.
        list(APPEND bisheng_args -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif ()
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -o ${bc_name})

    # Conditionally append sysroot if it exists
    if(CMAKE_SYSROOT)
        list(APPEND bisheng_args --sysroot=${CMAKE_SYSROOT})
    endif()

    # Derive a tag for grouping .bc outputs by (core_type, arch) for the link step.
    if (${core_type} STREQUAL "host")
        set(bc_arch_tag "host")
    else ()
        string(REPLACE "." "_" arch_suffix_tok "${bc_arch_suffix}")
        set(bc_arch_tag "${core_type}${arch_suffix_tok}")
    endif ()

    set(bc_output "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    string(REGEX REPLACE ";" " " compile_command "${BISHENG_TOOL} ${bisheng_args}")
    message("-- command of compile cpp to bc is ${compile_command}")

    # A custom command is not invalidated when its COMMAND arguments change.
    # Generate a content-addressed command signature and depend on it so a
    # changed ccec flag, compiler path, include path, or sysroot rebuilds this
    # output. file(GENERATE) updates the file only when its content changes.
    string(REPLACE ";" "\n" compile_signature_content "${BISHENG_TOOL}\n${bisheng_args}")
    set(compile_signature "${bc_output}.ccec.args")
    file(GENERATE OUTPUT "${compile_signature}" CONTENT "${compile_signature_content}\n")

    add_custom_command(
        OUTPUT "${bc_output}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${BISHENG_TOOL} ${bisheng_args}
        DEPENDS "${cpp_with_path}" ${BISHENGIR_TEMPLATE_HEADERS} "${compile_signature}"
        COMMENT "[ccec] ${cpp_name} -> ${bc_name}"
        VERBATIM
    )
    set_property(GLOBAL APPEND PROPERTY bishengir_bc_outputs_${bc_arch_tag} "${bc_output}")
    set(bc_output "${bc_output}" PARENT_SCOPE)
endfunction()

function(compile_shmem_to_bc cpp_with_path core_type platform_type)
    # Get name of xx.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})

    set(bisheng_args "-O2")
    list(APPEND bisheng_args -x cce)
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args --cce-auto-sync=off)
    list(APPEND bisheng_args --cce-aicore-only)
    list(APPEND bisheng_args --cce-generic-addrspace=off)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)
    list(APPEND bisheng_args --cce-enable-print)
    # open --cce-enable-sanitizer after shmem upgrade to 1.6.x

    if ((${core_type} STREQUAL "mix_aic") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args -cce-enable-mix)
        list(APPEND bisheng_args -mllvm -enable-mix=true)
    endif ()

    if ((${core_type} STREQUAL "aic") OR (${core_type} STREQUAL "mix_aic"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-cube)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-cube)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aic")
            string(REGEX REPLACE ".cpp" ".aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif ((${core_type} STREQUAL "aiv") OR (${core_type} STREQUAL "mix_aiv"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-vec)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-vec)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aiv")
            string(REGEX REPLACE ".cpp" ".aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
    else ()
        message(FATAL_ERROR "Core type must be aic/mix_aic/aiv/mix_aiv or host.")
    endif ()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -I ${PROJECT_SOURCE_DIR}/third-party/shmem/include/)
    list(APPEND bisheng_args -I ${PROJECT_SOURCE_DIR}/third-party/shmem/src/device/)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
        list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/)
    else ()
        message(FATAL_ERROR "Unknown architecture")
    endif ()
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/)
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl)
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -o ${bc_name})

    # Conditionally append sysroot if it exists
    if(CMAKE_SYSROOT)
        list(APPEND bisheng_args --sysroot=${CMAKE_SYSROOT})
    endif()

    # Shmem may be built for both C220 and C310. Keep each architecture in
    # its matching meta_op group; mixing them makes hivmc consume C310 device
    # bitcode while compiling an Ascend910B module.
    string(REPLACE "." "_" arch_suffix_tok "${bc_arch_suffix}")
    set(bc_arch_tag "${core_type}${arch_suffix_tok}")

    set(bc_output "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    string(REGEX REPLACE ";" " " compile_command "${BISHENG_TOOL} ${bisheng_args}")
    message("-- command of compile cpp to bc is ${compile_command}")

    string(REPLACE ";" "\n" compile_signature_content "${BISHENG_TOOL}\n${bisheng_args}")
    set(compile_signature "${bc_output}.ccec.args")
    file(GENERATE OUTPUT "${compile_signature}" CONTENT "${compile_signature_content}\n")

    add_custom_command(
        OUTPUT "${bc_output}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${BISHENG_TOOL} ${bisheng_args}
        DEPENDS "${cpp_with_path}" ${BISHENGIR_TEMPLATE_HEADERS} "${compile_signature}"
        COMMENT "[ccec] ${cpp_name} -> ${bc_name}"
        VERBATIM
    )
    set_property(GLOBAL APPEND PROPERTY bishengir_bc_outputs_${bc_arch_tag} "${bc_output}")
    bishengir_register_bcgen_target("${bc_output}")
endfunction()

#[[Compile single xx.cpp to xx_debug.bc
 Core_type: "aic" "aiv" "host"
 Platform_type: "DEBUG"]]
function(compile_single_cpp_to_bc_debug cpp_with_path core_type platform_type)
    # Get name of xx_debug.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})
    set_source_files_properties(${cpp_with_path} PROPERTIES LANGUAGE c++)

    set(bisheng_args "-O2")
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if (${core_type} STREQUAL "aic")
        string(REGEX REPLACE ".cpp" ".aic.bc" bc_name ${cpp_name})
    elseif (${core_type} STREQUAL "aiv")
        string(REGEX REPLACE ".cpp" ".aiv.bc" bc_name ${cpp_name})
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})    
    else ()
        message(FATAL_ERROR "Core type must be aic/aiv or host.")
    endif ()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -DENABLE_CPU_TRACE_INTRINSIC)
    list(APPEND bisheng_args -o ${bc_name})

    # Debug-mode .bc files (no arch suffix) are not linked into meta_op:
    # link_all_to_meta_op only handles c220/c310/host. Tag host outputs as
    # "host" so they participate in the host link; aic/aiv debug outputs are
    # tagged separately and left unlinked, matching the old behavior.
    if (${core_type} STREQUAL "host")
        set(bc_arch_tag "host")
    else ()
        set(bc_arch_tag "${core_type}_debug")
    endif ()

    set(bc_output "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    string(REGEX REPLACE ";" " " compile_command "${BISHENG_TOOL} ${bisheng_args}")
    message("-- command of compile cpp to bc is ${compile_command}")

    string(REPLACE ";" "\n" compile_signature_content "${BISHENG_TOOL}\n${bisheng_args}")
    set(compile_signature "${bc_output}.ccec.args")
    file(GENERATE OUTPUT "${compile_signature}" CONTENT "${compile_signature_content}\n")

    add_custom_command(
        OUTPUT "${bc_output}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${BISHENG_TOOL} ${bisheng_args}
        DEPENDS "${cpp_with_path}" ${BISHENGIR_TEMPLATE_HEADERS} "${compile_signature}"
        COMMENT "[ccec] ${cpp_name} -> ${bc_name}"
        VERBATIM
    )
    set_property(GLOBAL APPEND PROPERTY bishengir_bc_outputs_${bc_arch_tag} "${bc_output}")
    set(bc_output "${bc_output}" PARENT_SCOPE)
endfunction()

#[[Compile multi xx.cpp to multi xx.bc
 Core_type: "aic" "aiv"
 Platform_type: "Ascend910B"]]
function(compile_cpp_to_bc)
    set (core_type " ")
    set (platform_type " ")

    set(core_type ${ARGV0})
    if(NOT ${ARGV0} STREQUAL "host")
        set(platform_type ${ARGV1})
    endif ()

    if (ARGC GREATER 2)
        list(SUBLIST ARGV 2 -1 cpp_list)
        set(explicit_cpp_list)
        foreach (cpp_source ${cpp_list})
            if (IS_ABSOLUTE "${cpp_source}")
                list(APPEND explicit_cpp_list "${cpp_source}")
            else ()
                list(APPEND explicit_cpp_list
                     "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_source}")
            endif ()
        endforeach ()
        set(cpp_list ${explicit_cpp_list})
    else ()
        file(GLOB cpp_list CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    endif ()
    set(this_call_outputs "")
    foreach (cpp_with_path ${cpp_list})
        if (ENABLE_CPU_TRACE_INTRINSIC AND
            (NOT ${core_type} STREQUAL "mix_aic") AND
            (NOT ${core_type} STREQUAL "mix_aiv"))
            compile_single_cpp_to_bc_debug(${cpp_with_path} ${core_type} "DEBUG")
        else ()
            compile_single_cpp_to_bc(${cpp_with_path} ${core_type} ${platform_type})
        endif ()
        list(APPEND this_call_outputs ${bc_output})
    endforeach ()
    bishengir_register_bcgen_target("${this_call_outputs}")
endfunction()

# Register a link rule that merges all per-file .bc registered under
# bishengir_bc_outputs_<tag> (populated by compile_*_to_bc) into `out_file`.
# Using add_custom_command(OUTPUT ...) makes the link incremental: ninja skips
# llvm-link when `out_file` is newer than its .bc inputs. The DEPENDS on the
# subdir .bc files resolves because the bcgen targets created in each subdir
# (bishengir_register_bcgen_target) make those custom-command rules emitted
# into the build. The ALL driver target just triggers the custom command.
# Each group is its own ALL target so ninja can run the link steps in parallel.
function(link_one_group_to_meta_op tag out_file)
    get_property(inputs GLOBAL PROPERTY bishengir_bc_outputs_${tag})
    if(NOT inputs)
        message(STATUS "No .bc files registered for ${tag}, skip linking ${out_file}")
        return()
    endif()
    string(REGEX REPLACE "[^0-9A-Za-z_]" "_" target_suffix "${tag}")
    set(link_target "bishengir_link_${target_suffix}")
    string(REPLACE ";" "\n" link_signature_content "${LLVM_LINK_TOOL}\n${inputs}\n-o\n${out_file}")
    set(link_signature "${out_file}.llvm-link.args")
    file(GENERATE OUTPUT "${link_signature}" CONTENT "${link_signature_content}\n")
    add_custom_command(
        OUTPUT "${out_file}"
        COMMAND ${LLVM_LINK_TOOL} ${inputs} -o "${out_file}"
        DEPENDS ${inputs} "${link_signature}"
        COMMENT "[llvm-link] ${tag} -> ${out_file}"
        VERBATIM
    )
    add_custom_target(${link_target} ALL DEPENDS "${out_file}")
    set_property(GLOBAL APPEND PROPERTY bishengir_meta_op_targets "${link_target}")
endfunction()

function(link_all_to_meta_op)
    set(BITCODE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
    file(MAKE_DIRECTORY ${BITCODE_OUTPUT_DIR})

    # c220 (Ascend910B / A3): use .c220 suffix.
    link_one_group_to_meta_op("aic_c220"     "${BITCODE_OUTPUT_DIR}/meta_op.aic.c220.bc")
    link_one_group_to_meta_op("aiv_c220"     "${BITCODE_OUTPUT_DIR}/meta_op.aiv.c220.bc")
    link_one_group_to_meta_op("mix_aic_c220" "${BITCODE_OUTPUT_DIR}/meta_op.mix.aic.c220.bc")
    link_one_group_to_meta_op("mix_aiv_c220" "${BITCODE_OUTPUT_DIR}/meta_op.mix.aiv.c220.bc")

    # c310 (Ascend910D / A5): use .c310 suffix.
    link_one_group_to_meta_op("aic_c310"     "${BITCODE_OUTPUT_DIR}/meta_op.aic.c310.bc")
    link_one_group_to_meta_op("aiv_c310"     "${BITCODE_OUTPUT_DIR}/meta_op.aiv.c310.bc")
    link_one_group_to_meta_op("mix_aic_c310" "${BITCODE_OUTPUT_DIR}/meta_op.mix.aic.c310.bc")
    link_one_group_to_meta_op("mix_aiv_c310" "${BITCODE_OUTPUT_DIR}/meta_op.mix.aiv.c310.bc")

    # Host bitcode (arch-agnostic).
    link_one_group_to_meta_op("host" "${BITCODE_OUTPUT_DIR}/host.bc")

    # NOTE: the install-component -> link-target dependency is wired after the
    # install() call below (the install-bishengir-publish-products target only
    # exists once install(FILES ... COMPONENT bishengir-publish-products) runs).
endfunction()

if (BISHENGIR_BUILD_TEMPLATE STREQUAL "ON")
    message("-- BISHENGIR_BUILD_TEMPLATE Value: ${BISHENGIR_BUILD_TEMPLATE}")
    message("-- BISHENG_COMPILER_PATH Value: ${BISHENG_COMPILER_PATH}")

    set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(BISHENG_TOOL "${BISHENG_COMPILER_PATH}/ccec")
    set(LLVM_LINK_TOOL "${BISHENG_COMPILER_PATH}/llvm-link")
    if ((NOT EXISTS "${BISHENG_TOOL}") OR (NOT EXISTS "${LLVM_LINK_TOOL}"))
        message(FATAL_ERROR "Can not find ccec or llvm-link in ${BISHENG_COMPILER_PATH},\
please check ${BISHENG_TOOL} or ${LLVM_LINK_TOOL} exist or not!")
    endif ()

    add_subdirectory(lib)
    link_all_to_meta_op()

    install(FILES
            ${CMAKE_BINARY_DIR}/lib/meta_op.aic.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.aiv.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aic.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aiv.c220.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.aic.c310.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.aiv.c310.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aic.c310.bc
            ${CMAKE_BINARY_DIR}/lib/meta_op.mix.aiv.c310.bc
            ${CMAKE_BINARY_DIR}/lib/host.bc
            DESTINATION lib
            COMPONENT bishengir-publish-products)
    # Wire the install component to depend on every link target so that
    # `cmake --build . --target install-bishengir-publish-products` builds the
    # .bc files first. Done here (after install()) because the install-<component>
    # target only comes into existence once the install() call above is processed.
    # build.sh always runs `ninja` (ALL) before install, so this is extra safety.
    get_property(_bishengir_link_targets GLOBAL PROPERTY bishengir_meta_op_targets)
    if(TARGET install-bishengir-publish-products)
        foreach(_t ${_bishengir_link_targets})
            add_dependencies(install-bishengir-publish-products ${_t})
        endforeach()
    endif()
endif ()

option(ENABLE_SYCL "If enable sycl, use bishengcpp. Otherwise use bisheng." OFF)
message("-- ENABLE_SYCL Value: ${ENABLE_SYCL}")

if (ENABLE_SYCL)
    add_definitions(-DENABLE_SYCL)
endif ()
