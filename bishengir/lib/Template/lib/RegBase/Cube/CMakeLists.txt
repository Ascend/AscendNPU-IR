set(REGBASE_CUBE_COMMON_SOURCES
    Copy1D.cpp
    Fixpipe.cpp
    GlobalMmad.cpp
    LocalConv.cpp
    LocalMmad.cpp
    LocalMmadMx.cpp
    TcbPool.cpp
    l12ub.cpp
    loadMXScale.cpp
    nchw2c1hwnc0.cpp
    nchw2nc1hwc0.cpp
    nd2nz.cpp
    set2d.cpp
)

# TODO: Extend standalone cube operations to A6 and A2/A3 with target-specific
# implementations. Keep these A5-only wrappers separate to preserve the
# existing A6 build.
set(A5_REGBASE_CUBE_SOURCES
    ${REGBASE_CUBE_COMMON_SOURCES}
    l12l0.cpp
    l12bt.cpp
    MmadL0.cpp
)
compile_cpp_to_bc("aic" "Ascend910D" ${A5_REGBASE_CUBE_SOURCES})

# 920r1 ccec does provide GM->CBUF (L1) MTE2 intrinsics:
# copy_gm_to_cbuf_multi_nd2nz (10-arg pre_allocation form) and
# copy_gm_to_cbuf_align_v2 (12-arg pre_allocation form); the corresponding
# 920r1 branches in ND2NZ.h / DMAUtils.h select them.
set(A6_REGBASE_CUBE_SOURCES
    ${REGBASE_CUBE_COMMON_SOURCES}
)
compile_cpp_to_bc("aic" "Ascend960" ${A6_REGBASE_CUBE_SOURCES})
