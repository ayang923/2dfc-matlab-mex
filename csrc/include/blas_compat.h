#ifndef __BLAS_COMPAT_H__
#define __BLAS_COMPAT_H__

/*
 * Portable CBLAS access. The original 2dfc-c required the Intel oneAPI
 * compiler + MKL. This header only requires a standard CBLAS interface,
 * satisfied by:
 *   - macOS:  the system Accelerate framework (-framework Accelerate)
 *   - Linux/other: OpenBLAS, or any other library shipping <cblas.h>
 *     (e.g. `apt install libopenblas-dev`), linked with -lopenblas or -lcblas
 *   - Linux/other, no OpenBLAS available: MKL's LP64 CBLAS interface
 *     (`-DUSE_MKL_CBLAS`, set automatically by build_mex.m/csrc/Makefile
 *     when $MKLROOT is set). This still uses plain `int` throughout (MKL's
 *     default LP64 interface), not MKL_INT/ILP64, so it's a drop-in swap.
 *
 * No LAPACK/LAPACKE dependency is needed anywhere in this port -- see
 * patch_kernels.c for the one 2x2 linear solve, which is hand-rolled.
 */
#if defined(__APPLE__)
    /* Use Apple's modern (non-deprecated) Accelerate headers, but stay in
     * LP64 mode (32-bit ints) since that's what this codebase uses
     * throughout -- do not define ACCELERATE_LAPACK_ILP64. */
    #ifndef ACCELERATE_NEW_LAPACK
        #define ACCELERATE_NEW_LAPACK
    #endif
    #include <Accelerate/Accelerate.h>
#elif defined(USE_MKL_CBLAS)
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#endif
