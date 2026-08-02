/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2026, Yves Gallot\n" \
"\n" \
"genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#if __OPENCL_VERSION__ >= 120\n" \
"	#define INLINE	static inline\n" \
"#else\n" \
"	#define INLINE\n" \
"#endif\n" \
"\n" \
"#if defined(__NV_CL_C_VERSION)\n" \
"	#define PTX_ASM	1\n" \
"#endif\n" \
"\n" \
"#if !defined(N_SZ)\n" \
"#define N_SZ		65536u\n" \
"#define LN_SZ		16\n" \
"#define VSIZE		8\n" \
"// #define IS32		1\n" \
"#endif\n" \
"\n" \
"typedef uint	sz_t;\n" \
"typedef uint	uint_32;\n" \
"typedef int		int_32;\n" \
"\n" \
"__kernel\n" \
"void backward8(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg, const uint_32 m, const uint_32 s)\n" \
"{\n" \
"	// DECLARE_VAR_REG();\n" \
"	// const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id; DECLARE_IVAR(s, j);\n" \
"	// backward4io(pq, m, &z[k], wi, sji);\n" \
"}\n" \
"";
