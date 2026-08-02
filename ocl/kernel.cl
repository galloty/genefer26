/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#if __OPENCL_VERSION__ >= 120
	#define INLINE	static inline
#else
	#define INLINE
#endif

#if defined(__NV_CL_C_VERSION)
	#define PTX_ASM	1
#endif

#if !defined(N_SZ)
#define N_SZ		65536u
#define LN_SZ		16
#define VSIZE		8
// #define IS32		1
#endif

typedef uint	sz_t;
typedef uint	uint_32;
typedef int		int_32;

__kernel
void backward8(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg, const uint_32 m, const uint_32 s)
{
	// DECLARE_VAR_REG();
	// const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id; DECLARE_IVAR(s, j);
	// backward4io(pq, m, &z[k], wi, sji);
}
