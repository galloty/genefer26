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
#define P1			2130706433u
#define Q1			2164260865u
#define RSQ1		402124772u
#define IM1			2063729671u
#define MFIM1		1930170389u
#define SQRTI1		1626730317u
#define ISQRTI1		856006302u
#define P2			2113929217u
#define Q2			2181038081u
#define RSQ2		2111798781u
#define IM2			530075385u
#define MFIM2		1036950657u
#define SQRTI2		338852760u
#define ISQRTI2		1090446030u
#define P3			2013265921u
#define Q3			2281701377u
#define RSQ3		1172168163u
#define IM3			473486609u
#define MFIM3		734725699u
#define SQRTI3		1032137103u
#define ISQRTI3		1964242958u
#define INVP2_P1	2130706177u
#define INVP3_P1	608773230u
#define INVP3_P2	1409286102u
#define P1P2P3L		1962934273u
#define P1P2P3H		2111326211158966273ul
#define P1P2P3_2L	3128950784u
#define P1P2P3_2H	1055663105579483136ul
#define NORM1		2130641409u
#define NORM2		2113864705u
#define NORM3		2013204481u
#define W_SZ		32768u
#endif

#define VN_SZ		(N_SZ * VSIZE)

typedef uint	sz_t;
typedef uint	uint_32;
typedef int		int_32;
typedef ulong	uint_64;
typedef uint2	uint2_32;
typedef uint4	uint4_32;

// --- modular arithmetic

#define	PQ1		(uint2_32)(P1, Q1)
#define	PQ2		(uint2_32)(P2, Q2)
#define	PQ3		(uint2_32)(P3, Q3)

__constant uint2_32 g_pq[3] = { PQ1, PQ2, PQ3 };

INLINE uint_32 addmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)
{
#if defined(IS32)
	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);
#else
	const uint_32 t = lhs + rhs;
	return t - ((t >= p) ? p : 0);
#endif
}

INLINE uint_32 submod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)
{
#if defined(IS32)
	return lhs - rhs + ((lhs < rhs) ? p : 0);
#else
	const uint_32 t = lhs - rhs;
	return t + (((int_32)(t) < 0) ? p : 0);
#endif
}

// 2 mul + 2 mul_hi
INLINE uint_32 mulmod(const uint_32 lhs, const uint_32 rhs, const uint2_32 pq)
{
	const uint_64 t = lhs * (uint_64)(rhs);
	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
	const uint_32 mp = mul_hi(lo * pq.s1, pq.s0);
	return submod(hi, mp, pq.s0);
}

INLINE uint_32 sqrmod(const uint_32 lhs, const uint2_32 pq) { return mulmod(lhs, lhs, pq); }

INLINE int_32 get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }
INLINE uint_32 set_int(const int_32 i, const uint_32 p) { return (uint_32)(i + ((i < 0) ? p : 0)); }

// --- transform/macro ---

#define FWD2(z0, z1, w) \
{ \
	const uint_32 t = mulmod(z1, w, pq); \
	z1 = submod(z0, t, pq.s0); z0 = addmod(z0, t, pq.s0); \
}

#define BCK2(z0, z1, wi) \
{ \
	const uint_32 t = submod(z1, z0, pq.s0); z0 = addmod(z0, z1, pq.s0); \
	z1 = mulmod(t, wi, pq); \
}

INLINE void _loadg(const sz_t n, uint_32 * const zl, __global const uint_32 * restrict const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _storeg(const sz_t n, __global uint_32 * restrict const z, const sz_t s, const uint_32 * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

// ---

INLINE void _backward8r(const uint2_32 pq, uint_32 z[8], const uint_32 wi1, const uint_32 wi2r[2], const uint_32 wi4r[4])
{
	BCK2(z[0], z[1], wi4r[3]); BCK2(z[2], z[3], wi4r[2]); BCK2(z[4], z[5], wi4r[1]); BCK2(z[6], z[7], wi4r[0]);
	BCK2(z[0], z[2], wi2r[1]); BCK2(z[1], z[3], wi2r[1]); BCK2(z[4], z[6], wi2r[0]); BCK2(z[5], z[7], wi2r[0]);
	BCK2(z[0], z[4], wi1); BCK2(z[2], z[6], wi1); BCK2(z[1], z[5], wi1); BCK2(z[3], z[7], wi1);
}

INLINE void backward8io(const uint2_32 pq, const sz_t vm, __global uint_32 * restrict const z, __global const uint_32 * restrict const w, const sz_t sji)
{
	const uint_32 wi1 = w[sji];
	uint_32 wi2r[2]; { const uint2_32 t = ((__global const uint2_32 *)w)[sji]; wi2r[0] = t.s0; wi2r[1] = t.s1; }
	uint_32 wi4r[4]; { const uint4_32 t = ((__global const uint4_32 *)w)[sji]; wi4r[0] = t.s0; wi4r[1] = t.s1; wi4r[2] = t.s2; wi4r[3] = t.s3; }

	uint_32 zl[8]; _loadg(8, zl, z, vm);
	_backward8r(pq, zl, wi1, wi2r, wi4r);
	_storeg(8, z, vm, zl);
}

__kernel
void backward8(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg, const int_32 lm, const uint_32 s)
{
	// DECLARE_VAR_REGv1
	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (VN_SZ / 8), id = gid % (VN_SZ / 8); \
	const uint2_32 pq = g_pq[lid]; \
	__global uint_32 * restrict const z = &zg[lid * VN_SZ]; \
	__global const uint_32 * restrict const w = &wg[lid * W_SZ];

	const sz_t vm = VSIZE << lm, j = (id / VSIZE) >> lm, k = 7 * (id & ~(vm - 1)) + id;

	const sz_t ji = s - j - 1;
	backward8io(pq, vm, &z[k], w, s + ji);
}
