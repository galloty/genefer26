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
"#define P1			2130706433u\n" \
"#define Q1			2164260865u\n" \
"#define RSQ1		402124772u\n" \
"#define IM1			2063729671u\n" \
"#define MFIM1		1930170389u\n" \
"#define SQRTI1		1626730317u\n" \
"#define ISQRTI1		856006302u\n" \
"#define P2			2113929217u\n" \
"#define Q2			2181038081u\n" \
"#define RSQ2		2111798781u\n" \
"#define IM2			530075385u\n" \
"#define MFIM2		1036950657u\n" \
"#define SQRTI2		338852760u\n" \
"#define ISQRTI2		1090446030u\n" \
"#define P3			2013265921u\n" \
"#define Q3			2281701377u\n" \
"#define RSQ3		1172168163u\n" \
"#define IM3			473486609u\n" \
"#define MFIM3		734725699u\n" \
"#define SQRTI3		1032137103u\n" \
"#define ISQRTI3		1964242958u\n" \
"#define INVP2_P1	2130706177u\n" \
"#define INVP3_P1	608773230u\n" \
"#define INVP3_P2	1409286102u\n" \
"#define P1P2P3L		1962934273u\n" \
"#define P1P2P3H		2111326211158966273ul\n" \
"#define P1P2P3_2L	3128950784u\n" \
"#define P1P2P3_2H	1055663105579483136ul\n" \
"#define NORM1		2130641409u\n" \
"#define NORM2		2113864705u\n" \
"#define NORM3		2013204481u\n" \
"#define W_SZ		32768u\n" \
"#define CARRY_WG_SZ	256u\n" \
"#endif\n" \
"\n" \
"#define VN_SZ		(N_SZ * VSIZE)\n" \
"\n" \
"typedef uint	sz_t;\n" \
"typedef uint	uint_32;\n" \
"typedef int		int_32;\n" \
"typedef ulong	uint_64;\n" \
"typedef long	int_64;\n" \
"typedef uint2	uint2_32;\n" \
"typedef uint4	uint4_32;\n" \
"// typedef int4	int4_32;\n" \
"\n" \
"// --- modular arithmetic\n" \
"\n" \
"#define	PQ1		(uint2_32)(P1, Q1)\n" \
"#define	PQ2		(uint2_32)(P2, Q2)\n" \
"#define	PQ3		(uint2_32)(P3, Q3)\n" \
"\n" \
"__constant uint2_32 g_pq[3] = { PQ1, PQ2, PQ3 };\n" \
"__constant uint4_32 g_f0[3] = { (uint4_32)(RSQ1, MFIM1, SQRTI1, ISQRTI1), (uint4_32)(RSQ2, MFIM2, SQRTI2, ISQRTI2), (uint4_32)(RSQ3, MFIM3, SQRTI3, ISQRTI3) };\n" \
"\n" \
"INLINE uint_32 addmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)\n" \
"{\n" \
"#if defined(IS32)\n" \
"	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);\n" \
"#else\n" \
"	const uint_32 t = lhs + rhs;\n" \
"	return t - ((t >= p) ? p : 0);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE uint_32 submod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)\n" \
"{\n" \
"#if defined(IS32)\n" \
"	return lhs - rhs + ((lhs < rhs) ? p : 0);\n" \
"#else\n" \
"	const uint_32 t = lhs - rhs;\n" \
"	return t + (((int_32)(t) < 0) ? p : 0);\n" \
"#endif\n" \
"}\n" \
"\n" \
"// 2 mul + 2 mul_hi\n" \
"INLINE uint_32 mulmod(const uint_32 lhs, const uint_32 rhs, const uint2_32 pq)\n" \
"{\n" \
"	const uint_64 t = lhs * (uint_64)(rhs);\n" \
"	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);\n" \
"	const uint_32 mp = mul_hi(lo * pq.s1, pq.s0);\n" \
"	return submod(hi, mp, pq.s0);\n" \
"}\n" \
"\n" \
"INLINE uint_32 sqrmod(const uint_32 lhs, const uint2_32 pq) { return mulmod(lhs, lhs, pq); }\n" \
"\n" \
"INLINE int_32 get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }\n" \
"INLINE uint_32 set_int(const int_32 i, const uint_32 p) { return (uint_32)(i + ((i < 0) ? p : 0)); }\n" \
"\n" \
"// --- v2\n" \
"\n" \
"// INLINE uint2_32 mulmod2(const uint2_32 lhs, const uint2_32 rhs, const uint2_32 pq)\n" \
"// {\n" \
"// 	return (uint2_32)(mulmod(lhs.s0, rhs.s0, pq), mulmod(lhs.s1, rhs.s1, pq));\n" \
"// }\n" \
"\n" \
"// --- v4\n" \
"\n" \
"// INLINE uint4_32 mulmod4(const uint4_32 lhs, const uint4_32 rhs, const uint2_32 pq)\n" \
"// {\n" \
"// 	return (uint4_32)(mulmod2(lhs.s01, rhs.s01, pq), mulmod2(lhs.s23, rhs.s23, pq));\n" \
"// }\n" \
"\n" \
"// --- uint96/int96 ---\n" \
"\n" \
"typedef struct { uint_32 s0; uint_64 s1; } uint96;\n" \
"typedef struct { uint_32 s0; int_64 s1; } int96;\n" \
"\n" \
"INLINE int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int_64)(x.s1); return r; }\n" \
"\n" \
"INLINE uint96 uint96_set(const uint_32 s0, const uint_64 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }\n" \
"\n" \
"INLINE int96 int96_set_si(const int_64 n) { int96 r; r.s0 = (uint_32)(n); r.s1 = n >> 32; return r; }\n" \
"\n" \
"INLINE bool int96_is_neg(const int96 x) { return (x.s1 < 0); }\n" \
"\n" \
"INLINE bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }\n" \
"\n" \
"INLINE uint96 uint96_add_64(const uint96 x, const uint_64 y)\n" \
"{\n" \
"	const uint_32 yl = (uint_32)(y); const uint_64 yh = y >> 32;\n" \
"	uint96 r;\n" \
"#if defined(PTX_ASM)\n" \
"	asm volatile (\"add.cc.u32 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (yl));\n" \
"	asm volatile (\"addc.u64 %0, %1, %2;\" : \"=l\" (r.s1) : \"l\" (x.s1), \"l\" (yh));\n" \
"#else\n" \
"	const uint_32 s0 = x.s0 + yl;\n" \
"	r.s0 = s0; r.s1 = x.s1 + yh + ((s0 < x.s0) ? 1 : 0);\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int96 int96_add_64(const int96 x, const int_64 y)\n" \
"{\n" \
"	const uint_32 yl = (uint_32)(y); const int_64 yh = y >> 32;\n" \
"	int96 r;\n" \
"#if defined(PTX_ASM)\n" \
"	asm volatile (\"add.cc.u32 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (yl));\n" \
"	asm volatile (\"addc.s64 %0, %1, %2;\" : \"=l\" (r.s1) : \"l\" (x.s1), \"l\" (yh));\n" \
"#else\n" \
"	const uint_32 s0 = x.s0 + yl;\n" \
"	r.s0 = s0; r.s1 = x.s1 + yh + ((s0 < x.s0) ? 1 : 0);\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int96 int96_add(const int96 x, const int96 y)\n" \
"{\n" \
"	int96 r;\n" \
"#if defined(PTX_ASM)\n" \
"	asm volatile (\"add.cc.u32 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (y.s0));\n" \
"	asm volatile (\"addc.s64 %0, %1, %2;\" : \"=l\" (r.s1) : \"l\" (x.s1), \"l\" (y.s1));\n" \
"#else\n" \
"	const uint_32 s0 = x.s0 + y.s0;\n" \
"	r.s0 = s0; r.s1 = x.s1 + y.s1 + ((s0 < x.s0) ? 1 : 0);\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint96 uint96_sub(const uint96 x, const uint96 y)\n" \
"{\n" \
"	uint96 r;\n" \
"#if defined(PTX_ASM)\n" \
"	asm volatile (\"sub.cc.u32 %0, %1, %2;\" : \"=r\" (r.s0) : \"r\" (x.s0), \"r\" (y.s0));\n" \
"	asm volatile (\"subc.u64 %0, %1, %2;\" : \"=l\" (r.s1) : \"l\" (x.s1), \"l\" (y.s1));\n" \
"#else\n" \
"	r.s0 = x.s0 - y.s0; r.s1 = (int_64)(x.s1 - y.s1 - ((x.s0 < y.s0) ? 1 : 0));\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint96 int96_abs(const int96 x)\n" \
"{\n" \
"	const bool is_neg = int96_is_neg(x);\n" \
"	const uint96 mask = uint96_set(is_neg ? ~0u : 0u, is_neg ? ~0ul : 0ul);\n" \
"	const uint96 t = uint96_set(x.s0 ^ mask.s0, (uint_64)(x.s1) ^ mask.s1);\n" \
"	return uint96_sub(t, mask);\n" \
"}\n" \
"\n" \
"INLINE uint96 uint96_mul_64_32(const uint_64 x, const uint_32 y)\n" \
"{\n" \
"	const uint_64 l = (uint_32)(x) * (uint_64)(y);\n" \
"	uint96 r; r.s0 = (uint_32)(l); r.s1 = (x >> 32) * y + (l >> 32);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// --- I/O ---\n" \
"\n" \
"INLINE void _loadg(const sz_t n, uint_32 * const zl, __global const uint_32 * restrict const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }\n" \
"INLINE void _storeg(const sz_t n, __global uint_32 * restrict const z, const sz_t s, const uint_32 * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }\n" \
"\n" \
"INLINE uint2_32 _loadg2(__global const uint_32 * restrict const w, const sz_t j) { return ((__global const uint2_32 *)w)[j]; }\n" \
"INLINE uint4_32 _loadg4(__global const uint_32 * restrict const w, const sz_t j) { return ((__global const uint4_32 *)w)[j]; }\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define FWD2(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint_32 t = mulmod(z1, w, pq); \\\n" \
"	z1 = submod(z0, t, pq.s0); z0 = addmod(z0, t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define BCK2(z0, z1, wi) \\\n" \
"{ \\\n" \
"	const uint_32 t = submod(z1, z0, pq.s0); z0 = addmod(z0, z1, pq.s0); \\\n" \
"	z1 = mulmod(t, wi, pq); \\\n" \
"}\n" \
"\n" \
"#define SQR2(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint_32 t = mulmod(sqrmod(z1, pq), w, pq); \\\n" \
"	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \\\n" \
"	z0 = addmod(sqrmod(z0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define SQR2N(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint_32 t = mulmod(sqrmod(z1, pq), w, pq); \\\n" \
"	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \\\n" \
"	z0 = submod(sqrmod(z0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2(z0, z1, zp0, zp1, w) \\\n" \
"{ \\\n" \
"	const uint_32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \\\n" \
"	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \\\n" \
"	z0 = addmod(mulmod(z0, zp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2N(z0, z1, zp0, zp1, w) \\\n" \
"{ \\\n" \
"	const uint_32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \\\n" \
"	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \\\n" \
"	z0 = submod(mulmod(z0, zp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"// --- transform/inline ---\n" \
"\n" \
"INLINE void _forward8(const uint2_32 pq, uint_32 z[8], const uint_32 w1, const uint2_32 w2, const uint4_32 w4)\n" \
"{\n" \
"	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);\n" \
"	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);\n" \
"	FWD2(z[0], z[1], w4.s0); FWD2(z[2], z[3], w4.s1); FWD2(z[4], z[5], w4.s2); FWD2(z[6], z[7], w4.s3);\n" \
"}\n" \
"\n" \
"INLINE void _backward8r(const uint2_32 pq, uint_32 z[8], const uint_32 wi1, const uint2_32 wi2r, const uint4_32 wi4r)\n" \
"{\n" \
"	BCK2(z[0], z[1], wi4r.s3); BCK2(z[2], z[3], wi4r.s2); BCK2(z[4], z[5], wi4r.s1); BCK2(z[6], z[7], wi4r.s0);\n" \
"	BCK2(z[0], z[2], wi2r.s1); BCK2(z[1], z[3], wi2r.s1); BCK2(z[4], z[6], wi2r.s0); BCK2(z[5], z[7], wi2r.s0);\n" \
"	BCK2(z[0], z[4], wi1); BCK2(z[2], z[6], wi1); BCK2(z[1], z[5], wi1); BCK2(z[3], z[7], wi1);\n" \
"}\n" \
"\n" \
"INLINE void _forward8_0(const uint2_32 pq, const uint4_32 f0, uint_32 z[8], const uint4_32 w4)\n" \
"{\n" \
"	z[0] = mulmod(z[0], f0.s0, pq); z[1] = mulmod(z[1], f0.s0, pq); z[2] = mulmod(z[2], f0.s0, pq); z[3] = mulmod(z[3], f0.s0, pq);\n" \
"	_forward8(pq, z, f0.s1, f0.s23, w4);\n" \
"}\n" \
"\n" \
"INLINE void _square2x4(const uint2_32 pq, uint_32 z[8], const uint2_32 w2)\n" \
"{\n" \
"	SQR2(z[0], z[1], w2.s0); SQR2N(z[2], z[3], w2.s0); SQR2(z[4], z[5], w2.s1); SQR2N(z[6], z[7], w2.s1);\n" \
"}\n" \
"\n" \
"INLINE void _square4x2r(const uint2_32 pq, uint_32 z[8], const uint2_32 w2, const uint2_32 wi2r)\n" \
"{\n" \
"	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);\n" \
"	_square2x4(pq, z, w2);\n" \
"	BCK2(z[0], z[2], wi2r.s1); BCK2(z[1], z[3], wi2r.s1); BCK2(z[4], z[6], wi2r.s0); BCK2(z[5], z[7], wi2r.s0);\n" \
"}\n" \
"\n" \
"INLINE void _square8r(const uint2_32 pq, uint_32 z[8], const uint_32 w1, const uint_32 wi1, const uint2_32 w2, const uint2_32 wi2r)\n" \
"{\n" \
"	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);\n" \
"	_square4x2r(pq, z, w2, wi2r);\n" \
"	BCK2(z[0], z[4], wi1); BCK2(z[2], z[6], wi1); BCK2(z[1], z[5], wi1); BCK2(z[3], z[7], wi1);\n" \
"}\n" \
"\n" \
"INLINE void _fwd4x2(const uint2_32 pq, uint_32 z[8], const uint2_32 w2)\n" \
"{\n" \
"	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);\n" \
"}\n" \
"\n" \
"INLINE void _fwd8(const uint2_32 pq, uint_32 z[8], const uint_32 w1, const uint2_32 w2)\n" \
"{\n" \
"	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);\n" \
"	_fwd4x2(pq, z, w2);\n" \
"}\n" \
"\n" \
"INLINE void _mul2x4(const uint2_32 pq, uint_32 z[8], const uint_32 zp[8], const uint2_32 w2)\n" \
"{\n" \
"	MUL2(z[0], z[1], zp[0], zp[1], w2.s0); MUL2N(z[2], z[3], zp[2], zp[3], w2.s0);\n" \
"	MUL2(z[4], z[5], zp[4], zp[5], w2.s1); MUL2N(z[6], z[7], zp[6], zp[7], w2.s1);\n" \
"}\n" \
"\n" \
"INLINE void _mul4x2r(const uint2_32 pq, uint_32 z[8], const uint_32 zp[8],\n" \
"	const uint2_32 w2, const uint2_32 wi2r, const bool bmask)\n" \
"{\n" \
"	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);\n" \
"	if (bmask) _mul2x4(pq, z, zp, w2);\n" \
"	BCK2(z[0], z[2], wi2r.s1); BCK2(z[1], z[3], wi2r.s1); BCK2(z[4], z[6], wi2r.s0); BCK2(z[5], z[7], wi2r.s0);\n" \
"}\n" \
"\n" \
"INLINE void _mul8r(const uint2_32 pq, uint_32 z[8], const uint_32 zp[8],\n" \
"	const uint_32 w1, const uint_32 wi1, const uint2_32 w2, const uint2_32 wi2r, const bool bmask)\n" \
"{\n" \
"	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);\n" \
"	_mul4x2r(pq, z, zp, w2, wi2r, bmask);\n" \
"	BCK2(z[0], z[4], wi1); BCK2(z[2], z[6], wi1); BCK2(z[1], z[5], wi1); BCK2(z[3], z[7], wi1);\n" \
"}\n" \
"\n" \
"// ---\n" \
"\n" \
"INLINE void forward8io(const uint2_32 pq, const sz_t vm, __global uint_32 * restrict const z,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	const uint_32 w1 = w[sj];\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"	const uint4_32 w4 = _loadg4(w, sj);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, vm);\n" \
"	_forward8(pq, zl, w1, w2, w4);\n" \
"	_storeg(8, z, vm, zl);\n" \
"}\n" \
"\n" \
"INLINE void backward8io(const uint2_32 pq, const sz_t vm, __global uint_32 * restrict const z,\n" \
"	__global const uint_32 * restrict const w, const sz_t sji)\n" \
"{\n" \
"	const uint_32 wi1 = w[sji];\n" \
"	const uint2_32 wi2r = _loadg2(w, sji);\n" \
"	const uint4_32 wi4r = _loadg4(w, sji);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, vm);\n" \
"	_backward8r(pq, zl, wi1, wi2r, wi4r);\n" \
"	_storeg(8, z, vm, zl);\n" \
"}\n" \
"\n" \
"INLINE void forward8_0io(const uint2_32 pq, const uint4_32 f0, const sz_t vn_8,\n" \
"	__global uint_32 * restrict const z, __global const uint_32 * restrict const w)\n" \
"{\n" \
"	const uint4_32 w4 = _loadg4(w, 1);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, vn_8);\n" \
"	_forward8_0(pq, f0, zl, w4);\n" \
"	_storeg(8, z, vn_8, zl);\n" \
"}\n" \
"\n" \
"INLINE void square2x4io(const uint2_32 pq, __global uint_32 * restrict const z,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	_square2x4(pq, zl, w2);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"INLINE void square4x2io(const uint2_32 pq, __global uint_32 * restrict const z,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"	const uint2_32 wi2r = _loadg2(w, sji);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	_square4x2r(pq, zl, w2, wi2r);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"INLINE void square8io(const uint2_32 pq, __global uint_32 * restrict const z,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	const uint_32 w1 = w[sj], wi1 = w[sji];\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"	const uint2_32 wi2r = _loadg2(w, sji);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	_square8r(pq, zl, w1, wi1, w2, wi2r);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x2io(const uint2_32 pq, __global uint_32 * restrict const z,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	_fwd4x2(pq, zl, w2);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd8io(const uint2_32 pq, __global uint_32 * restrict const z,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	const uint_32 w1 = w[sj];\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	_fwd8(pq, zl, w1, w2);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul2x4io(const uint2_32 pq, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	uint_32 zpl[8]; _loadg(8, zpl, zp, VSIZE);\n" \
"	_mul2x4(pq, zl, zpl, w2);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul4x2io(const uint2_32 pq, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)\n" \
"{\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"	const uint2_32 wi2r = _loadg2(w, sji);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	uint_32 zpl[8]; _loadg(8, zpl, zp, VSIZE);\n" \
"	_mul4x2r(pq, zl, zpl, w2, wi2r, bmask);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul8io(const uint2_32 pq, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,\n" \
"	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)\n" \
"{\n" \
"	const uint_32 w1 = w[sj], wi1 = w[sji];\n" \
"	const uint2_32 w2 = _loadg2(w, sj);\n" \
"	const uint2_32 wi2r = _loadg2(w, sji);\n" \
"\n" \
"	uint_32 zl[8]; _loadg(8, zl, z, VSIZE);\n" \
"	uint_32 zpl[8]; _loadg(8, zpl, zp, VSIZE);\n" \
"	_mul8r(pq, zl, zpl, w1, wi1, w2, wi2r, bmask);\n" \
"	_storeg(8, z, VSIZE, zl);\n" \
"}\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define DECLARE_VAR_REG() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (VN_SZ / 8), id = gid % (VN_SZ / 8); \\\n" \
"	const uint2_32 pq = g_pq[lid]; \\\n" \
"	__global uint_32 * restrict const z = &zg[lid * VN_SZ]; \\\n" \
"	__global const uint_32 * restrict const w = &wg[lid * W_SZ];\n" \
"\n" \
"#define DECLARE_VARP_REG() \\\n" \
"	__global const uint_32 * restrict const zp = &zpg[lid * VN_SZ];\n" \
"\n" \
"// --- transform without local mem ---\n" \
"\n" \
"__kernel\n" \
"void forward8(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t vm = VSIZE << lm, j = (id / VSIZE) >> lm, k = 7 * (id & ~(vm - 1)) + id;\n" \
"	forward8io(pq, vm, &z[k], w, s + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward8(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg, const int_32 lm, const uint_32 s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t vm = VSIZE << lm, j = (id / VSIZE) >> lm, k = 7 * (id & ~(vm - 1)) + id;\n" \
"	const sz_t ji = s - j - 1;\n" \
"	backward8io(pq, vm, &z[k], w, s + ji);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward8_0(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t vn_8 = VSIZE * N_SZ / 8, k = id;\n" \
"	forward8_0io(pq, g_f0[lid], vn_8, &z[k], w);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square2x4(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	square2x4io(pq, &z[k], w, n_8 + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4x2(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const sz_t ji = n_8 - j - 1;\n" \
"	square4x2io(pq, &z[k], w, n_8 + j, n_8 + ji);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square8(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const sz_t ji = n_8 - j - 1;\n" \
"	square8io(pq, &z[k], w, n_8 + j, n_8 + ji);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd4x2(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	fwd4x2io(pq, &z[k], w, n_8 + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd8(__global uint_32 * restrict const zg, __global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	fwd8io(pq, &z[k], w, n_8 + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2x4(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,\n" \
"	__global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	mul2x4io(pq, &z[k], &zp[k], w, n_8 + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4x2(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,\n" \
"	__global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const sz_t ji = n_8 - j - 1;\n" \
"	mul4x2io(pq, &z[k], &zp[k], w, n_8 + j, n_8 + ji, true);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul8(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,\n" \
"	__global const uint_32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const sz_t ji = n_8 - j - 1;\n" \
"	mul8io(pq, &z[k], &zp[k], w, n_8 + j, n_8 + ji, true);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2x4_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,\n" \
"	__global const uint_32 * restrict const wg, const uint_32 mask)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	if ((mask & (1u << (id % VSIZE))) != 0)\n" \
"	{\n" \
"		DECLARE_VARP_REG();\n" \
"		const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"		mul2x4io(pq, &z[k], &zp[k], w, n_8 + j);\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4x2_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,\n" \
"	__global const uint_32 * restrict const wg, const uint_32 mask)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const sz_t ji = n_8 - j - 1;\n" \
"	mul4x2io(pq, &z[k], &zp[k], w, n_8 + j, n_8 + ji, (mask & (1u << (id % VSIZE))) != 0);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul8_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,\n" \
"	__global const uint_32 * restrict const wg, const uint_32 mask)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t n_8 = N_SZ / 8, j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const sz_t ji = n_8 - j - 1;\n" \
"	mul8io(pq, &z[k], &zp[k], w, n_8 + j, n_8 + ji, (mask & (1u << (id % VSIZE))) != 0);\n" \
"}\n" \
"\n" \
"// --- carry ---\n" \
"\n" \
"INLINE uint_32 barrett(const uint_64 a, const uint_32 b, const uint_32 b_inv, const int_32 b_s, uint_32 * a_p)\n" \
"{\n" \
"	// Using notations of Modular SIMD arithmetic in Mathemagix, Joris van der Hoeven, Grégoire Lecerf, Guillaume Quintin, 2014, HAL.\n" \
"	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.\n" \
"	// b < 2^31, alpha = 2^29 => a < 2^29 b\n" \
"	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31\n" \
"	// b_inv = [2^{s + 32} / b]\n" \
"	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31\n" \
"	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].\n" \
"	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])\n" \
"	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,\n" \
"	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.\n" \
"\n" \
"	const uint_32 d = mul_hi((uint_32)(a >> b_s), b_inv), r = (uint_32)(a) - d * b;\n" \
"	const bool o = (r >= b);\n" \
"	*a_p = d + (o ? 1 : 0);\n" \
"	return r - (o ? b : 0);\n" \
"}\n" \
"\n" \
"INLINE int_32 reduce64(int_64 * f, const uint_32 b, const uint_32 b_inv, const int_32 b_s)\n" \
"{\n" \
"	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32\n" \
"	// 2- t < 2^23 b^2 => t_h < b^2 / 2^6. If 2 <= b < 32 then t_h < 32^2 / 2^6 = 16 < 2^29 b\n" \
"	const uint_64 t = abs(*f);\n" \
"	const uint_64 t_h = t >> 29;\n" \
"	const uint_32 t_l = (uint_32)(t) % (1u << 29);\n" \
"\n" \
"	uint_32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint_32 d_l, r_l = barrett(((uint_64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint_64 d = ((uint_64)(d_h) << 29) | d_l;\n" \
"\n" \
"	const bool s = (*f < 0);\n" \
"	*f = s ? -(int_64)(d) : (int_64)(d);\n" \
"	return s ? -(int_32)(r_l) : (int_32)(r_l);\n" \
"}\n" \
"\n" \
"INLINE int_32 reduce96(int_64 * f, const int96 l, const uint_32 b, const uint_32 b_inv, const int_32 b_s)\n" \
"{\n" \
"	const uint96 t = int96_abs(l);\n" \
"	const uint_64 t_h = (t.s1 << (32 - 29)) | (t.s0 >> 29);\n" \
"	const uint_32 t_l = t.s0 % (1u << 29);\n" \
"\n" \
"	uint_32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint_32 d_l, r_l = barrett(((uint_64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint_64 d = ((uint_64)(d_h) << 29) | d_l;\n" \
"\n" \
"	const bool s = int96_is_neg(l);\n" \
"	*f = s ? -(int_64)(d) : (int_64)(d);\n" \
"	return s ? -(int_32)(r_l) : (int_32)(r_l);\n" \
"}\n" \
"\n" \
"INLINE int96 garner3(const uint_32 r1, const uint_32 r2, const uint_32 r3)\n" \
"{\n" \
"	const uint_32 u13 = mulmod(submod(r1, r3, P1), INVP3_P1, PQ1);\n" \
"	const uint_32 u23 = mulmod(submod(r2, r3, P2), INVP3_P2, PQ2);\n" \
"	const uint_32 u123 = mulmod(submod(u13, u23, P1), INVP2_P1, PQ1);\n" \
"	const uint96 n = uint96_add_64(uint96_mul_64_32(P2 * (uint_64)(P3), u123), u23 * (uint_64)(P3) + r3);\n" \
"	const bool b = uint96_is_greater(n, uint96_set(P1P2P3_2L, P1P2P3_2H));\n" \
"	return uint96_i(b ? uint96_sub(n, uint96_set(P1P2P3L, P1P2P3H)) : n);\n" \
"}\n" \
"\n" \
"INLINE void write_rns(__global uint_32 * restrict const z, const int_32 r)\n" \
"{\n" \
"	z[0 * VN_SZ] = set_int(r, P1);\n" \
"	z[1 * VN_SZ] = set_int(r, P2);\n" \
"	z[2 * VN_SZ] = set_int(r, P3);\n" \
"}\n" \
"\n" \
"// INLINE int4_32 carry_1(__global uint4_32 * restrict const zi, __global int_64 * restrict const c,\n" \
"// 	__local int_64 * const cl, const sz_t gid, const sz_t lid,\n" \
"// 	const uint_32 b, const uint_32 b_inv, const int b_s, const uint_32 dup)\n" \
"// {\n" \
"// 	const uint4_32 u1 = mulmod4(zi[0 * VN_SZ / 4], NORM1, PQ1), u2 = mulmod4(zi[1 * VN_SZ / 4], NORM2, PQ2), u3 = mulmod4(zi[2 * VN_SZ / 4], NORM3, PQ3);\n" \
"// 	int4_32 r;\n" \
"\n" \
"// 	int96 l0 = garner3(u1.s0, u2.s0, u3.s0), l1 = garner3(u1.s1, u2.s1, u3.s1);\n" \
"// 	int96 l2 = garner3(u1.s2, u2.s2, u3.s2), l3 = garner3(u1.s3, u2.s3, u3.s3);\n" \
"\n" \
"// 	if (dup) { l0 = int96_add(l0, l0); l1 = int96_add(l1, l1);  l2 = int96_add(l2, l2); l3 = int96_add(l3, l3); }	// TODO\n" \
"\n" \
"// 	int96 f96 = l0; r.s0 = reduce96(&f96, b, b_inv, b_s);\n" \
"// 	f96 = int96_add(f96, l1); r.s1 = reduce96(&f96, b, b_inv, b_s);\n" \
"// 	f96 = int96_add(f96, l2); r.s2 = reduce96(&f96, b, b_inv, b_s);\n" \
"// 	f96 = int96_add(f96, l3); r.s3 = reduce96(&f96, b, b_inv, b_s);\n" \
"// 	int_64 f = int96_get_si(f96);\n" \
"\n" \
"// 	cl[lid] = f;\n" \
"\n" \
"// 	if (lid == CARRY_WG_SZ - 1)\n" \
"// 	{\n" \
"// 		const sz_t i = (gid / CARRY_WG_SZ + 1) % (N_SZ / 4 / CARRY_WG_SZ);\n" \
"// 		c[i] = (i == 0) ? -f : f;\n" \
"// 	}\n" \
"\n" \
"// 	return r;\n" \
"// }\n" \
"\n" \
"// INLINE void carry_2(__global uint4_32 * restrict const zi, __local int_64 * const cl, const sz_t lid,\n" \
"// 	const int4_32 r, const uint_32 b, const uint_32 b_inv, const int b_s)\n" \
"// {\n" \
"// 	int_64 f = (lid == 0) ? 0 : cl[lid - 1];\n" \
"// 	int4_32 ro;\n" \
"// 	f += r.s0; ro.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"// 	f += r.s1; ro.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"// 	f += r.s2; ro.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"// 	f += r.s3; ro.s3 = (sz_t)(f);\n" \
"\n" \
"// 	write_rns(zi, ro);\n" \
"// }\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))\n" \
"void carry1(const __global uint2_32 * restrict const bb_inv, const __global int_32 * restrict const bs,\n" \
"	__global uint_32 * restrict const z, __global int_64 * restrict const c, const uint_32 dup)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), i = id % VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const uint2_32 bb_inv_i = bb_inv[i]; const int_32 bs_i = bs[i];\n" \
"\n" \
"	int_64 f = 0;\n" \
"	for (sz_t j = 0; j < 8; ++j)\n" \
"	{\n" \
"		const uint_32 u1 = mulmod(z[k + j * VSIZE + 0 * VN_SZ], NORM1, PQ1);\n" \
"		const uint_32 u2 = mulmod(z[k + j * VSIZE + 1 * VN_SZ], NORM2, PQ2);\n" \
"		const uint_32 u3 = mulmod(z[k + j * VSIZE + 2 * VN_SZ], NORM3, PQ3);\n" \
"		int96 l = garner3(u1, u2, u3);\n" \
"		if ((dup & (1u << i)) != 0) l = int96_add(l, l);\n" \
"		l = int96_add_64(l, f);\n" \
"		const int_32 r = reduce96(&f, l, bb_inv_i.s0, bb_inv_i.s1, bs_i);\n" \
"		write_rns(&z[k + j * VSIZE], r);\n" \
"	}\n" \
"\n" \
"	const sz_t vid = ((id / VSIZE) + 1) % (N_SZ / 8);\n" \
"	c[vid * VSIZE + i] = (vid == 0) ? -f : f;\n" \
"\n" \
"\n" \
"	// const sz_t gid = (sz_t)get_global_id(0), lid = gid % CARRY_WG_SZ;\n" \
"	// __global uint4_32 * restrict const zi = &z[gid];\n" \
"	// __local int_64 cl[CARRY_WG_SZ];\n" \
"\n" \
"	// const int4_32 r = carry_1(zi, c, cl, gid, lid, b, b_inv, b_s, dup);\n" \
"\n" \
"	// barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	// carry_2(zi, cl, lid, r, b, b_inv, b_s);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void carry2(const __global uint2_32 * restrict const bb_inv, const __global int_32 * restrict const bs,\n" \
"	__global uint_32 * restrict const z, const __global int_64 * restrict const c)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), i = id % VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;\n" \
"	const uint2_32 bb_inv_i = bb_inv[i]; const int_32 bs_i = bs[i];\n" \
"\n" \
"	int_64 f = c[id];\n" \
"	for (size_t j = 0; j < 7; ++j)\n" \
"	{\n" \
"		f += get_int(z[k + j * VSIZE], P1);\n" \
"		const int_32 r = reduce64(&f, bb_inv_i.s0, bb_inv_i.s1, bs_i);\n" \
"		write_rns(&z[k + j * VSIZE], r);\n" \
"		if (f == 0) return;\n" \
"	}\n" \
"\n" \
"	f += get_int(z[k + 7 * VSIZE], P1);\n" \
"	const int_32 r = (int_32)(f);\n" \
"	write_rns(&z[k + 7 * VSIZE], r);\n" \
"\n" \
"// 	const sz_t gid = (sz_t)get_global_id(0);\n" \
"// 	__global uint4_32 * restrict const zi = &z[CARRY_WG_SZ * gid];\n" \
"\n" \
"// 	const uint4_32 u1 = zi[0 * N_SZ / 4];\n" \
"// 	int4_32 r;\n" \
"\n" \
"// 	int_64 f = c[gid] + get_int(u1.s0, P1);\n" \
"// 	r.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"// 	f += get_int(u1.s1, P1);\n" \
"// 	r.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"// 	f += get_int(u1.s2, P1);\n" \
"// 	r.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"// 	f += get_int(u1.s3, P1);\n" \
"// 	r.s3 = (int_32)(f);\n" \
"\n" \
"// 	write_rns(zi, r);\n" \
"}\n" \
"\n" \
"// --- misc ---\n" \
"\n" \
"__kernel\n" \
"void set(__global uint_32 * restrict const z, const uint_32 a)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	z[id] = (id % VN_SZ < VSIZE) ? a : 0;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy(__global uint_32 * restrict const z, const sz_t dst, const sz_t src)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	z[3 * VN_SZ * dst + id] = z[3 * VN_SZ * src + id];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copyp(__global uint_32 * restrict const zp, __global const uint_32 * restrict const z, const sz_t src)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	zp[id] = z[3 * VN_SZ * src + id];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy_mask(__global uint_32 * restrict const z, const sz_t dst, const sz_t src, const uint_32 mask)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	if ((mask & (1u << (id % VSIZE))) != 0) z[3 * VN_SZ * dst + id] = z[3 * VN_SZ * src + id];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void cosmic_ray(__global uint_32 * restrict const z)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	if (id == VN_SZ / 2) z[id] = addmod(z[id], 1, P1);\n" \
"}\n" \
"";
