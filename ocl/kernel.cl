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
#define OCL_VSIZE		4
#define OCL_CARRY_VSIZE	1
#define CARRY_LENGTH	4
#define CARRY_WG_SZ		128u
#define BLK16			32
#define BLK32			16
#define BLK64			8
#define BLK128			4
#define BLK256			2
#define BLK512			1
#define CHUNK64			8
#define CHUNK512		1
// #define QVALID			1
#endif

#define N_VSIZE		(N_SZ * VSIZE)
#define N_VLEN		(N_SZ * VSIZE / OCL_VSIZE)

#define CARRY_VSIZE		(OCL_VSIZE / OCL_CARRY_VSIZE)
#define N_CARRY_VSIZE	(N_VSIZE / OCL_CARRY_VSIZE)

typedef uint	sz_t;
typedef uint	uint_32;
typedef int		int_32;
typedef ulong	uint_64;
typedef long	int_64;
typedef uint2	uint2_32;
typedef int2	int2_32;
typedef long2	int2_64;
typedef uint4	uint4_32;
typedef int4	int4_32;
typedef long4	int4_64;
typedef uint8	uint8_32;

// --- modular arithmetic

#define	PQ1		(uint2_32)(P1, Q1)
#define	PQ2		(uint2_32)(P2, Q2)
#define	PQ3		(uint2_32)(P3, Q3)

__constant uint2_32 g_pq[3] = { PQ1, PQ2, PQ3 };
__constant uint4_32 g_f0[3] = { (uint4_32)(RSQ1, MFIM1, SQRTI1, ISQRTI1), (uint4_32)(RSQ2, MFIM2, SQRTI2, ISQRTI2), (uint4_32)(RSQ3, MFIM3, SQRTI3, ISQRTI3) };
__constant uint4_32 g_f0i[3] = { (uint4_32)(NORM1, IM1, SQRTI1, ISQRTI1), (uint4_32)(NORM2, IM2, SQRTI2, ISQRTI2), (uint4_32)(NORM3, IM3, SQRTI3, ISQRTI3) };

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

INLINE int_32 get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }
INLINE uint_32 set_int(const int_32 i, const uint_32 p) { return (uint_32)(i + ((i < 0) ? p : 0)); }

// --- v2

INLINE uint2_32 addmod2(const uint2_32 lhs, const uint2_32 rhs, const uint_32 p)
{
	return (uint2_32)(addmod(lhs.s0, rhs.s0, p), addmod(lhs.s1, rhs.s1, p));
}

INLINE uint2_32 submod2(const uint2_32 lhs, const uint2_32 rhs, const uint_32 p)
{
	return (uint2_32)(submod(lhs.s0, rhs.s0, p), submod(lhs.s1, rhs.s1, p));
}

INLINE uint2_32 mulmod2(const uint2_32 lhs, const uint2_32 rhs, const uint2_32 pq)
{
	return (uint2_32)(mulmod(lhs.s0, rhs.s0, pq), mulmod(lhs.s1, rhs.s1, pq));
}

INLINE uint2_32 mulmods2(const uint2_32 lhs, const uint_32 rhs, const uint2_32 pq)
{
	return (uint2_32)(mulmod(lhs.s0, rhs, pq), mulmod(lhs.s1, rhs, pq));
}

INLINE uint2_32 set_int2(const int2_32 i, const uint_32 p) { return (uint2_32)(set_int(i.s0, p), set_int(i.s1, p)); }

// --- v4

INLINE uint4_32 addmod4(const uint4_32 lhs, const uint4_32 rhs, const uint_32 p)
{
	return (uint4_32)(addmod2(lhs.s01, rhs.s01, p), addmod2(lhs.s23, rhs.s23, p));
}

INLINE uint4_32 submod4(const uint4_32 lhs, const uint4_32 rhs, const uint_32 p)
{
	return (uint4_32)(submod2(lhs.s01, rhs.s01, p), submod2(lhs.s23, rhs.s23, p));
}

INLINE uint4_32 mulmod4(const uint4_32 lhs, const uint4_32 rhs, const uint2_32 pq)
{
	return (uint4_32)(mulmod2(lhs.s01, rhs.s01, pq), mulmod2(lhs.s23, rhs.s23, pq));
}

INLINE uint4_32 mulmods4(const uint4_32 lhs, const uint_32 rhs, const uint2_32 pq)
{
	return (uint4_32)(mulmods2(lhs.s01, rhs, pq), mulmods2(lhs.s23, rhs, pq));
}

INLINE uint4_32 set_int4(const int4_32 i, const uint_32 p) { return (uint4_32)(set_int2(i.s01, p), set_int2(i.s23, p)); }

// --- uint96/int96 ---

typedef struct { uint_32 s0; uint_64 s1; } uint96;
typedef struct { uint_32 s0; int_64 s1; } int96;

INLINE int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int_64)(x.s1); return r; }

INLINE uint96 uint96_set(const uint_32 s0, const uint_64 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }

INLINE int96 int96_set_si(const int_64 n) { int96 r; r.s0 = (uint_32)(n); r.s1 = n >> 32; return r; }

INLINE bool int96_is_neg(const int96 x) { return (x.s1 < 0); }

INLINE bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }

INLINE uint96 uint96_add_64(const uint96 x, const uint_64 y)
{
	const uint_32 yl = (uint_32)(y); const uint_64 yh = y >> 32;
	uint96 r;
#if defined(PTX_ASM)
	asm volatile ("add.cc.u32 %0, %1, %2;" : "=r" (r.s0) : "r" (x.s0), "r" (yl));
	asm volatile ("addc.u64 %0, %1, %2;" : "=l" (r.s1) : "l" (x.s1), "l" (yh));
#else
	const uint_32 s0 = x.s0 + yl;
	r.s0 = s0; r.s1 = x.s1 + yh + ((s0 < x.s0) ? 1 : 0);
#endif
	return r;
}

INLINE int96 int96_add_64(const int96 x, const int_64 y)
{
	const uint_32 yl = (uint_32)(y); const int_64 yh = y >> 32;
	int96 r;
#if defined(PTX_ASM)
	asm volatile ("add.cc.u32 %0, %1, %2;" : "=r" (r.s0) : "r" (x.s0), "r" (yl));
	asm volatile ("addc.s64 %0, %1, %2;" : "=l" (r.s1) : "l" (x.s1), "l" (yh));
#else
	const uint_32 s0 = x.s0 + yl;
	r.s0 = s0; r.s1 = x.s1 + yh + ((s0 < x.s0) ? 1 : 0);
#endif
	return r;
}

INLINE int96 int96_add(const int96 x, const int96 y)
{
	int96 r;
#if defined(PTX_ASM)
	asm volatile ("add.cc.u32 %0, %1, %2;" : "=r" (r.s0) : "r" (x.s0), "r" (y.s0));
	asm volatile ("addc.s64 %0, %1, %2;" : "=l" (r.s1) : "l" (x.s1), "l" (y.s1));
#else
	const uint_32 s0 = x.s0 + y.s0;
	r.s0 = s0; r.s1 = x.s1 + y.s1 + ((s0 < x.s0) ? 1 : 0);
#endif
	return r;
}

INLINE uint96 uint96_sub(const uint96 x, const uint96 y)
{
	uint96 r;
#if defined(PTX_ASM)
	asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r" (r.s0) : "r" (x.s0), "r" (y.s0));
	asm volatile ("subc.u64 %0, %1, %2;" : "=l" (r.s1) : "l" (x.s1), "l" (y.s1));
#else
	r.s0 = x.s0 - y.s0; r.s1 = (int_64)(x.s1 - y.s1 - ((x.s0 < y.s0) ? 1 : 0));
#endif
	return r;
}

INLINE uint96 int96_abs(const int96 x)
{
	const bool is_neg = int96_is_neg(x);
	const uint96 mask = uint96_set(is_neg ? ~0u : 0u, is_neg ? ~0ul : 0ul);
	const uint96 t = uint96_set(x.s0 ^ mask.s0, (uint_64)(x.s1) ^ mask.s1);
	return uint96_sub(t, mask);
}

INLINE uint96 uint96_mul_64_32(const uint_64 x, const uint_32 y)
{
	const uint_64 l = (uint_32)(x) * (uint_64)(y);
	uint96 r; r.s0 = (uint_32)(l); r.s1 = (x >> 32) * y + (l >> 32);
	return r;
}

// --- internal vector size (1, 2 or 4) ---

#if OCL_VSIZE == 4
#define VTYPE		uint4_32
#define addmodv		addmod4
#define submodv		submod4
#define mulmodv		mulmod4
#define mulmodsv	mulmods4
#elif OCL_VSIZE == 2
#define VTYPE		uint2_32
#define addmodv		addmod2
#define submodv		submod2
#define mulmodv		mulmod2
#define mulmodsv	mulmods2
#else
#define VTYPE		uint_32
#define addmodv		addmod
#define submodv		submod
#define mulmodv		mulmod
#define mulmodsv	mulmod
#endif

// --- I/O ---

INLINE void _loadg(const sz_t n, VTYPE * const zl, __global const VTYPE * restrict const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _storeg(const sz_t n, __global VTYPE * restrict const z, const sz_t s, const VTYPE * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

INLINE void _loadg_1(const sz_t n, uint_32 * const zl, __global const uint_32 * restrict const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _storeg_1(const sz_t n, __global uint_32 * restrict const z, const sz_t s, const uint_32 * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

INLINE uint2_32 _load2g(__global const uint_32 * restrict const w, const sz_t j) { return ((__global const uint2_32 *)w)[j]; }
INLINE uint4_32 _load4g(__global const uint_32 * restrict const w, const sz_t j) { return ((__global const uint4_32 *)w)[j]; }

INLINE void _loadl(const sz_t n, VTYPE * const zl, __local const VTYPE * restrict const Z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }
INLINE void _storel(const sz_t n, __local VTYPE * restrict const Z, const sz_t s, const VTYPE * const zl) { for (sz_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }

// --- transform/macro ---

#define FWD2(z0, z1, w) \
{ \
	const VTYPE t = mulmodsv(z1, w, pq); \
	z1 = submodv(z0, t, pq.s0); z0 = addmodv(z0, t, pq.s0); \
}

#define BCK2(z0, z1, wi) \
{ \
	const VTYPE t = submodv(z1, z0, pq.s0); z0 = addmodv(z0, z1, pq.s0); \
	z1 = mulmodsv(t, wi, pq); \
}

#define SQR2(z0, z1, w) \
{ \
	const VTYPE t = mulmodsv(mulmodv(z1, z1, pq), w, pq); \
	z1 = mulmodv(addmodv(z0, z0, pq.s0), z1, pq); \
	z0 = addmodv(mulmodv(z0, z0, pq), t, pq.s0); \
}

#define SQR2N(z0, z1, w) \
{ \
	const VTYPE t = mulmodsv(mulmodv(z1, z1, pq), w, pq); \
	z1 = mulmodv(addmodv(z0, z0, pq.s0), z1, pq); \
	z0 = submodv(mulmodv(z0, z0, pq), t, pq.s0); \
}

#define MUL2(z0, z1, zp0, zp1, w) \
{ \
	const VTYPE t = mulmodsv(mulmodv(z1, zp1, pq), w, pq); \
	z1 = addmodv(mulmodv(z0, zp1, pq), mulmodv(zp0, z1, pq), pq.s0); \
	z0 = addmodv(mulmodv(z0, zp0, pq), t, pq.s0); \
}

#define MUL2N(z0, z1, zp0, zp1, w) \
{ \
	const VTYPE t = mulmodsv(mulmodv(z1, zp1, pq), w, pq); \
	z1 = addmodv(mulmodv(z0, zp1, pq), mulmodv(zp0, z1, pq), pq.s0); \
	z0 = submodv(mulmodv(z0, zp0, pq), t, pq.s0); \
}

#define FWD2_1(z0, z1, w) \
{ \
	const uint_32 t = mulmod(z1, w, pq); \
	z1 = submod(z0, t, pq.s0); z0 = addmod(z0, t, pq.s0); \
}

#define BCK2_1(z0, z1, wi) \
{ \
	const uint_32 t = submod(z1, z0, pq.s0); z0 = addmod(z0, z1, pq.s0); \
	z1 = mulmod(t, wi, pq); \
}

#define MUL2_1(z0, z1, zp0, zp1, w) \
{ \
	const uint_32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \
	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \
	z0 = addmod(mulmod(z0, zp0, pq), t, pq.s0); \
}

#define MUL2N_1(z0, z1, zp0, zp1, w) \
{ \
	const uint_32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \
	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \
	z0 = submod(mulmod(z0, zp0, pq), t, pq.s0); \
}

// --- transform/inline ---

INLINE void _forward8(const uint2_32 pq, VTYPE z[8], const uint_32 w1, const uint2_32 w2, const uint4_32 w4)
{
	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);
	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);
	FWD2(z[0], z[1], w4.s0); FWD2(z[2], z[3], w4.s1); FWD2(z[4], z[5], w4.s2); FWD2(z[6], z[7], w4.s3);
}

INLINE void _backward8r(const uint2_32 pq, VTYPE z[8], const uint_32 wi1, const uint2_32 wi2r, const uint4_32 wi4r)
{
	BCK2(z[0], z[1], wi4r.s3); BCK2(z[2], z[3], wi4r.s2); BCK2(z[4], z[5], wi4r.s1); BCK2(z[6], z[7], wi4r.s0);
	BCK2(z[0], z[2], wi2r.s1); BCK2(z[1], z[3], wi2r.s1); BCK2(z[4], z[6], wi2r.s0); BCK2(z[5], z[7], wi2r.s0);
	BCK2(z[0], z[4], wi1); BCK2(z[2], z[6], wi1); BCK2(z[1], z[5], wi1); BCK2(z[3], z[7], wi1);
}

INLINE void _forward8_0(const uint2_32 pq, const uint4_32 f0, VTYPE z[8], const uint4_32 w4)
{
	for (sz_t i = 0; i < 4; ++i) z[i] = mulmodsv(z[i], f0.s0, pq);
	_forward8(pq, z, f0.s1, f0.s23, w4);
}

INLINE void _backward8r_0(const uint2_32 pq, const uint4_32 f0i, VTYPE z[8], const uint4_32 wi4r)
{
	_backward8r(pq, z, f0i.s1, f0i.s23, wi4r);
	for (sz_t i = 0; i < 8; ++i) z[i] = mulmodsv(z[i], f0i.s0, pq);
}

INLINE void _square2x4(const uint2_32 pq, VTYPE z[8], const uint2_32 w2)
{
	SQR2(z[0], z[1], w2.s0); SQR2N(z[2], z[3], w2.s0); SQR2(z[4], z[5], w2.s1); SQR2N(z[6], z[7], w2.s1);
}

INLINE void _square4x2r(const uint2_32 pq, VTYPE z[8], const uint2_32 w2, const uint2_32 wi2r)
{
	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);
	_square2x4(pq, z, w2);
	BCK2(z[0], z[2], wi2r.s1); BCK2(z[1], z[3], wi2r.s1); BCK2(z[4], z[6], wi2r.s0); BCK2(z[5], z[7], wi2r.s0);
}

INLINE void _square8r(const uint2_32 pq, VTYPE z[8], const uint_32 w1, const uint_32 wi1, const uint2_32 w2, const uint2_32 wi2r)
{
	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);
	_square4x2r(pq, z, w2, wi2r);
	BCK2(z[0], z[4], wi1); BCK2(z[2], z[6], wi1); BCK2(z[1], z[5], wi1); BCK2(z[3], z[7], wi1);
}

INLINE void _fwd4x2(const uint2_32 pq, VTYPE z[8], const uint2_32 w2)
{
	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);
}

INLINE void _fwd8(const uint2_32 pq, VTYPE z[8], const uint_32 w1, const uint2_32 w2)
{
	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);
	_fwd4x2(pq, z, w2);
}

INLINE void _mul2x4(const uint2_32 pq, VTYPE z[8], const VTYPE zp[8], const uint2_32 w2)
{
	MUL2(z[0], z[1], zp[0], zp[1], w2.s0); MUL2N(z[2], z[3], zp[2], zp[3], w2.s0);
	MUL2(z[4], z[5], zp[4], zp[5], w2.s1); MUL2N(z[6], z[7], zp[6], zp[7], w2.s1);
}

INLINE void _mul4x2r(const uint2_32 pq, VTYPE z[8], const VTYPE zp[8],
	const uint2_32 w2, const uint2_32 wi2r)
{
	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);
	_mul2x4(pq, z, zp, w2);
	BCK2(z[0], z[2], wi2r.s1); BCK2(z[1], z[3], wi2r.s1); BCK2(z[4], z[6], wi2r.s0); BCK2(z[5], z[7], wi2r.s0);
}

INLINE void _mul8r(const uint2_32 pq, VTYPE z[8], const VTYPE zp[8],
	const uint_32 w1, const uint_32 wi1, const uint2_32 w2, const uint2_32 wi2r)
{
	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);
	_mul4x2r(pq, z, zp, w2, wi2r);
	BCK2(z[0], z[4], wi1); BCK2(z[2], z[6], wi1); BCK2(z[1], z[5], wi1); BCK2(z[3], z[7], wi1);
}

INLINE void _mul2x4_1(const uint2_32 pq, uint_32 z[8], const uint_32 zp[8], const uint2_32 w2)
{
	MUL2_1(z[0], z[1], zp[0], zp[1], w2.s0); MUL2N_1(z[2], z[3], zp[2], zp[3], w2.s0);
	MUL2_1(z[4], z[5], zp[4], zp[5], w2.s1); MUL2N_1(z[6], z[7], zp[6], zp[7], w2.s1);
}

INLINE void _mul4x2r_1(const uint2_32 pq, uint_32 z[8], const uint_32 zp[8],
	const uint2_32 w2, const uint2_32 wi2r, const bool bmask)
{
	FWD2_1(z[0], z[2], w2.s0); FWD2_1(z[1], z[3], w2.s0); FWD2_1(z[4], z[6], w2.s1); FWD2_1(z[5], z[7], w2.s1);
	if (bmask) _mul2x4_1(pq, z, zp, w2);
	BCK2_1(z[0], z[2], wi2r.s1); BCK2_1(z[1], z[3], wi2r.s1); BCK2_1(z[4], z[6], wi2r.s0); BCK2_1(z[5], z[7], wi2r.s0);
}

INLINE void _mul8r_1(const uint2_32 pq, uint_32 z[8], const uint_32 zp[8],
	const uint_32 w1, const uint_32 wi1, const uint2_32 w2, const uint2_32 wi2r, const bool bmask)
{
	FWD2_1(z[0], z[4], w1); FWD2_1(z[2], z[6], w1); FWD2_1(z[1], z[5], w1); FWD2_1(z[3], z[7], w1);
	_mul4x2r_1(pq, z, zp, w2, wi2r, bmask);
	BCK2_1(z[0], z[4], wi1); BCK2_1(z[2], z[6], wi1); BCK2_1(z[1], z[5], wi1); BCK2_1(z[3], z[7], wi1);
}

// ---

#define LOADG(s)	VTYPE zl[8]; _loadg(8, zl, z, s);
#define LOADPG(s)	VTYPE zpl[8]; _loadg(8, zpl, zp, s);
#define STOREG(s)	_storeg(8, z, s, zl);
#define LOADL(s) \
	barrier(CLK_LOCAL_MEM_FENCE); \
	VTYPE zl[8]; _loadl(8, zl, Z, s);
#define STOREL(s)	_storel(8, Z, s, zl);
#define LOADG1(s) \
	uint_32 zl[8];	_loadg_1(8, zl, z, s); \
	uint_32 zpl[8];	_loadg_1(8, zpl, zp, s);
#define STOREG1(s)	_storeg_1(8, z, s, zl);

#define DECLARE_W124() \
	const uint_32 w1 = w[sj]; \
	const uint2_32 w2 = _load2g(w, sj); \
	const uint4_32 w4 = _load4g(w, sj);

#define DECLARE_WI124() \
	const uint_32 wi1 = w[sji]; \
	const uint2_32 wi2r = _load2g(w, sji); \
	const uint4_32 wi4r = _load4g(w, sji);

#define DECLARE_W2I2()	const uint2_32 w2 = _load2g(w, sj), wi2r = _load2g(w, sji);

#define DECLARE_W12I12() \
	const uint_32 w1 = w[sj], wi1 = w[sji]; \
	DECLARE_W2I2();

INLINE void forward8g(const uint2_32 pq, const sz_t m, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADG(m);
	_forward8(pq, zl, w1, w2, w4);
	STOREG(m);
}

INLINE void forward8i(const uint2_32 pq, const sz_t ml, __local VTYPE * restrict const Z, const sz_t mg,
	__global VTYPE * restrict const z, __global const uint_32 * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADG(mg);
	_forward8(pq, zl, w1, w2, w4);
	STOREL(ml);
}

INLINE void forward8l(const uint2_32 pq, const sz_t m, __local VTYPE * restrict const Z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADL(m);
	_forward8(pq, zl, w1, w2, w4);
	STOREL(m);
}

INLINE void forward8o(const uint2_32 pq, const sz_t mg, __global VTYPE * restrict const z, const sz_t ml,
	__local VTYPE * restrict const Z, __global const uint_32 * restrict const w, const sz_t sj)
{
	DECLARE_W124();
	LOADL(ml);
	_forward8(pq, zl, w1, w2, w4);
	STOREG(mg);
}

INLINE void backward8g(const uint2_32 pq, const sz_t m, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sji)
{
	DECLARE_WI124();
	LOADG(m);
	_backward8r(pq, zl, wi1, wi2r, wi4r);
	STOREG(m);
}

INLINE void backward8i(const uint2_32 pq, const sz_t ml, __local VTYPE * restrict const Z, const sz_t mg,
	__global VTYPE * restrict const z, __global const uint_32 * restrict const w, const sz_t sji)
{
	DECLARE_WI124();
	LOADG(mg);
	_backward8r(pq, zl, wi1, wi2r, wi4r);
	STOREL(ml);
}

INLINE void backward8l(const uint2_32 pq, const sz_t m, __local VTYPE * restrict const Z,
	__global const uint_32 * restrict const w, const sz_t sji)
{
	DECLARE_WI124();
	LOADL(m);
	_backward8r(pq, zl, wi1, wi2r, wi4r);
	STOREL(m);
}

INLINE void backward8o(const uint2_32 pq, const sz_t mg, __global VTYPE * restrict const z, const sz_t ml,
	__local VTYPE * restrict const Z, __global const uint_32 * restrict const w, const sz_t sji)
{
	DECLARE_WI124();
	LOADL(ml);
	_backward8r(pq, zl, wi1, wi2r, wi4r);
	STOREG(mg);
}

INLINE void forward8_0g(const uint2_32 pq, const uint4_32 f0, const sz_t m,
	__global VTYPE * restrict const z, __global const uint_32 * restrict const w)
{
	const uint4_32 w4 = _load4g(w, 1);
	LOADG(m);
	_forward8_0(pq, f0, zl, w4);
	STOREG(m);
}

INLINE void forward8_0i(const uint2_32 pq, const uint4_32 f0, const sz_t ml, __local VTYPE * restrict const Z,
	const sz_t mg, __global VTYPE * restrict const z, __global const uint_32 * restrict const w)
{
	const uint4_32 w4 = _load4g(w, 1);
	LOADG(mg);
	_forward8_0(pq, f0, zl, w4);
	STOREL(ml);
}

INLINE void backward8_0g(const uint2_32 pq, const uint4_32 f0i, const sz_t m,
	__global VTYPE * restrict const z, __global const uint_32 * restrict const w)
{
	const uint4_32 wi4r = _load4g(w, 1);
	LOADG(m);
	_backward8r_0(pq, f0i, zl, wi4r);
	STOREG(m);
}

INLINE void backward8_0o(const uint2_32 pq, const uint4_32 f0i, const sz_t mg, __global VTYPE * restrict const z,
	const sz_t ml, __local VTYPE * restrict const Z, __global const uint_32 * restrict const w)
{
	const uint4_32 wi4r = _load4g(w, 1);
	LOADL(ml);
	_backward8r_0(pq, f0i, zl, wi4r);
	STOREG(mg);
}

INLINE void square2x4g(const uint2_32 pq, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);
	LOADG(1);
	_square2x4(pq, zl, w2);
	STOREG(1);
}

INLINE void square2x4l(const uint2_32 pq, __local VTYPE * restrict const Z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);
	LOADL(1);
	_square2x4(pq, zl, w2);
	STOREL(1);
}

INLINE void square4x2g(const uint2_32 pq, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W2I2();
	LOADG(1);
	_square4x2r(pq, zl, w2, wi2r);
	STOREG(1);
}

INLINE void square4x2l(const uint2_32 pq, __local VTYPE * restrict const Z,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W2I2();
	LOADL(1);
	_square4x2r(pq, zl, w2, wi2r);
	STOREL(1);
}

INLINE void square8g(const uint2_32 pq, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W12I12();
	LOADG(1);
	_square8r(pq, zl, w1, wi1, w2, wi2r);
	STOREG(1);
}

INLINE void square8l(const uint2_32 pq, __local VTYPE * restrict const Z,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W12I12();
	LOADL(1);
	_square8r(pq, zl, w1, wi1, w2, wi2r);
	STOREL(1);
}

INLINE void fwd4x2g(const uint2_32 pq, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);
	LOADG(1);
	_fwd4x2(pq, zl, w2);
	STOREG(1);
}

INLINE void fwd4x2o(const uint2_32 pq, __global VTYPE * restrict const z, __local VTYPE * restrict const Z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);
	LOADL(1);
	_fwd4x2(pq, zl, w2);
	STOREG(1);
}

INLINE void fwd8g(const uint2_32 pq, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint_32 w1 = w[sj];
	const uint2_32 w2 = _load2g(w, sj);
	LOADG(1);
	_fwd8(pq, zl, w1, w2);
	STOREG(1);
}

INLINE void fwd8o(const uint2_32 pq, __global VTYPE * restrict const z, __local VTYPE * restrict const Z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint_32 w1 = w[sj];
	const uint2_32 w2 = _load2g(w, sj);
	LOADL(1);
	_fwd8(pq, zl, w1, w2);
	STOREG(1);
}

INLINE void mul2x4g(const uint2_32 pq, __global VTYPE * restrict const z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);
	LOADG(1);
	LOADPG(1);
	_mul2x4(pq, zl, zpl, w2);
	STOREG(1);
}

INLINE void mul2x4l(const uint2_32 pq, __local VTYPE * restrict const Z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);
	LOADL(1);
	LOADPG(1);
	_mul2x4(pq, zl, zpl, w2);
	STOREL(1);
}

INLINE void mul4x2g(const uint2_32 pq, __global VTYPE * restrict const z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W2I2();
	LOADG(1);
	LOADPG(1);
	_mul4x2r(pq, zl, zpl, w2, wi2r);
	STOREG(1);
}

INLINE void mul4x2l(const uint2_32 pq, __local VTYPE * restrict const Z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W2I2();
	LOADL(1);
	LOADPG(1);
	_mul4x2r(pq, zl, zpl, w2, wi2r);
	STOREL(1);
}

INLINE void mul8g(const uint2_32 pq, __global VTYPE * restrict const z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W12I12();
	LOADG(1);
	LOADPG(1);
	_mul8r(pq, zl, zpl, w1, wi1, w2, wi2r);
	STOREG(1);
}

INLINE void mul8l(const uint2_32 pq, __local VTYPE * restrict const Z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	DECLARE_W12I12();
	LOADL(1);
	LOADPG(1);
	_mul8r(pq, zl, zpl, w1, wi1, w2, wi2r);
	STOREL(1);
}

INLINE void mul2x4g_1(const uint2_32 pq, const sz_t m, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);
	LOADG1(m);
	_mul2x4_1(pq, zl, zpl, w2);
	STOREG1(m);
}

INLINE void mul4x2g_1(const uint2_32 pq, const sz_t m, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)
{
	DECLARE_W2I2();
	LOADG1(m);
	_mul4x2r_1(pq, zl, zpl, w2, wi2r, bmask);
	STOREG1(m);
}

INLINE void mul8g_1(const uint2_32 pq, const sz_t m, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)
{
	DECLARE_W12I12();
	LOADG1(m);
	_mul8r_1(pq, zl, zpl, w1, wi1, w2, wi2r, bmask);
	STOREG1(m);
}

// --- transform/macro ---

#define DECLARE_VAR_REG_N(LEN, TYPE) \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (LEN / 8), vid = gid % (LEN / 8), id = gid % (N_SZ / 8); \
	const uint2_32 pq = g_pq[lid]; \
	__global TYPE * restrict const z = &zg[lid * LEN]; \
	__global const uint_32 * restrict const w = &wg[lid * W_SZ];

#define DECLARE_VAR_REG() DECLARE_VAR_REG_N(N_VLEN, VTYPE)
#define DECLARE_VARP_REG() __global const VTYPE * restrict const zp = &zpg[lid * N_VLEN];

#define DECLARE_VAR_REG_1() DECLARE_VAR_REG_N(N_VSIZE, uint_32)
#define DECLARE_VARP_REG_1() __global const uint_32 * restrict const zp = &zpg[N_VSIZE * lid];

// --- transform without local mem ---

__kernel
void forward8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg, const int_32 lm, const uint_32 s)
{
	DECLARE_VAR_REG();
	const sz_t m = 1u << lm, j = id >> lm, k = 7 * (vid & ~(m - 1)) + vid;
	forward8g(pq, m, &z[k], w, s + j);
}

__kernel
void backward8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg, const int_32 lm, const uint_32 s)
{
	DECLARE_VAR_REG();
	const sz_t m = 1u << lm, j = id >> lm, ji = s - j - 1, k = 7 * (vid & ~(m - 1)) + vid;
	backward8g(pq, m, &z[k], w, s + ji);
}

/*__kernel
void forward8_0(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t m = N_SZ / 8, k = 7 * (vid & ~(m - 1)) + vid;
	forward8_0g(pq, g_f0[lid], m, &z[k], w);
}

__kernel
void backward8_0(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t m = N_SZ / 8, k = 7 * (vid & ~(m - 1)) + vid;
	backward8_0g(pq, g_f0i[lid], m, &z[k], w);
}*/

#define DECLARE_VAR_FB(N, CHUNK_N) \
	DECLARE_VAR_REG(); \
	const sz_t mid = vid & ~(N_SZ / 8 - 1); \
	__global VTYPE * restrict const zv = &z[8 * mid]; \
	const size_t chunk_id = id % CHUNK_N, local_id = (id / CHUNK_N) % (N / 8), block_id = id & ~(N / 8 * CHUNK_N - 1); \
	__global VTYPE * restrict const zt = &zv[block_id / (N / 8) + chunk_id]; \
	__local VTYPE * const Zt = &Z[N * chunk_id];

#if (LN_SZ <= 15)

#define DECLARE_VAR_FB64() \
	const sz_t kl0 = local_id, k0 = kl0 * (N_SZ / 64), ml0 = 8, m0 = ml0 * (N_SZ / 64); \
	const sz_t kl8 = local_id * 8, k8 = kl8 * (N_SZ / 64), ml8 = 1, m8 = ml8 * (N_SZ / 64);

__kernel __attribute__((reqd_work_group_size(64 / 8 * CHUNK64, 1, 1)))
void forward64_0(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	__local VTYPE Z[64 * CHUNK64];
	DECLARE_VAR_FB(64, CHUNK64);
	DECLARE_VAR_FB64();

	forward8_0i(pq, g_f0[lid], ml0, &Zt[kl0], m0, &zt[k0], w);

	const sz_t j8 = local_id;
	forward8o(pq, m8, &zt[k8], ml8, &Zt[kl8], w, 8 + j8);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * CHUNK64, 1, 1)))
void backward64_0(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	__local VTYPE Z[64 * CHUNK64];
	DECLARE_VAR_FB(64, CHUNK64);
	DECLARE_VAR_FB64();

	const sz_t j8 = local_id, j8i = 8 - j8 - 1;
	backward8i(pq, ml8, &Zt[kl8], m8, &zt[k8], w, 8 + j8i);

	backward8_0o(pq, g_f0i[lid], m0, &zt[k0], ml0, &Zt[kl0], w);
}

#endif

#define DECLARE_VAR_FB512() \
	const sz_t kl0 = local_id, k0 = kl0 * (N_SZ / 512), ml0 = 64, m0 = ml0 * (N_SZ / 512); \
	const sz_t kl8 = (local_id % 8) + (local_id / 8) * 64, ml8 = 8; \
	const sz_t kl64 = local_id * 8, k64 = kl64 * (N_SZ / 512), ml64 = 1, m64 = ml64 * (N_SZ / 512);

__kernel __attribute__((reqd_work_group_size(512 / 8 * CHUNK512, 1, 1)))
void forward512_0(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg, __local VTYPE * const Z)
{
	// __local VTYPE Z[512 * CHUNK512];
	DECLARE_VAR_FB(512, CHUNK512);
	DECLARE_VAR_FB512();

	forward8_0i(pq, g_f0[lid], ml0, &Zt[kl0], m0, &zt[k0], w);

	const sz_t j8 = (local_id / 8) % 8;
	forward8l(pq, ml8, &Zt[kl8], w, 8 + j8);

	const sz_t j64 = local_id;
	forward8o(pq, m64, &zt[k64], ml64, &Zt[kl64], w, 64 + j64);
}

__kernel __attribute__((reqd_work_group_size(512 / 8 * CHUNK512, 1, 1)))
void backward512_0(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	__local VTYPE Z[512 * CHUNK512];
	DECLARE_VAR_FB(512, CHUNK512);
	DECLARE_VAR_FB512();

	const sz_t j64 = local_id, j64i = 64 - j64 - 1;
	backward8i(pq, ml64, &Zt[kl64], m64, &zt[k64], w, 64 + j64i);

	const sz_t j8 = (local_id / 8) % 8, j8i = 8 - j8 - 1;
	backward8l(pq, ml8, &Zt[kl8], w, 8 + j8i);

	backward8_0o(pq, g_f0i[lid], m0, &zt[k0], ml0, &Zt[kl0], w);
}

/*__kernel
void square2x4(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t s = N_SZ / 8, j = id, k = 8 * vid;
	square2x4g(pq, &z[k], w, s + j);
}

__kernel
void square4x2(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t s = N_SZ / 8, j = id, ji = s - j - 1, k = 8 * vid;
	square4x2g(pq, &z[k], w, s + j, s + ji);
}

__kernel
void square8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t s = N_SZ / 8, j = id, ji = s - j - 1, k = 8 * vid;
	square8g(pq, &z[k], w, s + j, s + ji);
}*/

#define DECLARE_VAR_SQRMUL(N, BLK_N) \
	__local VTYPE Z[N * BLK_N]; \
	DECLARE_VAR_REG(); \
	const sz_t block_id = (vid / (N / 8)) % BLK_N; \
	__local VTYPE * const Zb = &Z[N * block_id]; \
	const sz_t s = N_SZ / 8, j = id, ji = s - j - 1, k = 8 * vid;

#ifdef QVALID

__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))
void square16(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(16, BLK16);
	const sz_t k2 = 7 * (vid & ~(2 - 1)) + vid;

	forward8i(pq, 2, &Zb[k2 % 16], 2, &z[k2], w, (s + j) / 2);
	square2x4l(pq, &Zb[k % 16], w, s + j);
	backward8o(pq, 2, &z[k2], 2, &Zb[k2 % 16], w, (s + ji) / 2);
}

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void square32(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(32, BLK32);
	const sz_t k4 = 7 * (vid & ~(4 - 1)) + vid;

	forward8i(pq, 4, &Zb[k4 % 32], 4, &z[k4], w, (s + j) / 4);
	square4x2l(pq, &Zb[k % 32], w, s + j, s + ji);
	backward8o(pq, 4, &z[k4], 4, &Zb[k4 % 32], w, (s + ji) / 4);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void square64(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(64, BLK64);
	const sz_t k8 = 7 * (vid & ~(8 - 1)) + vid;

	forward8i(pq, 8, &Zb[k8 % 64], 8, &z[k8], w, (s + j) / 8);
	square8l(pq, &Zb[k % 64], w, s + j, s + ji);
	backward8o(pq, 8, &z[k8], 8, &Zb[k8 % 64], w, (s + ji) / 8);
}

#else

#if (LN_SZ % 3 == 1)

__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))
void square128(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(128, BLK128);
	const sz_t k2 = 7 * (vid & ~(2 - 1)) + vid, k16 = 7 * (vid & ~(16 - 1)) + vid;

	forward8i(pq, 16, &Zb[k16 % 128], 16, &z[k16], w, (s + j) / 16);
	forward8l(pq, 2, &Zb[k2 % 128], w, (s + j) / 2);
	square2x4l(pq, &Zb[k % 128], w, s + j);
	backward8l(pq, 2, &Zb[k2 % 128], w, (s + ji) / 2);
	backward8o(pq, 16, &z[k16], 16, &Zb[k16 % 128], w, (s + ji) / 16);
}

#elif (LN_SZ % 3 == 2)

__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))
void square256(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(256, BLK256);
	const sz_t k4 = 7 * (vid & ~(4 - 1)) + vid, k32 = 7 * (vid & ~(32 - 1)) + vid;

	forward8i(pq, 32, &Zb[k32 % 256], 32, &z[k32], w, (s + j) / 32);
	forward8l(pq, 4, &Zb[k4 % 256], w, (s + j) / 4);
	square4x2l(pq, &Zb[k % 256], w, s + j, s + ji);
	backward8l(pq, 4, &Zb[k4 % 256], w, (s + ji) / 4);
	backward8o(pq, 32, &z[k32], 32, &Zb[k32 % 256], w, (s + ji) / 32);
}

#else

__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))
void square512(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(512, BLK512);
	const sz_t k8 = 7 * (vid & ~(8 - 1)) + vid, k64 = 7 * (vid & ~(64 - 1)) + vid;

	forward8i(pq, 64, &Zb[k64 % 512], 64, &z[k64], w, (s + j) / 64);
	forward8l(pq, 8, &Zb[k8 % 512], w, (s + j) / 8);
	square8l(pq, &Zb[k % 512], w, s + j, s + ji);
	backward8l(pq, 8, &Zb[k8 % 512], w, (s + ji) / 8);
	backward8o(pq, 64, &z[k64], 64, &Zb[k64 % 512], w, (s + ji) / 64);
}

#endif
#endif

/*__kernel
void fwd4x2(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id, k = 8 * vid;
	fwd4x2g(pq, &z[k], w, n_8 + j);
}

__kernel
void fwd8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id, k = 8 * vid;
	fwd8g(pq, &z[k], w, n_8 + j);
}*/

#ifdef QVALID

__kernel
void fwd16(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id, k2 = 7 * (vid & ~(2 - 1)) + vid;
	forward8g(pq, 2, &z[k2], w, (n_8 + j) / 2);
}

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void fwd32(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(32, BLK32);
	const sz_t k4 = 7 * (vid & ~(4 - 1)) + vid;

	forward8i(pq, 4, &Zb[k4 % 32], 4, &z[k4], w, (s + j) / 4);
	fwd4x2o(pq, &z[k], &Zb[k % 32], w, s + j);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void fwd64(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(64, BLK64);
	const sz_t k8 = 7 * (vid & ~(8 - 1)) + vid;

	forward8i(pq, 8, &Zb[k8 % 64], 8, &z[k8], w, (s + j) / 8);
	fwd8o(pq, &z[k], &Zb[k % 64], w, s + j);
}

#else

#if (LN_SZ % 3 == 1)

__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))
void fwd128(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(128, BLK128);
	const sz_t k2 = 7 * (vid & ~(2 - 1)) + vid, k16 = 7 * (vid & ~(16 - 1)) + vid;

	forward8i(pq, 16, &Zb[k16 % 128], 16, &z[k16], w, (s + j) / 16);
	forward8o(pq, 2, &z[k2], 2, &Zb[k2 % 128], w, (s + j) / 2);
}

#elif (LN_SZ % 3 == 2)

__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))
void fwd256(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(256, BLK256);
	const sz_t k4 = 7 * (vid & ~(4 - 1)) + vid, k32 = 7 * (vid & ~(32 - 1)) + vid;

	forward8i(pq, 32, &Zb[k32 % 256], 32, &z[k32], w, (s + j) / 32);
	forward8l(pq, 4, &Zb[k4 % 256], w, (s + j) / 4);
	fwd4x2o(pq, &z[k], &Zb[k % 256], w, s + j);
}

#else

__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))
void fwd512(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(512, BLK512);
	const sz_t k8 = 7 * (vid & ~(8 - 1)) + vid, k64 = 7 * (vid & ~(64 - 1)) + vid;

	forward8i(pq, 64, &Zb[k64 % 512], 64, &z[k64], w, (s + j) / 64);
	forward8l(pq, 8, &Zb[k8 % 512], w, (s + j) / 8);
	fwd8o(pq, &z[k], &Zb[k % 512], w, s + j);
}

#endif
#endif

/*__kernel
void mul2x4(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t n_8 = N_SZ / 8, j = id, k = 8 * vid;
	mul2x4g(pq, &z[k], &zp[k], w, n_8 + j);
}

__kernel
void mul4x2(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t n_8 = N_SZ / 8, j = id, k = 8 * vid;
	const sz_t ji = n_8 - j - 1;
	mul4x2g(pq, &z[k], &zp[k], w, n_8 + j, n_8 + ji);
}

__kernel
void mul8(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t n_8 = N_SZ / 8, j = id, k = 8 * vid;
	const sz_t ji = n_8 - j - 1;
	mul8g(pq, &z[k], &zp[k], w, n_8 + j, n_8 + ji);
}*/

#ifdef QVALID

__kernel __attribute__((reqd_work_group_size(16 / 8 * BLK16, 1, 1)))
void mul16(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(16, BLK16);
	DECLARE_VARP_REG();
	const sz_t k2 = 7 * (vid & ~(2 - 1)) + vid;

	forward8i(pq, 2, &Zb[k2 % 16], 2, &z[k2], w, (s + j) / 2);
	mul2x4l(pq, &Zb[k % 16], &zp[k], w, s + j);
	backward8o(pq, 2, &z[k2], 2, &Zb[k2 % 16], w, (s + ji) / 2);
}

__kernel __attribute__((reqd_work_group_size(32 / 8 * BLK32, 1, 1)))
void mul32(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(32, BLK32);
	DECLARE_VARP_REG();
	const sz_t k4 = 7 * (vid & ~(4 - 1)) + vid;

	forward8i(pq, 4, &Zb[k4 % 32], 4, &z[k4], w, (s + j) / 4);
	mul4x2l(pq, &Zb[k % 32], &zp[k], w, s + j, s + ji);
	backward8o(pq, 4, &z[k4], 4, &Zb[k4 % 32], w, (s + ji) / 4);
}

__kernel __attribute__((reqd_work_group_size(64 / 8 * BLK64, 1, 1)))
void mul64(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(64, BLK64);
	DECLARE_VARP_REG();
	const sz_t k8 = 7 * (vid & ~(8 - 1)) + vid;

	forward8i(pq, 8, &Zb[k8 % 64], 8, &z[k8], w, (s + j) / 8);
	mul8l(pq, &Zb[k % 64], &zp[k], w, s + j, s + ji);
	backward8o(pq, 8, &z[k8], 8, &Zb[k8 % 64], w, (s + ji) / 8);
}

#else

#if (LN_SZ % 3 == 1)

__kernel __attribute__((reqd_work_group_size(128 / 8 * BLK128, 1, 1)))
void mul128(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(128, BLK128);
	DECLARE_VARP_REG();
	const sz_t k2 = 7 * (vid & ~(2 - 1)) + vid, k16 = 7 * (vid & ~(16 - 1)) + vid;

	forward8i(pq, 16, &Zb[k16 % 128], 16, &z[k16], w, (s + j) / 16);
	forward8l(pq, 2, &Zb[k2 % 128], w, (s + j) / 2);
	mul2x4l(pq, &Zb[k % 128], &zp[k], w, s + j);
	backward8l(pq, 2, &Zb[k2 % 128], w, (s + ji) / 2);
	backward8o(pq, 16, &z[k16], 16, &Zb[k16 % 128], w, (s + ji) / 16);
}

#elif (LN_SZ % 3 == 2)

__kernel __attribute__((reqd_work_group_size(256 / 8 * BLK256, 1, 1)))
void mul256(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(256, BLK256);
	DECLARE_VARP_REG();
	const sz_t k4 = 7 * (vid & ~(4 - 1)) + vid, k32 = 7 * (vid & ~(32 - 1)) + vid;

	forward8i(pq, 32, &Zb[k32 % 256], 32, &z[k32], w, (s + j) / 32);
	forward8l(pq, 4, &Zb[k4 % 256], w, (s + j) / 4);
	mul4x2l(pq, &Zb[k % 256], &zp[k], w, s + j, s + ji);
	backward8l(pq, 4, &Zb[k4 % 256], w, (s + ji) / 4);
	backward8o(pq, 32, &z[k32], 32, &Zb[k32 % 256], w, (s + ji) / 32);
}

#else

__kernel __attribute__((reqd_work_group_size(512 / 8 * BLK512, 1, 1)))
void mul512(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_SQRMUL(512, BLK512);
	DECLARE_VARP_REG();
	const sz_t k8 = 7 * (vid & ~(8 - 1)) + vid, k64 = 7 * (vid & ~(64 - 1)) + vid;

	forward8i(pq, 64, &Zb[k64 % 512], 64, &z[k64], w, (s + j) / 64);
	forward8l(pq, 8, &Zb[k8 % 512], w, (s + j) / 8);
	mul8l(pq, &Zb[k % 512], &zp[k], w, s + j, s + ji);
	backward8l(pq, 8, &Zb[k8 % 512], w, (s + ji) / 8);
	backward8o(pq, 64, &z[k64], 64, &Zb[k64 % 512], w, (s + ji) / 64);
}

#endif
#endif

#define DECLARE_VAR_MUL_MASK() \
	DECLARE_VAR_REG_1(); \
	DECLARE_VARP_REG_1(); \
	const sz_t oid = vid / OCL_VSIZE, i = (vid % OCL_VSIZE) + OCL_VSIZE * (oid / (N_SZ / 8)); \
	const sz_t n_8 = N_SZ / 8, j = oid % (N_SZ / 8), ji = n_8 - j - 1, k = 7 * (vid & ~(OCL_VSIZE - 1)) + vid;

__kernel
void mul2x4_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,
	__global const uint_32 * restrict const wg, const uint_32 mask)
{
	DECLARE_VAR_MUL_MASK();
	if ((mask & (1u << i)) != 0) mul2x4g_1(pq, OCL_VSIZE, &z[k], &zp[k], w, n_8 + j);
}

__kernel
void mul4x2_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,
	__global const uint_32 * restrict const wg, const uint_32 mask)
{
	DECLARE_VAR_MUL_MASK();
	mul4x2g_1(pq, OCL_VSIZE, &z[k], &zp[k], w, n_8 + j, n_8 + ji, (mask & (1u << i)) != 0);
}

__kernel
void mul8_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,
	__global const uint_32 * restrict const wg, const uint_32 mask)
{
	DECLARE_VAR_MUL_MASK();
	mul8g_1(pq, OCL_VSIZE, &z[k], &zp[k], w, n_8 + j, n_8 + ji, (mask & (1u << i)) != 0);
}

// --- carry ---

INLINE uint_32 barrett(const uint_64 a, const uint_32 b, const uint_32 b_inv, const int_32 b_s, uint_32 * a_p)
{
	// Using notations of Modular SIMD arithmetic in Mathemagix, Joris van der Hoeven, Grégoire Lecerf, Guillaume Quintin, 2014, HAL.
	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.
	// b < 2^31, alpha = 2^29 => a < 2^29 b
	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31
	// b_inv = [2^{s + 32} / b]
	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31
	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].
	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])
	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,
	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.

	const uint_32 d = mul_hi((uint_32)(a >> b_s), b_inv), r = (uint_32)(a) - d * b;
	const bool o = (r >= b);
	*a_p = d + (o ? 1 : 0);
	return r - (o ? b : 0);
}

INLINE int_32 reduce64(int_64 * f, const uint_32 b, const uint_32 b_inv, const int_32 b_s)
{
	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
	// 2- t < 2^23 b^2 => t_h < b^2 / 2^6. If 2 <= b < 32 then t_h < 32^2 / 2^6 = 16 < 2^29 b
	const uint_64 t = abs(*f);
	const uint_64 t_h = t >> 29;
	const uint_32 t_l = (uint_32)(t) % (1u << 29);

	uint_32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint_32 d_l, r_l = barrett(((uint_64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint_64 d = ((uint_64)(d_h) << 29) | d_l;

	const bool s = (*f < 0);
	*f = s ? -(int_64)(d) : (int_64)(d);
	return s ? -(int_32)(r_l) : (int_32)(r_l);
}

INLINE int_32 reduce96(int_64 * f, const int96 l, const uint_32 b, const uint_32 b_inv, const int_32 b_s)
{
	const uint96 t = int96_abs(l);
	const uint_64 t_h = (t.s1 << (32 - 29)) | (t.s0 >> 29);
	const uint_32 t_l = t.s0 % (1u << 29);

	uint_32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint_32 d_l, r_l = barrett(((uint_64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint_64 d = ((uint_64)(d_h) << 29) | d_l;

	const bool s = int96_is_neg(l);
	*f = s ? -(int_64)(d) : (int_64)(d);
	return s ? -(int_32)(r_l) : (int_32)(r_l);
}

INLINE int96 garner3(const uint_32 r1, const uint_32 r2, const uint_32 r3)
{
	const uint_32 u13 = mulmod(submod(r1, r3, P1), INVP3_P1, PQ1);
	const uint_32 u23 = mulmod(submod(r2, r3, P2), INVP3_P2, PQ2);
	const uint_32 u123 = mulmod(submod(u13, u23, P1), INVP2_P1, PQ1);
	const uint96 n = uint96_add_64(uint96_mul_64_32(P2 * (uint_64)(P3), u123), u23 * (uint_64)(P3) + r3);
	const bool b = uint96_is_greater(n, uint96_set(P1P2P3_2L, P1P2P3_2H));
	return uint96_i(b ? uint96_sub(n, uint96_set(P1P2P3L, P1P2P3H)) : n);
}

INLINE int_32 reduce(int_64 * f, const uint_32 u1, const uint_32 u2, const uint_32 u3,
	const uint2_32 bb_inv, const int_32 bs, const bool dup)
{
	int96 l = garner3(u1, u2, u3);
	if (dup) l = int96_add(l, l);
	l = int96_add_64(l, *f);
	return reduce96(f, l, bb_inv.s0, bb_inv.s1, bs);
}

INLINE void write_rns(__global uint_32 * restrict const z, const int_32 r)
{
	z[0 * N_VSIZE] = set_int(r, P1);
	z[1 * N_VSIZE] = set_int(r, P2);
	z[2 * N_VSIZE] = set_int(r, P3);
}

#if OCL_CARRY_VSIZE == 4

INLINE void carry_1x4(const __global uint4_32 * restrict const zk, __global int4_64 * restrict const c, __local int4_64 * const cl,
	int4_32 r[CARRY_LENGTH], const sz_t id, const uint8_32 bb_inv_i, const int4_32 bs_i, const uint_32 dup)
{
	int_64 f0 = 0, f1 = 0, f2 = 0, f3 = 0;
	for (sz_t j = 0; j < CARRY_LENGTH; ++j)
	{
		const uint4_32 u1 = zk[j * CARRY_VSIZE + 0 * N_CARRY_VSIZE];
		const uint4_32 u2 = zk[j * CARRY_VSIZE + 1 * N_CARRY_VSIZE];
		const uint4_32 u3 = zk[j * CARRY_VSIZE + 2 * N_CARRY_VSIZE];
		r[j].s0 = reduce(&f0, u1.s0, u2.s0, u3.s0, bb_inv_i.s01, bs_i.s0, (dup & 1u) != 0);
		r[j].s1 = reduce(&f1, u1.s1, u2.s1, u3.s1, bb_inv_i.s23, bs_i.s1, (dup & 2u) != 0);
		r[j].s2 = reduce(&f2, u1.s2, u2.s2, u3.s2, bb_inv_i.s45, bs_i.s2, (dup & 4u) != 0);
		r[j].s3 = reduce(&f3, u1.s3, u2.s3, u3.s3, bb_inv_i.s67, bs_i.s3, (dup & 8u) != 0);
	}
	const int4_64 f = (int4_64)(f0, f1, f2, f3);

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;

	if (lid >= CARRY_WG_SZ - CARRY_VSIZE)
	{
		const sz_t svid = (id / CARRY_VSIZE) & ~(N_SZ / CARRY_LENGTH - 1);
		const sz_t vid = (id / CARRY_VSIZE + 1) % (N_SZ / CARRY_LENGTH);
		const sz_t cid = (id % CARRY_VSIZE) + CARRY_VSIZE * ((svid + vid) / (CARRY_WG_SZ / CARRY_VSIZE));
		c[cid] = (vid == 0) ? -f : f;
	}
}

INLINE void carry_2x4(__global uint4_32 * restrict const zk, const __local int4_64 * const cl,
	int4_32 r[CARRY_LENGTH], const sz_t id, const uint8_32 bb_inv_i, const int4_32 bs_i)
{
	const sz_t lid = id % CARRY_WG_SZ;
	if (lid >= CARRY_VSIZE)
	{
		const int4_64 f = cl[lid - CARRY_VSIZE];
		int_64 f0 = f.s0, f1 = f.s1, f2 = f.s2, f3 = f.s3;
		for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
		{
			if (f0 != 0)
			{
				f0 += r[j].s0;
				r[j].s0 = reduce64(&f0, bb_inv_i.s0, bb_inv_i.s1, bs_i.s0);
			}
			if (f1 != 0)
			{
				f1 += r[j].s1;
				r[j].s1 = reduce64(&f1, bb_inv_i.s2, bb_inv_i.s3, bs_i.s1);
			}
			if (f2 != 0)
			{
				f2 += r[j].s2;
				r[j].s2 = reduce64(&f2, bb_inv_i.s4, bb_inv_i.s5, bs_i.s2);
			}
			if (f3 != 0)
			{
				f3 += r[j].s3;
				r[j].s3 = reduce64(&f3, bb_inv_i.s6, bb_inv_i.s7, bs_i.s3);
			}
		}
		if (f0 != 0) r[CARRY_LENGTH - 1].s0 = (int_32)(f0 + r[CARRY_LENGTH - 1].s0);
		if (f1 != 0) r[CARRY_LENGTH - 1].s1 = (int_32)(f1 + r[CARRY_LENGTH - 1].s1);
		if (f2 != 0) r[CARRY_LENGTH - 1].s2 = (int_32)(f2 + r[CARRY_LENGTH - 1].s2);
		if (f3 != 0) r[CARRY_LENGTH - 1].s3 = (int_32)(f3 + r[CARRY_LENGTH - 1].s3);
	}

	for (size_t j = 0; j < CARRY_LENGTH; ++j)
	{
		zk[j * CARRY_VSIZE + 0 * N_CARRY_VSIZE] = set_int4(r[j], P1);
		zk[j * CARRY_VSIZE + 1 * N_CARRY_VSIZE] = set_int4(r[j], P2);
		zk[j * CARRY_VSIZE + 2 * N_CARRY_VSIZE] = set_int4(r[j], P3);
	}
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry1(const __global uint8_32 * restrict const bb_inv, const __global int4_32 * restrict const bs,
	__global uint4_32 * restrict const z, __global int4_64 * restrict const c, const uint_32 dup)
{
	__local int4_64 cl[CARRY_WG_SZ];

	const sz_t id = (sz_t)get_global_id(0), i = (id % CARRY_VSIZE) + ((id / (N_SZ / CARRY_LENGTH)) & ~(CARRY_VSIZE - 1));
	const sz_t k = (CARRY_LENGTH - 1) * (id & ~(CARRY_VSIZE - 1)) + id;
	const uint8_32 bb_inv_i = bb_inv[i]; const int4_32 bs_i = bs[i];
	int4_32 r[CARRY_LENGTH];

	carry_1x4(&z[k], c, cl, r, id, bb_inv_i, bs_i, dup >> (4 * i));

	barrier(CLK_LOCAL_MEM_FENCE);

	carry_2x4(&z[k], cl, r, id, bb_inv_i, bs_i);
}

#elif OCL_CARRY_VSIZE == 2

INLINE void carry_1x2(const __global uint2_32 * restrict const zk, __global int2_64 * restrict const c, __local int2_64 * const cl,
	int2_32 r[CARRY_LENGTH], const sz_t id, const uint4_32 bb_inv_i, const int2_32 bs_i, const uint_32 dup)
{
	int_64 f0 = 0, f1 = 0;
	for (sz_t j = 0; j < CARRY_LENGTH; ++j)
	{
		const uint2_32 u1 = zk[j * CARRY_VSIZE + 0 * N_CARRY_VSIZE];
		const uint2_32 u2 = zk[j * CARRY_VSIZE + 1 * N_CARRY_VSIZE];
		const uint2_32 u3 = zk[j * CARRY_VSIZE + 2 * N_CARRY_VSIZE];
		r[j].s0 = reduce(&f0, u1.s0, u2.s0, u3.s0, bb_inv_i.s01, bs_i.s0, (dup & 1u) != 0);
		r[j].s1 = reduce(&f1, u1.s1, u2.s1, u3.s1, bb_inv_i.s23, bs_i.s1, (dup & 2u) != 0);
	}
	const int2_64 f = (int2_64)(f0, f1);

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;

	if (lid >= CARRY_WG_SZ - CARRY_VSIZE)
	{
		const sz_t svid = (id / CARRY_VSIZE) & ~(N_SZ / CARRY_LENGTH - 1);
		const sz_t vid = (id / CARRY_VSIZE + 1) % (N_SZ / CARRY_LENGTH);
		const sz_t cid = (id % CARRY_VSIZE) + CARRY_VSIZE * ((svid + vid) / (CARRY_WG_SZ / CARRY_VSIZE));
		c[cid] = (vid == 0) ? -f : f;
	}
}

INLINE void carry_2x2(__global uint2_32 * restrict const zk, const __local int2_64 * const cl,
	int2_32 r[CARRY_LENGTH], const sz_t id, const uint4_32 bb_inv_i, const int2_32 bs_i)
{
	const sz_t lid = id % CARRY_WG_SZ;
	if (lid >= CARRY_VSIZE)
	{
		const int2_64 f = cl[lid - CARRY_VSIZE];
		int_64 f0 = f.s0, f1 = f.s1;
		for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
		{
			if (f0 != 0)
			{
				f0 += r[j].s0;
				r[j].s0 = reduce64(&f0, bb_inv_i.s0, bb_inv_i.s1, bs_i.s0);
			}
			if (f1 != 0)
			{
				f1 += r[j].s1;
				r[j].s1 = reduce64(&f1, bb_inv_i.s2, bb_inv_i.s3, bs_i.s1);
			}
		}
		if (f0 != 0) r[CARRY_LENGTH - 1].s0 = (int_32)(f0 + r[CARRY_LENGTH - 1].s0);
		if (f1 != 0) r[CARRY_LENGTH - 1].s1 = (int_32)(f1 + r[CARRY_LENGTH - 1].s1);
	}

	for (size_t j = 0; j < CARRY_LENGTH; ++j)
	{
		zk[j * CARRY_VSIZE + 0 * N_CARRY_VSIZE] = set_int2(r[j], P1);
		zk[j * CARRY_VSIZE + 1 * N_CARRY_VSIZE] = set_int2(r[j], P2);
		zk[j * CARRY_VSIZE + 2 * N_CARRY_VSIZE] = set_int2(r[j], P3);
	}
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry1(const __global uint4_32 * restrict const bb_inv, const __global int2_32 * restrict const bs,
	__global uint2_32 * restrict const z, __global int2_64 * restrict const c, const uint_32 dup)
{
	__local int2_64 cl[CARRY_WG_SZ];

	const sz_t id = (sz_t)get_global_id(0), i = (id % CARRY_VSIZE) + ((id / (N_SZ / CARRY_LENGTH)) & ~(CARRY_VSIZE - 1));
	const sz_t k = (CARRY_LENGTH - 1) * (id & ~(CARRY_VSIZE - 1)) + id;
	const uint4_32 bb_inv_i = bb_inv[i]; const int2_32 bs_i = bs[i];
	int2_32 r[CARRY_LENGTH];

	carry_1x2(&z[k], c, cl, r, id, bb_inv_i, bs_i, dup >> (2 * i));

	barrier(CLK_LOCAL_MEM_FENCE);

	carry_2x2(&z[k], cl, r, id, bb_inv_i, bs_i);
}

#else	// OCL_CARRY_VSIZE = 1

INLINE void carry_1x1(const __global uint_32 * restrict const zk, __global int_64 * restrict const c, __local int_64 * const cl,
	int_32 r[CARRY_LENGTH], const sz_t id, const uint2_32 bb_inv_i, const int_32 bs_i, const uint_32 dup)
{
	int_64 f = 0;
	for (sz_t j = 0; j < CARRY_LENGTH; ++j)
	{
		const uint_32 u1 = zk[j * CARRY_VSIZE + 0 * N_CARRY_VSIZE];
		const uint_32 u2 = zk[j * CARRY_VSIZE + 1 * N_CARRY_VSIZE];
		const uint_32 u3 = zk[j * CARRY_VSIZE + 2 * N_CARRY_VSIZE];
		r[j] = reduce(&f, u1, u2, u3, bb_inv_i, bs_i, (dup & 1u) != 0);
	}

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;

	if (lid >= CARRY_WG_SZ - CARRY_VSIZE)
	{
		const sz_t svid = (id / CARRY_VSIZE) & ~(N_SZ / CARRY_LENGTH - 1);
		const sz_t vid = (id / CARRY_VSIZE + 1) % (N_SZ / CARRY_LENGTH);
		const sz_t cid = (id % CARRY_VSIZE) + CARRY_VSIZE * ((svid + vid) / (CARRY_WG_SZ / CARRY_VSIZE));
		c[cid] = (vid == 0) ? -f : f;
	}
}

INLINE void carry_2x1(__global uint_32 * restrict const zk, const __local int_64 * const cl,
	int_32 r[CARRY_LENGTH], const sz_t id, const uint2_32 bb_inv_i, const int_32 bs_i)
{
	const sz_t lid = id % CARRY_WG_SZ;
	if (lid >= CARRY_VSIZE)
	{
		int_64 f = cl[lid - CARRY_VSIZE];
		for (size_t j = 0; j < CARRY_LENGTH - 1; ++j)
		{
			if (f != 0)
			{
				f += r[j];
				r[j] = reduce64(&f, bb_inv_i.s0, bb_inv_i.s1, bs_i);
			}
		}
		if (f != 0) r[CARRY_LENGTH - 1] = (int_32)(f + r[CARRY_LENGTH - 1]);
	}

	for (size_t j = 0; j < CARRY_LENGTH; ++j) write_rns(&zk[j * CARRY_VSIZE], r[j]);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry1(const __global uint2_32 * restrict const bb_inv, const __global int_32 * restrict const bs,
	__global uint_32 * restrict const z, __global int_64 * restrict const c, const uint_32 dup)
{
	__local int_64 cl[CARRY_WG_SZ];

	const sz_t id = (sz_t)get_global_id(0), i = (id % CARRY_VSIZE) + ((id / (N_SZ / CARRY_LENGTH)) & ~(CARRY_VSIZE - 1));
	const sz_t k = (CARRY_LENGTH - 1) * (id & ~(CARRY_VSIZE - 1)) + id;
	const uint2_32 bb_inv_i = bb_inv[i]; const int_32 bs_i = bs[i];
	int_32 r[CARRY_LENGTH];

	carry_1x1(&z[k], c, cl, r, id, bb_inv_i, bs_i, dup >> i);

	barrier(CLK_LOCAL_MEM_FENCE);

	carry_2x1(&z[k], cl, r, id, bb_inv_i, bs_i);
}
#endif

__kernel
void carry2(const __global uint2_32 * restrict const bb_inv, const __global int_32 * restrict const bs,
	__global uint_32 * restrict const z, const __global int_64 * restrict const c)
{
	const sz_t gid = (sz_t)get_global_id(0), id = (gid / OCL_VSIZE) * CARRY_WG_SZ * OCL_CARRY_VSIZE + (gid % OCL_VSIZE);
	const sz_t i = (id % OCL_VSIZE) + ((id / (N_SZ / CARRY_LENGTH)) & ~(OCL_VSIZE - 1));
	const sz_t k = (CARRY_LENGTH - 1) * (id & ~(OCL_VSIZE - 1)) + id;
	const uint2_32 bb_inv_i = bb_inv[i]; const int_32 bs_i = bs[i];

	int_64 f = c[gid];
	for (size_t j = 0; j < 3; ++j)
	{
		if (f != 0)
		{
			f += get_int(z[k + j * OCL_VSIZE], P1);
			const int_32 r = reduce64(&f, bb_inv_i.s0, bb_inv_i.s1, bs_i);
			write_rns(&z[k + j * OCL_VSIZE], r);
		}
	}

	if (f != 0)
	{
		f += get_int(z[k + 3 * OCL_VSIZE], P1);
		write_rns(&z[k + 3 * OCL_VSIZE], (int_32)(f));
	}
}

// --- misc ---

__kernel
void set(__global VTYPE * restrict const z, const uint_32 a)
{
	const sz_t id = (sz_t)get_global_id(0);
	z[id] = (id % N_SZ == 0) ? (VTYPE)(a) : (VTYPE)(0);
}

__kernel
void copy(__global VTYPE * restrict const z, const sz_t dst, const sz_t src)
{
	const sz_t id = (sz_t)get_global_id(0);
	z[3 * N_VLEN * dst + id] = z[3 * N_VLEN * src + id];
}

__kernel
void copyp(__global VTYPE * restrict const zp, __global const VTYPE * restrict const z, const sz_t src)
{
	const sz_t id = (sz_t)get_global_id(0);
	zp[id] = z[3 * N_VLEN * src + id];
}

__kernel
void copy_mask(__global uint_32 * restrict const z, const sz_t dst, const sz_t src, const uint_32 mask)
{
	const sz_t id = (sz_t)get_global_id(0), i = (id % OCL_VSIZE) + (((id / N_SZ) % VSIZE) & ~(OCL_VSIZE - 1));
	if ((mask & (1u << i)) != 0) z[3 * N_VSIZE * dst + id] = z[3 * N_VSIZE * src + id];
}

#ifdef QVALID

__kernel
void cosmic_ray(__global uint_32 * restrict const z)
{
	const sz_t id = (sz_t)get_global_id(0);
	if (id == (N_SZ * OCL_VSIZE) / 2) z[id] = addmod(z[id], 1, P1);
}

#endif
