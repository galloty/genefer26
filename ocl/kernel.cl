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
#define OCL_VSIZE	2
#define CARRY_WG_SZ	256u
#endif

#define N_VSIZE		(N_SZ * VSIZE)
#define N_VLEN		(N_SZ * VSIZE / OCL_VSIZE)

typedef uint	sz_t;
typedef uint	uint_32;
typedef int		int_32;
typedef ulong	uint_64;
typedef long	int_64;
typedef uint2	uint2_32;
typedef uint4	uint4_32;

// --- modular arithmetic

#define	PQ1		(uint2_32)(P1, Q1)
#define	PQ2		(uint2_32)(P2, Q2)
#define	PQ3		(uint2_32)(P3, Q3)

__constant uint2_32 g_pq[3] = { PQ1, PQ2, PQ3 };
__constant uint4_32 g_f0[3] = { (uint4_32)(RSQ1, MFIM1, SQRTI1, ISQRTI1), (uint4_32)(RSQ2, MFIM2, SQRTI2, ISQRTI2), (uint4_32)(RSQ3, MFIM3, SQRTI3, ISQRTI3) };

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
#define VTYPE				uint4_32
#define addmodv				addmod4
#define submodv				submod4
#define mulmodv				mulmod4
#define mulmodsv			mulmods4
#elif OCL_VSIZE == 2
#define VTYPE				uint2_32
#define addmodv				addmod2
#define submodv				submod2
#define mulmodv				mulmod2
#define mulmodsv			mulmods2
#else
#define VTYPE				uint_32
#define addmodv				addmod
#define submodv				submod
#define mulmodv				mulmod
#define mulmodsv			mulmod
#endif

// --- I/O ---

INLINE void _loadg(const sz_t n, VTYPE * const zl, __global const VTYPE * restrict const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _storeg(const sz_t n, __global VTYPE * restrict const z, const sz_t s, const VTYPE * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

INLINE void _loadg_1(const sz_t n, uint_32 * const zl, __global const uint_32 * restrict const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _storeg_1(const sz_t n, __global uint_32 * restrict const z, const sz_t s, const uint_32 * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

INLINE uint2_32 _load2g(__global const uint_32 * restrict const w, const sz_t j) { return ((__global const uint2_32 *)w)[j]; }
INLINE uint4_32 _load4g(__global const uint_32 * restrict const w, const sz_t j) { return ((__global const uint4_32 *)w)[j]; }

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
	z[0] = mulmodsv(z[0], f0.s0, pq); z[1] = mulmodsv(z[1], f0.s0, pq); z[2] = mulmodsv(z[2], f0.s0, pq); z[3] = mulmodsv(z[3], f0.s0, pq);
	_forward8(pq, z, f0.s1, f0.s23, w4);
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
	const uint2_32 w2, const uint2_32 wi2r, const bool bmask)
{
	FWD2(z[0], z[2], w2.s0); FWD2(z[1], z[3], w2.s0); FWD2(z[4], z[6], w2.s1); FWD2(z[5], z[7], w2.s1);
	if (bmask) _mul2x4(pq, z, zp, w2);
	BCK2(z[0], z[2], wi2r.s1); BCK2(z[1], z[3], wi2r.s1); BCK2(z[4], z[6], wi2r.s0); BCK2(z[5], z[7], wi2r.s0);
}

INLINE void _mul8r(const uint2_32 pq, VTYPE z[8], const VTYPE zp[8],
	const uint_32 w1, const uint_32 wi1, const uint2_32 w2, const uint2_32 wi2r, const bool bmask)
{
	FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);
	_mul4x2r(pq, z, zp, w2, wi2r, bmask);
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

INLINE void forward8io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint_32 w1 = w[sj];
	const uint2_32 w2 = _load2g(w, sj);
	const uint4_32 w4 = _load4g(w, sj);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_forward8(pq, zl, w1, w2, w4);
	_storeg(8, z, vm, zl);
}

INLINE void backward8io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sji)
{
	const uint_32 wi1 = w[sji];
	const uint2_32 wi2r = _load2g(w, sji);
	const uint4_32 wi4r = _load4g(w, sji);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_backward8r(pq, zl, wi1, wi2r, wi4r);
	_storeg(8, z, vm, zl);
}

INLINE void forward8_0io(const uint2_32 pq, const uint4_32 f0, const sz_t vm,
	__global VTYPE * restrict const z, __global const uint_32 * restrict const w)
{
	const uint4_32 w4 = _load4g(w, 1);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_forward8_0(pq, f0, zl, w4);
	_storeg(8, z, vm, zl);
}

INLINE void square2x4io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_square2x4(pq, zl, w2);
	_storeg(8, z, vm, zl);
}

INLINE void square4x2io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	const uint2_32 w2 = _load2g(w, sj), wi2r = _load2g(w, sji);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_square4x2r(pq, zl, w2, wi2r);
	_storeg(8, z, vm, zl);
}

INLINE void square8io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji)
{
	const uint_32 w1 = w[sj], wi1 = w[sji];
	const uint2_32 w2 = _load2g(w, sj), wi2r = _load2g(w, sji);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_square8r(pq, zl, w1, wi1, w2, wi2r);
	_storeg(8, z, vm, zl);
}

INLINE void fwd4x2io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_fwd4x2(pq, zl, w2);
	_storeg(8, z, vm, zl);
}

INLINE void fwd8io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint_32 w1 = w[sj];
	const uint2_32 w2 = _load2g(w, sj);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	_fwd8(pq, zl, w1, w2);
	_storeg(8, z, vm, zl);
}

INLINE void mul2x4io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	VTYPE zpl[8]; _loadg(8, zpl, zp, vm);
	_mul2x4(pq, zl, zpl, w2);
	_storeg(8, z, vm, zl);
}

INLINE void mul4x2io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)
{
	const uint2_32 w2 = _load2g(w, sj), wi2r = _load2g(w, sji);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	VTYPE zpl[8]; _loadg(8, zpl, zp, vm);
	_mul4x2r(pq, zl, zpl, w2, wi2r, bmask);
	_storeg(8, z, vm, zl);
}

INLINE void mul8io(const uint2_32 pq, const sz_t vm, __global VTYPE * restrict const z, __global const VTYPE * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)
{
	const uint_32 w1 = w[sj], wi1 = w[sji];
	const uint2_32 w2 = _load2g(w, sj), wi2r = _load2g(w, sji);

	VTYPE zl[8]; _loadg(8, zl, z, vm);
	VTYPE zpl[8]; _loadg(8, zpl, zp, vm);
	_mul8r(pq, zl, zpl, w1, wi1, w2, wi2r, bmask);
	_storeg(8, z, vm, zl);
}

INLINE void mul2x4io_1(const uint2_32 pq, const sz_t vm, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj)
{
	const uint2_32 w2 = _load2g(w, sj);

	uint_32 zl[8]; _loadg_1(8, zl, z, vm);
	uint_32 zpl[8]; _loadg_1(8, zpl, zp, vm);
	_mul2x4_1(pq, zl, zpl, w2);
	_storeg_1(8, z, vm, zl);
}

INLINE void mul4x2io_1(const uint2_32 pq, const sz_t vm, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)
{
	const uint2_32 w2 = _load2g(w, sj), wi2r = _load2g(w, sji);

	uint_32 zl[8]; _loadg_1(8, zl, z, vm);
	uint_32 zpl[8]; _loadg_1(8, zpl, zp, vm);
	_mul4x2r_1(pq, zl, zpl, w2, wi2r, bmask);
	_storeg_1(8, z, vm, zl);
}

INLINE void mul8io_1(const uint2_32 pq, const sz_t vm, __global uint_32 * restrict const z, __global const uint_32 * restrict const zp,
	__global const uint_32 * restrict const w, const sz_t sj, const sz_t sji, const bool bmask)
{
	const uint_32 w1 = w[sj], wi1 = w[sji];
	const uint2_32 w2 = _load2g(w, sj), wi2r = _load2g(w, sji);

	uint_32 zl[8]; _loadg_1(8, zl, z, vm);
	uint_32 zpl[8]; _loadg_1(8, zpl, zp, vm);
	_mul8r_1(pq, zl, zpl, w1, wi1, w2, wi2r, bmask);
	_storeg_1(8, z, vm, zl);
}

// --- transform/macro ---

#define DECLARE_VAR_REG() \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (N_VLEN / 8), id = gid % (N_VLEN / 8); \
	const uint2_32 pq = g_pq[lid]; \
	__global VTYPE * restrict const z = &zg[lid * N_VLEN]; \
	__global const uint_32 * restrict const w = &wg[lid * W_SZ];

#define DECLARE_VARP_REG() \
	__global const VTYPE * restrict const zp = &zpg[lid * N_VLEN];

#define DECLARE_VAR_REG_1() \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid / (N_VSIZE / 8), id = gid % (N_VSIZE / 8); \
	const uint2_32 pq = g_pq[lid]; \
	__global uint_32 * restrict const z = &zg[lid * N_VSIZE]; \
	__global const uint_32 * restrict const w = &wg[lid * W_SZ];

#define DECLARE_VARP_REG_1() \
	__global const uint_32 * restrict const zp = &zpg[lid * N_VSIZE];

// --- transform without local mem ---

__kernel
void forward8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg, const int_32 lm, const uint_32 s)
{
	DECLARE_VAR_REG();
	const sz_t vm = 1u << lm, j = (id % (N_SZ / 8)) >> lm, k = 7 * (id & ~(vm - 1)) + id;
	forward8io(pq, vm, &z[k], w, s + j);
}

__kernel
void backward8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg, const int_32 lm, const uint_32 s)
{
	DECLARE_VAR_REG();
	const sz_t vm = 1u << lm, j = (id % (N_SZ / 8)) >> lm, k = 7 * (id & ~(vm - 1)) + id;
	const sz_t ji = s - j - 1;
	backward8io(pq, vm, &z[k], w, s + ji);
}

__kernel
void forward8_0(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t vn_8 = N_SZ / 8, k = 7 * (id & ~(vn_8 - 1)) + id;
	forward8_0io(pq, g_f0[lid], vn_8, &z[k], w);
}

__kernel
void square2x4(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	square2x4io(pq, 1, &z[k], w, n_8 + j);
}

__kernel
void square4x2(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	const sz_t ji = n_8 - j - 1;
	square4x2io(pq, 1, &z[k], w, n_8 + j, n_8 + ji);
}

__kernel
void square8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	const sz_t ji = n_8 - j - 1;
	square8io(pq, 1, &z[k], w, n_8 + j, n_8 + ji);
}

__kernel
void fwd4x2(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	fwd4x2io(pq, 1, &z[k], w, n_8 + j);
}

__kernel
void fwd8(__global VTYPE * restrict const zg, __global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	fwd8io(pq, 1, &z[k], w, n_8 + j);
}

__kernel
void mul2x4(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg,
	__global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	mul2x4io(pq, 1, &z[k], &zp[k], w, n_8 + j);
}

__kernel
void mul4x2(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg,
	__global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	const sz_t ji = n_8 - j - 1;
	mul4x2io(pq, 1, &z[k], &zp[k], w, n_8 + j, n_8 + ji, true);
}

__kernel
void mul8(__global VTYPE * restrict const zg, const __global VTYPE * restrict const zpg,
	__global const uint_32 * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t n_8 = N_SZ / 8, j = id % (N_SZ / 8), k = 8 * id;
	const sz_t ji = n_8 - j - 1;
	mul8io(pq, 1, &z[k], &zp[k], w, n_8 + j, n_8 + ji, true);
}

__kernel
void mul2x4_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,
	__global const uint_32 * restrict const wg, const uint_32 mask)
{
	DECLARE_VAR_REG_1();
	const sz_t vid = id / OCL_VSIZE, i = (id % OCL_VSIZE) + OCL_VSIZE * (vid / (N_SZ / 8));
	if ((mask & (1u << i)) != 0)
	{
		DECLARE_VARP_REG_1();
		const sz_t n_8 = N_SZ / 8, j = vid % (N_SZ / 8), k = 7 * (id & ~(OCL_VSIZE - 1)) + id;
		mul2x4io_1(pq, OCL_VSIZE, &z[k], &zp[k], w, n_8 + j);
	}
}

__kernel
void mul4x2_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,
	__global const uint_32 * restrict const wg, const uint_32 mask)
{
	DECLARE_VAR_REG_1();
	DECLARE_VARP_REG_1();
	const sz_t vid = id / OCL_VSIZE, i = (id % OCL_VSIZE) + OCL_VSIZE * (vid / (N_SZ / 8));
	const sz_t n_8 = N_SZ / 8, j = vid % (N_SZ / 8), k = 7 * (id & ~(OCL_VSIZE - 1)) + id;
	const sz_t ji = n_8 - j - 1;
	mul4x2io_1(pq, OCL_VSIZE, &z[k], &zp[k], w, n_8 + j, n_8 + ji, (mask & (1u << i)) != 0);
}

__kernel
void mul8_mask(__global uint_32 * restrict const zg, const __global uint_32 * restrict const zpg,
	__global const uint_32 * restrict const wg, const uint_32 mask)
{
	DECLARE_VAR_REG_1();
	DECLARE_VARP_REG_1();
	const sz_t vid = id / OCL_VSIZE, i = (id % OCL_VSIZE) + OCL_VSIZE * (vid / (N_SZ / 8));
	const sz_t n_8 = N_SZ / 8, j = vid % (N_SZ / 8), k = 7 * (id & ~(OCL_VSIZE - 1)) + id;
	const sz_t ji = n_8 - j - 1;
	mul8io_1(pq, OCL_VSIZE, &z[k], &zp[k], w, n_8 + j, n_8 + ji, (mask & (1u << i)) != 0);
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

INLINE void write_rns(__global uint_32 * restrict const z, const int_32 r)
{
	z[0 * N_VSIZE] = set_int(r, P1);
	z[1 * N_VSIZE] = set_int(r, P2);
	z[2 * N_VSIZE] = set_int(r, P3);
}

INLINE void carry_1(__global uint_32 * restrict const zk, __global int_64 * restrict const c, __local int_64 * const cl,
	int_32 r[8], const sz_t id, const uint2_32 bb_inv_i, const int bs_i, const bool dup)
{
	// Tn vloadn(size_t offset, const Q T *p)

	int_64 f = 0;
	for (sz_t j = 0; j < 8; ++j)
	{
		const uint_32 u1 = mulmod(zk[j * OCL_VSIZE + 0 * N_VSIZE], NORM1, PQ1);
		const uint_32 u2 = mulmod(zk[j * OCL_VSIZE + 1 * N_VSIZE], NORM2, PQ2);
		const uint_32 u3 = mulmod(zk[j * OCL_VSIZE + 2 * N_VSIZE], NORM3, PQ3);
		int96 l = garner3(u1, u2, u3);
		if (dup) l = int96_add(l, l);
		l = int96_add_64(l, f);
		r[j] = reduce96(&f, l, bb_inv_i.s0, bb_inv_i.s1, bs_i);
	}

	const sz_t lid = id % CARRY_WG_SZ;
	cl[lid] = f;

	if (lid >= CARRY_WG_SZ - OCL_VSIZE)	//VSIZE)
	{
		// const sz_t vid = ((id / VSIZE) + 1) % (N_SZ / 8);
		// const sz_t cid = vid / (CARRY_WG_SZ / VSIZE) * VSIZE + id % VSIZE;
		// c[cid] = (vid == 0) ? -f : f;
		const sz_t svid = (id / OCL_VSIZE) / (N_SZ / 8);
		const sz_t vid = ((id / OCL_VSIZE) + 1) % (N_SZ / 8);
		const sz_t cid = (vid + svid * (N_SZ / 8)) / (CARRY_WG_SZ / OCL_VSIZE) * OCL_VSIZE + (id % OCL_VSIZE);
		c[cid] = (vid == 0) ? -f : f;
	}
}

INLINE void carry_2(__global uint_32 * restrict const zk, const __local int_64 * const cl,
	int_32 r[8], const sz_t id, const uint2_32 bb_inv_i, const int bs_i)
{
	const sz_t lid = id % CARRY_WG_SZ;
	if (lid >= OCL_VSIZE)	// VSIZE)
	{
		int_64 f = cl[lid - OCL_VSIZE];	// VSIZE];
		for (size_t j = 0; j < 7; ++j)
		{
			f += r[j];
			r[j] = reduce64(&f, bb_inv_i.s0, bb_inv_i.s1, bs_i);
			if (f == 0) break;
		}
		r[7] += (int_32)(f);
	}

	for (size_t j = 0; j < 8; ++j) write_rns(&zk[j * OCL_VSIZE], r[j]);
}

__kernel __attribute__((reqd_work_group_size(CARRY_WG_SZ, 1, 1)))
void carry1(const __global uint2_32 * restrict const bb_inv, const __global int_32 * restrict const bs,
	__global uint_32 * restrict const z, __global int_64 * restrict const c, const uint_32 dup)
{
	__local int_64 cl[CARRY_WG_SZ];

	const sz_t id = (sz_t)get_global_id(0), i = (id % OCL_VSIZE) + ((id / (N_SZ / 8)) & ~(OCL_VSIZE - 1)), k = 7 * (id & ~(OCL_VSIZE - 1)) + id;
	const uint2_32 bb_inv_i = bb_inv[i]; const int_32 bs_i = bs[i];
	int_32 r[8];

	carry_1(&z[k], c, cl, r, id, bb_inv_i, bs_i, (dup & (1u << i)) != 0);

	barrier(CLK_LOCAL_MEM_FENCE);

	carry_2(&z[k], cl, r, id, bb_inv_i, bs_i);
}

__kernel
void carry2(const __global uint2_32 * restrict const bb_inv, const __global int_32 * restrict const bs,
	__global uint_32 * restrict const z, const __global int_64 * restrict const c)
{
	const sz_t gid = (sz_t)get_global_id(0), id = (gid / OCL_VSIZE) * CARRY_WG_SZ + (gid % OCL_VSIZE);
	const sz_t i = (id % OCL_VSIZE) + ((id / (N_SZ / 8)) & ~(OCL_VSIZE - 1)), k = 7 * (id & ~(OCL_VSIZE - 1)) + id;
	const uint2_32 bb_inv_i = bb_inv[i]; const int_32 bs_i = bs[i];

	int_64 f = c[gid];
	for (size_t j = 0; j < 7; ++j)
	{
		f += get_int(z[k + j * OCL_VSIZE], P1);
		const int_32 r = reduce64(&f, bb_inv_i.s0, bb_inv_i.s1, bs_i);
		write_rns(&z[k + j * OCL_VSIZE], r);
		if (f == 0) return;
	}

	f += get_int(z[k + 7 * OCL_VSIZE], P1);
	const int_32 r = (int_32)(f);
	write_rns(&z[k + 7 * OCL_VSIZE], r);
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

__kernel
void cosmic_ray(__global uint_32 * restrict const z)
{
	const sz_t id = (sz_t)get_global_id(0);
	if (id == (N_SZ * OCL_VSIZE) / 2) z[id] = addmod(z[id], 1, P1);
}