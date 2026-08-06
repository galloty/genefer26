/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "transform.h"
#include "engine.h"
#include "ocl/kernel.h"

#define	P1S			(127 * (uint32(1) << 24) + 1)
#define	Q1S			2164260865u		// p * q = 1 (mod 2^32)
#define	R1S			33554430u		// 2^32 mod p
#define	RSQ1S		402124772u		// (2^32)^2 mod p
#define	H1S			100663290u		// Montgomery form of the primitive root 3
#define	IM1S		2063729671u		// MF of I = 3^{(p - 1)/4}
#define	MFIM1S		1930170389u		// MF of MF of I to convert input into MF
#define	SQRTI1S		1626730317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1S	856006302u		// MF of i * sqrt(i)

#define	P2S			(63 * (uint32(1) << 25) + 1)
#define	Q2S			2181038081u
#define	R2S			67108862u
#define	RSQ2S		2111798781u
#define	H2S			335544310u		// MF of the primitive root 5
#define	IM2S		530075385u
#define	MFIM2S		1036950657u
#define	SQRTI2S		338852760u
#define	ISQRTI2S	1090446030u

#define	P3S			(15 * (uint32(1) << 27) + 1)
#define	Q3S			2281701377u
#define	R3S			268435454u
#define	RSQ3S		1172168163u
#define	H3S			268435390u		// MF of the primitive root 31
#define	IM3S		473486609u
#define	MFIM3S		734725699u
#define	SQRTI3S		1032137103u
#define	ISQRTI3S	1964242958u

#define	INVP2_P1S	2130706177u		// MF of 1 / P2 (mod P1)
#define	INVP3_P1S	608773230u		// MF of 1 / P3 (mod P1)
#define	INVP3_P2S	1409286102u		// MF of 1 / P3 (mod P2)

#define	P1P2P3LS	1962934273u					// (P1 * P2 * P3) mod 2^32
#define	P1P2P3HS	2111326211158966273ul		// (P1 * P2 * P3) >> 32

#define	P1P2P3_2LS	3128950784u					// (P1 * P2 * P3 / 2) mod 2^32
#define	P1P2P3_2HS	1055663105579483136ul		// (P1 * P2 * P3 / 2) >> 32

#define P1U			(125 * (uint32(1) << 25) + 1)
#define	Q1U			100663297u		// p * q = 1 (mod 2^32)
#define	R1U			100663295u		// 2^32 mod p
#define	RSQ1U		232465106u		// (2^32)^2 mod p
#define	H1U			301989885u		// Montgomery form of the primitive root 3
#define	IM1U		1486287593u		// MF of I = 3^{(p - 1)/4}
#define	MFIM1U		3645424034u		// MF of MF of I to convert input into MF
#define	SQRTI1U		3580437317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1U	2017881188u		// MF of i * sqrt(i)

#define P2U			(243 * (uint32(1) << 24) + 1)
#define	Q2U			218103809u
#define	R2U			218103807u
#define	RSQ2U		3444438393u
#define	H2U			1526726649u		// MF of the primitive root 7
#define	IM2U		99906823u
#define	MFIM2U		1773796560u
#define	SQRTI2U		2024944857u
#define	ISQRTI2U	2119710515u

#define P3U			(235 * (uint32(1) << 24) + 1)
#define	Q3U			352321537u
#define	R3U			352321535u
#define	RSQ3U		3810498414u
#define	H3U			1056964605u		// MF of the primitive root 3
#define	IM3U		2213106415u
#define	MFIM3U		2454519270u
#define	SQRTI3U		3448990025u
#define	ISQRTI3U	3659377330u

#define	INVP2_P1U	1797558821u		// MF of 1 / P2 (mod P1)
#define	INVP3_P1U	3075822917u		// MF of 1 / P3 (mod P1)
#define	INVP3_P2U	4076863457u		// MF of 1 / P3 (mod P2)

#define	P1P2P3LU	3623878657u					// (P1 * P2 * P3) mod 2^32
#define	P1P2P3HU	15696902887611105282ul		// (P1 * P2 * P3) >> 32

#define	P1P2P3_2LU	1811939328u					// (P1 * P2 * P3 / 2) mod 2^32
#define	P1P2P3_2HU	7848451443805552641ul		// (P1 * P2 * P3 / 2) >> 32


template<uint32 P, uint32 Q, uint32 R, uint32 RSQ, uint32 H, uint32 IM, uint32 SQRTI, uint32 ISQRTI>
class Zp : public ZP
{
private:
	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= P) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? P : 0); }

	// 2 mul + 2 mul_hi
	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

	static void _load(const size_t n, Zp * const zl, const Zp * const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
	static void _store(const size_t n, Zp * const z, const size_t s, const Zp * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

	static void _fwd2(Zp & z0, Zp & z1, const Zp & w) { const Zp t = z1.mul(w); z1 = z0.sub(t); z0 = z0.add(t); }
	static void _bck2(Zp & z0, Zp & z1, const Zp & wi) { const Zp t = z1.sub(z0); z0 = z0.add(z1), z1 = t.mul(wi); }

	static void _sqr2(Zp & z0, Zp & z1, const Zp & w) { const Zp t = z1.sqr().mul(w); z1 = z0.add(z0).mul(z1); z0 = z0.sqr().add(t); }
	static void _sqr2n(Zp & z0, Zp & z1, const Zp & w) { const Zp t = z1.sqr().mul(w); z1 = z0.add(z0).mul(z1); z0 = z0.sqr().sub(t); }

	static void _mul2(Zp & z0, Zp & z1, const Zp & zp0, const Zp & zp1, const Zp & w) { const Zp t = z1.mul(zp1).mul(w); z1 = z0.mul(zp1).add(zp0.mul(z1)); z0 = z0.mul(zp0).add(t); }
	static void _mul2n(Zp & z0, Zp & z1, const Zp & zp0, const Zp & zp1, const Zp & w) { const Zp t = z1.mul(zp1).mul(w); z1 = z0.mul(zp1).add(zp0.mul(z1)); z0 = z0.mul(zp0).sub(t); }

	static void _forward8(Zp z[8], const Zp w1, const Zp w2[2], const Zp w4[4])
	{
		_fwd2(z[0], z[4], w1); _fwd2(z[2], z[6], w1); _fwd2(z[1], z[5], w1); _fwd2(z[3], z[7], w1);
		_fwd2(z[0], z[2], w2[0]); _fwd2(z[1], z[3], w2[0]); _fwd2(z[4], z[6], w2[1]); _fwd2(z[5], z[7], w2[1]);
		_fwd2(z[0], z[1], w4[0]); _fwd2(z[2], z[3], w4[1]); _fwd2(z[4], z[5], w4[2]); _fwd2(z[6], z[7], w4[3]);
	}

	static void _backward8r(Zp z[8], const Zp wi1, const Zp wi2r[2], const Zp wi4r[4])
	{
		_bck2(z[0], z[1], wi4r[3]); _bck2(z[2], z[3], wi4r[2]); _bck2(z[4], z[5], wi4r[1]); _bck2(z[6], z[7], wi4r[0]);
		_bck2(z[0], z[2], wi2r[1]); _bck2(z[1], z[3], wi2r[1]); _bck2(z[4], z[6], wi2r[0]); _bck2(z[5], z[7], wi2r[0]);
		_bck2(z[0], z[4], wi1); _bck2(z[2], z[6], wi1); _bck2(z[1], z[5], wi1); _bck2(z[3], z[7], wi1);
	}

	static void _forward8_0(Zp z[4], const Zp w2[2], const Zp w4[4])
	{
		z[0] = z[0].toMonty(); z[1] = z[1].toMonty(); z[2] = z[2].toMonty(); z[3] = z[3].toMonty();
		_forward8(z, Zp(IM), w2, w4);
	}

	static void _square2x4(Zp z[8], const Zp w2[2])
	{
		_sqr2(z[0], z[1], w2[0]); _sqr2n(z[2], z[3], w2[0]); _sqr2(z[4], z[5], w2[1]); _sqr2n(z[6], z[7], w2[1]);
	}

	static void _square4x2r(Zp z[8], const Zp w2[2], const Zp wi2r[2])
	{
		_fwd2(z[0], z[2], w2[0]); _fwd2(z[1], z[3], w2[0]); _fwd2(z[4], z[6], w2[1]); _fwd2(z[5], z[7], w2[1]);
		_square2x4(z, w2);
		_bck2(z[0], z[2], wi2r[1]); _bck2(z[1], z[3], wi2r[1]); _bck2(z[4], z[6], wi2r[0]); _bck2(z[5], z[7], wi2r[0]);
	}

	static void _square8r(Zp z[8], const Zp w1, const Zp  wi1, const Zp w2[2], const Zp wi2r[2])
	{
		_fwd2(z[0], z[4], w1); _fwd2(z[2], z[6], w1); _fwd2(z[1], z[5], w1); _fwd2(z[3], z[7], w1);
		_square4x2r(z, w2, wi2r);
		_bck2(z[0], z[4], wi1); _bck2(z[2], z[6], wi1); _bck2(z[1], z[5], wi1); _bck2(z[3], z[7], wi1);
	}

	static void _mul2x4(Zp z[8], const Zp zp[8], const Zp w2[2])
	{
		_mul2(z[0], z[1], zp[0], zp[1], w2[0]); _mul2n(z[2], z[3], zp[2], zp[3], w2[0]); _mul2(z[4], z[5], zp[4], zp[5], w2[1]); _mul2n(z[6], z[7], zp[6], zp[7], w2[1]);
	}

	static void _fwd4x2(Zp zp[8], const Zp w2[2])
	{
		_fwd2(zp[0], zp[2], w2[0]); _fwd2(zp[1], zp[3], w2[0]); _fwd2(zp[4], zp[6], w2[1]); _fwd2(zp[5], zp[7], w2[1]);
	}

	static void _mul4x2r(Zp z[8], const Zp zp[8], const Zp w2[2], const Zp wi2r[2])
	{
		_fwd2(z[0], z[2], w2[0]); _fwd2(z[1], z[3], w2[0]); _fwd2(z[4], z[6], w2[1]); _fwd2(z[5], z[7], w2[1]);
		_mul2x4(z, zp, w2);
		_bck2(z[0], z[2], wi2r[1]); _bck2(z[1], z[3], wi2r[1]); _bck2(z[4], z[6], wi2r[0]); _bck2(z[5], z[7], wi2r[0]);
	}

	static void _fwd8(Zp zp[8], const Zp w1, const Zp w2[2])
	{
		_fwd2(zp[0], zp[4], w1); _fwd2(zp[2], zp[6], w1); _fwd2(zp[1], zp[5], w1); _fwd2(zp[3], zp[7], w1);
		_fwd4x2(zp, w2);
	}

	static void _mul8r(Zp z[8], const Zp zp[8], const Zp w1, const Zp  wi1, const Zp w2[2], const Zp wi2r[2])
	{
		_fwd2(z[0], z[4], w1); _fwd2(z[2], z[6], w1); _fwd2(z[1], z[5], w1); _fwd2(z[3], z[7], w1);
		_mul4x2r(z, zp, w2, wi2r);
		_bck2(z[0], z[4], wi1); _bck2(z[2], z[6], wi1); _bck2(z[1], z[5], wi1); _bck2(z[3], z[7], wi1);
	}

public:
	Zp() {}
	explicit Zp(const uint32 n) : ZP(n) {}

	int32 get_int() const { return (_n >= P / 2) ? int32(_n - P) : int32(_n); }
	Zp & set_int(const int32 i) { _n = (i < 0) ? (uint32(i) + P) : uint32(i); return *this; }

	Zp add(const Zp & rhs) const { return Zp(_add(_n, rhs._n)); }
	Zp sub(const Zp & rhs) const { return Zp(_sub(_n, rhs._n)); }
	Zp mul(const Zp & rhs) const { return Zp(_mul(_n, rhs._n)); }
	Zp sqr() const { return mul(*this); }

	// Conversion into / out of Montgomery form
	Zp toMonty() const { return Zp(_mul(_n, RSQ)); }
	// Zp fromMonty() const { return Zp(_mul(_n, 1)); }

	Zp pow(const size_t e) const
	{
		if (e == 0) return Zp(R);	// MF of one is R
		Zp r = Zp(R), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		r = r.mul(y);
		return r;
	}

	static const Zp primroot_n(const uint32 n) { return Zp(H).pow((P - 1) / n); }
	static Zp norm(const uint32 n) { return Zp(P - (P - 1) / n); }

private:
	template<size_t VSIZE>
	static void forward8(Zp * const z, const Zp * const w, const int lm, const size_t s, const int ln_8)
	{
		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t vm = VSIZE << lm, j = (id / VSIZE) >> lm, k = 7 * (id & ~(vm - 1)) + id;

			const Zp w1 = w[1 * (s + j)];
			Zp w2[2]; _load(2, w2, &w[2 * (s + j)], 1);
			Zp w4[4]; _load(4, w4, &w[4 * (s + j)], 1);

			Zp zl[8]; _load(8, zl, &z[k], vm);
			_forward8(zl, w1, w2, w4);
			_store(8, &z[k], vm, zl);
		}
	}

	template<size_t VSIZE>
	static void backward8(Zp * const z, const Zp * const w, const int lm, const size_t s, const int ln_8)
	{
		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t vm = VSIZE << lm, j = (id / VSIZE) >> lm, k = 7 * (id & ~(vm - 1)) + id;

			const size_t ji = s - j - 1;
			const Zp wi1 = w[1 * (s + ji)];
			Zp wi2r[2]; _load(2, wi2r, &w[2 * (s + ji)], 1);
			Zp wi4r[4]; _load(4, wi4r, &w[4 * (s + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[k], vm);
			_backward8r(zl, wi1, wi2r, wi4r);
			_store(8, &z[k], vm, zl);
		}
	}

	template<size_t VSIZE>
	static void forward8_0(Zp * const z, const Zp * const w, const int ln_8)
	{
		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t vn_8 = VSIZE << ln_8, k = id;

			Zp w2[2]; _load(2, w2, &w[2], 1);
			Zp w4[4]; _load(4, w4, &w[4], 1);

			Zp zl[8]; _load(8, zl, &z[k], vn_8);
			_forward8_0(zl, w2, w4);
			_store(8, &z[k], vn_8, zl);
		}
	}

	template<size_t VSIZE>
	static void square2x4(Zp * const z, const Zp * const w, const int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			_square2x4(zl, w2);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void square4x2(Zp * const z, const Zp * const w, const int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			const size_t ji = n_8 - j - 1;
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2r[2]; _load(2, wi2r, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			_square4x2r(zl, w2, wi2r);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void square8(Zp * const z, const Zp * const w, const int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			const size_t ji = n_8 - j - 1;
			const Zp w1 = w[1 * (n_8 + j)], wi1 = w[1 * (n_8 + ji)];
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2r[2]; _load(2, wi2r, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			_square8r(zl, w1, wi1, w2, wi2r);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void mul2x4(Zp * const z, const Zp * const zp, const Zp * const w, const int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			Zp w2[2]; _load(2, w2, &w[2 * n_8 + 2 * j], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			_mul2x4(zl, zpl, w2);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void mul2x4_mask(Zp * const z, const Zp * const zp, const Zp * const w, const int ln_8, const uint8_t mask)
	{
		const size_t n_8 = size_t(1) << ln_8;
		const Zp one = Zp(1).toMonty(), zero = Zp(0);
		const Zp z1l[8] = { one, zero, one, zero, one, zero, one, zero };

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			Zp w2[2]; _load(2, w2, &w[2 * n_8 + 2 * j], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			const Zp * const zml = ((mask & (uint8_t(1) << (id % VSIZE))) != 0) ? zpl : z1l;
			_mul2x4(zl, zml, w2);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void fwd4x2(Zp * const zp, const Zp * const w, const int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			_fwd4x2(zpl, w2);
			_store(8, &zp[k], VSIZE, zpl);
		}
	}

	template<size_t VSIZE>
	static void mul4x2(Zp * const z, const Zp * const zp, const Zp * const w, const int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			const size_t ji = n_8 - j - 1;
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2r[2]; _load(2, wi2r, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			_mul4x2r(zl, zpl, w2, wi2r);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void mul4x2_mask(Zp * const z, const Zp * const zp, const Zp * const w, const int ln_8, const uint8_t mask)
	{
		const size_t n_8 = size_t(1) << ln_8;
		const Zp one = Zp(1).toMonty(), zero = Zp(0);
		const Zp z1l[8] = { one, zero, one, zero, one, zero, one, zero };

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			const size_t ji = n_8 - j - 1;
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2r[2]; _load(2, wi2r, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			const Zp * const zml = ((mask & (uint8_t(1) << (id % VSIZE))) != 0) ? zpl : z1l;
			_mul4x2r(zl, zml, w2, wi2r);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void fwd8(Zp * const zp, const Zp * const w, const int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			const Zp w1 = w[1 * (n_8 + j)];
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			_fwd8(zpl, w1, w2);
			_store(8, &zp[k], VSIZE, zpl);
		}
	}

	template<size_t VSIZE>
	static void mul8(Zp * const z, const Zp * const zp, const Zp * const w, int ln_8)
	{
		const size_t n_8 = size_t(1) << ln_8;

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			const size_t ji = n_8 - j - 1;
			const Zp w1 = w[1 * (n_8 + j)], wi1 = w[1 * (n_8 + ji)];
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2r[2]; _load(2, wi2r, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			_mul8r(zl, zpl, w1, wi1, w2, wi2r);
			_store(8, &z[k], VSIZE, zl);
		}
	}

	template<size_t VSIZE>
	static void mul8_mask(Zp * const z, const Zp * const zp, const Zp * const w, const int ln_8, const uint8_t mask)
	{
		const size_t n_8 = size_t(1) << ln_8;
		const Zp one = Zp(1).toMonty(), zero = Zp(0);
		const Zp z1l[8] = { one, zero, one, zero, one, zero, one, zero };

		for (size_t id = 0; id < (VSIZE << ln_8); ++id)
		{
			const size_t j = id / VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			const size_t ji = n_8 - j - 1;
			const Zp w1 = w[1 * (n_8 + j)], wi1 = w[1 * (n_8 + ji)];
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2r[2]; _load(2, wi2r, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[k], VSIZE);
			Zp zpl[8]; _load(8, zpl, &zp[k], VSIZE);
			const Zp * const zml = ((mask & (uint8_t(1) << (id % VSIZE))) != 0) ? zpl : z1l;
			_mul8r(zl, zml, w1, wi1, w2, wi2r);
			_store(8, &z[k], VSIZE, zl);
		}
	}

public:
	template<size_t VSIZE>
	static void forward0(Zp * const z, const Zp * const w, const int ln_8)
	{
		forward8_0<VSIZE>(z, w, ln_8);
	}

	template<size_t VSIZE>
	static int forward(Zp * const z, const Zp * const w, const int ln_8)
	{
		int lm = ln_8;
		for (size_t s = 8; lm > 3; lm -= 3, s *= 8) forward8<VSIZE>(z, w, lm - 3, s, ln_8);
		return lm;
	}

	template<size_t VSIZE>
	static void backward(Zp * const z, const Zp * const w, const int lm0, const int ln_8)
	{
		int lm = lm0;
		for (size_t s = size_t(1) << (ln_8 - lm0); s >= 1; lm += 3, s /= 8) backward8<VSIZE>(z, w, lm, s, ln_8);
	}

	template<size_t VSIZE>
	static void square(Zp * const z, const Zp * const w, const int lm0, const int ln_8)
	{
		if (lm0 == 3) square8<VSIZE>(z, w, ln_8);
		else if (lm0 == 2) square4x2<VSIZE>(z, w, ln_8);
		else if (lm0 == 1) square2x4<VSIZE>(z, w, ln_8);
	}

	template<size_t VSIZE>
	static void fwd(Zp * const zp, const Zp * const w, const int lm0, const int ln_8)
	{
		if (lm0 == 3) fwd8<VSIZE>(zp, w, ln_8);
		else if (lm0 == 2) fwd4x2<VSIZE>(zp, w, ln_8);
	}

	template<size_t VSIZE>
	static void mul(Zp * const z, const Zp * const zp, const Zp * const w, const int lm0, const int ln_8)
	{
		if (lm0 == 3) mul8<VSIZE>(z, zp, w, ln_8);
		else if (lm0 == 2) mul4x2<VSIZE>(z, zp, w, ln_8);
		else if (lm0 == 1) mul2x4<VSIZE>(z, zp, w, ln_8);
	}

	template<size_t VSIZE>
	static void mul_mask(Zp * const z, const Zp * const zp, const Zp * const w, const int lm0, const int ln_8, const uint8_t mask)
	{
		if (lm0 == 3) mul8_mask<VSIZE>(z, zp, w, ln_8, mask);
		else if (lm0 == 2) mul4x2_mask<VSIZE>(z, zp, w, ln_8, mask);
		else if (lm0 == 1) mul2x4_mask<VSIZE>(z, zp, w, ln_8, mask);
	}
};

typedef Zp<P1S, Q1S, R1S, RSQ1S, H1S, MFIM1S, SQRTI1S, ISQRTI1S> Zp1;
typedef Zp<P2S, Q2S, R2S, RSQ2S, H2S, MFIM2S, SQRTI2S, ISQRTI2S> Zp2;
typedef Zp<P3S, Q3S, R3S, RSQ3S, H3S, MFIM3S, SQRTI3S, ISQRTI3S> Zp3;

template<size_t VSIZE, bool IS32>
class transformGPU : public transform
{
	using ZP1 = ZPT<IS32 ? P1U : P1S, IS32 ? Q1U : Q1S, IS32 ? R1U : R1S, IS32 ? H1U : H1S>;
	using ZP2 = ZPT<IS32 ? P2U : P2S, IS32 ? Q2U : Q2S, IS32 ? R2U : R2S, IS32 ? H2U : H2S>;
	using ZP3 = ZPT<IS32 ? P3U : P3S, IS32 ? Q3U : Q3S, IS32 ? R3U : R3S, IS32 ? H3U : H3S>;

private:
	const size_t _num_regs;
	const int _lsize;
	const size_t _size;
	const Zp1 _norm1;
	const Zp2 _norm2;
	const Zp3 _norm3;
	ZP * const _z;
	ZP * const _zp;
	Zp1 * const _w1;
	Zp2 * const _w2;
	Zp3 * const _w3;
	int64_t * const _c;
	engine<VSIZE, IS32> * _engine = nullptr;

public:
	transformGPU(const UInt32_8 & b, const uint32_t n, const size_t num_regs, const size_t device_id,
				 const bool is_boinc, const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id)
				: transform(b, n, EKind::GPU), _num_regs(num_regs), _lsize(int(n)), _size(size_t(1) << n),
				_norm1(Zp1::norm(uint32(_size / 2))), _norm2(Zp2::norm(uint32(_size / 2))), _norm3(Zp3::norm(uint32(_size / 2))),
				_z(new ZP[3 * VSIZE * num_regs * _size]), _zp(new ZP[3 * VSIZE * _size]),
				_w1(new Zp1[_size / 2]), _w2(new Zp2[_size / 2]), _w3(new Zp3[_size / 2]),
				_c(new int64_t[VSIZE * _size / 8])
	{
		const size_t size = _size;

		const bool is_boinc_platform = is_boinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_engine = new engine<VSIZE, IS32>(eng_platform, is_boinc_platform ? 0 : device_id, static_cast<int>(n), is_boinc, num_regs);
		set_type(_engine->getType());

		std::ostringstream src;

		src << "#define N_SZ\t" << size << "u" << std::endl;
		src << "#define LN_SZ\t" << n << std::endl;
		src << "#define VSIZE\t" << VSIZE << std::endl;
		if (IS32) src << "#define IS32\t" << 1 << std::endl;

		src << "#define P1\t" << (IS32 ? P1U : P1S) << "u" << std::endl;
		src << "#define Q1\t" << (IS32 ? Q1U : Q1S) << "u" << std::endl;
		src << "#define RSQ1\t" << (IS32 ? RSQ1U : RSQ1S) << "u" << std::endl;
		src << "#define IM1\t" << (IS32 ? IM1U : IM1S) << "u" << std::endl;
		src << "#define MFIM1\t" << (IS32 ? MFIM1U : MFIM1S) << "u" << std::endl;
		src << "#define SQRTI1\t" << (IS32 ? SQRTI1U : SQRTI1S) << "u" << std::endl;
		src << "#define ISQRTI1\t" << (IS32 ? ISQRTI1U : ISQRTI1S) << "u" << std::endl;

		src << "#define P2\t" << (IS32 ? P2U : P2S) << "u" << std::endl;
		src << "#define Q2\t" << (IS32 ? Q2U : Q2S) << "u" << std::endl;
		src << "#define RSQ2\t" << (IS32 ? RSQ2U : RSQ2S) << "u" << std::endl;
		src << "#define IM2\t" << (IS32 ? IM2U : IM2S) << "u" << std::endl;
		src << "#define MFIM2\t" << (IS32 ? MFIM2U : MFIM2S) << "u" << std::endl;
		src << "#define SQRTI2\t" << (IS32 ? SQRTI2U : SQRTI2S) << "u" << std::endl;
		src << "#define ISQRTI2\t" << (IS32 ? ISQRTI2U : ISQRTI2S) << "u" << std::endl;

		src << "#define P3\t" << (IS32 ? P3U : P3S) << "u" << std::endl;
		src << "#define Q3\t" << (IS32 ? Q3U : Q3S) << "u" << std::endl;
		src << "#define RSQ3\t" << (IS32 ? RSQ3U : RSQ3S) << "u" << std::endl;
		src << "#define IM3\t" << (IS32 ? IM3U : IM3S) << "u" << std::endl;
		src << "#define MFIM3\t" << (IS32 ? MFIM3U : MFIM3S) << "u" << std::endl;
		src << "#define SQRTI3\t" << (IS32 ? SQRTI3U : SQRTI3S) << "u" << std::endl;
		src << "#define ISQRTI3\t" << (IS32 ? ISQRTI3U : ISQRTI3S) << "u" << std::endl;

		src << "#define INVP2_P1\t" << (IS32 ? INVP2_P1U : INVP2_P1S) << "u" << std::endl;
		src << "#define INVP3_P1\t" << (IS32 ? INVP3_P1U : INVP3_P1S) << "u" << std::endl;
		src << "#define INVP3_P2\t" << (IS32 ? INVP3_P2U : INVP3_P2S) << "u" << std::endl;

		src << "#define P1P2P3L\t" << (IS32 ? P1P2P3LU : P1P2P3LS) << "u" << std::endl;
		src << "#define P1P2P3H\t" << (IS32 ? P1P2P3HU : P1P2P3HS) << "ul" << std::endl;
		src << "#define P1P2P3_2L\t" << (IS32 ? P1P2P3_2LU : P1P2P3_2LS) << "u" << std::endl;
		src << "#define P1P2P3_2H\t" << (IS32 ? P1P2P3_2HU : P1P2P3_2HS) << "ul" << std::endl;

		// Not converted into Montgomery form such that output is converted out of MF
		src << "#define NORM1\t" << ZP1::norm(uint32(size / 2)).get() << "u" << std::endl;
		src << "#define NORM2\t" << ZP2::norm(uint32(size / 2)).get() << "u" << std::endl;
		src << "#define NORM3\t" << ZP3::norm(uint32(size / 2)).get() << "u" << std::endl;

		src << "#define W_SZ\t" << size / 2 << "u" << std::endl;

		src << "#define CARRY_WG_SZ\t" << _engine->get_carry_workgroup_size() << "u" << std::endl;

// std::cout << src.str() << std::endl;

		if (is_boinc || !_engine->readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine->loadProgram(src.str());
		_engine->alloc_memory();
		_engine->create_kernels();

		Zp1 * const w1 = _w1;
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const Zp1 r_s = Zp1::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				w1[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		Zp2 * const w2 = _w2;
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const Zp2 r_s = Zp2::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				w2[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		Zp3 * const w3 = _w3;
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const Zp3 r_s = Zp3::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				w3[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		_engine->write_memory_w(w1, 0);
		_engine->write_memory_w(w2, 1);
		_engine->write_memory_w(w3, 2);

		uint32_t bu[VSIZE], b_inv[VSIZE]; int b_s[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i)
		{
			const uint32_t bi = b[i];
			const int s = 31 - __builtin_clz(bi) - 1;
			bu[i] = bi;
			b_inv[i] = static_cast<uint32_t>((static_cast<uint64_t>(1) << (s + 32)) / bi);
			b_s[i] = s;
		}

		_engine->write_memory_b(bu, b_inv, b_s);
	}

	virtual ~transformGPU()
	{
		_engine->release_kernels();
		_engine->release_memory();
		_engine->clearProgram();
		delete _engine;

		delete[] _z;
		delete[] _zp;
		delete[] _w1;
		delete[] _w2;
		delete[] _w3;
		delete[] _c;
	}

private:
	static __int128_t garner3(const Zp1 r1, const Zp2 r2, const Zp3 r3)
	{
		// Montgomery form of 1 / Pi (mod Pj)
		const uint32 mfInvP3_P1 = 608773230u, mfInvP2_P1 = 2130706177u, mfInvP3_P2 = 1409286102u;
		const uint64 P2P3 = P2S * uint64(P3S);
		const __uint128_t P1P2P3 = P1S * __uint128_t(P2P3);

		const Zp1 u13 = r1.sub(Zp1(r3.get())).mul(Zp1(mfInvP3_P1));	// P3 < P1
		const Zp2 u23 = r2.sub(Zp2(r3.get())).mul(Zp2(mfInvP3_P2));	// P3 < P2
		const Zp1 u123 = u13.sub(Zp1(u23.get())).mul(Zp1(mfInvP2_P1));	// P3 < P1
		const __uint128_t n = __uint128_t(P2P3) * u123.get() + (u23.get() * uint64(P3S) + r3.get());
		return (n > P1P2P3 / 2) ? __int128_t(n - P1P2P3) : __int128_t(n);
	}

	void carry(const uint32_t dup)
	{
		const size_t vsize = VSIZE * _size, n = _size;
		Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * vsize]);
		Zp2 * const z2 = reinterpret_cast<Zp2 *>(&_z[1 * vsize]);
		Zp3 * const z3 = reinterpret_cast<Zp3 *>(&_z[2 * vsize]);
		int64_t * const c = _c;

		const Zp1 norm1 = _norm1; const Zp2 norm2 = _norm2;	const Zp3 norm3 = _norm3;

		for (size_t id = 0; id < VSIZE * n / 8; ++id)
		{
			const size_t i = id % VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;
			int64_t f = 0;
			for (size_t j = 0; j < 8; ++j)
			{
				const Zp1 u1 = z1[k + j * VSIZE].mul(norm1);
				const Zp2 u2 = z2[k + j * VSIZE].mul(norm2);
				const Zp3 u3 = z3[k + j * VSIZE].mul(norm3);
				__int128_t l = garner3(u1, u2, u3);
				if ((dup & (1u << i)) != 0) l += l;
				l += f;
				const int32 base = static_cast<int32>(get_b()[i]);
				const int64_t r = int64_t(l / base);
				const int32 ri = int32(l - __int128_t(r) * base);
				f = r;
				z1[k + j * VSIZE].set_int(ri);
				z2[k + j * VSIZE].set_int(ri);
				z3[k + j * VSIZE].set_int(ri);
			}

			const size_t vid = ((id / VSIZE) + 1) % (n / 8);
			c[vid * VSIZE + i] = (vid == 0) ? -f : f;
		}

		for (size_t id = 0; id < VSIZE * n / 8; ++id)
		{
			const size_t i = id % VSIZE, k = 7 * (id & ~(VSIZE - 1)) + id;

			int64_t f = c[id];
			for (size_t j = 0; j < 7; ++j)
			{
				if (f != 0)
				{
					f += z1[k + j * VSIZE].get_int();
					const int32 base = static_cast<int32>(get_b()[i]);
					const int64_t r = f / base;
					const int32 ri = int32(f - r * base);
					f = r;
					z1[k + j * VSIZE].set_int(ri);
					z2[k + j * VSIZE].set_int(ri);
					z3[k + j * VSIZE].set_int(ri);
				}
			}
			if (f != 0)
			{
				f += z1[k + 7 * VSIZE].get_int();
				const int32 ri = int32(f);
				z1[k + 7 * VSIZE].set_int(ri);
				z2[k + 7 * VSIZE].set_int(ri);
				z3[k + 7 * VSIZE].set_int(ri);
			}
		}
	}

protected:
	void getZi(Int32_8 * const zi) const override
	{
		const size_t size = _size;
		const Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * VSIZE * _size]);

		for (size_t k = 0; k < size; ++k)
		{
			int32 d[VSIZE]; for (size_t j = 0; j < VSIZE; ++j) d[j] = z1[VSIZE * k + j].get_int();
			zi[k] = Int32_8(d);
		}
	}

	void setZi(const Int32_8 * const zi) override
	{
		const size_t size = _size, vsize = VSIZE * size;
		Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * vsize]);
		Zp2 * const z2 = reinterpret_cast<Zp2 *>(&_z[1 * vsize]);
		Zp3 * const z3 = reinterpret_cast<Zp3 *>(&_z[2 * vsize]);

		for (size_t k = 0; k < size; ++k)
		{
			const Int32_8 zk = zi[k];
			for (size_t j = 0; j < VSIZE; ++j)
			{
				const int32 d = zk[j];
				z1[VSIZE * k + j].set_int(d);
				z2[VSIZE * k + j].set_int(d);
				z3[VSIZE * k + j].set_int(d);
			}
		}
	}

public:
	void set(const uint32_t a) override
	{
		const size_t size = _size, vsize = VSIZE * size;

		Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * vsize]);
		for (size_t j = 0; j < VSIZE; ++j) z1[j] = Zp1(a);
		for (size_t k = 1; k < size; ++k) for (size_t j = 0; j < VSIZE; ++j) z1[VSIZE * k + j] = Zp1(0);

		Zp2 * const z2 = reinterpret_cast<Zp2 *>(&_z[1 * vsize]);
		for (size_t j = 0; j < VSIZE; ++j) z2[j] = Zp2(a);
		for (size_t k = 1; k < size; ++k) for (size_t j = 0; j < VSIZE; ++j) z2[VSIZE * k + j] = Zp2(0);

		Zp3 * const z3 = reinterpret_cast<Zp3 *>(&_z[2 * vsize]);
		for (size_t j = 0; j < VSIZE; ++j) z3[j] = Zp3(a);
		for (size_t k = 1; k < size; ++k) for (size_t j = 0; j < VSIZE; ++j) z3[VSIZE * k + j] = Zp3(0);
	}

	void square_dup(const uint32_t dup) override
	{
		const int ln_8 = _lsize - 3;

		// const size_t vsize = VSIZE * _size;

		// Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * vsize]);
		// Zp2 * const z2 = reinterpret_cast<Zp2 *>(&_z[1 * vsize]);
		// Zp3 * const z3 = reinterpret_cast<Zp3 *>(&_z[2 * vsize]);

		// const Zp1 * const w1 = _w1;
		// const Zp2 * const w2 = _w2;
		// const Zp3 * const w3 = _w3;

		// Zp1::forward0<VSIZE>(z1, w1, ln_8);
		// Zp2::forward0<VSIZE>(z2, w2, ln_8);
		// Zp3::forward0<VSIZE>(z3, w3, ln_8);

		// const int lm0 = Zp1::forward<VSIZE>(z1, w1, ln_8);
		// Zp2::forward<VSIZE>(z2, w2, ln_8);
		// Zp3::forward<VSIZE>(z3, w3, ln_8);

		// Zp1::square<VSIZE>(z1, w1, lm0, ln_8);
		// Zp2::square<VSIZE>(z2, w2, lm0, ln_8);
		// Zp3::square<VSIZE>(z3, w3, lm0, ln_8);

		// Zp1::backward<VSIZE>(z1, w1, lm0, ln_8);
		// Zp2::backward<VSIZE>(z2, w2, lm0, ln_8);
		// Zp3::backward<VSIZE>(z3, w3, lm0, ln_8);

		// carry(dup);

		_engine->write_memory_z(_z);
		_engine->forward8_0();

		int lm = ln_8;
		for (size_t s = 8; lm > 3; lm -= 3, s *= 8) _engine->forward8(lm - 3, s);

		if (lm == 3) _engine->square8();
		else if (lm == 2) _engine->square4x2();
		else if (lm == 1) _engine->square2x4();

		for (size_t s = size_t(1) << (ln_8 - lm); s >= 1; lm += 3, s /= 8) _engine->backward8(lm, s);

		_engine->carry1(dup);
		_engine->carry2();
		_engine->read_memory_z(_z);
	}

	void init_multiplicand(const size_t src) override
	{
		const size_t vsize = VSIZE * _size;
		const int ln_8 = _lsize - 3;

		const Zp1 * const z1_src = reinterpret_cast<Zp1 *>(&_z[(3 * src + 0) * vsize]);
		Zp1 * const zp1 = reinterpret_cast<Zp1 *>(&_zp[0 * vsize]);
		for (size_t k = 0; k < vsize; ++k) zp1[k] = z1_src[k];
		const Zp2 * const z2_src = reinterpret_cast<Zp2 *>(&_z[(3 * src + 1) * vsize]);
		Zp2 * const zp2 = reinterpret_cast<Zp2 *>(&_zp[1 * vsize]);
		for (size_t k = 0; k < vsize; ++k) zp2[k] = z2_src[k];
		const Zp3 * const z3_src = reinterpret_cast<Zp3 *>(&_z[(3 * src + 2) * vsize]);
		Zp3 * const zp3 = reinterpret_cast<Zp3 *>(&_zp[2 * vsize]);
		for (size_t k = 0; k < vsize; ++k) zp3[k] = z3_src[k];

		const Zp1 * const w1 = _w1;
		const Zp2 * const w2 = _w2;
		const Zp3 * const w3 = _w3;

		Zp1::forward0<VSIZE>(zp1, w1, ln_8);
		Zp2::forward0<VSIZE>(zp2, w2, ln_8);
		Zp3::forward0<VSIZE>(zp3, w3, ln_8);

		const int lm0 = Zp1::forward<VSIZE>(zp1, w1, ln_8);
		Zp2::forward<VSIZE>(zp2, w2, ln_8);
		Zp3::forward<VSIZE>(zp3, w3, ln_8);

		Zp1::fwd<VSIZE>(zp1, w1, lm0, ln_8);
		Zp2::fwd<VSIZE>(zp2, w2, lm0, ln_8);
		Zp3::fwd<VSIZE>(zp3, w3, lm0, ln_8);
	}

	void mul() override
	{
		const size_t vsize = VSIZE * _size;
		const int ln_8 = _lsize - 3;

		Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * vsize]);
		Zp2 * const z2 = reinterpret_cast<Zp2 *>(&_z[1 * vsize]);
		Zp3 * const z3 = reinterpret_cast<Zp3 *>(&_z[2 * vsize]);

		const Zp1 * const w1 = _w1;
		const Zp2 * const w2 = _w2;
		const Zp3 * const w3 = _w3;

		Zp1::forward0<VSIZE>(z1, w1, ln_8);
		Zp2::forward0<VSIZE>(z2, w2, ln_8);
		Zp3::forward0<VSIZE>(z3, w3, ln_8);

		const int lm0 = Zp1::forward<VSIZE>(z1, w1, ln_8);
		Zp2::forward<VSIZE>(z2, w2, ln_8);
		Zp3::forward<VSIZE>(z3, w3, ln_8);

		Zp1 * const zp1 = reinterpret_cast<Zp1 *>(&_zp[0 * vsize]);
		Zp2 * const zp2 = reinterpret_cast<Zp2 *>(&_zp[1 * vsize]);
		Zp3 * const zp3 = reinterpret_cast<Zp3 *>(&_zp[2 * vsize]);

		Zp1::mul<VSIZE>(z1, zp1, w1, lm0, ln_8);
		Zp2::mul<VSIZE>(z2, zp2, w2, lm0, ln_8);
		Zp3::mul<VSIZE>(z3, zp3, w3, lm0, ln_8);

		Zp1::backward<VSIZE>(z1, w1, lm0, ln_8);
		Zp2::backward<VSIZE>(z2, w2, lm0, ln_8);
		Zp3::backward<VSIZE>(z3, w3, lm0, ln_8);

		carry(0);
	}

	void mul_mask(const uint8_t mask) override
	{
		const size_t vsize = VSIZE * _size;
		const int ln_8 = _lsize - 3;

		Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * vsize]);
		Zp2 * const z2 = reinterpret_cast<Zp2 *>(&_z[1 * vsize]);
		Zp3 * const z3 = reinterpret_cast<Zp3 *>(&_z[2 * vsize]);

		const Zp1 * const w1 = _w1;
		const Zp2 * const w2 = _w2;
		const Zp3 * const w3 = _w3;

		Zp1::forward0<VSIZE>(z1, w1, ln_8);
		Zp2::forward0<VSIZE>(z2, w2, ln_8);
		Zp3::forward0<VSIZE>(z3, w3, ln_8);

		const int lm0 = Zp1::forward<VSIZE>(z1, w1, ln_8);
		Zp2::forward<VSIZE>(z2, w2, ln_8);
		Zp3::forward<VSIZE>(z3, w3, ln_8);

		Zp1 * const zp1 = reinterpret_cast<Zp1 *>(&_zp[0 * vsize]);
		Zp2 * const zp2 = reinterpret_cast<Zp2 *>(&_zp[1 * vsize]);
		Zp3 * const zp3 = reinterpret_cast<Zp3 *>(&_zp[2 * vsize]);

		Zp1::mul_mask<VSIZE>(z1, zp1, w1, lm0, ln_8, mask);
		Zp2::mul_mask<VSIZE>(z2, zp2, w2, lm0, ln_8, mask);
		Zp3::mul_mask<VSIZE>(z3, zp3, w3, lm0, ln_8, mask);

		Zp1::backward<VSIZE>(z1, w1, lm0, ln_8);
		Zp2::backward<VSIZE>(z2, w2, lm0, ln_8);
		Zp3::backward<VSIZE>(z3, w3, lm0, ln_8);

		carry(0);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const size_t vsize = VSIZE * _size;

		const Zp1 * const z1_src = reinterpret_cast<Zp1 *>(&_z[(3 * src + 0) * vsize]);
		Zp1 * const z1_dst =  reinterpret_cast<Zp1 *>(&_z[(3 * dst + 0) * vsize]);
		for (size_t k = 0; k < vsize; ++k) z1_dst[k] = z1_src[k];

		const Zp2 * const z2_src = reinterpret_cast<Zp2 *>(&_z[(3 * src + 1) * vsize]);
		Zp2 * const z2_dst =  reinterpret_cast<Zp2 *>(&_z[(3 * dst + 1) * vsize]);
		for (size_t k = 0; k < vsize; ++k) z2_dst[k] = z2_src[k];

		const Zp3 * const z3_src = reinterpret_cast<Zp3 *>(&_z[(3 * src + 2) * vsize]);
		Zp3 * const z3_dst =  reinterpret_cast<Zp3 *>(&_z[(3 * dst + 2) * vsize]);
		for (size_t k = 0; k < vsize; ++k) z3_dst[k] = z3_src[k];
	}

	void copy_mask(const size_t dst, const size_t src, const uint8_t mask) const override
	{
		if (mask == 0) return;
		if (mask == uint8_t(-1)) { copy(dst, src); return; }

		pio::print("copy_mask");

		const size_t size = _size, vsize = VSIZE * size;

		const Zp1 * const z1_src = reinterpret_cast<Zp1 *>(&_z[(3 * src + 0) * vsize]);
		Zp1 * const z1_dst =  reinterpret_cast<Zp1 *>(&_z[(3 * dst + 0) * vsize]);
		for (size_t k = 0; k < size; ++k)
		{
			for (size_t j = 0; j < VSIZE; ++j) if ((mask & (uint8_t(1) << j)) != 0) z1_dst[VSIZE * k + j] = z1_src[VSIZE * k + j];
		} 

		const Zp2 * const z2_src = reinterpret_cast<Zp2 *>(&_z[(3 * src + 1) * vsize]);
		Zp2 * const z2_dst =  reinterpret_cast<Zp2 *>(&_z[(3 * dst + 1) * vsize]);
		for (size_t k = 0; k < size; ++k)
		{
			for (size_t j = 0; j < VSIZE; ++j) if ((mask & (uint8_t(1) << j)) != 0) z2_dst[VSIZE * k + j] = z2_src[VSIZE * k + j];
		} 

		const Zp3 * const z3_src = reinterpret_cast<Zp3 *>(&_z[(3 * src + 2) * vsize]);
		Zp3 * const z3_dst =  reinterpret_cast<Zp3 *>(&_z[(3 * dst + 2) * vsize]);
		for (size_t k = 0; k < size; ++k)
		{
			for (size_t j = 0; j < VSIZE; ++j) if ((mask & (uint8_t(1) << j)) != 0) z3_dst[VSIZE * k + j] = z3_src[VSIZE * k + j];
		} 
	}

	void power(const size_t src, const uint32_t e) override { _power(src, e); }
	void power_vec(const size_t src, const UInt32_8 & e) override { _power_vec(src, e); }

	bool read_checkpoint(file & cFile) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;
		if (!cFile.read(reinterpret_cast<char *>(_z), 3 * VSIZE * _num_regs * _size * sizeof(ZP))) return false;
		return true;
	}

	void save_checkpoint(file & cFile) const override
	{
		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;
		if (!cFile.write(reinterpret_cast<const char *>(_z), 3 * VSIZE * _num_regs * _size * sizeof(ZP))) return;
	}

	size_t get_data_size() const override { return (3 * VSIZE * ((_num_regs + 1) * _size + _size / 2)) * sizeof(ZP); }
	size_t get_cache_size() const override { return (3 * VSIZE * (_size + _size / 2)) * sizeof(ZP); }
	double get_error() const override { return 0; }

	void is_one(bool b[8], UInt64_8 & res64) const override { _is_one(b, res64); }
	UInt64_8 gethash64() const override { return _gethash64(); }
	UInt32_8 gethash32() const override { return _gethash32(); }

	void cosmic_ray() override
	{
		const size_t vsize = VSIZE * _size;
		Zp1 * const z1 = reinterpret_cast<Zp1 *>(&_z[0 * vsize]);
		Zp1 & z = z1[vsize / 2];
		z = z.add(Zp1(1));
	}
};
