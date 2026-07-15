/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "transform.h"
#include "ocl.h"

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

#define	P1		(127 * (uint32(1) << 24) + 1)
#define	Q1		2164260865u		// p * q = 1 (mod 2^32)
#define	R1		33554430u		// 2^32 mod p
#define	RSQ1	402124772u		// (2^32)^2 mod p
#define	H1		100663290u		// Montgomery form of the primitive root 3
#define	IM1		1930170389u		// MF of MF of I = 3^{(p - 1)/4} to convert input into MF
#define	SQRTI1	1626730317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1	856006302u		// MF of i * sqrt(i)

#define	P2		(63 * (uint32(1) << 25) + 1)
#define	Q2		2181038081u
#define	R2		67108862u
#define	RSQ2	2111798781u
#define	H2		335544310u		// MF of the primitive root 5
#define	IM2		1036950657u
#define	SQRTI2	338852760u
#define	ISQRTI2	1090446030u

#define	P3		(15 * (uint32(1) << 27) + 1)
#define	Q3		2281701377u
#define	R3		268435454u
#define	RSQ3	1172168163u
#define	H3		268435390u		// MF of the primitive root 31
#define	IM3		734725699u
#define	SQRTI3	1032137103u
#define	ISQRTI3	1964242958u

template<uint32 P, uint32 Q, uint32 R, uint32 RSQ, uint32 H, uint32 IM, uint32 SQRTI, uint32 ISQRTI>
class Zp
{
private:
	uint32 _n;

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
	static void _loadr(const size_t n, Zp * const zl, const Zp * const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[n - l - 1] = z[l * s]; }
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

	static void _backward8(Zp z[8], const Zp wi1, const Zp wi2[2], const Zp wi4[4])
	{
		_bck2(z[0], z[1], wi4[0]); _bck2(z[2], z[3], wi4[1]); _bck2(z[4], z[5], wi4[2]); _bck2(z[6], z[7], wi4[3]);
		_bck2(z[0], z[2], wi2[0]); _bck2(z[1], z[3], wi2[0]); _bck2(z[4], z[6], wi2[1]); _bck2(z[5], z[7], wi2[1]);
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

	static void _square4x2(Zp z[8], const Zp w2[2], const Zp wi2[2])
	{
		_fwd2(z[0], z[2], w2[0]); _fwd2(z[1], z[3], w2[0]); _fwd2(z[4], z[6], w2[1]); _fwd2(z[5], z[7], w2[1]);
		_square2x4(z, w2);
		_bck2(z[0], z[2], wi2[0]); _bck2(z[1], z[3], wi2[0]); _bck2(z[4], z[6], wi2[1]); _bck2(z[5], z[7], wi2[1]);
	}

	static void _square8(Zp z[8], const Zp w1, const Zp  wi1, const Zp w2[2], const Zp wi2[2])
	{
		_fwd2(z[0], z[4], w1); _fwd2(z[2], z[6], w1); _fwd2(z[1], z[5], w1); _fwd2(z[3], z[7], w1);
		_square4x2(z, w2, wi2);
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

	static void _mul4x2(Zp z[8], const Zp zp[8], const Zp w2[2], const Zp wi2[2])
	{
		_fwd2(z[0], z[2], w2[0]); _fwd2(z[1], z[3], w2[0]); _fwd2(z[4], z[6], w2[1]); _fwd2(z[5], z[7], w2[1]);
		_mul2x4(z, zp, w2);
		_bck2(z[0], z[2], wi2[0]); _bck2(z[1], z[3], wi2[0]); _bck2(z[4], z[6], wi2[1]); _bck2(z[5], z[7], wi2[1]);
	}

	static void _fwd8(Zp zp[8], const Zp w1, const Zp w2[2])
	{
		_fwd2(zp[0], zp[4], w1); _fwd2(zp[2], zp[6], w1); _fwd2(zp[1], zp[5], w1); _fwd2(zp[3], zp[7], w1);
		_fwd4x2(zp, w2);
	}

	static void _mul8(Zp z[8], const Zp zp[8], const Zp w1, const Zp  wi1, const Zp w2[2], const Zp wi2[2])
	{
		_fwd2(z[0], z[4], w1); _fwd2(z[2], z[6], w1); _fwd2(z[1], z[5], w1); _fwd2(z[3], z[7], w1);
		_mul4x2(z, zp, w2, wi2);
		_bck2(z[0], z[4], wi1); _bck2(z[2], z[6], wi1); _bck2(z[1], z[5], wi1); _bck2(z[3], z[7], wi1);
	}

public:
	Zp() {}
	explicit Zp(const uint32 n) : _n(n) {}

	uint32 get() const { return _n; }

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
	static void forward8(Zp * const z, const Zp * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Zp w1 = w[1 * (s + j)];
			Zp w2[2]; _load(2, w2, &w[2 * (s + j)], 1);
			Zp w4[4]; _load(4, w4, &w[4 * (s + j)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				Zp zl[8]; _load(8, zl, &z[k], m);
				_forward8(zl, w1, w2, w4);
				_store(8, &z[k], m, zl);
			}
		}
	}

	static void backward8(Zp * const z, const Zp * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const size_t ji = s - j - 1;
			const Zp wi1 = w[1 * (s + ji)];
			Zp wi2[2]; _loadr(2, wi2, &w[2 * (s + ji)], 1);
			Zp wi4[4]; _loadr(4, wi4, &w[4 * (s + ji)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				Zp zl[8]; _load(8, zl, &z[k], m);
				_backward8(zl, wi1, wi2, wi4);
				_store(8, &z[k], m, zl);
			}
		}
	}

	static void forward8_0(Zp * const z, const Zp * const w, const size_t n_8)
	{
		Zp w2[2]; _load(2, w2, &w[2], 1);
		Zp w4[4]; _load(4, w4, &w[4], 1);

		for (size_t i = 0; i < n_8; ++i)
		{
			const size_t k = i;
			Zp zl[8]; _load(8, zl, &z[k], n_8);
			_forward8_0(zl, w2, w4);
			_store(8, &z[k], n_8, zl);
		}
	}

	static void square2x4(Zp * const z, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			Zp w2[2]; _load(2, w2, &w[2 * n_8 + 2 * j], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			_square2x4(zl, w2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

	static void square4x2(Zp * const z, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const size_t ji = n_8 - j - 1;
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2[2]; _loadr(2, wi2, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			_square4x2(zl, w2, wi2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

	static void square8(Zp * const z, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const size_t ji = n_8 - j - 1;
			const Zp w1 = w[1 * (n_8 + j)];
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			const Zp wi1 = w[1 * (n_8 + ji)];
			Zp wi2[2]; _loadr(2, wi2, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			_square8(zl, w1, wi1, w2, wi2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

	static void mul2x4(Zp * const z, const Zp * const zp, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			Zp w2[2]; _load(2, w2, &w[2 * n_8 + 2 * j], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			Zp zpl[8]; _load(8, zpl, &zp[8 * j], 1);
			_mul2x4(zl, zpl, w2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

	static void fwd4x2(Zp * const zp, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			Zp zpl[8]; _load(8, zpl, &zp[8 * j], 1);
			_fwd4x2(zpl, w2);
			_store(8, &zp[8 * j], 1, zpl);
		}
	}

	static void mul4x2(Zp * const z, const Zp * const zp, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const size_t ji = n_8 - j - 1;
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			Zp wi2[2]; _loadr(2, wi2, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			Zp zpl[8]; _load(8, zpl, &zp[8 * j], 1);
			_mul4x2(zl, zpl, w2, wi2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

	static void fwd8(Zp * const zp, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const Zp w1 = w[1 * (n_8 + j)];
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);

			Zp zpl[8]; _load(8, zpl, &zp[8 * j], 1);
			_fwd8(zpl, w1, w2);
			_store(8, &zp[8 * j], 1, zpl);
		}
	}

	static void mul8(Zp * const z, const Zp * const zp, const Zp * const w, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const size_t ji = n_8 - j - 1;
			const Zp w1 = w[1 * (n_8 + j)];
			Zp w2[2]; _load(2, w2, &w[2 * (n_8 + j)], 1);
			const Zp wi1 = w[1 * (n_8 + ji)];
			Zp wi2[2]; _loadr(2, wi2, &w[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			Zp zpl[8]; _load(8, zpl, &zp[8 * j], 1);
			_mul8(zl, zpl, w1, wi1, w2, wi2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

public:
	static size_t forward(Zp * const z, const Zp * const w, const size_t n_8)
	{
		forward8_0(z, w, n_8);
		size_t m = n_8, s = 8;
		for (; m > 8; m /= 8, s *= 8) forward8(z, w, m / 8, s);
		return m;
	}

	static void backward(Zp * const z, const Zp * const w, const size_t n_8, const size_t m0)
	{
		for (size_t m = m0, s = n_8 / m0; s >= 1; m *= 8, s /= 8) backward8(z, w, m, s);
	}

	static void square(Zp * const z, const Zp * const w, const size_t n_8, const size_t m0)
	{
		if (m0 == 8) square8(z, w, n_8);
		else if (m0 == 4) square4x2(z, w, n_8);
		else if (m0 == 2) square2x4(z, w, n_8);
	}

	static void fwd(Zp * const zp, const Zp * const w, const size_t n_8, const size_t m0)
	{
		if (m0 == 8) fwd8(zp, w, n_8);
		else if (m0 == 4) fwd4x2(zp, w, n_8);
	}

	static void mul(Zp * const z, const Zp * const zp, const Zp * const w, const size_t n_8, const size_t m0)
	{
		if (m0 == 8) mul8(z, zp, w, n_8);
		else if (m0 == 4) mul4x2(z, zp, w, n_8);
		else if (m0 == 2) mul2x4(z, zp, w, n_8);
	}
};

typedef Zp<P1, Q1, R1, RSQ1, H1, IM1, SQRTI1, ISQRTI1> Zp1;
typedef Zp<P2, Q2, R2, RSQ2, H2, IM2, SQRTI2, ISQRTI2> Zp2;
typedef Zp<P3, Q3, R3, RSQ3, H3, IM3, SQRTI3, ISQRTI3> Zp3;

template<bool IS32>
class transformGPU : public transform
{
private:
	const size_t _num_regs;
	const size_t _size;
	const Zp1 _norm1;
	const Zp2 _norm2;
	const Zp3 _norm3;
	Zp1 * const _z1;
	Zp2 * const _z2;
	Zp3 * const _z3;
	Zp1 * const _zp1;
	Zp2 * const _zp2;
	Zp3 * const _zp3;
	Zp1 * const _w1;
	Zp2 * const _w2;
	Zp3 * const _w3;
	device * _pdevice;

public:
	transformGPU(const vuint32 & b, const uint32_t n, const size_t num_regs, const size_t d,	// device
				 const bool is_boinc, const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id)
				: transform(b, n, EKind::GPU), _num_regs(num_regs), _size(size_t(1) << n),
				_norm1(Zp1::norm(uint32(_size / 2))), _norm2(Zp2::norm(uint32(_size / 2))), _norm3(Zp3::norm(uint32(_size / 2))),
				_z1(new Zp1[num_regs * _size]), _z2(new Zp2[num_regs * _size]), _z3(new Zp3[num_regs * _size]),
				_zp1(new Zp1[_size]), _zp2(new Zp2[_size]), _zp3(new Zp3[_size]),
				_w1(new Zp1[_size]), _w2(new Zp2[_size]), _w3(new Zp3[_size])
	{
		const bool is_boinc_platform = is_boinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pdevice = new device(eng_platform, is_boinc_platform ? 0 : d);
		set_type(_pdevice->getType());

		const size_t size = _size;

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
	}

	virtual ~transformGPU()
	{
		delete[] _z1;
		delete[] _z2;
		delete[] _z3;
		delete[] _zp1;
		delete[] _zp2;
		delete[] _zp3;
		delete[] _w1;
		delete[] _w2;
		delete[] _w3;
		delete _pdevice;
	}

private:
	static __int128_t garner3(const Zp1 r1, const Zp2 r2, const Zp3 r3)
	{
		// Montgomery form of 1 / Pi (mod Pj)
		const uint32 mfInvP3_P1 = 608773230u, mfInvP2_P1 = 2130706177u, mfInvP3_P2 = 1409286102u;
		const uint64 P2P3 = P2 * uint64(P3);
		const __uint128_t P1P2P3 = P1 * __uint128_t(P2P3);

		const Zp1 u13 = r1.sub(Zp1(r3.get())).mul(Zp1(mfInvP3_P1));	// P3 < P1
		const Zp2 u23 = r2.sub(Zp2(r3.get())).mul(Zp2(mfInvP3_P2));	// P3 < P2
		const Zp1 u123 = u13.sub(Zp1(u23.get())).mul(Zp1(mfInvP2_P1));	// P3 < P1
		const __uint128_t n = __uint128_t(P2P3) * u123.get() + (u23.get() * uint64(P3) + r3.get());
		return (n > P1P2P3 / 2) ? __int128_t(n - P1P2P3) : __int128_t(n);
	}

	void carry(const uint32_t dup)
	{
		const size_t n = _size;
		Zp1 * const z1 = _z1;
		Zp2 * const z2 = _z2;
		Zp3 * const z3 = _z3;

		// Not converted into Montgomery form such that output is converted out of MF
		const Zp1 norm1 = _norm1; const Zp2 norm2 = _norm2;	const Zp3 norm3 = _norm3;
		const int32 base = static_cast<int32>(get_b()[0]);	// TODO
		__int128_t f = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const Zp1 u1 = z1[k].mul(norm1);
			const Zp2 u2 = z2[k].mul(norm2);
			const Zp3 u3 = z3[k].mul(norm3);
			__int128_t l = garner3(u1, u2, u3);
			if ((dup & 1) != 0) l += l;
			f += l;
			const __int128_t r = f / base;
			const int32 i = int32(f - r * base);
			f = r;
			z1[k].set_int(i); z2[k].set_int(i); z3[k].set_int(i);
		}

		while (f != 0)
		{
			f = -f;		// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				f += z1[k].get_int();
				const __int128_t r = f / base;
				const int32 i = int32(f - r * base);
				z1[k].set_int(i); z2[k].set_int(i); z3[k].set_int(i);
				f = r;
				if (r == 0) break;
			}
		}
	}

protected:
	void getZi(vint32 * const zi) const override
	{
		const size_t size = _size;
		Zp1 * const z1 = _z1;

		for (size_t k = 0; k < size; ++k) zi[k][0] = z1[k].get_int();
	}

	void setZi(const vint32 * const zi) override
	{
		const size_t size = _size;
		Zp1 * const z1 = _z1;
		Zp2 * const z2 = _z2;
		Zp3 * const z3 = _z3;

		for (size_t k = 0; k < size; ++k)
		{
			const int32_t v = zi[k][0];
			z1[k].set_int(v); z2[k].set_int(v); z3[k].set_int(v);
		}
	}

public:
	void set(const uint32_t a) override
	{
		const size_t size = _size;

		Zp1 * const z1 = _z1;
		z1[0] = Zp1(a); for (size_t k = 1; k < size; ++k) z1[k] = Zp1(0);

		Zp2 * const z2 = _z2;
		z2[0] = Zp2(a); for (size_t k = 1; k < size; ++k) z2[k] = Zp2(0);

		Zp3 * const z3 = _z3;
		z3[0] = Zp3(a); for (size_t k = 1; k < size; ++k) z3[k] = Zp3(0);
	}

	void square_dup(const uint32_t dup) override
	{
		const size_t n_8 = _size / 8;

		Zp1 * const z1 = _z1;
		const Zp1 * const w1 = _w1;
		const size_t m0 = Zp1::forward(z1, w1, n_8);
		Zp1::square(z1, w1, n_8, m0);
		Zp1::backward(z1, w1, n_8, m0);

		Zp2 * const z2 = _z2;
		const Zp2 * const w2 = _w2;
		Zp2::forward(z2, w2, n_8);
		Zp2::square(z2, w2, n_8, m0);
		Zp2::backward(z2, w2, n_8, m0);

		Zp3 * const z3 = _z3;
		const Zp3 * const w3 = _w3;
		Zp3::forward(z3, w3, n_8);
		Zp3::square(z3, w3, n_8, m0);
		Zp3::backward(z3, w3, n_8, m0);

		carry(dup);
	}

	void init_multiplicand(const size_t src) override
	{
		const size_t size = _size, n_8 = size / 8;

		const Zp1 * const z1_src = &_z1[src * size];
		Zp1 * const zp1 = _zp1;
		for (size_t k = 0; k < size; ++k) zp1[k] = z1_src[k];
		const Zp1 * const w1 = _w1;
		const size_t m0 = Zp1::forward(zp1, w1, n_8);
		Zp1::fwd(zp1, w1, n_8, m0);

		const Zp2 * const z2_src = &_z2[src * size];
		Zp2 * const zp2 = _zp2;
		for (size_t k = 0; k < size; ++k) zp2[k] = z2_src[k];
		const Zp2 * const w2 = _w2;
		Zp2::forward(zp2, w2, n_8);
		Zp2::fwd(zp2, w2, n_8, m0);

		const Zp3 * const z3_src = &_z3[src * size];
		Zp3 * const zp3 = _zp3;
		for (size_t k = 0; k < size; ++k) zp3[k] = z3_src[k];
		const Zp3 * const w3 = _w3;
		Zp3::forward(zp3, w3, n_8);
		Zp3::fwd(zp3, w3, n_8, m0);
	}

	void mul_mask(const uint8_t mask) override	// TODO
	{
		const size_t n_8 = _size / 8;

		Zp1 * const z1 = _z1;
		const Zp1 * const w1 = _w1;
		const size_t m0 = Zp1::forward(z1, w1, n_8);
		Zp1::mul(z1, _zp1, w1, n_8, m0);
		Zp1::backward(z1, w1, n_8, m0);

		Zp2 * const z2 = _z2;
		const Zp2 * const w2 = _w2;
		Zp2::forward(z2, w2, n_8);
		Zp2::mul(z2, _zp2, w2, n_8, m0);
		Zp2::backward(z2, w2, n_8, m0);

		Zp3 * const z3 = _z3;
		const Zp3 * const w3 = _w3;
		Zp3::forward(z3, w3, n_8);
		Zp3::mul(z3, _zp3, w3, n_8, m0);
		Zp3::backward(z3, w3, n_8, m0);

		carry(false);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const size_t size = _size;

		const Zp1 * const z1_src = &_z1[src * size];
		Zp1 * const z1_dst =  &_z1[dst * size];
		for (size_t k = 0; k < size; ++k) z1_dst[k] = z1_src[k];

		const Zp2 * const z2_src = &_z2[src * size];
		Zp2 * const z2_dst =  &_z2[dst * size];
		for (size_t k = 0; k < size; ++k) z2_dst[k] = z2_src[k];

		const Zp3 * const z3_src = &_z3[src * size];
		Zp3 * const z3_dst =  &_z3[dst * size];
		for (size_t k = 0; k < size; ++k) z3_dst[k] = z3_src[k];
	}

	void copy_mask(const size_t dst, const size_t src, const uint8_t mask) const override	// TODO
	{
		copy(dst, src);
	}

	bool read_checkpoint(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs, size = _size;
		if (!cFile.read(reinterpret_cast<char *>(_z1), num_regs * size * sizeof(Zp1))) return false;
		if (!cFile.read(reinterpret_cast<char *>(_z2), num_regs * size * sizeof(Zp2))) return false;
		if (!cFile.read(reinterpret_cast<char *>(_z3), num_regs * size * sizeof(Zp3))) return false;

		return true;
	}

	void save_checkpoint(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs, size = _size;
		if (!cFile.write(reinterpret_cast<const char *>(_z1), num_regs * size * sizeof(Zp1))) return;
		if (!cFile.write(reinterpret_cast<const char *>(_z2), num_regs * size * sizeof(Zp2))) return;
		if (!cFile.write(reinterpret_cast<const char *>(_z3), num_regs * size * sizeof(Zp3))) return;
	}

	size_t get_cache_size() const override { return (sizeof(Zp1) + sizeof(Zp2) + sizeof(Zp3)) * _size; }
	double get_error() const override { return 0.0; }

	void cosmic_ray() override { Zp1 & z = _z1[_size / 2]; z = z.add(Zp1(1)); }
};
