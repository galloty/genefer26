/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include <immintrin.h>

#include "transform.h"
#include "alignment.h"

#define finline	__attribute__((always_inline))

#define CHECK_ERROR		true

namespace transformCPU_namespace
{

typedef double	double_2 __attribute__ ((vector_size(2 * sizeof(double))));
typedef double	double_4 __attribute__ ((vector_size(4 * sizeof(double))));
typedef double	double_8 __attribute__ ((vector_size(8 * sizeof(double))));

#define VSIZE	8
#define VTYPE	double_8

typedef union
{
	double_8	d8;
	double_4	d4[2];
	double_2	d2[4];
} double_8u;

#define vmadd(x, y, z)	(x * y + z)
#define vnmadd(x, y, z) (z - x * y)
#define vmsub(x, y, z) (x * y - z)

inline void vround(VTYPE & y, const VTYPE & x)
{
#if defined(__AVX512F__)
	y = _mm512_roundscale_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#elif defined(__AVX__)
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 2; ++i) r.d4[i] = _mm256_round_pd(xu.d4[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = r.d8;
#else
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 4; ++i) r.d2[i] = _mm_round_pd(xu.d2[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = r.d8;
#endif
}

inline void vabs(VTYPE & y, const VTYPE & x)
{
#if defined(__AVX512F__)
	y = _mm512_abs_pd(x);
#elif defined(__AVX__)
	const long long m = (long long)(uint64_t(1) << 63);
	const __m256d mask = _mm256_castsi256_pd((__m256i){m, m, m, m});
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 2; ++i) r.d4[i] = _mm256_andnot_pd(mask, xu.d4[i]);
	y = r.d8;
#else
	const long long m = (long long)(uint64_t(1) << 63);
	const __m128d mask = _mm_castsi128_pd((__m128i){m, m});
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 4; ++i) r.d2[i] = _mm_andnot_pd(mask, xu.d2[i]);
	y = r.d8;
#endif
}

inline void vmax(VTYPE & z, const VTYPE & x, const VTYPE & y)
{
#if defined(__AVX512F__)
	z = _mm512_max_pd(x, y);
#elif defined(__AVX__)
	double_8u xu, yu, r; xu.d8 = x; yu.d8 = y;
	for (size_t i = 0; i < 2; ++i) r.d4[i] = _mm256_max_pd(xu.d4[i], yu.d4[i]);
	z = r.d8;
#else
	double_8u xu, yu, r; xu.d8 = x; yu.d8 = y;
	for (size_t i = 0; i < 4; ++i) r.d2[i] = _mm_max_pd(xu.d2[i], yu.d2[i]);
	z = r.d8;
#endif
}

class TwiddleFactor
{
private:
	double_2 _z;

public:
	TwiddleFactor() {}
	TwiddleFactor(const size_t a, const size_t b)
	{
		const long double alpha = 6.2831853071795864769252867665590057684L * ((long double)(a) / (long double)(b));
		const double cs = static_cast<double>(std::cosl(alpha)), sn = static_cast<double>(std::sinl(alpha));
		_z[0] = cs; _z[1] = sn / cs;
	}

	double cos() const { return _z[0]; }
	double tan() const { return _z[1]; }

	static TwiddleFactor TF_1_16() { TwiddleFactor r; r._z = double_2{0.92387953251128675612818318939678828682, 0.41421356237309504880168872420969807857}; return r; }
};

class VComplex
{
private:
	static constexpr double _csqrt2_2 = 0.70710678118654752440084436210484903929;
	VTYPE _re, _im;

	finline constexpr explicit VComplex(const VTYPE & re, const VTYPE & im) : _re(re), _im(im) {}

public:
	finline explicit VComplex() {}
	finline explicit VComplex(const double f) : _re{f}, _im{0} {}

	finline double real(const size_t index) const { return _re[index]; }
	finline double imag(const size_t index) const { return _im[index]; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
	finline void set(const size_t index, const double re, const double im) { _re[index] = re; _im[index] = im; }
#pragma GCC diagnostic pop

	finline bool is_zero() const { return ((_re[0] == 0) && (_im[0] == 0)); }

	finline VComplex & operator+=(const VComplex & rhs) { _re += rhs._re; _im += rhs._im; return *this; }
	finline VComplex & operator-=(const VComplex & rhs) { _re -= rhs._re; _im -= rhs._im; return *this; }
	finline VComplex & operator*=(const double rhs) { _re *= rhs; _im *= rhs; return *this; }

	finline VComplex operator+(const VComplex & rhs) const { return VComplex(_re + rhs._re, _im + rhs._im); }
	finline VComplex operator-(const VComplex & rhs) const { return VComplex(_re - rhs._re, _im - rhs._im); }
	finline VComplex operator*(const double rhs) const { return VComplex(_re * rhs, _im * rhs); }
	finline VComplex addi(const VComplex & rhs) const { return VComplex(_re - rhs._im, _im + rhs._re); }
	finline VComplex subi(const VComplex & rhs) const { return VComplex(_re + rhs._im, _im - rhs._re); }
	// finline VComplex muli() const { return VComplex(-_im, _re); }

	finline VComplex addmul(const VComplex & rhs, const double c) const { return VComplex(vmadd(rhs._re, c, _re), vmadd(rhs._im, c, _im)); }
	finline VComplex submul(const VComplex & rhs, const double c) const { return VComplex(vnmadd(rhs._re, c, _re), vnmadd(rhs._im, c, _im)); }

	finline VComplex mulW_0() const { return VComplex((_re - _im) * _csqrt2_2, (_im + _re) * _csqrt2_2); }
	finline VComplex mulWconj_0() const { return VComplex((_re + _im) * _csqrt2_2, (_im - _re) * _csqrt2_2); }

	finline VComplex mulW(const TwiddleFactor & rhs) const 
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return VComplex(vnmadd(_im, tg, _re) * cs, vmadd(_re, tg, _im) * cs);
	}
	finline VComplex mulWconj(const TwiddleFactor & rhs) const
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return VComplex(vmadd(_im, tg, _re) * cs, vnmadd(_re, tg, _im) * cs);
	}
	finline VComplex mulWconji(const TwiddleFactor & rhs) const
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return VComplex(vmsub(_re, tg, _im) * cs, vmadd(_im, tg, _re) * cs);
	}

	finline VComplex sqr() const { const VTYPE t = _re * _im; return VComplex(_re * _re - _im * _im, t + t); }
	finline VComplex mul(const VComplex & rhs) const { return VComplex(_re * rhs._re - _im * rhs._im, _re * rhs._im + _im * rhs._re); }

	finline VComplex round() const { VComplex r; vround(r._re, _re); vround(r._im, _im); return r; }
	finline VComplex abs() const { VComplex r; vabs(r._re, _re); vabs(r._im, _im); return r; }
	finline VComplex max(const VComplex & rhs) const { VComplex r; vmax(r._re, _re, rhs._re); vmax(r._im, _im, rhs._im); return r; }
	finline double max() const { return std::max(_re[0], _im[0]); }

	finline VComplex shift() const { return VComplex(-_im, _re); }
};

class VComplexPair
{
private:
	static constexpr double split = 1 << 20, split_inv = 1.0 / split;
	VComplex _l, _h;

	finline constexpr explicit VComplexPair(const VComplex & l, const VComplex & h) : _l(l), _h(h) {}

public:
	finline explicit VComplexPair() {}
	finline explicit VComplexPair(const double f) : _l(f), _h(0) {}

	finline VComplex get() const { return _l + _h; }
	finline void set(const VComplex & rhs)
	{
		const VComplex h = (rhs * split_inv).round() * VComplexPair::split;
		_l = rhs - h; _h = h;
	}

	finline bool is_zero() const { return (_l.is_zero() && _h.is_zero()); }

	finline VComplexPair & operator+=(const VComplexPair & rhs) { _l += rhs._l; _h += rhs._h; return *this; }
	finline VComplexPair & operator-=(const VComplexPair & rhs) { _l -= rhs._l; _h -= rhs._h; return *this; }
	finline VComplexPair & operator*=(const double rhs) { _l *= rhs; _h *= rhs; return *this; }

	finline VComplexPair operator+(const VComplexPair & rhs) const { return VComplexPair(_l + rhs._l, _h + rhs._h); }
	finline VComplexPair operator-(const VComplexPair & rhs) const { return VComplexPair(_l - rhs._l, _h - rhs._h); }
	finline VComplexPair operator*(const double rhs) const { return VComplexPair(_l * rhs, _h * rhs); }
	finline VComplexPair addi(const VComplexPair & rhs) const { return VComplexPair(_l.addi(rhs._l), _h.addi(rhs._h)); }
	finline VComplexPair subi(const VComplexPair & rhs) const { return VComplexPair(_l.subi(rhs._l), _h.subi(rhs._h)); }
	// finline VComplexPair muli() const { return VComplexPair(_l.muli(), _h.muli()); }

	finline VComplexPair addmul(const VComplexPair & rhs, const double c) const { return VComplexPair(_l.addmul(rhs._l, c), _h.addmul(rhs._h, c)); }

	finline VComplexPair mulW_0() const { return VComplexPair(_l.mulW_0(), _h.mulW_0()); }
	finline VComplexPair mulWconj_0() const { return VComplexPair(_l.mulWconj_0(), _h.mulWconj_0()); }
	finline VComplexPair mulW(const TwiddleFactor & rhs) const { return VComplexPair(_l.mulW(rhs), _h.mulW(rhs)); }
	finline VComplexPair mulWconj(const TwiddleFactor & rhs) const { return VComplexPair(_l.mulWconj(rhs), _h.mulWconj(rhs)); }
	finline VComplexPair mulWconji(const TwiddleFactor & rhs) const { return VComplexPair(_l.mulWconji(rhs), _h.mulWconji(rhs)); }

	finline VComplexPair sqr() const { return VComplexPair(_l.sqr(), _l.mul(_h + _h) + _h.sqr()); }
	finline VComplexPair mul(const VComplexPair & rhs) const { return VComplexPair(_l.mul(rhs._l), _l.mul(rhs._h) + _h.mul(rhs.get())); }

	finline VComplexPair flatten() const { return VComplexPair(_l, _h * split_inv); }
	finline VComplexPair elevate() const { return VComplexPair(_l, _h * split); }

	finline VComplexPair round() const { return VComplexPair(_l.round(), _h.round()); }
	finline VComplexPair abs() const { return VComplexPair(_l.abs(), _h.abs()); }
	finline VComplexPair max(const VComplexPair & rhs) const { return VComplexPair(_l.max(rhs._l), _h.max(rhs._h)); }
	finline VComplex max() const { return _l.max(_h); }

	finline VComplexPair shift() const { return VComplexPair(_l.shift(), _h.shift()); }

	finline VComplex mod(const double m, const double m_inv)
	{
		const VComplex l_m = (_l * m_inv).round(), rl_m = _l.submul(l_m, m);
		const VComplex h_m = (_h * m_inv).round(), rh_m = _h.submul(h_m, m);
		_h = h_m;

		const VComplex rl_mp = rl_m.addmul(rh_m, split);
		const VComplex rl_m2 = (rl_mp * m_inv).round();
		_l = l_m + rl_m2;

		return rl_mp.submul(rl_m2, m);
	}

private:
	finline static void _load(const size_t n, VComplexPair * const zl, const VComplexPair * const z, const size_t s)
	{
		for (size_t l = 0; l < n; ++l) zl[l] = z[l * s];
	}
	finline static void _store(const size_t n, VComplexPair * const z, const size_t s, const VComplexPair * const zl)
	{
		for (size_t l = 0; l < n; ++l) z[l * s] = zl[l];
	}

	finline static void _fwd2_0(VComplexPair & z0, VComplexPair & z1)
	{
		const VComplexPair t = z1.mulW_0(); z1 = z0 - t; z0 += t;
	}
	finline static void _fwd2(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.mulW(w); z1 = z0 - t; z0 += t;
	}
	finline static void _fwd2i(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.mulW(w); z1 = z0.subi(t); z0 = z0.addi(t);
	}

	finline static void _bck2_0(VComplexPair & z0, VComplexPair & z1)
	{
		const VComplexPair t = z0 - z1; z0 += z1, z1 = t.mulWconj_0();
	}
	finline static void _bck2(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z0 - z1; z0 += z1, z1 = t.mulWconj(w);
	}
	finline static void _bck2i(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1 - z0; z0 += z1, z1 = t.mulWconji(w);
	}

	finline static void _sqr2(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.sqr().mulW(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr() + t;
	}
	finline static void _sqr2i(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.sqr().mulW(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr().addi(t);
	}
	finline static void _sqr2n(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.sqr().mulW(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr() - t;
	}
	finline static void _sqr2ni(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.sqr().mulW(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr().subi(t);
	}

	finline static void _mul2(VComplexPair & z0, VComplexPair & z1, const VComplexPair & zp0, const VComplexPair & zp1, const TwiddleFactor & w) 
	{
		const VComplexPair t = z1.mul(zp1).mulW(w); z1 = z0.mul(zp1) + zp0.mul(z1); z0 = z0.mul(zp0) + t;
	}
	finline static void _mul2i(VComplexPair & z0, VComplexPair & z1, const VComplexPair & zp0, const VComplexPair & zp1, const TwiddleFactor & w) 
	{
		const VComplexPair t = z1.mul(zp1).mulW(w); z1 = z0.mul(zp1) + zp0.mul(z1); z0 = z0.mul(zp0).addi(t);
	}
	finline static void _mul2n(VComplexPair & z0, VComplexPair & z1, const VComplexPair & zp0, const VComplexPair & zp1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.mul(zp1).mulW(w); z1 = z0.mul(zp1) + zp0.mul(z1); z0 = z0.mul(zp0) - t;
	}
	finline static void _mul2ni(VComplexPair & z0, VComplexPair & z1, const VComplexPair & zp0, const VComplexPair & zp1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.mul(zp1).mulW(w); z1 = z0.mul(zp1) + zp0.mul(z1); z0 = z0.mul(zp0).subi(t);
	}

public:
	finline static void forward4_0(VComplexPair * const z, const size_t m)
	{
		const TwiddleFactor w2 = TwiddleFactor::TF_1_16();
		VComplexPair zl[4]; _load(4, zl, z, m);
		_fwd2_0(zl[0], zl[2]); _fwd2_0(zl[1], zl[3]);
		_fwd2(zl[0], zl[1], w2); _fwd2i(zl[2], zl[3], w2);
		_store(4, z, m, zl);
	}

	// finline static void backward4_0(VComplexPair * const z, const size_t m)
	// {
	// 	const TwiddleFactor w2 = TwiddleFactor::TF_1_16();
	// 	VComplexPair zl[4]; _load(4, zl, z, m);
	// 	_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
	// 	_bck2_0(zl[0], zl[2]); _bck2_0(zl[1], zl[3]);
	// 	_store(4, z, m, zl);
	// }

	finline static void forward4e(VComplexPair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		VComplexPair zl[4]; _load(4, zl, z, m);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_fwd2(zl[0], zl[1], w2); _fwd2i(zl[2], zl[3], w2);
		_store(4, z, m, zl);
	}

	finline static void forward4o(VComplexPair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		VComplexPair zl[4]; _load(4, zl, z, m);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_fwd2(zl[0], zl[1], w2); _fwd2i(zl[2], zl[3], w2);
		_store(4, z, m, zl);
	}

	finline static void backward4e(VComplexPair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		VComplexPair zl[4]; _load(4, zl, z, m);
		_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
		_bck2(zl[0], zl[2], w1); _bck2(zl[1], zl[3], w1);
		_store(4, z, m, zl);
	}

	finline static void backward4o(VComplexPair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		VComplexPair zl[4]; _load(4, zl, z, m);
		_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
		_bck2i(zl[0], zl[2], w1); _bck2i(zl[1], zl[3], w1);
		_store(4, z, m, zl);
	}

	finline static void square2x2e(VComplexPair * const z, const TwiddleFactor & w1)
	{
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_sqr2(zl[0], zl[1], w1); _sqr2n(zl[2], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void square2x2o(VComplexPair * const z, const TwiddleFactor & w1)
	{
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_sqr2i(zl[0], zl[1], w1); _sqr2ni(zl[2], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void square4e(VComplexPair * const z, const TwiddleFactor & w1)
	{
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_sqr2(zl[0], zl[1], w1); _sqr2n(zl[2], zl[3], w1);
		_bck2(zl[0], zl[2], w1); _bck2(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void square4o(VComplexPair * const z, const TwiddleFactor & w1)
	{
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_sqr2i(zl[0], zl[1], w1); _sqr2ni(zl[2], zl[3], w1);
		_bck2i(zl[0], zl[2], w1); _bck2i(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul2x2e(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor & w1)
	{
		VComplexPair zpl[4]; _load(4, zpl, zp, 1);
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_mul2(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2n(zl[2], zl[3], zpl[2], zpl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul2x2o(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor & w1)
	{
		VComplexPair zpl[4]; _load(4, zpl, zp, 1);
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_mul2i(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2ni(zl[2], zl[3], zpl[2], zpl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void fwd2x2e(VComplexPair * const z, const TwiddleFactor & w1)
	{
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void fwd2x2o(VComplexPair * const z, const TwiddleFactor & w1)
	{
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul4e(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor & w1)
	{
		VComplexPair zpl[4]; _load(4, zpl, zp, 1);
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_mul2(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2n(zl[2], zl[3], zpl[2], zpl[3], w1);
		_bck2(zl[0], zl[2], w1); _bck2(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul4o(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor & w1)
	{
		VComplexPair zpl[4]; _load(4, zpl, zp, 1);
		VComplexPair zl[4]; _load(4, zl, z, 1);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_mul2i(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2ni(zl[2], zl[3], zpl[2], zpl[3], w1);
		_bck2i(zl[0], zl[2], w1); _bck2i(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void backward4_0_carry(VComplexPair * const z, const size_t m, VComplexPair f[4],
		const double base, const double base_inv, const double t2_n, const double g
#if defined(CHECK_ERROR)
		, VComplexPair & err
#endif
	)
	{
		const TwiddleFactor w2 = TwiddleFactor::TF_1_16();

		VComplexPair zl[4]; _load(4, zl, z, m);
	 	_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
	 	_bck2_0(zl[0], zl[2]); _bck2_0(zl[1], zl[3]);

		for (size_t i = 0; i < 4; ++i)
		{
			zl[i] = zl[i].flatten() * t2_n;
			const VComplexPair t = zl[i].round();
#if defined(CHECK_ERROR)
			err = err.max((zl[i] - t).abs());
#endif
			f[i] = f[i].addmul(t, g);
			zl[i].set(f[i].mod(base, base_inv));
		}

		_store(4, z, m, zl);
	}

	finline static void carry_fix(VComplexPair * const z, const size_t m, VComplexPair f[4], const double base, const double base_inv)
	{
		for (size_t k = 0; k < 2; ++k)
		{
			VComplexPair zl[4]; _load(4, zl, &z[k], m);
			for (size_t i = 0; i < 4; ++i)
			{
				f[i] += zl[i].flatten();
				zl[i].set(f[i].mod(base, base_inv));
			}
			_store(4, &z[k], m, zl);
		}

		VComplexPair zl[4]; _load(4, zl, &z[2], m);
		for (size_t i = 0; i < 4; ++i) zl[i] += f[i].elevate();
		_store(4, &z[2], m, zl);
	}
};

template<size_t N>
class transformCPU : public transform
{
private:
	const size_t _num_regs;
	const double _base, _base_inv;
	VComplexPair * const _z;
	VComplexPair * const _zp;
	TwiddleFactor * const _w;
	double _error;

public:
	transformCPU(const uint32_t b, const uint32_t n, const size_t num_regs) : transform(b, n, EKind::CPU),
		_num_regs(num_regs), _base(b), _base_inv(1.0 / b),
		_z(static_cast<VComplexPair *>(align_new(num_regs * N * sizeof(VComplexPair), 2 * 1024 * 1024))),
		_zp(static_cast<VComplexPair *>(align_new(N * sizeof(VComplexPair), 1024))),
		_w(static_cast<TwiddleFactor *>(align_new(N / 2 * sizeof(TwiddleFactor), 1024)))
	{
		TwiddleFactor * const w = _w;
		for (size_t s = 4; s < N / 2; s *= 2)
		{
			for (size_t j = 0; j < s / 2; ++j)
			{
				w[s + j] = TwiddleFactor(bitrev(2 * j, 4 * s) + 1, 2 * 4 * s);
			}
		}

		_error = 0;
	}

	virtual ~transformCPU()
	{
		align_delete(_z);
		align_delete(_zp);
		align_delete(_w);
	}

protected:
	void getZi(int32_t * const zi) const override
	{
		const VComplexPair * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			const VComplex zk = z[k].get();
			zi[k + 0 * N] = std::lround(zk.real(0));
			zi[k + 1 * N] = std::lround(zk.imag(0));
		}
	}

	void setZi(const int32_t * const zi) override
	{
		VComplexPair * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			VComplex zk;
			zk.set(0, static_cast<double>(zi[k + 0 * N]), static_cast<double>(zi[k + 1 * N]));
			zk.set(1, static_cast<double>(zi[k + 0 * N]), static_cast<double>(zi[k + 1 * N]));	// TODO
			z[k].set(zk);
		}
	}

private:
	static void forward4_0(VComplexPair * const z)
	{
		for (size_t k = 0; k < N / 4; ++k) VComplexPair::forward4_0(&z[k], N / 4);
	}

	static void square_e(VComplexPair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w20 = w[2 * sj + 0];

		for (size_t i = 0; i < m; ++i) VComplexPair::forward4e(&z[i], m, w1, w20);

		if (m > 4)
		{
			square_e(&z[0 * m], w, m / 4, 4 * sj + 0);
			square_o(&z[1 * m], w, m / 4, 4 * sj + 0);
			square_e(&z[2 * m], w, m / 4, 4 * sj + 1);
			square_o(&z[3 * m], w, m / 4, 4 * sj + 1);
		}
		else if (m == 4)
		{
			const TwiddleFactor w40 = w[4 * sj + 0], w41 = w[4 * sj + 1];

			VComplexPair::square4e(&z[0], w40);
			VComplexPair::square4o(&z[4], w40);
			VComplexPair::square4e(&z[8], w41);
			VComplexPair::square4o(&z[12], w41);
		}
		else // if (m == 2)
		{
			VComplexPair::square2x2e(&z[0], w20);
			VComplexPair::square2x2o(&z[4], w20);
		}

		for (size_t i = 0; i < m; ++i) VComplexPair::backward4e(&z[i], m, w1, w20);
	}

	static void square_o(VComplexPair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w21 = w[2 * sj + 1];

		for (size_t i = 0; i < m; ++i) VComplexPair::forward4o(&z[i], m, w1, w21);

		if (m > 4)
		{
			square_e(&z[0 * m], w, m / 4, 4 * sj + 2);
			square_o(&z[1 * m], w, m / 4, 4 * sj + 2);
			square_e(&z[2 * m], w, m / 4, 4 * sj + 3);
			square_o(&z[3 * m], w, m / 4, 4 * sj + 3);
		}
		else if (m == 4)
		{
			const TwiddleFactor w42 = w[4 * sj + 2], w43 = w[4 * sj + 3];

			VComplexPair::square4e(&z[0], w42);
			VComplexPair::square4o(&z[4], w42);
			VComplexPair::square4e(&z[8], w43);
			VComplexPair::square4o(&z[12], w43);
		}
		else // if (m == 2)
		{
			VComplexPair::square2x2e(&z[0], w21);
			VComplexPair::square2x2o(&z[4], w21);
		}

		for (size_t i = 0; i < m; ++i) VComplexPair::backward4o(&z[i], m, w1, w21);
	}

	static void forward_e(VComplexPair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w20 = w[2 * sj + 0];

		for (size_t i = 0; i < m; ++i) VComplexPair::forward4e(&z[i], m, w1, w20);

		if (m > 4)
		{
			forward_e(&z[0 * m], w, m / 4, 4 * sj + 0);
			forward_o(&z[1 * m], w, m / 4, 4 * sj + 0);
			forward_e(&z[2 * m], w, m / 4, 4 * sj + 1);
			forward_o(&z[3 * m], w, m / 4, 4 * sj + 1);
		}
		else if (m == 4)
		{
			const TwiddleFactor w40 = w[4 * sj + 0], w41 = w[4 * sj + 1];

			VComplexPair::fwd2x2e(&z[0], w40);
			VComplexPair::fwd2x2o(&z[4], w40);
			VComplexPair::fwd2x2e(&z[8], w41);
			VComplexPair::fwd2x2o(&z[12], w41);
		}
	}

	static void forward_o(VComplexPair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w21 = w[2 * sj + 1];

		for (size_t i = 0; i < m; ++i) VComplexPair::forward4o(&z[i], m, w1, w21);

		if (m > 4)
		{
			forward_e(&z[0 * m], w, m / 4, 4 * sj + 2);
			forward_o(&z[1 * m], w, m / 4, 4 * sj + 2);
			forward_e(&z[2 * m], w, m / 4, 4 * sj + 3);
			forward_o(&z[3 * m], w, m / 4, 4 * sj + 3);
		}
		else if (m == 4)
		{
			const TwiddleFactor w42 = w[4 * sj + 2], w43 = w[4 * sj + 3];

			VComplexPair::fwd2x2e(&z[0], w42);
			VComplexPair::fwd2x2o(&z[4], w42);
			VComplexPair::fwd2x2e(&z[8], w43);
			VComplexPair::fwd2x2o(&z[12], w43);
		}
	}

	static void mul_e(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w20 = w[2 * sj + 0];

		for (size_t i = 0; i < m; ++i) VComplexPair::forward4e(&z[i], m, w1, w20);

		if (m > 4)
		{
			mul_e(&z[0 * m], &zp[0 * m], w, m / 4, 4 * sj + 0);
			mul_o(&z[1 * m], &zp[1 * m], w, m / 4, 4 * sj + 0);
			mul_e(&z[2 * m], &zp[2 * m], w, m / 4, 4 * sj + 1);
			mul_o(&z[3 * m], &zp[3 * m], w, m / 4, 4 * sj + 1);
		}
		else if (m == 4)
		{
			const TwiddleFactor w40 = w[4 * sj + 0], w41 = w[4 * sj + 1];

			VComplexPair::mul4e(&z[0], &zp[0], w40);
			VComplexPair::mul4o(&z[4], &zp[4], w40);
			VComplexPair::mul4e(&z[8], &zp[8], w41);
			VComplexPair::mul4o(&z[12], &zp[12], w41);
		}
		else // if (m == 2)
		{
			VComplexPair::mul2x2e(&z[0], &zp[0], w20);
			VComplexPair::mul2x2o(&z[4], &zp[4], w20);
		}

		for (size_t i = 0; i < m; ++i) VComplexPair::backward4e(&z[i], m, w1, w20);
	}

	static void mul_o(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w21 = w[2 * sj + 1];

		for (size_t i = 0; i < m; ++i) VComplexPair::forward4o(&z[i], m, w1, w21);

		if (m > 4)
		{
			mul_e(&z[0 * m], &zp[0 * m], w, m / 4, 4 * sj + 2);
			mul_o(&z[1 * m], &zp[1 * m], w, m / 4, 4 * sj + 2);
			mul_e(&z[2 * m], &zp[2 * m], w, m / 4, 4 * sj + 3);
			mul_o(&z[3 * m], &zp[3 * m], w, m / 4, 4 * sj + 3);
		}
		else if (m == 4)
		{
			const TwiddleFactor w42 = w[4 * sj + 2], w43 = w[4 * sj + 3];

			VComplexPair::mul4e(&z[0], &zp[0], w42);
			VComplexPair::mul4o(&z[4], &zp[4], w42);
			VComplexPair::mul4e(&z[8], &zp[8], w43);
			VComplexPair::mul4o(&z[12], &zp[12], w43);
		}
		else // if (m == 2)
		{
			VComplexPair::mul2x2e(&z[0], &zp[0], w21);
			VComplexPair::mul2x2o(&z[4], &zp[4], w21);
		}

		for (size_t i = 0; i < m; ++i) VComplexPair::backward4o(&z[i], m, w1, w21);
	}

	static double backward4_0_carry(VComplexPair * const z, const double base, const double base_inv, const bool dup)
	{
		VComplexPair f[4]; for (size_t i = 0; i < 4; ++i) f[i] = VComplexPair(0.0);

#if defined(CHECK_ERROR)
		VComplexPair err = VComplexPair(0.0);
#endif
		for (size_t k = 0; k < N / 4; ++k)
		{
			VComplexPair::backward4_0_carry(&z[k], N / 4, f, base, base_inv, 2.0 / N, dup ? 2 : 1
#if defined(CHECK_ERROR)
				, err
#endif
			);
		}

		// modulo z^n + 1
		const VComplexPair t = f[3].shift(); f[3] = f[2]; f[2] = f[1]; f[1] = f[0]; f[0] = t;

		VComplexPair::carry_fix(z, N / 4, f, base, base_inv);

#if defined(CHECK_ERROR)
		return err.max().max();
#else
		return 0;
#endif
	}

public:
	void set(const uint32_t a) override
	{
		VComplexPair * const z = _z;

		z[0] = VComplexPair(static_cast<double>(a));
		for (size_t k = 1; k < N; ++k) z[k] = VComplexPair(0.0);
	}

	void square_dup(const bool dup) override
	{
		VComplexPair * const z = _z;
		const TwiddleFactor * const w = _w;

		forward4_0(z);
		square_e(&z[0 * (N / 4)], w, N / 16, 4 + 0);
		square_o(&z[1 * (N / 4)], w, N / 16, 4 + 0);
		square_e(&z[2 * (N / 4)], w, N / 16, 4 + 1);
		square_o(&z[3 * (N / 4)], w, N / 16, 4 + 1);
		const double err = backward4_0_carry(z, _base, _base_inv, dup);
		_error = std::max(_error, err);
	}

	void init_multiplicand(const size_t src) override
	{
		const VComplexPair * const z_src = &_z[src * N];
		VComplexPair * const zp = _zp;
		const TwiddleFactor * const w = _w;

		for (size_t k = 0; k < N; ++k) zp[k] = z_src[k];

		forward4_0(zp);
		forward_e(&zp[0 * (N / 4)], w, N / 16, 4 + 0);
		forward_o(&zp[1 * (N / 4)], w, N / 16, 4 + 0);
		forward_e(&zp[2 * (N / 4)], w, N / 16, 4 + 1);
		forward_o(&zp[3 * (N / 4)], w, N / 16, 4 + 1);
	}

	void mul() override
	{
		VComplexPair * const z = _z;
		const VComplexPair * const zp = _zp;
		const TwiddleFactor * const w = _w;

		forward4_0(z);
		mul_e(&z[0 * (N / 4)], &zp[0 * (N / 4)], w, N / 16, 4 + 0);
		mul_o(&z[1 * (N / 4)], &zp[1 * (N / 4)], w, N / 16, 4 + 0);
		mul_e(&z[2 * (N / 4)], &zp[2 * (N / 4)], w, N / 16, 4 + 1);
		mul_o(&z[3 * (N / 4)], &zp[3 * (N / 4)], w, N / 16, 4 + 1);
		const double err = backward4_0_carry(z, _base, _base_inv, false);
		_error = std::max(_error, err);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const VComplexPair * const z_src = &_z[src * N];
		VComplexPair * const z_dst =  &_z[dst * N];

		for (size_t k = 0; k < N; ++k) z_dst[k] = z_src[k];
	}

	bool read_checkpoint(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.read(reinterpret_cast<char *>(_z), num_regs * N * sizeof(VComplexPair))) return false;

		return true;
	}

	void save_checkpoint(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.write(reinterpret_cast<const char *>(_z), num_regs * N * sizeof(VComplexPair))) return;
	}

	size_t get_cache_size() const override { return N * sizeof(VComplexPair); }
	double get_error() const override { return _error; }
};

inline transform * create_transformCPU(const uint32_t b, const uint32_t n, const size_t num_regs)
{
	transform * ptransform = nullptr;
	if      (n ==  5) ptransform = new transformCPU<(1 <<  4)>(b, n, num_regs);
	else if (n ==  6) ptransform = new transformCPU<(1 <<  5)>(b, n, num_regs);
	else if (n ==  7) ptransform = new transformCPU<(1 <<  6)>(b, n, num_regs);
	else if (n ==  8) ptransform = new transformCPU<(1 <<  7)>(b, n, num_regs);
	else if (n ==  9) ptransform = new transformCPU<(1 <<  8)>(b, n, num_regs);
	else if (n == 10) ptransform = new transformCPU<(1 <<  9)>(b, n, num_regs);
	else if (n == 11) ptransform = new transformCPU<(1 << 10)>(b, n, num_regs);
	else if (n == 12) ptransform = new transformCPU<(1 << 11)>(b, n, num_regs);
	else if (n == 13) ptransform = new transformCPU<(1 << 12)>(b, n, num_regs);
	else if (n == 14) ptransform = new transformCPU<(1 << 13)>(b, n, num_regs);
	else if (n == 15) ptransform = new transformCPU<(1 << 14)>(b, n, num_regs);
	else if (n == 16) ptransform = new transformCPU<(1 << 15)>(b, n, num_regs);
	else if (n == 17) ptransform = new transformCPU<(1 << 16)>(b, n, num_regs);

	if (ptransform == nullptr) throw std::runtime_error("exponent is not supported");

	return ptransform;
}

}