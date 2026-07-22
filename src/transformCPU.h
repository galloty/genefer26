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
#include "vcomplex.h"

#ifndef finline
#define finline	__attribute__((always_inline)) inline
#endif

#define CHECK_ERROR		true

namespace arch_namespace
{

class Complex_8_pair
{
private:
	static constexpr double split = 1 << 20, split_inv = 1.0 / split;
	Complex_8 _l, _h;

	finline explicit Complex_8_pair(const Complex_8 & l, const Complex_8 & h) : _l(l), _h(h) {}

public:
	finline explicit Complex_8_pair() {}
	finline explicit Complex_8_pair(const double f) : _l(f), _h(0) {}

	finline void copy_mask(const Complex_8_pair & rhs, const uint8_t mask) { _l.copy_mask(rhs._l, mask); _h.copy_mask(rhs._h, mask); }

	finline const Complex_8 & low() const { return _l; }
	finline const Complex_8 & high() const { return _h; }

	finline Complex_8 get() const { return _l + _h; }
	finline void set(const Complex_8 & rhs)
	{
		const Complex_8 h = (rhs * split_inv).round() * Complex_8_pair::split;
		_l = rhs - h; _h = h;
	}

	finline Complex_8_pair & operator+=(const Complex_8_pair & rhs) { _l += rhs._l; _h += rhs._h; return *this; }
	finline Complex_8_pair & operator-=(const Complex_8_pair & rhs) { _l -= rhs._l; _h -= rhs._h; return *this; }
	finline Complex_8_pair & operator*=(const double rhs) { _l *= rhs; _h *= rhs; return *this; }
	finline Complex_8_pair & operator*=(const Double_8 & rhs) { _l *= rhs; _h *= rhs; return *this; }

	finline Complex_8_pair operator+(const Complex_8_pair & rhs) const { return Complex_8_pair(_l + rhs._l, _h + rhs._h); }
	finline Complex_8_pair operator-(const Complex_8_pair & rhs) const { return Complex_8_pair(_l - rhs._l, _h - rhs._h); }
	finline Complex_8_pair operator*(const double rhs) const { return Complex_8_pair(_l * rhs, _h * rhs); }
	finline Complex_8_pair operator*(const Double_8 rhs) const { return Complex_8_pair(_l * rhs, _h * rhs); }
	finline Complex_8_pair addi(const Complex_8_pair & rhs) const { return Complex_8_pair(_l.addi(rhs._l), _h.addi(rhs._h)); }
	finline Complex_8_pair subi(const Complex_8_pair & rhs) const { return Complex_8_pair(_l.subi(rhs._l), _h.subi(rhs._h)); }
	// finline Complex_8_pair muli() const { return Complex_8_pair(_l.muli(), _h.muli()); }

	finline Complex_8_pair mulTF_0() const { return Complex_8_pair(_l.mulTF_0(), _h.mulTF_0()); }
	finline Complex_8_pair mulTFconj_0() const { return Complex_8_pair(_l.mulTFconj_0(), _h.mulTFconj_0()); }
	finline Complex_8_pair mulTF(const TwiddleFactor & rhs) const { return Complex_8_pair(_l.mulTF(rhs), _h.mulTF(rhs)); }
	finline Complex_8_pair mulTFconj(const TwiddleFactor & rhs) const { return Complex_8_pair(_l.mulTFconj(rhs), _h.mulTFconj(rhs)); }
	finline Complex_8_pair mulTFconji(const TwiddleFactor & rhs) const { return Complex_8_pair(_l.mulTFconji(rhs), _h.mulTFconji(rhs)); }

	finline Complex_8_pair sqr() const { return Complex_8_pair(_l.sqr(), _l.mul(_h + _h) + _h.sqr()); }
	finline Complex_8_pair mul(const Complex_8_pair & rhs) const { return Complex_8_pair(_l.mul(rhs._l), _l.mul(rhs._h) + _h.mul(rhs.get())); }

	finline Complex_8_pair mul_mask(const Complex_8_pair & rhs, const uint8_t mask) const { return Complex_8_pair(_l.mul_mask(rhs._l, mask), _l.mul_maskz(rhs._h, mask) + _h.mul_mask(rhs.get(), mask)); }
	finline Complex_8_pair mul_maskz(const Complex_8_pair & rhs, const uint8_t mask) const { return Complex_8_pair(_l.mul_maskz(rhs._l, mask), _l.mul_maskz(rhs._h, mask) + _h.mul_maskz(rhs.get(), mask)); }

	finline Complex_8_pair flatten() const { return Complex_8_pair(_l, _h * split_inv); }
	finline Complex_8_pair elevate() const { return Complex_8_pair(_l, _h * split); }

	finline Complex_8_pair round() const { return Complex_8_pair(_l.round(), _h.round()); }
	finline Complex_8_pair abs() const { return Complex_8_pair(_l.abs(), _h.abs()); }
	finline Complex_8_pair max(const Complex_8_pair & rhs) const { return Complex_8_pair(_l.max(rhs._l), _h.max(rhs._h)); }
	finline Complex_8 max() const { return _l.max(_h); }

	finline Complex_8_pair shift() const { return Complex_8_pair(_l.shift(), _h.shift()); }

	finline Complex_8 mod(const Double_8 & m, const Double_8 & m_inv)
	{
		const Complex_8 h_m = (_h * m_inv).round(), rh_m = _h.submul(h_m, m);
		_h = h_m;

		const Complex_8 l_m = (_l * m_inv).round(), rl_m = _l.submul(l_m, m) + rh_m * split;
		const Complex_8 rl_m2 = (rl_m * m_inv).round();
		_l = l_m + rl_m2;

		return rl_m.submul(rl_m2, m);
	}

private:
	finline static void _load(const size_t n, Complex_8_pair * const zl, const Complex_8_pair * const z, const size_t s)
	{
		for (size_t l = 0; l < n; ++l) zl[l] = z[l * s];
	}
	finline static void _store(const size_t n, Complex_8_pair * const z, const size_t s, const Complex_8_pair * const zl)
	{
		for (size_t l = 0; l < n; ++l) z[l * s] = zl[l];
	}

	finline static void _fwd2_0(Complex_8_pair & z0, Complex_8_pair & z1)
	{
		const Complex_8_pair t = z1.mulTF_0(); z1 = z0 - t; z0 += t;
	}
	finline static void _fwd2(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.mulTF(w); z1 = z0 - t; z0 += t;
	}
	finline static void _fwd2i(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.mulTF(w); z1 = z0.subi(t); z0 = z0.addi(t);
	}

	finline static void _bck2_0(Complex_8_pair & z0, Complex_8_pair & z1)
	{
		const Complex_8_pair t = z0 - z1; z0 += z1, z1 = t.mulTFconj_0();
	}
	finline static void _bck2(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z0 - z1; z0 += z1, z1 = t.mulTFconj(w);
	}
	finline static void _bck2i(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1 - z0; z0 += z1, z1 = t.mulTFconji(w);
	}

	finline static void _sqr2(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.sqr().mulTF(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr() + t;
	}
	finline static void _sqr2i(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.sqr().mulTF(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr().addi(t);
	}
	finline static void _sqr2n(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.sqr().mulTF(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr() - t;
	}
	finline static void _sqr2ni(Complex_8_pair & z0, Complex_8_pair & z1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.sqr().mulTF(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr().subi(t);
	}

	finline static void _mul2(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const TwiddleFactor & w) 
	{
		const Complex_8_pair t = z1.mul(zp1).mulTF(w); z1 = z0.mul(zp1) + z1.mul(zp0); z0 = z0.mul(zp0) + t;
	}
	finline static void _mul2i(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const TwiddleFactor & w) 
	{
		const Complex_8_pair t = z1.mul(zp1).mulTF(w); z1 = z0.mul(zp1) + z1.mul(zp0); z0 = z0.mul(zp0).addi(t);
	}
	finline static void _mul2n(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.mul(zp1).mulTF(w); z1 = z0.mul(zp1) + z1.mul(zp0); z0 = z0.mul(zp0) - t;
	}
	finline static void _mul2ni(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.mul(zp1).mulTF(w); z1 = z0.mul(zp1) + z1.mul(zp0); z0 = z0.mul(zp0).subi(t);
	}

	finline static void _mul2_mask(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const uint8_t mask, const TwiddleFactor & w) 
	{
		const Complex_8_pair t = z1.mul_maskz(zp1, mask).mulTF(w); z1 = z0.mul_maskz(zp1, mask) + z1.mul_mask(zp0, mask); z0 = z0.mul_mask(zp0, mask) + t;
	}
	finline static void _mul2i_mask(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const uint8_t mask, const TwiddleFactor & w) 
	{
		const Complex_8_pair t = z1.mul_maskz(zp1, mask).mulTF(w); z1 = z0.mul_maskz(zp1, mask) + z1.mul_mask(zp0, mask); z0 = z0.mul_mask(zp0, mask).addi(t);
	}
	finline static void _mul2n_mask(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const uint8_t mask, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.mul_maskz(zp1, mask).mulTF(w); z1 = z0.mul_maskz(zp1, mask) + z1.mul_mask(zp0, mask); z0 = z0.mul_mask(zp0, mask) - t;
	}
	finline static void _mul2ni_mask(Complex_8_pair & z0, Complex_8_pair & z1, const Complex_8_pair & zp0, const Complex_8_pair & zp1, const uint8_t mask, const TwiddleFactor & w)
	{
		const Complex_8_pair t = z1.mul_maskz(zp1, mask).mulTF(w); z1 = z0.mul_maskz(zp1, mask) + z1.mul_mask(zp0, mask); z0 = z0.mul_mask(zp0, mask).subi(t);
	}

public:
	finline static void forward4_0(Complex_8_pair * const z, const size_t m)
	{
		const TwiddleFactor w2 = TwiddleFactor::TF_1_16();
		Complex_8_pair zl[4]; _load(4, zl, z, m);
		_fwd2_0(zl[0], zl[2]); _fwd2_0(zl[1], zl[3]);
		_fwd2(zl[0], zl[1], w2); _fwd2i(zl[2], zl[3], w2);
		_store(4, z, m, zl);
	}

	// finline static void backward4_0(Complex_8_pair * const z, const size_t m)
	// {
	// 	const TwiddleFactor w2 = TwiddleFactor::TF_1_16();
	// 	Complex_8_pair zl[4]; _load(4, zl, z, m);
	// 	_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
	// 	_bck2_0(zl[0], zl[2]); _bck2_0(zl[1], zl[3]);
	// 	_store(4, z, m, zl);
	// }

	finline static void forward4e(Complex_8_pair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, m);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_fwd2(zl[0], zl[1], w2); _fwd2i(zl[2], zl[3], w2);
		_store(4, z, m, zl);
	}

	finline static void forward4o(Complex_8_pair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, m);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_fwd2(zl[0], zl[1], w2); _fwd2i(zl[2], zl[3], w2);
		_store(4, z, m, zl);
	}

	finline static void backward4e(Complex_8_pair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, m);
		_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
		_bck2(zl[0], zl[2], w1); _bck2(zl[1], zl[3], w1);
		_store(4, z, m, zl);
	}

	finline static void backward4o(Complex_8_pair * const z, const size_t m, const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, m);
		_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
		_bck2i(zl[0], zl[2], w1); _bck2i(zl[1], zl[3], w1);
		_store(4, z, m, zl);
	}

	finline static void square2x2e(Complex_8_pair * const z, const TwiddleFactor & w1)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_sqr2(zl[0], zl[1], w1); _sqr2n(zl[2], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void square2x2o(Complex_8_pair * const z, const TwiddleFactor & w1)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_sqr2i(zl[0], zl[1], w1); _sqr2ni(zl[2], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void square4e(Complex_8_pair * const z, const TwiddleFactor & w1)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_sqr2(zl[0], zl[1], w1); _sqr2n(zl[2], zl[3], w1);
		_bck2(zl[0], zl[2], w1); _bck2(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void square4o(Complex_8_pair * const z, const TwiddleFactor & w1)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_sqr2i(zl[0], zl[1], w1); _sqr2ni(zl[2], zl[3], w1);
		_bck2i(zl[0], zl[2], w1); _bck2i(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul2x2e(Complex_8_pair * const z, const Complex_8_pair * const zp, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_mul2(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2n(zl[2], zl[3], zpl[2], zpl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul2x2o(Complex_8_pair * const z, const Complex_8_pair * const zp, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_mul2i(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2ni(zl[2], zl[3], zpl[2], zpl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul2x2e_mask(Complex_8_pair * const z, const Complex_8_pair * const zp, const uint8_t mask, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_mul2_mask(zl[0], zl[1], zpl[0], zpl[1], mask, w1); _mul2n_mask(zl[2], zl[3], zpl[2], zpl[3], mask, w1);
		_store(4, z, 1, zl);
	}

	finline static void mul2x2o_mask(Complex_8_pair * const z, const Complex_8_pair * const zp, const uint8_t mask, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_mul2i_mask(zl[0], zl[1], zpl[0], zpl[1], mask, w1); _mul2ni_mask(zl[2], zl[3], zpl[2], zpl[3], mask, w1);
		_store(4, z, 1, zl);
	}

	finline static void fwd2x2e(Complex_8_pair * const z, const TwiddleFactor & w1)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void fwd2x2o(Complex_8_pair * const z, const TwiddleFactor & w1)
	{
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul4e(Complex_8_pair * const z, const Complex_8_pair * const zp, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_mul2(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2n(zl[2], zl[3], zpl[2], zpl[3], w1);
		_bck2(zl[0], zl[2], w1); _bck2(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul4o(Complex_8_pair * const z, const Complex_8_pair * const zp, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_mul2i(zl[0], zl[1], zpl[0], zpl[1], w1); _mul2ni(zl[2], zl[3], zpl[2], zpl[3], w1);
		_bck2i(zl[0], zl[2], w1); _bck2i(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul4e_mask(Complex_8_pair * const z, const Complex_8_pair * const zp, const uint8_t mask, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2(zl[0], zl[2], w1); _fwd2(zl[1], zl[3], w1);
		_mul2_mask(zl[0], zl[1], zpl[0], zpl[1], mask, w1); _mul2n_mask(zl[2], zl[3], zpl[2], zpl[3], mask, w1);
		_bck2(zl[0], zl[2], w1); _bck2(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void mul4o_mask(Complex_8_pair * const z, const Complex_8_pair * const zp, const uint8_t mask, const TwiddleFactor & w1)
	{
		Complex_8_pair zpl[4]; _load(4, zpl, zp, 1);
		Complex_8_pair zl[4]; _load(4, zl, z, 1);
		_fwd2i(zl[0], zl[2], w1); _fwd2i(zl[1], zl[3], w1);
		_mul2i_mask(zl[0], zl[1], zpl[0], zpl[1], mask, w1); _mul2ni_mask(zl[2], zl[3], zpl[2], zpl[3], mask, w1);
		_bck2i(zl[0], zl[2], w1); _bck2i(zl[1], zl[3], w1);
		_store(4, z, 1, zl);
	}

	finline static void backward4_0_carry(Complex_8_pair * const z, const size_t m, Complex_8_pair f[4],
		const Double_8 & base, const Double_8 & base_inv, const double t2_n, const Double_8 & g
#if defined(CHECK_ERROR)
		, Complex_8_pair & err
#endif
	)
	{
		const TwiddleFactor w2 = TwiddleFactor::TF_1_16();

		Complex_8_pair zl[4]; _load(4, zl, z, m);
	 	_bck2(zl[0], zl[1], w2); _bck2i(zl[2], zl[3], w2);
	 	_bck2_0(zl[0], zl[2]); _bck2_0(zl[1], zl[3]);

		for (size_t i = 0; i < 4; ++i)
		{
			zl[i] = zl[i].flatten() * t2_n;
			const Complex_8_pair t = zl[i].round();
#if defined(CHECK_ERROR)
			err = err.max((zl[i] - t).abs());
#endif
			f[i] += t * g;
			zl[i].set(f[i].mod(base, base_inv));
		}

		_store(4, z, m, zl);
	}

	finline static void carry_fix(Complex_8_pair * const z, const size_t m, Complex_8_pair f[4], const Double_8 & base, const Double_8 & base_inv)
	{
		// f_n <= ((b-1)^2 * N + f_{n-1}) / b
		// f_max = (b-1)^2 * N * (1/b + 1/b^2 + 1/b^3 + ...) = (b-1) * N

		// f_0 <= (b-1) * N
		// f_1 <= ((b-1) * N + (b-1)) / b <= N + 1
		// f_{1 + k} <= (N / b^{k-1} + 1 + (b-1)) / b <= N / b^k + 1

		// GFN-20: b_min = 1024, N = 2^19 => f_3 <= 2^19 / (2^10)^2 + 1 = 1

		for (size_t k = 0; k < 3; ++k)
		{
			Complex_8_pair zl[4]; _load(4, zl, &z[k], m);
			for (size_t i = 0; i < 4; ++i)
			{
				f[i] += zl[i].flatten();
				zl[i].set(f[i].mod(base, base_inv));
			}
			_store(4, &z[k], m, zl);
		}

		Complex_8_pair zl[4]; _load(4, zl, &z[3], m);
		for (size_t i = 0; i < 4; ++i) zl[i] += f[i].elevate();
		_store(4, &z[3], m, zl);
	}
};

template<size_t N>
class transformCPU : public transform
{
private:
	const size_t _num_regs;
	const Double_8 _base, _base_inv;
	Complex_8_pair * const _z;
	Complex_8_pair * const _zp;
	TwiddleFactor * const _w;
	double _error;

public:
	transformCPU(const UInt32_8 & b, const uint32_t n, const size_t num_regs) : transform(b, n, EKind::CPU),
		_num_regs(num_regs), _base(UInt32_8_to_Double_8(b)), _base_inv(_base.inverse()),
		_z(static_cast<Complex_8_pair *>(align_new(num_regs * N * sizeof(Complex_8_pair), 2 * 1024 * 1024))),
		_zp(static_cast<Complex_8_pair *>(align_new(N * sizeof(Complex_8_pair), sizeof(Complex_8_pair)))),
		_w(static_cast<TwiddleFactor *>(align_new(N / 2 * sizeof(TwiddleFactor), sizeof(TwiddleFactor))))
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
	void getZi(Int32_8 * const d) const override
	{
		const Complex_8_pair * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			const Complex_8 zk = z[k].get();
			d[k + 0 * N] = Double_8_to_Int32_8_round(zk.real());
			d[k + 1 * N] = Double_8_to_Int32_8_round(zk.imag());
		}
	}

	void setZi(const Int32_8 * const d) override
	{
		Complex_8_pair * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			const Double_8 re = Int32_8_to_Double_8(d[k + 0 * N]), im = Int32_8_to_Double_8(d[k + 1 * N]);
			z[k].set(Complex_8(re, im));
		}
	}

private:
	static void forward4_0(Complex_8_pair * const z)
	{
		for (size_t k = 0; k < N / 4; ++k) Complex_8_pair::forward4_0(&z[k], N / 4);
	}

	static void square_e(Complex_8_pair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w20 = w[2 * sj + 0];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4e(&z[i], m, w1, w20);

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

			Complex_8_pair::square4e(&z[0], w40);
			Complex_8_pair::square4o(&z[4], w40);
			Complex_8_pair::square4e(&z[8], w41);
			Complex_8_pair::square4o(&z[12], w41);
		}
		else // if (m == 2)
		{
			Complex_8_pair::square2x2e(&z[0], w20);
			Complex_8_pair::square2x2o(&z[4], w20);
		}

		for (size_t i = 0; i < m; ++i) Complex_8_pair::backward4e(&z[i], m, w1, w20);
	}

	static void square_o(Complex_8_pair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w21 = w[2 * sj + 1];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4o(&z[i], m, w1, w21);

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

			Complex_8_pair::square4e(&z[0], w42);
			Complex_8_pair::square4o(&z[4], w42);
			Complex_8_pair::square4e(&z[8], w43);
			Complex_8_pair::square4o(&z[12], w43);
		}
		else // if (m == 2)
		{
			Complex_8_pair::square2x2e(&z[0], w21);
			Complex_8_pair::square2x2o(&z[4], w21);
		}

		for (size_t i = 0; i < m; ++i) Complex_8_pair::backward4o(&z[i], m, w1, w21);
	}

	static void forward_e(Complex_8_pair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w20 = w[2 * sj + 0];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4e(&z[i], m, w1, w20);

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

			Complex_8_pair::fwd2x2e(&z[0], w40);
			Complex_8_pair::fwd2x2o(&z[4], w40);
			Complex_8_pair::fwd2x2e(&z[8], w41);
			Complex_8_pair::fwd2x2o(&z[12], w41);
		}
	}

	static void forward_o(Complex_8_pair * const z, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w21 = w[2 * sj + 1];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4o(&z[i], m, w1, w21);

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

			Complex_8_pair::fwd2x2e(&z[0], w42);
			Complex_8_pair::fwd2x2o(&z[4], w42);
			Complex_8_pair::fwd2x2e(&z[8], w43);
			Complex_8_pair::fwd2x2o(&z[12], w43);
		}
	}

	static void mul_e(Complex_8_pair * const z, const Complex_8_pair * const zp, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w20 = w[2 * sj + 0];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4e(&z[i], m, w1, w20);

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

			Complex_8_pair::mul4e(&z[0], &zp[0], w40);
			Complex_8_pair::mul4o(&z[4], &zp[4], w40);
			Complex_8_pair::mul4e(&z[8], &zp[8], w41);
			Complex_8_pair::mul4o(&z[12], &zp[12], w41);
		}
		else // if (m == 2)
		{
			Complex_8_pair::mul2x2e(&z[0], &zp[0], w20);
			Complex_8_pair::mul2x2o(&z[4], &zp[4], w20);
		}

		for (size_t i = 0; i < m; ++i) Complex_8_pair::backward4e(&z[i], m, w1, w20);
	}

	static void mul_o(Complex_8_pair * const z, const Complex_8_pair * const zp, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w21 = w[2 * sj + 1];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4o(&z[i], m, w1, w21);

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

			Complex_8_pair::mul4e(&z[0], &zp[0], w42);
			Complex_8_pair::mul4o(&z[4], &zp[4], w42);
			Complex_8_pair::mul4e(&z[8], &zp[8], w43);
			Complex_8_pair::mul4o(&z[12], &zp[12], w43);
		}
		else // if (m == 2)
		{
			Complex_8_pair::mul2x2e(&z[0], &zp[0], w21);
			Complex_8_pair::mul2x2o(&z[4], &zp[4], w21);
		}

		for (size_t i = 0; i < m; ++i) Complex_8_pair::backward4o(&z[i], m, w1, w21);
	}

	static void mul_e_mask(Complex_8_pair * const z, const Complex_8_pair * const zp, const uint8_t mask, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w20 = w[2 * sj + 0];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4e(&z[i], m, w1, w20);

		if (m > 4)
		{
			mul_e_mask(&z[0 * m], &zp[0 * m], mask, w, m / 4, 4 * sj + 0);
			mul_o_mask(&z[1 * m], &zp[1 * m], mask, w, m / 4, 4 * sj + 0);
			mul_e_mask(&z[2 * m], &zp[2 * m], mask, w, m / 4, 4 * sj + 1);
			mul_o_mask(&z[3 * m], &zp[3 * m], mask, w, m / 4, 4 * sj + 1);
		}
		else if (m == 4)
		{
			const TwiddleFactor w40 = w[4 * sj + 0], w41 = w[4 * sj + 1];

			Complex_8_pair::mul4e_mask(&z[0], &zp[0], mask, w40);
			Complex_8_pair::mul4o_mask(&z[4], &zp[4], mask, w40);
			Complex_8_pair::mul4e_mask(&z[8], &zp[8], mask, w41);
			Complex_8_pair::mul4o_mask(&z[12], &zp[12], mask, w41);
		}
		else // if (m == 2)
		{
			Complex_8_pair::mul2x2e_mask(&z[0], &zp[0], mask, w20);
			Complex_8_pair::mul2x2o_mask(&z[4], &zp[4], mask, w20);
		}

		for (size_t i = 0; i < m; ++i) Complex_8_pair::backward4e(&z[i], m, w1, w20);
	}

	static void mul_o_mask(Complex_8_pair * const z, const Complex_8_pair * const zp, const uint8_t mask, const TwiddleFactor * const w, const size_t m, const size_t sj)
	{
		const TwiddleFactor w1 = w[sj], w21 = w[2 * sj + 1];

		for (size_t i = 0; i < m; ++i) Complex_8_pair::forward4o(&z[i], m, w1, w21);

		if (m > 4)
		{
			mul_e_mask(&z[0 * m], &zp[0 * m], mask, w, m / 4, 4 * sj + 2);
			mul_o_mask(&z[1 * m], &zp[1 * m], mask, w, m / 4, 4 * sj + 2);
			mul_e_mask(&z[2 * m], &zp[2 * m], mask, w, m / 4, 4 * sj + 3);
			mul_o_mask(&z[3 * m], &zp[3 * m], mask, w, m / 4, 4 * sj + 3);
		}
		else if (m == 4)
		{
			const TwiddleFactor w42 = w[4 * sj + 2], w43 = w[4 * sj + 3];

			Complex_8_pair::mul4e_mask(&z[0], &zp[0], mask, w42);
			Complex_8_pair::mul4o_mask(&z[4], &zp[4], mask, w42);
			Complex_8_pair::mul4e_mask(&z[8], &zp[8], mask, w43);
			Complex_8_pair::mul4o_mask(&z[12], &zp[12], mask, w43);
		}
		else // if (m == 2)
		{
			Complex_8_pair::mul2x2e_mask(&z[0], &zp[0], mask, w21);
			Complex_8_pair::mul2x2o_mask(&z[4], &zp[4], mask, w21);
		}

		for (size_t i = 0; i < m; ++i) Complex_8_pair::backward4o(&z[i], m, w1, w21);
	}

	static double backward4_0_carry(Complex_8_pair * const z, const Double_8 & base, const Double_8 & base_inv, const uint32_t dup)
	{
		Complex_8_pair f[4]; for (size_t i = 0; i < 4; ++i) f[i] = Complex_8_pair(0.0);

		Double_8 g; g.copy_mask(Double_8(2), Double_8(1), uint8_t(dup));

#if defined(CHECK_ERROR)
		Complex_8_pair err = Complex_8_pair(0.0);
#endif
		for (size_t k = 0; k < N / 4; ++k)
		{
			Complex_8_pair::backward4_0_carry(&z[k], N / 4, f, base, base_inv, 2.0 / N, g
#if defined(CHECK_ERROR)
				, err
#endif
			);
		}

		// modulo z^n + 1
		const Complex_8_pair t = f[3].shift(); f[3] = f[2]; f[2] = f[1]; f[1] = f[0]; f[0] = t;

		Complex_8_pair::carry_fix(z, N / 4, f, base, base_inv);

#if defined(CHECK_ERROR)
		return err.max().max();
#else
		return 0;
#endif
	}

public:
	void set(const uint32_t a) override
	{
		Complex_8_pair * const z = _z;

		z[0] = Complex_8_pair(static_cast<double>(a));
		for (size_t k = 1; k < N; ++k) z[k] = Complex_8_pair(0.0);
	}

	void square_dup(const uint32_t dup) override
	{
		Complex_8_pair * const z = _z;
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
		const Complex_8_pair * const z_src = &_z[src * N];
		Complex_8_pair * const zp = _zp;
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
		Complex_8_pair * const z = _z;
		const Complex_8_pair * const zp = _zp;
		const TwiddleFactor * const w = _w;

		forward4_0(z);
		mul_e(&z[0 * (N / 4)], &zp[0 * (N / 4)], w, N / 16, 4 + 0);
		mul_o(&z[1 * (N / 4)], &zp[1 * (N / 4)], w, N / 16, 4 + 0);
		mul_e(&z[2 * (N / 4)], &zp[2 * (N / 4)], w, N / 16, 4 + 1);
		mul_o(&z[3 * (N / 4)], &zp[3 * (N / 4)], w, N / 16, 4 + 1);
		const double err = backward4_0_carry(z, _base, _base_inv, 0);
		_error = std::max(_error, err);
	}

	void mul_mask(const uint8_t mask) override
	{
		Complex_8_pair * const z = _z;
		const Complex_8_pair * const zp = _zp;
		const TwiddleFactor * const w = _w;

		forward4_0(z);
		mul_e_mask(&z[0 * (N / 4)], &zp[0 * (N / 4)], mask, w, N / 16, 4 + 0);
		mul_o_mask(&z[1 * (N / 4)], &zp[1 * (N / 4)], mask, w, N / 16, 4 + 0);
		mul_e_mask(&z[2 * (N / 4)], &zp[2 * (N / 4)], mask, w, N / 16, 4 + 1);
		mul_o_mask(&z[3 * (N / 4)], &zp[3 * (N / 4)], mask, w, N / 16, 4 + 1);
		const double err = backward4_0_carry(z, _base, _base_inv, 0);
		_error = std::max(_error, err);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const Complex_8_pair * const z_src = &_z[src * N];
		Complex_8_pair * const z_dst =  &_z[dst * N];

		for (size_t k = 0; k < N; ++k) z_dst[k] = z_src[k];
	}

	void copy_mask(const size_t dst, const size_t src, const uint8_t mask) const override
	{
		const Complex_8_pair * const z_src = &_z[src * N];
		Complex_8_pair * const z_dst =  &_z[dst * N];

		for (size_t k = 0; k < N; ++k) z_dst[k].copy_mask(z_src[k], mask);
	}

	bool read_checkpoint(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.read(reinterpret_cast<char *>(_z), num_regs * N * sizeof(Complex_8_pair))) return false;

		return true;
	}

	void save_checkpoint(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.write(reinterpret_cast<const char *>(_z), num_regs * N * sizeof(Complex_8_pair))) return;
	}

	size_t get_cache_size() const override { return N * sizeof(Complex_8_pair); }
	double get_error() const override { return _error; }

	void cosmic_ray() override { const Complex_8 z = _z[N / 2].get(); Double_8 x = z.real(); x.cosmic_ray(); _z[N / 2].set(Complex_8(x, z.imag())); }
};

inline transform * create_transformCPU(const UInt32_8 & b, const uint32_t n, const size_t num_regs)
{
	transform * ptransform = nullptr;
	/*if      (n ==  5) ptransform = new transformCPU<(1 <<  4)>(b, n, num_regs);
	else if (n ==  6) ptransform = new transformCPU<(1 <<  5)>(b, n, num_regs);
	else if (n ==  7) ptransform = new transformCPU<(1 <<  6)>(b, n, num_regs);
	else*/ if (n ==  8) ptransform = new transformCPU<(1 <<  7)>(b, n, num_regs);
	else if (n ==  9) ptransform = new transformCPU<(1 <<  8)>(b, n, num_regs);
	/*else if (n == 10) ptransform = new transformCPU<(1 <<  9)>(b, n, num_regs);
	else if (n == 11) ptransform = new transformCPU<(1 << 10)>(b, n, num_regs);
	else if (n == 12) ptransform = new transformCPU<(1 << 11)>(b, n, num_regs);
	else if (n == 13) ptransform = new transformCPU<(1 << 12)>(b, n, num_regs);
	else if (n == 14) ptransform = new transformCPU<(1 << 13)>(b, n, num_regs);
	else if (n == 15) ptransform = new transformCPU<(1 << 14)>(b, n, num_regs);
	else if (n == 16) ptransform = new transformCPU<(1 << 15)>(b, n, num_regs);
	else if (n == 17) ptransform = new transformCPU<(1 << 16)>(b, n, num_regs);*/

	if (ptransform == nullptr) throw std::runtime_error("exponent is not supported");

	return ptransform;
}

}