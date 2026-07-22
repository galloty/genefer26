/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include "vdouble.h"

#ifndef finline
#define finline	__attribute__((always_inline)) inline
#endif

namespace arch_namespace
{

#define vmadd(x, y, z)	(x * y + z)
#define vnmadd(x, y, z) (z - x * y)
#define vmsub(x, y, z)  (x * y - z)

class TwiddleFactor
{
	typedef double	double_2 __attribute__ ((vector_size(2 * sizeof(double))));

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

class Complex_8
{
private:
	static constexpr double _csqrt2_2 = 0.70710678118654752440084436210484903929;
	Double_8 _re, _im;

public:
	finline explicit Complex_8() {}
	finline explicit Complex_8(const Double_8 & re, const Double_8 & im) : _re(re), _im(im) {}
	finline explicit Complex_8(const double f) : _re(f), _im(0) {}

	finline const Double_8 & real() const { return _re; }
	finline const Double_8 & imag() const { return _im; }

	finline void copy_mask(const Complex_8 & rhs, const uint8_t mask) { _re.copy_mask(rhs._re, _re, mask); _im.copy_mask(rhs._im, _im, mask); }

	finline Complex_8 & operator+=(const Complex_8 & rhs) { _re += rhs._re; _im += rhs._im; return *this; }
	finline Complex_8 & operator-=(const Complex_8 & rhs) { _re -= rhs._re; _im -= rhs._im; return *this; }
	finline Complex_8 & operator*=(const double rhs) { _re *= rhs; _im *= rhs; return *this; }
	finline Complex_8 & operator*=(const Double_8 & rhs) { _re *= rhs; _im *= rhs; return *this; }

	finline Complex_8 operator+(const Complex_8 & rhs) const { return Complex_8(_re + rhs._re, _im + rhs._im); }
	finline Complex_8 operator-(const Complex_8 & rhs) const { return Complex_8(_re - rhs._re, _im - rhs._im); }
	finline Complex_8 operator*(const double rhs) const { return Complex_8(_re * rhs, _im * rhs); }
	finline Complex_8 operator*(const Double_8 & rhs) const { return Complex_8(_re * rhs, _im * rhs); }
	finline Complex_8 addi(const Complex_8 & rhs) const { return Complex_8(_re - rhs._im, _im + rhs._re); }
	finline Complex_8 subi(const Complex_8 & rhs) const { return Complex_8(_re + rhs._im, _im - rhs._re); }
	// finline Complex_8 muli() const { return Complex_8(-_im, _re); }

	finline Complex_8 submul(const Complex_8 & rhs, const Double_8 & c) const { return Complex_8(vnmadd(rhs._re, c, _re), vnmadd(rhs._im, c, _im)); }

	finline Complex_8 mulTF_0() const { return Complex_8((_re - _im) * _csqrt2_2, (_im + _re) * _csqrt2_2); }
	finline Complex_8 mulTFconj_0() const { return Complex_8((_re + _im) * _csqrt2_2, (_im - _re) * _csqrt2_2); }

	finline Complex_8 mulTF(const TwiddleFactor & rhs) const 
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return Complex_8(vnmadd(_im, tg, _re) * cs, vmadd(_re, tg, _im) * cs);
	}
	finline Complex_8 mulTFconj(const TwiddleFactor & rhs) const
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return Complex_8(vmadd(_im, tg, _re) * cs, vnmadd(_re, tg, _im) * cs);
	}
	finline Complex_8 mulTFconji(const TwiddleFactor & rhs) const
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return Complex_8(vmsub(_re, tg, _im) * cs, vmadd(_im, tg, _re) * cs);
	}

	finline Complex_8 sqr() const { const Double_8 t = _re * _im; return Complex_8(_re * _re - _im * _im, t + t); }
	finline Complex_8 mul(const Complex_8 & rhs) const { return Complex_8(_re * rhs._re - _im * rhs._im, _re * rhs._im + _im * rhs._re); }

	finline Complex_8 mul_mask(const Complex_8 & rhs, const uint8_t mask) const
	{
		const Double_8 re = _re.mul_mask(rhs._re, mask) - _im.mul_maskz(rhs._im, mask);
		const Double_8 im = _re.mul_maskz(rhs._im, mask) + _im.mul_mask(rhs._re, mask);
		return Complex_8(re, im);
	}
	finline Complex_8 mul_maskz(const Complex_8 & rhs, const uint8_t mask) const
	{
		const Double_8 re = _re.mul_maskz(rhs._re, mask) - _im.mul_maskz(rhs._im, mask);
		const Double_8 im = _re.mul_maskz(rhs._im, mask) + _im.mul_maskz(rhs._re, mask);
		return Complex_8(re, im);
	}

	finline Complex_8 round() const { return Complex_8(_re.round(), _im.round()); }
	finline Complex_8 abs() const { return Complex_8(_re.abs(), _im.abs()); }
	finline Complex_8 max(const Complex_8 & rhs) const { return Complex_8(_re.max(rhs._re), _im.max(rhs._im)); }
	finline double max() const { return std::max(_re.max(), _im.max()); }

	finline Complex_8 shift() const { return Complex_8(-_im, _re); }
};

}
