/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include "transform.h"

namespace transformCPU_namespace
{

class Complex
{
private:
	double _re, _im;

public:
	Complex() {}
	Complex(const double & re, const double & im = 0) : _re(re), _im(im) {}

	double real() const { return _re; }
	double imag() const { return _im; }

	bool is_zero() const { const bool r = (_re == 0) && (_im == 0); return r; }

	Complex & operator+=(const Complex & rhs) { _re += rhs._re; _im += rhs._im; return *this; }
	Complex & operator-=(const Complex & rhs) { _re -= rhs._re; _im -= rhs._im; return *this; }
	Complex & operator*=(const double rhs) { _re *= rhs; _im *= rhs; return *this; }
	Complex & operator*=(const Complex & rhs) { const double t = _re * rhs._im + _im * rhs._re; _re = _re * rhs._re - _im * rhs._im; _im = t; return *this; }

	Complex operator+(const Complex & rhs) const { return Complex(_re + rhs._re, _im + rhs._im); }
	Complex operator-(const Complex & rhs) const { return Complex(_re - rhs._re, _im - rhs._im); }
	Complex operator*(const double rhs) const { return Complex(_re * rhs, _im * rhs); }
	Complex operator*(const Complex & rhs) const { return Complex(_re * rhs._re - _im * rhs._im, _re * rhs._im + _im * rhs._re); }

	Complex mulW(const Complex & rhs) const { return Complex((_re - _im * rhs._im) * rhs._re, (_im + _re * rhs._im) * rhs._re); }
	Complex mulWconj(const Complex & rhs) const { return Complex((_re + _im * rhs._im) * rhs._re, (_im - _re * rhs._im) * rhs._re); }

	Complex muli() const { return Complex(-_im, _re); }

	Complex round() const { return Complex(std::round(_re), std::round(_im)); }
	Complex abs() const { return Complex(std::fabs(_re), std::fabs(_im)); }
	Complex max(const Complex & rhs) const { return Complex(std::max(_re, rhs._re), std::max(_im, rhs._im)); }

	static Complex exp2iPi(const size_t a, const size_t b)
	{
#define	C2PI	6.2831853071795864769252867665590057684L
		const long double alpha = C2PI * (long double)a / (long double)b;
		const double cs = static_cast<double>(std::cosl(alpha)), sn = static_cast<double>(std::sinl(alpha));
		return Complex(cs, sn / cs);
	}
};

class ComplexPair
{
private:
	static constexpr double split = 1 << 20, split_inv = 1.0 / split;
	Complex _l, _h;

public:
	ComplexPair() {}
	ComplexPair(const Complex & l, const Complex & h = 0.0) : _l(l), _h(h) {}

	void set(const Complex & rhs) { _h = (rhs * split_inv).round() * ComplexPair::split; _l = rhs - _h; }
	Complex get() const { return _l + _h; }

	bool is_zero() const { const bool r = _l.is_zero() && _h.is_zero(); return r; }

	ComplexPair & operator+=(const ComplexPair & rhs) { _l += rhs._l; _h += rhs._h; return *this; }
	ComplexPair & operator-=(const ComplexPair & rhs) { _l -= rhs._l; _h -= rhs._h; return *this; }
	ComplexPair & operator*=(const double rhs) { _l *= rhs; _h *= rhs; return *this; }

	ComplexPair operator+(const ComplexPair & rhs) const { return ComplexPair(_l + rhs._l, _h + rhs._h); }
	ComplexPair operator-(const ComplexPair & rhs) const { return ComplexPair(_l - rhs._l, _h - rhs._h); }
	ComplexPair operator*(const double rhs) const { return ComplexPair(_l * rhs, _h * rhs); }

	ComplexPair mulW(const Complex & rhs) const { return ComplexPair(_l.mulW(rhs), _h.mulW(rhs)); }
	ComplexPair mulWconj(const Complex & rhs) const { return ComplexPair(_l.mulWconj(rhs), _h.mulWconj(rhs)); }

	ComplexPair & operator*=(const ComplexPair & rhs)
	{
		_h = _l * rhs._h + _h * (rhs._l + rhs._h);
		_l *= rhs._l;
		return *this;
	}

	ComplexPair adjust() const { return ComplexPair(_l, _h * split_inv); }

	ComplexPair round() const { return ComplexPair(_l.round(), _h.round()); }
	ComplexPair abs() const { return ComplexPair(_l.abs(), _h.abs()); }
	Complex max() const { return _l.max(_h); }

	ComplexPair shift() const { return ComplexPair(_l.muli(), _h.muli()); }

	Complex mod(const double m, const double m_inv)
	{
		Complex l_m = (_l * m_inv).round(), rl_m = _l - l_m * m;
		const Complex h_m = (_h * m_inv).round(), rh_m = _h - h_m * m;
		_h = h_m;

		rl_m += rh_m * split;
		const Complex rl_m2 = (rl_m * m_inv).round(); rl_m -= rl_m2 * m; l_m += rl_m2;
		_l = l_m;

		return rl_m;
	}
};

template<size_t N>
class transformCPU : public transform
{
private:
	const size_t _num_regs;
	ComplexPair * const _z;
	ComplexPair * const _zp;
	Complex * const _w;
	double _error;

public:
	transformCPU(const uint32_t b, const uint32_t n, const size_t num_regs) : transform(b, n, EKind::CPU),
		_num_regs(num_regs), _z(new ComplexPair[num_regs * N]), _zp(new ComplexPair[N]), _w(new Complex[N])
	{
		Complex * const w = _w;
		for (size_t s = 1; s < N; s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				w[s + j] = Complex::exp2iPi(bitrev(j, 4 * s) + 1, 2 * 4 * s);
			}
		}

		_error = 0;
	}

	virtual ~transformCPU()
	{
		delete[] _z;
		delete[] _zp;
		delete[] _w;
	}

private:
	void forward(ComplexPair * const z)
	{
		const Complex * const w = _w;

		for (size_t m = N / 2, s = 1; m >= 1; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Complex wsj = w[s + j];
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const ComplexPair u = z[k + 0 * m], v = z[k + 1 * m].mulW(wsj);
					z[k + 0 * m] = u + v; z[k + 1 * m] = u - v;
				}
			}
		}
	}

	void backward(ComplexPair * const z)
	{
		const Complex * const w = _w;

		for (size_t m = 1, s = N / 2; s >= 1; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Complex wsj = w[s + j];
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const ComplexPair u = z[k + 0 * m], v = z[k + 1 * m];
					z[k + 0 * m] = u + v; z[k + 1 * m] = (u - v).mulWconj(wsj);
				}
			}
		}
	}

	static void mult(ComplexPair * const z, const ComplexPair * const zp)
	{
		for (size_t k = 0; k < N; ++k) z[k] *= zp[k];
	}

	void carry(const bool dup)
	{
		ComplexPair * const z = _z;

		const double base = get_b(), base_inv = 1.0 / base, t1_n = 1.0 / N;
		const double g = dup ? 2 : 1;

		ComplexPair f(0.0);
		Complex err = 0.0;

		for (size_t k = 0; k < N; ++k)
		{
			ComplexPair of = z[k].adjust() * t1_n;
			const ComplexPair o = of.round();
			err = err.max((of - o).abs().max());
			f += o * g;

			z[k].set(f.mod(base, base_inv));
		}

		_error = std::fmax(_error, std::fmax(err.real(), err.imag()));

		while (!f.is_zero())
		{
			f = f.shift();	// a[n] = -a[0]

			for (size_t k = 0; k < N; ++k)
			{
				f += z[k].adjust();

				z[k].set(f.mod(base, base_inv));

				if (f.is_zero()) return;
			}
		}
	}

protected:
	void getZi(int32_t * const zi) const override
	{
		const ComplexPair * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			const Complex zk = z[k].get();
			zi[k + 0 * N] = std::lround(zk.real());
			zi[k + 1 * N] = std::lround(zk.imag());
		}
	}

	void setZi(const int32_t * const zi) override
	{
		ComplexPair * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			z[k].set(Complex(static_cast<double>(zi[k + 0 * N]), static_cast<double>(zi[k + 1 * N])));
		}
	}

public:
	void set(const uint32_t a) override
	{
		ComplexPair * const z = _z;

		z[0] = ComplexPair(static_cast<double>(a));
		for (size_t k = 1; k < N; ++k) z[k] = ComplexPair(0.0);
	}

	void square_dup(const bool dup) override
	{
		ComplexPair * const z = _z;

		forward(z);
		mult(z, z);
		backward(z);
		carry(dup);
	}

	void init_multiplicand(const size_t src) override
	{
		const ComplexPair * const z_src = &_z[src * N];
		ComplexPair * const zp = _zp;

		for (size_t k = 0; k < N; ++k) zp[k] = z_src[k];
		forward(zp);
	}

	void mul() override
	{
		ComplexPair * const z = _z;

		forward(z);
		mult(z, _zp);
		backward(z);
		carry(false);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const ComplexPair * const z_src = &_z[src * N];
		ComplexPair * const z_dst =  &_z[dst * N];

		for (size_t k = 0; k < N; ++k) z_dst[k] = z_src[k];
	}

	bool read_checkpoint(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.read(reinterpret_cast<char *>(_z), num_regs * N * sizeof(ComplexPair))) return false;

		return true;
	}

	void save_checkpoint(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.write(reinterpret_cast<const char *>(_z), num_regs * N * sizeof(ComplexPair))) return;
	}

	size_t get_cache_size() const override { return N * sizeof(ComplexPair); }
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