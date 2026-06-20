/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "transform.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
typedef std::complex<double> Complex;

inline Complex fabs(const Complex & rhs)
{
	return Complex(std::fabs(rhs.real()), std::fabs(rhs.imag()));
}

inline Complex fmax(const Complex & z0, const Complex & z1)
{
	return Complex(std::fmax(z0.real(), z1.real()), std::fmax(z0.imag(), z1.imag()));
}

inline Complex round(const Complex & rhs)
{
	return Complex(std::round(rhs.real()), std::round(rhs.imag()));
}

inline Complex mod(Complex & rhs, const double m, const double m_inv)
{
	Complex r = rhs;
	rhs = round(rhs * m_inv);
	return r - rhs * m;
}

namespace transformCPU_namespace
{

template<size_t N>
class transformCPU : public transform
{
private:
	const size_t _num_regs;
	Complex * const _z;
	Complex * const _zp;
	Complex * const _w;
	double _error;

public:
	transformCPU(const uint32_t b, const uint32_t n, const size_t num_regs) : transform(b, n, EKind::CPU),
		_num_regs(num_regs), _z(new Complex[num_regs * N]), _zp(new Complex[N]), _w(new Complex[N])
	{
		Complex * const w = _w;
		for (size_t s = 1; s < N; s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t e = bitrev(j, 4 * s) + 1, n = 2 * 4 * s;
				w[s + j] = std::polar(1.0, (2 * M_PI / n) * e);
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
	void forward(Complex * const z)
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
					const Complex u = z[k + 0 * m], v = z[k + 1 * m] * wsj;
					z[k + 0 * m] = u + v; z[k + 1 * m] = u - v;
				}
			}
		}
	}

	void backward(Complex * const z)
	{
		const Complex * const w = _w;

		for (size_t m = 1, s = N / 2; s >= 1; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Complex wsj_inv = std::conj(w[s + j]);
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const Complex u = z[k + 0 * m], v = z[k + 1 * m];
					z[k + 0 * m] = u + v; z[k + 1 * m] = (u - v) * wsj_inv;
				}
			}
		}
	}

	static void mult(Complex * const z, const Complex * const zp)
	{
		for (size_t k = 0; k < N; ++k) z[k] *= zp[k];
	}

	void carry(const bool dup)
	{
		Complex * const z = _z;

		const double base = get_b(), base_inv = 1.0 / base, t1_n = 1.0 / N;
		const double g = dup ? 2 : 1;

		Complex f = 0.0, e = 0.0;

		for (size_t k = 0; k < N; ++k)
		{
			const Complex u = z[k] * t1_n;
			const Complex u_i = round(u);
			e = fmax(e, fabs(u_i - u));
			f += u_i * g;
			z[k] = mod(f, base, base_inv);
		}

		_error = std::fmax(_error, std::fmax(e.real(), e.imag()));

		while (f != 0.0)
		{
			f = Complex(-f.imag(), f.real());	// a[n] = -a[0]

			for (size_t k = 0; k < N; ++k)
			{
				f += z[k];
				z[k] = mod(f, base, base_inv);
				if (f == 0.0) break;
			}
		}
	}

protected:
	void getZi(int32_t * const zi) const override
	{
		const Complex * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			zi[k + 0 * N] = std::lround(z[k].real());
			zi[k + 1 * N] = std::lround(z[k].imag());
		}
	}

	void setZi(const int32_t * const zi) override
	{
		Complex * const z = _z;

		for (size_t k = 0; k < N; ++k)
		{
			z[k] = Complex(static_cast<double>(zi[k + 0 * N]), static_cast<double>(zi[k + 1 * N]));
		}
	}

public:
	void set(const uint32_t a) override
	{
		Complex * const z = _z;

		z[0] = Complex(a);
		for (size_t k = 1; k < N; ++k) z[k] = 0.0;
	}

	void square_dup(const bool dup) override
	{
		Complex * const z = _z;

		forward(z);
		mult(z, z);
		backward(z);
		carry(dup);
	}

	void init_multiplicand(const size_t src) override
	{
		const Complex * const z_src = &_z[src * N];
		Complex * const zp = _zp;

		for (size_t k = 0; k < N; ++k) zp[k] = z_src[k];
		forward(zp);
	}

	void mul() override
	{
		Complex * const z = _z;

		forward(z);
		mult(z, _zp);
		backward(z);
		carry(false);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const Complex * const z_src = &_z[src * N];
		Complex * const z_dst =  &_z[dst * N];

		for (size_t k = 0; k < N; ++k) z_dst[k] = z_src[k];
	}

	bool read_checkpoint(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.read(reinterpret_cast<char *>(_z), num_regs * N * sizeof(Complex))) return false;

		return true;
	}

	void save_checkpoint(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		if (!cFile.write(reinterpret_cast<const char *>(_z), num_regs * N * sizeof(Complex))) return;
	}

	size_t get_cache_size() const override { return N * sizeof(Complex); }
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