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

#define finline	__attribute__((always_inline))

namespace transformCPU_namespace
{

typedef double	double_2 __attribute__ ((vector_size(2 * sizeof(double))));
typedef double	double_8 __attribute__ ((vector_size(8 * sizeof(double))));

inline double_2 addmul_2(const double_2 & x, const double_2 & y, const double c) { return x + y * c; }
inline double_2 submul_2(const double_2 & x, const double_2 & y, const double c) { return x - y * c; }

inline double_2 round_2(const double_2 & x) { return _mm_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }

inline double_2 abs_2(const double_2 & x)
{
	const long long m = (long long)(uint64_t(1) << 63);
	return _mm_andnot_pd(_mm_castsi128_pd((__m128i){m, m}), x);
}

inline double_2 max_2(const double_2 & x, const double_2 & y) { return _mm_max_pd(x, y); }

class TwiddleFactor
{
private:
	double_2 _z;

public:
	TwiddleFactor() {}
	TwiddleFactor(const size_t a, const size_t b)
	{
#define	C2PI	6.2831853071795864769252867665590057684L
		const long double alpha = C2PI * (long double)a / (long double)b;
		const double cs = static_cast<double>(std::cosl(alpha)), sn = static_cast<double>(std::sinl(alpha));
		_z[0] = cs; _z[1] = sn / cs;
	}

	double cos() const { return _z[0]; }
	double tan() const { return _z[1]; }
};

class VComplex
{
private:
	double_2 _re, _im;

	finline constexpr explicit VComplex(const double_2 & re, const double_2 & im) : _re(re), _im(im) {}

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

	finline VComplex addmul(const VComplex & rhs, const double c) const { return VComplex(addmul_2(_re, rhs._re, c), addmul_2(_im, rhs._im, c)); }
	finline VComplex submul(const VComplex & rhs, const double c) const { return VComplex(submul_2(_re, rhs._re, c), submul_2(_im, rhs._im, c)); }

	finline VComplex mulW(const TwiddleFactor & rhs) const 
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return VComplex(submul_2(_re, _im, tg) * cs, addmul_2(_im, _re, tg) * cs);
	}
	finline VComplex mulWconj(const TwiddleFactor & rhs) const
	{
		const double cs = rhs.cos(), tg = rhs.tan();
		return VComplex(addmul_2(_re, _im, tg) * cs, submul_2(_im, _re, tg) * cs);
	}

	finline VComplex muli() const { return VComplex(-_im, _re); }	// TODO remove

	finline VComplex sqr() const { const double_2 t = _re * _im; return VComplex(_re * _re - _im * _im, t + t); }
	finline VComplex mul(const VComplex & rhs) const { return VComplex(_re * rhs._re - _im * rhs._im, _re * rhs._im + _im * rhs._re); }

	finline VComplex round() const { return VComplex(round_2(_re), round_2(_im)); }
	finline VComplex abs() const { return VComplex(abs_2(_re), abs_2(_im)); }
	finline VComplex max(const VComplex & rhs) const { return VComplex(max_2(_re, rhs._re), max_2(_im, rhs._im)); }
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

	finline VComplexPair mulW(const TwiddleFactor & rhs) const { return VComplexPair(_l.mulW(rhs), _h.mulW(rhs)); }
	finline VComplexPair mulWconj(const TwiddleFactor & rhs) const { return VComplexPair(_l.mulWconj(rhs), _h.mulWconj(rhs)); }

	finline VComplexPair muli() const { return VComplexPair(_l.muli(), _h.muli()); }	// TODO remove

	finline VComplexPair sqr() const { return VComplexPair(_l.sqr(), _l.mul(_h + _h) + _h.sqr()); }
	finline VComplexPair mul(const VComplexPair & rhs) const { return VComplexPair(_l.mul(rhs._l), _l.mul(rhs._h) + _h.mul(rhs.get())); }

	finline VComplexPair adjust() const { return VComplexPair(_l, _h * split_inv); }

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
	static void _load(const size_t n, VComplexPair * const zl, const VComplexPair * const z, const size_t s)
	{
		for (size_t l = 0; l < n; ++l) zl[l] = z[l * s];
	}
	static void _store(const size_t n, VComplexPair * const z, const size_t s, const VComplexPair * const zl)
	{
		for (size_t l = 0; l < n; ++l) z[l * s] = zl[l];
	}

	static void _fwd2(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.mulW(w); z1 = z0 - t; z0 += t;
	}
	static void _fwd2i(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.mulW(w); z1 = z0.subi(t); z0 = z0.addi(t);
	}
	static void _bck2(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z0 - z1; z0 += z1, z1 = t.mulWconj(w);
	}

	static void _bck2i(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z0.addi(z1); z0 = z0.subi(z1), z1 = t.mulWconj(w);
	}

	static void _sqr2(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.sqr().mulW(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr() + t;
	}
	static void _sqr2n(VComplexPair & z0, VComplexPair & z1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.sqr().mulW(w); z1 = z0.mul(z1 + z1); z0 = z0.sqr() - t;
	}

	static void _mul2(VComplexPair & z0, VComplexPair & z1, const VComplexPair & zp0, const VComplexPair & zp1, const TwiddleFactor & w) 
	{
		const VComplexPair t = z1.mul(zp1).mulW(w); z1 = z0.mul(zp1) + zp0.mul(z1); z0 = z0.mul(zp0) + t;
	}
	static void _mul2n(VComplexPair & z0, VComplexPair & z1, const VComplexPair & zp0, const VComplexPair & zp1, const TwiddleFactor & w)
	{
		const VComplexPair t = z1.mul(zp1).mulW(w); z1 = z0.mul(zp1) + zp0.mul(z1); z0 = z0.mul(zp0) - t;
	}

	static void _forward4(VComplexPair z[4], const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		_fwd2(z[0], z[2], w1); _fwd2(z[1], z[3], w1);
		_fwd2(z[0], z[1], w2); _fwd2i(z[2], z[3], w2);
	}

	static void _backward4(VComplexPair z[4], const TwiddleFactor & w1, const TwiddleFactor & w2)
	{
		_bck2(z[0], z[1], w2); _bck2(z[2], z[3], w2);
		_bck2(z[0], z[2], w1); _bck2i(z[1], z[3], w1);
	}

	static void _square2x2(VComplexPair z[4], const TwiddleFactor & w1)
	{
		_sqr2(z[0], z[1], w1); _sqr2n(z[2], z[3], w1);
	}

	static void _square4(VComplexPair z[4], const TwiddleFactor & w1)
	{
		_fwd2(z[0], z[2], w1); _fwd2(z[1], z[3], w1);
		_square2x2(z, w1);
		_bck2(z[0], z[2], w1); _bck2(z[1], z[3], w1);
	}

	static void _mul2x2(VComplexPair z[4], const VComplexPair zp[4], const TwiddleFactor & w1)
	{
		_mul2(z[0], z[1], zp[0], zp[1], w1); _mul2n(z[2], z[3], zp[2], zp[3], w1);
	}

	static void _fwd2x2(VComplexPair zp[4], const TwiddleFactor & w1)
	{
		_fwd2(zp[0], zp[2], w1); _fwd2(zp[1], zp[3], w1);
	}

	static void _mul4(VComplexPair z[4], const VComplexPair zp[4], const TwiddleFactor & w1)
	{
		_fwd2(z[0], z[2], w1); _fwd2(z[1], z[3], w1);
		_mul2x2(z, zp, w1);
		_bck2(z[0], z[2], w1); _bck2(z[1], z[3], w1);
	}

	static void forward4(VComplexPair * const z, const TwiddleFactor * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const TwiddleFactor w1 = w[s + j], w2 = w[2 * (s + j)];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				VComplexPair zl[4]; _load(4, zl, &z[k], m);
				_forward4(zl, w1, w2);
				_store(4, &z[k], m, zl);
			}
		}
	}

	static void backward4(VComplexPair * const z, const TwiddleFactor * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const TwiddleFactor w1 = w[s + j], w2 = w[2 * (s + j)];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				VComplexPair zl[4]; _load(4, zl, &z[k], m);
				_backward4(zl, w1, w2);
				_store(4, &z[k], m, zl);
			}
		}
	}

public:
	static size_t forward(VComplexPair * const z, const TwiddleFactor * const w, const size_t n_4)
	{
		size_t m = 4 * n_4, s = 1;
		for (; m >= 8; m /= 4, s *= 4) forward4(z, w, m / 4, s);
		return m;
	}

	static void backward(VComplexPair * const z, const TwiddleFactor * const w, const size_t n_4, const size_t m0)
	{
		for (size_t m = m0, s = n_4 / m; s >= 1; m *= 4, s /= 4) backward4(z, w, m, s);
	}

	static void square2(VComplexPair * const z, const TwiddleFactor * const w, const size_t n_4)
	{
		for (size_t j = 0; j < n_4; ++j)
		{
			const TwiddleFactor w1 = w[n_4 + j];

			VComplexPair zl[4]; _load(4, zl, &z[4 * j], 1);
			_square2x2(zl, w1);
			_store(4, &z[4 * j], 1, zl);
		}
	}

	static void square4(VComplexPair * const z, const TwiddleFactor * const w, const size_t n_4)
	{
		for (size_t j = 0; j < n_4; ++j)
		{
			const TwiddleFactor w1 = w[n_4 + j];

			VComplexPair zl[4]; _load(4, zl, &z[4 * j], 1);
			_square4(zl, w1);
			_store(4, &z[4 * j], 1, zl);
		}
	}

	static void mul2(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor * const w, const size_t n_4)
	{
		for (size_t j = 0; j < n_4; ++j)
		{
			const TwiddleFactor w1 = w[n_4 + j];

			VComplexPair zl[4]; _load(4, zl, &z[4 * j], 1);
			VComplexPair zpl[4]; _load(4, zpl, &zp[4 * j], 1);
			_mul2x2(zl, zpl, w1);
			_store(4, &z[4 * j], 1, zl);
		}
	}

	static void fwd2(VComplexPair * const zp, const TwiddleFactor * const w, const size_t n_4)
	{
		for (size_t j = 0; j < n_4; ++j)
		{
			const TwiddleFactor w1 = w[n_4 + j];

			VComplexPair zl[4]; _load(4, zl, &zp[4 * j], 1);
			_fwd2x2(zl, w1);
			_store(4, &zp[4 * j], 1, zl);
		}
	}

	static void mul4(VComplexPair * const z, const VComplexPair * const zp, const TwiddleFactor * const w, const size_t n_4)
	{
		for (size_t j = 0; j < n_4; ++j)
		{
			const TwiddleFactor w1 = w[n_4 + j];

			VComplexPair zl[4]; _load(4, zl, &z[4 * j], 1);
			VComplexPair zpl[4]; _load(4, zpl, &zp[4 * j], 1);
			_mul4(zl, zpl, w1);
			_store(4, &z[4 * j], 1, zl);
		}
	}
};

template<size_t N>
class transformCPU : public transform
{
private:
	const size_t _num_regs;
	VComplexPair * const _z;
	VComplexPair * const _zp;
	TwiddleFactor * const _w;
	double _error;

public:
	transformCPU(const uint32_t b, const uint32_t n, const size_t num_regs) : transform(b, n, EKind::CPU),
		_num_regs(num_regs), _z(new VComplexPair[num_regs * N]), _zp(new VComplexPair[N]), _w(new TwiddleFactor[N / 2])
	{
		TwiddleFactor * const w = _w;
		for (size_t s = 1; s < N / 2; s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				w[s + j] = TwiddleFactor(bitrev(j, 4 * s) + 1, 2 * 4 * s);
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
	void carry(const bool dup)
	{
		VComplexPair * const z = _z;

		const double base = get_b(), base_inv = 1.0 / base, t2_n = 2.0 / N;
		const double g = dup ? 2 : 1;

		VComplexPair f(0.0), err(0.0);

		for (size_t k = 0; k < N; ++k)
		{
			VComplexPair of = z[k].adjust() * t2_n;
			const VComplexPair o = of.round();
			err = err.max((of - o).abs());
			f += o * g;

			z[k].set(f.mod(base, base_inv));
		}

		_error = std::fmax(_error, err.max().max());

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

		const size_t m0 = VComplexPair::forward(z, w, N / 4);
		if (m0 == 4) VComplexPair::square4(z, w, N / 4); else VComplexPair::square2(z, w, N / 4);
		VComplexPair::backward(z, w, N / 4, m0);
		carry(dup);
	}

	void init_multiplicand(const size_t src) override
	{
		const VComplexPair * const z_src = &_z[src * N];
		VComplexPair * const zp = _zp;

		for (size_t k = 0; k < N; ++k) zp[k] = z_src[k];
		const TwiddleFactor * const w = _w;
		const size_t m0 = VComplexPair::forward(zp, w, N / 4);
		if (m0 == 4) VComplexPair::fwd2(zp, w, N / 4);
	}

	void mul() override
	{
		VComplexPair * const z = _z;
		const TwiddleFactor * const w = _w;

		const size_t m0 = VComplexPair::forward(z, w, N / 4);
		if (m0 == 4) VComplexPair::mul4(z, _zp, w, N / 4); else VComplexPair::mul2(z, _zp, w, N / 4);
		VComplexPair::backward(z, w, N / 4, m0);
		carry(false);
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