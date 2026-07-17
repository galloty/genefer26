/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <immintrin.h>

#include "vint.h"

#define DEF_MASK4()	static const __m256i mask4[16] = \
	{ \
		_mm256_set_epi64x(0,       0, 0, 0), _mm256_set_epi64x(0,       0, 0, -1ll), _mm256_set_epi64x(0,       0, -1ll, 0), _mm256_set_epi64x(0,       0, -1ll, -1ll), \
		_mm256_set_epi64x(0,    -1ll, 0, 0), _mm256_set_epi64x(0,    -1ll, 0, -1ll), _mm256_set_epi64x(0,    -1ll, -1ll, 0), _mm256_set_epi64x(0,    -1ll, -1ll, -1ll), \
		_mm256_set_epi64x(-1ll,    0, 0, 0), _mm256_set_epi64x(-1ll,    0, 0, -1ll), _mm256_set_epi64x(-1ll,    0, -1ll, 0), _mm256_set_epi64x(-1ll,    0, -1ll, -1ll), \
		_mm256_set_epi64x(-1ll, -1ll, 0, 0), _mm256_set_epi64x(-1ll, -1ll, 0, -1ll), _mm256_set_epi64x(-1ll, -1ll, -1ll, 0), _mm256_set_epi64x(-1ll, -1ll, -1ll, -1ll), \
	}

#define DEF_MASK2()	static const __m128i mask2[4] = \
	{ _mm_set_epi64x(0, 0), _mm_set_epi64x(0, -1ll), _mm_set_epi64x(-1ll, 0), _mm_set_epi64x(-1ll, -1ll) }

class Double_8
{
	typedef double	double_2 __attribute__ ((vector_size(2 * sizeof(double))));
	typedef double	double_4 __attribute__ ((vector_size(4 * sizeof(double))));
	typedef double	double_8 __attribute__ ((vector_size(8 * sizeof(double))));

	typedef union
	{
		double_8	d8;
		double_4	d4[2];
		double_2	d2[4];
	} double_8u;

private:
	double_8 _x;

	finline constexpr explicit Double_8(const double_8 & x) : _x(x) {}

public:
	finline explicit Double_8() {}
	finline explicit Double_8(const double f) : _x(double_8{f, f, f, f, f, f, f, f}) {}
	finline explicit Double_8(const vint32 & rhs) : _x(__builtin_convertvector(rhs, double_8)) {}
	finline explicit Double_8(const vuint32 & rhs) : _x(__builtin_convertvector(rhs, double_8)) {}

	finline void to_int(vint32 & rhs) const
	{
#if defined(__AVX512F__)
		rhs = (vint32)_mm512_cvt_roundpd_epi32(_x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
		const Double_8 r = round();
		rhs = __builtin_convertvector(r._x, vint32);
#endif
	}

	finline bool is_zero() const
	{
		bool b = true;
#if defined(__AVX512F__)
		b = (_mm512_cmp_pd_mask((__m512d)_x, _mm512_setzero_pd(), _CMP_NEQ_OQ) == 0);
#elif defined(__AVX__)
		double_8u xu; xu.d8 = _x;
		for (size_t i = 0; i < 2; ++i) b &= (_mm256_movemask_pd(_mm256_cmp_pd((__m256d)xu.d4[i], _mm256_setzero_pd(), _CMP_NEQ_OQ)) == 0);
#else
		double_8u xu; xu.d8 = _x;
		for (size_t i = 0; i < 4; ++i) b &= (_mm_movemask_pd(_mm_cmpneq_pd((__m128d)xu.d2[i], _mm_setzero_pd())) == 0);
#endif
		return b;
	}

	finline Double_8 operator-() const { return Double_8(-_x); }

	finline Double_8 & operator+=(const Double_8 & rhs) { _x += rhs._x; return *this; }
	finline Double_8 & operator-=(const Double_8 & rhs) { _x -= rhs._x; return *this; }
	finline Double_8 & operator*=(const double rhs) { _x *= rhs; return *this; }
	finline Double_8 & operator*=(const Double_8 & rhs) { _x *= rhs._x; return *this; }

	finline Double_8 operator+(const Double_8 & rhs) const { return Double_8(_x + rhs._x); }
	finline Double_8 operator-(const Double_8 & rhs) const { return Double_8(_x - rhs._x); }
	finline Double_8 operator*(const double rhs) const { return Double_8(_x * rhs); }
	finline Double_8 operator*(const Double_8 & rhs) const { return Double_8(_x * rhs._x); }

	finline void copy_mask(const Double_8 & src_1, const Double_8 & src_0, const uint8_t mask)
	{
#if defined(__AVX512F__)
		_x = _mm512_mask_mov_pd((__m512d)src_0._x, mask, (__m512d)src_1._x);
#elif defined(__AVX__)
		DEF_MASK4();
		double_8u src_0u, src_1u, xu; src_0u.d8 = src_0._x; src_1u.d8 = src_1._x;
		for (size_t i = 0; i < 2; ++i)
		{
			xu.d4[i] = _mm256_blendv_pd((__m256d)src_0u.d4[i], (__m256d)src_1u.d4[i], _mm256_castsi256_pd(mask4[(mask >> (4 * i)) % 16]));
		}
		_x = xu.d8;
#else
		DEF_MASK2();
		double_8u src_0u, src_1u, xu; src_0u.d8 = src_0._x; src_1u.d8 = src_1._x;
		for (size_t i = 0; i < 4; ++i)
		{
			xu.d2[i] = _mm_blendv_pd((__m128d)src_0u.d2[i], (__m128d)src_1u.d2[i], _mm_castsi128_pd(mask2[(mask >> (2 * i)) % 4]));
		}
		_x = xu.d8;
#endif
	}

	finline Double_8 mul_mask(const Double_8 & rhs, const uint8_t mask) const
	{
		double_8u yu;
#if defined(__AVX512F__)
		yu.d8 = _mm512_mask_mul_pd((__m512d)_x, mask, (__m512d)_x, (__m512d)rhs._x);
#elif defined(__AVX__)
		DEF_MASK4();
		double_8u xu; xu.d8 = _x; yu.d8 = _x * rhs._x;
		for (size_t i = 0; i < 2; ++i)
		{
			yu.d4[i] = _mm256_blendv_pd((__m256d)xu.d4[i], (__m256d)yu.d4[i], _mm256_castsi256_pd(mask4[(mask >> (4 * i)) % 16]));
		}
#else
		DEF_MASK2();
		double_8u xu; xu.d8 = _x; yu.d8 = _x * rhs._x;
		for (size_t i = 0; i < 4; ++i)
		{
			yu.d2[i] = _mm_blendv_pd((__m128d)xu.d2[i], (__m128d)yu.d2[i], _mm_castsi128_pd(mask2[(mask >> (2 * i)) % 4]));
		}
#endif
		return Double_8(yu.d8);
	}

	finline Double_8 mul_maskz(const Double_8 & rhs, const uint8_t mask) const
	{
		double_8u yu;
#if defined(__AVX512F__)
		yu.d8 = _mm512_maskz_mul_pd(mask, (__m512d)_x, (__m512d)rhs._x);
#elif defined(__AVX__)
		DEF_MASK4();
		yu.d8 = _x * rhs._x;
		for (size_t i = 0; i < 2; ++i)
		{
			yu.d4[i] = _mm256_blendv_pd(_mm256_setzero_pd(), (__m256d)yu.d4[i], _mm256_castsi256_pd(mask4[(mask >> (4 * i)) % 16]));
		}
#else
		DEF_MASK2();
		yu.d8 = _x * rhs._x;
		for (size_t i = 0; i < 4; ++i)
		{
			yu.d2[i] = _mm_blendv_pd(_mm_setzero_pd(), (__m128d)yu.d2[i], _mm_castsi128_pd(mask2[(mask >> (2 * i)) % 4]));
		}
#endif
		return Double_8(yu.d8);
	}

	finline Double_8 inverse() const { return Double_8(1 / _x); }

	finline Double_8 round() const
	{
		double_8u yu;
#if defined(__AVX512F__)
		yu.d8 = (double_8)_mm512_roundscale_pd((__m512d)_x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#elif defined(__AVX__)
		double_8u xu; xu.d8 = _x;
		for (size_t i = 0; i < 2; ++i) yu.d4[i] = (double_4)_mm256_round_pd((__m256d)xu.d4[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
		double_8u xu; xu.d8 = _x;
		for (size_t i = 0; i < 4; ++i) yu.d2[i] = (double_2)_mm_round_pd((__m128d)xu.d2[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
		return Double_8(yu.d8);
	}

	finline Double_8 abs() const
	{
		double_8u yu;
#if defined(__AVX512F__)
		yu.d8 = (double_8)_mm512_abs_pd((__m512d)_x);
#elif defined(__AVX__)
		const long long m = (long long)(uint64_t(1) << 63);
		const __m256d mask = _mm256_castsi256_pd((__m256i){m, m, m, m});
		double_8u xu; xu.d8 = _x;
		for (size_t i = 0; i < 2; ++i) yu.d4[i] = (double_4)_mm256_andnot_pd(mask, (__m256d)xu.d4[i]);
#else
		const long long m = (long long)(uint64_t(1) << 63);
		const __m128d mask = _mm_castsi128_pd((__m128i){m, m});
		double_8u xu; xu.d8 = _x;
		for (size_t i = 0; i < 4; ++i) yu.d2[i] = (double_2)_mm_andnot_pd(mask, (__m128d)xu.d2[i]);
#endif
		return Double_8(yu.d8);
	}

	finline Double_8 max(const Double_8 & rhs) const
	{
		double_8u yu;
#if defined(__AVX512F__)
		yu.d8 = (double_8)_mm512_max_pd((__m512d)_x, (__m512d)rhs._x);
#elif defined(__AVX__)
		double_8u xu, rxu; xu.d8 = _x; rxu.d8 = rhs._x;
		for (size_t i = 0; i < 2; ++i) yu.d4[i] = (double_4)_mm256_max_pd((__m256d)xu.d4[i], (__m256d)rxu.d4[i]);
#else
		double_8u xu, rxu; xu.d8 = _x; rxu.d8 = rhs._x;
		for (size_t i = 0; i < 4; ++i) yu.d2[i] = (double_2)_mm_max_pd((__m128d)xu.d2[i], (__m128d)rxu.d2[i]);
#endif
		return Double_8(yu.d8);
	}

	finline double max() const
	{
		double y;
#if defined(__AVX512F__)
		y = _mm512_reduce_max_pd((__m512d)_x);
#else
		double_8u xu; xu.d8 = _x;
		const __m128d m02 = _mm_max_pd((__m128d)xu.d2[0], (__m128d)xu.d2[2]);
		const __m128d m13 = _mm_max_pd((__m128d)xu.d2[1], (__m128d)xu.d2[3]);
		const double_2 m0123 = (double_2)_mm_max_pd(m02, m13);
		y = std::max(m0123[0], m0123[1]);
#endif
		return y;
	}

	finline void cosmic_ray() { _x[3] += 1; }
};
