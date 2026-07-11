/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <immintrin.h>

#include "vint.h"

typedef double	double_2 __attribute__ ((vector_size(2 * sizeof(double))));
typedef double	double_4 __attribute__ ((vector_size(4 * sizeof(double))));
typedef double	double_8 __attribute__ ((vector_size(8 * sizeof(double))));

using vdouble = double_8;

typedef union
{
	double_8	d8;
	double_4	d4[2];
	double_2	d2[4];
} double_8u;

finline void set1(double_8 & x, const double f)
{
#if defined(__AVX512F__)
	x = (double_8)_mm512_set1_pd(f);
#elif defined(__AVX__)
	double_8u xu;
	for (size_t i = 0; i < 2; ++i) xu.d4[i] = (double_4)_mm256_set1_pd(f);
	x = xu.d8;
#else
	double_8u xu;
	for (size_t i = 0; i < 4; ++i) xu.d2[i] = (double_2)_mm_set1_pd(f);
	x = xu.d8;
#endif
}

finline bool cmp_zero(const double_8 & x)
{
#if defined(__AVX512F__)
	return (_mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_NEQ_OQ) == 0);
#elif defined(__AVX__)
	double_8u xu; xu.d8 = x;
	bool b = true;
	for (size_t i = 0; i < 2; ++i) b &= (_mm256_movemask_pd(_mm256_cmp_pd(xu.d4[i], _mm256_setzero_pd(), _CMP_NEQ_OQ)) == 0);
	return b;
#else
	double_8u xu; xu.d8 = x;
	bool b = true;
	for (size_t i = 0; i < 4; ++i) b &= (_mm_movemask_pd(_mm_cmpneq_pd(xu.d2[i], _mm_setzero_pd())) == 0);
	return b;
#endif
}

finline void vround(double_8 & y, const double_8 & x)
{
#if defined(__AVX512F__)
	y = (double_8)_mm512_roundscale_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#elif defined(__AVX__)
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 2; ++i) r.d4[i] = (double_4)_mm256_round_pd(xu.d4[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = r.d8;
#else
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 4; ++i) r.d2[i] = (double_2)_mm_round_pd(xu.d2[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	y = r.d8;
#endif
}

finline void vabs(double_8 & y, const double_8 & x)
{
#if defined(__AVX512F__)
	y = (double_8)_mm512_abs_pd(x);
#elif defined(__AVX__)
	const long long m = (long long)(uint64_t(1) << 63);
	const __m256d mask = _mm256_castsi256_pd((__m256i){m, m, m, m});
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 2; ++i) r.d4[i] = (double_4)_mm256_andnot_pd(mask, xu.d4[i]);
	y = r.d8;
#else
	const long long m = (long long)(uint64_t(1) << 63);
	const __m128d mask = _mm_castsi128_pd((__m128i){m, m});
	double_8u xu, r; xu.d8 = x;
	for (size_t i = 0; i < 4; ++i) r.d2[i] = (double_2)_mm_andnot_pd(mask, xu.d2[i]);
	y = r.d8;
#endif
}

finline void vmax(double_8 & z, const double_8 & x, const double_8 & y)
{
#if defined(__AVX512F__)
	z = (double_8)_mm512_max_pd(x, y);
#elif defined(__AVX__)
	double_8u xu, yu, r; xu.d8 = x; yu.d8 = y;
	for (size_t i = 0; i < 2; ++i) r.d4[i] = (double_4)_mm256_max_pd(xu.d4[i], yu.d4[i]);
	z = r.d8;
#else
	double_8u xu, yu, r; xu.d8 = x; yu.d8 = y;
	for (size_t i = 0; i < 4; ++i) r.d2[i] = (double_2)_mm_max_pd(xu.d2[i], yu.d2[i]);
	z = r.d8;
#endif
}

finline void double_to_int(vint32 & y, const double_8 & x)
{
#if defined(__AVX512F__)
	y = (vint32)_mm512_cvt_roundpd_epi32(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#elif defined(__AVX__)
	double_8u xu; xu.d8 = x;
	int32_8u yu;
	for (size_t i = 0; i < 2; ++i) yu.i4[i] = (int32_4)_mm256_cvtpd_epi32(_mm256_round_pd(xu.d4[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
	y = yu.i8;
#else
	double_8u xu; xu.d8 = x;
	int32_8u yu;
	for (size_t i = 0; i < 4; ++i) yu.i2[i] = (int32_2)_mm_cvtpd_pi32(_mm_round_pd(xu.d2[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
	y = yu.i8;
#endif
}

finline void int_to_double(double_8 & y, const vint32 & x)
{
#if defined(__AVX512F__)
	y = (double_8)_mm512_cvtepi32_pd((__m256i)x);
#elif defined(__AVX__)
	int32_8u xu; xu.i8 = x;
	double_8u yu;
	for (size_t i = 0; i < 2; ++i) yu.d4[i] = (double_4)_mm256_cvtepi32_pd((__m128i)xu.i4[i]);
	y = yu.d8;
#else
	int32_8u xu; xu.i8 = x;
	double_8u yu;
	for (size_t i = 0; i < 4; ++i) yu.d2[i] = (double_2)_mm_cvtpi32_pd((__m64)xu.i2[i]);
	y = yu.d8;
#endif
}
