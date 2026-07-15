/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <immintrin.h>

#ifndef finline
#define finline	__attribute__((always_inline)) inline
#endif

typedef int32_t		int32_2  __attribute__ ((vector_size(2 * sizeof(int32_t))));
typedef int32_t		int32_4  __attribute__ ((vector_size(4 * sizeof(int32_t))));
typedef int32_t		int32_8  __attribute__ ((vector_size(8 * sizeof(int32_t))));
typedef uint32_t	uint32_4 __attribute__ ((vector_size(4 * sizeof(uint32_t))));
typedef uint32_t	uint32_8 __attribute__ ((vector_size(8 * sizeof(uint32_t))));
typedef int64_t		int64_2  __attribute__ ((vector_size(2 * sizeof(int64_t))));
typedef int64_t		int64_4  __attribute__ ((vector_size(4 * sizeof(int64_t))));
typedef int64_t		int64_8  __attribute__ ((vector_size(8 * sizeof(int64_t))));
typedef uint64_t	uint64_2 __attribute__ ((vector_size(2 * sizeof(uint64_t))));
typedef uint64_t	uint64_8 __attribute__ ((vector_size(8 * sizeof(uint64_t))));

#define VSIZE	8
using vint32 = int32_8;
using vuint32 = uint32_8;
using vint64 = int64_8;
using vuint64 = uint64_8;

typedef union
{
	int32_8	i8;
	int32_4	i4[2];
	int32_2	i2[4];
} int32_8u;

typedef union
{
	uint32_8	i8;
	uint32_4	i4[2];
} uint32_8u;

typedef union
{
	int64_8	i8;
	int64_4	i4[2];
	int64_2	i2[4];
} int64_8u;

typedef union
{
	uint64_8	i8;
	// uint64_4	i4[2];
	uint64_2	i2[4];
} uint64_8u;

finline void set1(vint32 & x, const int32_t a)
{
#if defined(__AVX__)
	x = (int32_8)_mm256_set1_epi32(a);
#else
	int32_8u xu;
	for (size_t i = 0; i < 2; ++i) xu.i4[i] = (int32_4)_mm_set1_epi32(a);
	x = xu.i8;
#endif
}

finline void set1(vuint32 & x, const uint32_t a)
{
#if defined(__AVX__)
	x = (uint32_8)_mm256_set1_epi32(int(a));
#else
	uint32_8u xu;
	for (size_t i = 0; i < 2; ++i) xu.i4[i] = (uint32_4)_mm_set1_epi32(int(a));
	x = xu.i8;
#endif
}

finline void set_zero(vint64 & x)
{
#if defined(__AVX__)
	int64_8u xu; xu.i8 = x;
	for (size_t i = 0; i < 2; ++i) xu.i4[i] = (int64_4)_mm256_setzero_si256();
	x = xu.i8;
#else
	int64_8u xu; xu.i8 = x;
	for (size_t i = 0; i < 4; ++i) xu.i2[i] = (int64_2)_mm_setzero_si128();
	x = xu.i8;
#endif
}

finline bool cmp(const vuint32 & x, const vuint32 & y)
{
#if defined(__AVX2__)
	return (_mm256_movemask_epi8(_mm256_cmpeq_epi32((__m256i)x, (__m256i)y)) == (int)0xffffffff);
#else
	uint32_8u xu, yu; xu.i8 = x; yu.i8 = y;
	const int m0 = _mm_movemask_epi8(_mm_cmpeq_epi32((__m128i)xu.i4[0], (__m128i)yu.i4[0]));
	const int m1 = _mm_movemask_epi8(_mm_cmpeq_epi32((__m128i)xu.i4[1], (__m128i)yu.i4[1]));
	return ((m0 & m1) == (int)0xffff);
#endif
}

finline bool cmp(const vuint64 & x, const vuint64 & y)
{
// #if defined(__AVX2__)
// TODO
// 	return (_mm256_movemask_epi8(_mm256_cmpeq_epi32((__m256i)x, (__m256i)y)) == (int)0xffffffff);
// #else
	uint64_8u xu, yu; xu.i8 = x; yu.i8 = y;
	int m = (int)0xffff;
	for (size_t i = 0; i < 4; ++i) m &= _mm_movemask_epi8(_mm_cmpeq_epi32((__m128i)xu.i2[i], (__m128i)yu.i2[i]));
	return (m == (int)0xffff);
// #endif
}

finline bool cmp_zero(const vint64 & x)
{
// #if defined(__AVX2__)
// TODO
// #else
	int64_8u xu; xu.i8 = x;
	int m = (int)0xffff;
	for (size_t i = 0; i < 4; ++i) m &= _mm_movemask_epi8(_mm_cmpeq_epi32((__m128i)xu.i2[i], _mm_setzero_si128()));
	return (m == (int)0xffff);
// #endif
}

finline uint32_t reduce_max(const vuint32 & x)
{
// TODO
	uint32_t x_max = 0; for (size_t j = 0; j < VSIZE; ++j) x_max = std::max(x_max, x[j]);
	return x_max;
}
