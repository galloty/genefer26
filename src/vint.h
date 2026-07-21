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


// typedef uint32_t	uint32_4 __attribute__ ((vector_size(4 * sizeof(uint32_t))));
// typedef uint32_t	uint32_8 __attribute__ ((vector_size(8 * sizeof(uint32_t))));
// typedef int64_t		int64_2  __attribute__ ((vector_size(2 * sizeof(int64_t))));
// typedef int64_t		int64_4  __attribute__ ((vector_size(4 * sizeof(int64_t))));
// typedef int64_t		int64_8  __attribute__ ((vector_size(8 * sizeof(int64_t))));
// typedef uint64_t	uint64_2 __attribute__ ((vector_size(2 * sizeof(uint64_t))));
// typedef uint64_t	uint64_8 __attribute__ ((vector_size(8 * sizeof(uint64_t))));

#define VSIZE	8
// using vint32 = int32_8;
// using vuint32 = uint32_8;
// using vuint64 = uint64_8;

class UInt32_8
{
public:
	typedef uint32_t	uint32_8 __attribute__ ((vector_size(8 * sizeof(uint32_t))));

private:
	uint32_8 _n;

public:
	finline explicit UInt32_8() {}
	finline constexpr explicit UInt32_8(const uint32_8 & n) : _n(n) {}
	finline explicit UInt32_8(const uint32_t d) : _n(uint32_8{d, d, d, d, d, d, d, d}) {}
	finline explicit UInt32_8(const uint32_t d[8]) : _n(uint32_8{d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]}) {}


	// Needed to convert vector
	finline const uint32_8 & get() const { return _n; }
	// finline void set(const uint32_8 & n) { _n = n; }

	uint32_t operator[](const size_t i) const { return _n[i]; }
	
	// finline uint32_8 operator==(const UInt32_8 & rhs) const { return (_n == rhs._n); }
	finline bool operator!=(const UInt32_8 & rhs) const { const uint32_8 c = (_n != rhs._n); bool b = true; for (size_t i = 0; i < 8; ++i) b &= (c[i] != 0); return b; }

	finline uint32_t max() const
	{
 		uint32_t m = 0; for (size_t i = 0; i < 8; ++i) m = std::max(m, _n[i]);
		return m;
	}
};

class Int32_8
{
public:
	typedef int32_t 	int32_8  __attribute__ ((vector_size(8 * sizeof(int32_t))));

private:
	// typedef int32_t	int32_2  __attribute__ ((vector_size(2 * sizeof(int32_t))));
	// typedef int32_t	int32_4  __attribute__ ((vector_size(4 * sizeof(int32_t))));

	// typedef union
	// {
	// 	int32_8	i8;
	// 	int32_4	i4[2];
	// 	int32_2	i2[4];
	// } int32_8u;

private:
	int32_8 _n;

	finline constexpr explicit Int32_8(const int32_8 & n) : _n(n) {}

public:
	finline explicit Int32_8() {}
	finline explicit Int32_8(const int32_t d) : _n(int32_8{d, d, d, d, d, d, d, d}) {}
	finline explicit Int32_8(const UInt32_8 & rhs) : _n(__builtin_convertvector(rhs.get(), int32_8)) {}

	// Needed to convert vector
	finline const int32_8 & get() const { return _n; }
	finline void set(const int32_8 & n) { _n = n; }

	int32_t operator[](const size_t i) const { return _n[i]; }	// TODO
	void set_i(const size_t i, const int32_t d) { _n[i] = d; }	// TODO

	finline int32_8 operator==(const Int32_8 & rhs) const { return (_n == rhs._n); }
	finline int32_8 operator!=(const Int32_8 & rhs) const { return (_n != rhs._n); }

	finline Int32_8 operator-() const { return Int32_8(-_n); }

	finline Int32_8 & operator+=(const Int32_8 & rhs) { _n += rhs._n; return *this; }
	finline Int32_8 & operator-=(const Int32_8 & rhs) { _n -= rhs._n; return *this; }

	finline Int32_8 operator+(const Int32_8 & rhs) const { return Int32_8(_n + rhs._n); }
	finline Int32_8 operator-(const Int32_8 & rhs) const { return Int32_8(_n - rhs._n); }

	finline Int32_8 operator/(const int32_t d) const { return Int32_8(_n / d); }
};

class UInt64_8
{
public:
	typedef uint64_t	uint64_8 __attribute__ ((vector_size(8 * sizeof(uint64_t))));

private:
	uint64_8 _n;

public:
	finline explicit UInt64_8() {}
	finline constexpr explicit UInt64_8(const uint64_8 & n) : _n(n) {}
	finline explicit UInt64_8(const Int32_8 & rhs) : _n(__builtin_convertvector(rhs.get(), uint64_8)) {}
	finline explicit UInt64_8(const UInt32_8 & rhs) : _n(__builtin_convertvector(rhs.get(), uint64_8)) {}

	finline const uint64_8 & get() const { return _n; }

	uint64_t operator[](const size_t i) const { return _n[i]; }

	finline bool operator==(const UInt64_8 & rhs) const { const uint64_8 c = (_n == rhs._n); bool b = true; for (size_t i = 0; i < 8; ++i) b &= (c[i] != 0); return b; }
	finline bool operator!=(const UInt64_8 & rhs) const { const uint64_8 c = (_n != rhs._n); bool b = true; for (size_t i = 0; i < 8; ++i) b &= (c[i] != 0); return b; }

	finline UInt64_8 & operator+=(const UInt64_8 & rhs) { _n += rhs._n; return *this; }
	finline UInt64_8 & operator*=(const UInt64_8 & rhs) { _n *= rhs._n; return *this; }

	finline UInt64_8 operator*(const UInt64_8 & rhs) const { return UInt64_8(_n * rhs._n); }

};




// typedef union
// {
// 	uint32_8	i8;
// 	uint32_4	i4[2];
// } uint32_8u;

// typedef union
// {
// 	uint64_8	i8;
// 	// uint64_4	i4[2];
// 	uint64_2	i2[4];
// } uint64_8u;

// finline void set1(vint32 & x, const int32_t a)
// {
// #if defined(__AVX__)
// 	x = (int32_8)_mm256_set1_epi32(a);
// #else
// 	int32_8u xu;
// 	for (size_t i = 0; i < 2; ++i) xu.i4[i] = (int32_4)_mm_set1_epi32(a);
// 	x = xu.i8;
// #endif
// }

// finline void set1(vuint32 & x, const uint32_t a)
// {
// #if defined(__AVX__)
// 	x = (uint32_8)_mm256_set1_epi32(int(a));
// #else
// 	uint32_8u xu;
// 	for (size_t i = 0; i < 2; ++i) xu.i4[i] = (uint32_4)_mm_set1_epi32(int(a));
// 	x = xu.i8;
// #endif
// }

// finline bool cmp(const vuint32 & x, const vuint32 & y)
// {
// #if defined(__AVX2__)
// 	return (_mm256_movemask_epi8(_mm256_cmpeq_epi32((__m256i)x, (__m256i)y)) == (int)0xffffffff);
// #else
// 	uint32_8u xu, yu; xu.i8 = x; yu.i8 = y;
// 	const int m0 = _mm_movemask_epi8(_mm_cmpeq_epi32((__m128i)xu.i4[0], (__m128i)yu.i4[0]));
// 	const int m1 = _mm_movemask_epi8(_mm_cmpeq_epi32((__m128i)xu.i4[1], (__m128i)yu.i4[1]));
// 	return ((m0 & m1) == (int)0xffff);
// #endif
// }

// finline bool cmp(const vuint64 & x, const vuint64 & y)
// {
// // #if defined(__AVX2__)
// // TODO
// // 	return (_mm256_movemask_epi8(_mm256_cmpeq_epi32((__m256i)x, (__m256i)y)) == (int)0xffffffff);
// // #else
// 	uint64_8u xu, yu; xu.i8 = x; yu.i8 = y;
// 	int m = (int)0xffff;
// 	for (size_t i = 0; i < 4; ++i) m &= _mm_movemask_epi8(_mm_cmpeq_epi32((__m128i)xu.i2[i], (__m128i)yu.i2[i]));
// 	return (m == (int)0xffff);
// // #endif
// }

// finline uint32_t reduce_max(const vuint32 & x)
// {
// // TODO
// 	uint32_t x_max = 0; for (size_t j = 0; j < VSIZE; ++j) x_max = std::max(x_max, x[j]);
// 	return x_max;
// }
