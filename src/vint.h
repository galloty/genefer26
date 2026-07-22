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

template<typename stype, size_t SIZE, typename mtype>
class Vint
{
public:
	typedef stype	vtype __attribute__ ((vector_size(SIZE * sizeof(stype))));

private:
	typedef stype	vtype_2 __attribute__ ((vector_size(SIZE / 2 * sizeof(stype))));
	typedef stype	vtype_4 __attribute__ ((vector_size(SIZE / 4 * sizeof(stype))));

	typedef union
	{
		vtype	i;
		vtype_2	i_2[2];
		vtype_4	i_4[4];
	} vtype_u;

protected:
	vtype _n;

public:
	finline explicit Vint() {}
	finline Vint(const Vint & rhs) : _n(rhs._n) {}
	finline explicit Vint(const vtype & n) : _n(n) {}
	finline explicit Vint(const stype d) { for (size_t i = 0; i < SIZE; ++i) _n[i] = d; }
	finline explicit Vint(const stype d[SIZE]) { for (size_t i = 0; i < SIZE; ++i) _n[i] = d[i]; }

	finline Vint & operator=(const Vint & rhs) { _n = rhs._n; return *this; }

	finline const vtype & get() const { return _n; }

	stype operator[](const size_t i) const { return _n[i]; }

	finline vtype operator==(const Vint & rhs) const { return (_n == rhs._n); }
	finline vtype operator!=(const Vint & rhs) const { return (_n != rhs._n); }
	finline vtype operator>(const Vint & rhs) const { return (_n > rhs._n); }
	finline vtype operator<(const Vint & rhs) const { return (_n < rhs._n); }
	finline vtype operator>=(const Vint & rhs) const { return (_n >= rhs._n); }
	finline vtype operator<=(const Vint & rhs) const { return (_n <= rhs._n); }

	finline bool is_true() const { bool b = true; for (size_t i = 0; i < SIZE; ++i) b &= (_n[i] == stype(-1)); return b; }

	finline bool is_equal(const Vint & rhs) const { const vtype c = (_n == rhs._n); bool b = true; for (size_t i = 0; i < SIZE; ++i) b &= (c[i] != 0); return b; }
	finline bool is_zero() const { bool b = true; for (size_t i = 0; i < SIZE; ++i) b &= (_n[i] == 0); return b; }

	finline Vint operator-() const { return Vint(-_n); }

	finline Vint & operator&=(const Vint & rhs) { _n &= rhs._n; return *this; }
	finline Vint & operator|=(const Vint & rhs) { _n |= rhs._n; return *this; }
	finline Vint & operator^=(const Vint & rhs) { _n ^= rhs._n; return *this; }
	finline Vint & operator+=(const Vint & rhs) { _n += rhs._n; return *this; }
	finline Vint & operator-=(const Vint & rhs) { _n -= rhs._n; return *this; }
	finline Vint & operator*=(const Vint & rhs) { _n *= rhs._n; return *this; }

	finline Vint operator&(const Vint & rhs) const { return Vint(_n & rhs._n); }
	finline Vint operator|(const Vint & rhs) const { return Vint(_n | rhs._n); }
	finline Vint operator^(const Vint & rhs) const { return Vint(_n ^ rhs._n); }
	finline Vint operator+(const Vint & rhs) const { return Vint(_n + rhs._n); }
	finline Vint operator-(const Vint & rhs) const { return Vint(_n - rhs._n); }
	finline Vint operator*(const Vint & rhs) const { return Vint(_n * rhs._n); }

	finline Vint operator/(const stype d) const { return Vint(_n / d); }

	finline Vint operator>>(const int s) const { return Vint(_n >> s); }
	finline Vint operator<<(const int s) const { return Vint(_n << s); }

	finline Vint rotl(const Vint & rhs) const { return Vint((_n << rhs._n) | (_n >> (-rhs._n & (8 * sizeof(stype) - 1)))); }

	finline Vint max(const Vint & rhs) const { return Vint((_n >= rhs._n) ? _n : rhs._n); }

	finline stype min(const stype m_min) const { stype m = m_min; for (size_t i = 0; i < SIZE; ++i) m = std::min(m, _n[i]); return m; }
	finline stype max(const stype m_max) const { stype m = m_max; for (size_t i = 0; i < SIZE; ++i) m = std::max(m, _n[i]); return m; }

	finline mtype get_bit_mask(const int i) const
	{
		const Vint r = *this & (Vint(stype(1)) << i);
		mtype m = 0;
		for (size_t j = 0; j < SIZE; ++j) m |= ((r[j] != 0) ? mtype(1) : mtype(0)) << j;
		return m;
	}
};

using UInt32_8 = Vint<uint32_t, 8, uint8_t>;
using Int32_8 = Vint<int32_t, 8, uint8_t>;
using UInt64_8 = Vint<uint64_t, 8, uint8_t>;

// specializations
template<>
finline UInt32_8::Vint(const uint32_t d)
{
#if defined(__AVX__)
	_n = (vtype)_mm256_set1_epi32(int(d));
#else
	vtype_u nu;
	for (size_t i = 0; i < 2; ++i) nu.i_2[i] = (vtype_2)_mm_set1_epi32(int(d));
	_n = nu.i;
#endif
}

template<>
finline Int32_8::Vint(const int32_t d)
{
#if defined(__AVX__)
	_n = (vtype)_mm256_set1_epi32(int(d));
#else
	vtype_u nu;
	for (size_t i = 0; i < 2; ++i) nu.i_2[i] = (vtype_2)_mm_set1_epi32(int(d));
	_n = nu.i;
#endif
}



// typedef uint32_t	uint32_4 __attribute__ ((vector_size(4 * sizeof(uint32_t))));
// typedef uint32_t	uint32_8 __attribute__ ((vector_size(8 * sizeof(uint32_t))));
// typedef int64_t		int64_2  __attribute__ ((vector_size(2 * sizeof(int64_t))));
// typedef int64_t		int64_4  __attribute__ ((vector_size(4 * sizeof(int64_t))));
// typedef int64_t		int64_8  __attribute__ ((vector_size(8 * sizeof(int64_t))));
// typedef uint64_t	uint64_2 __attribute__ ((vector_size(2 * sizeof(uint64_t))));
// typedef uint64_t	uint64_8 __attribute__ ((vector_size(8 * sizeof(uint64_t))));

#define VSIZE	8
// using Vint32 = int32_8;
// using vuint32 = uint32_8;
// using vuint64 = uint64_8;

// typedef uint32_t	uint32_8 __attribute__ ((vector_size(8 * sizeof(uint32_t))));



/*class UInt32_8 : public Vint<uint32_t, 8>
{
public:
	// typedef uint32_t	uint32_8 __attribute__ ((vector_size(8 * sizeof(uint32_t))));

private:
	// uint32_8 _n;

public:
	finline UInt32_8() : Vint() {}
	finline UInt32_8(const Vint::vint & n) : Vint(n) {}
	finline UInt32_8(const Vint & rhs) : Vint(rhs) {}
	finline UInt32_8(const UInt32_8 & rhs) : Vint(rhs._n) {}
	finline UInt32_8(const uint32_t d) : Vint(d) {}	// Vint(Vint::vint{d, d, d, d, d, d, d, d}) {}
	finline UInt32_8(const uint32_t d[8]) : Vint(d) {}	//: Vint(Vint::vint{d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]}) {}

	finline UInt32_8 & operator=(const UInt32_8 & rhs) { _n = rhs._n; return *this; }

	// finline explicit UInt32_8() {}
	// finline constexpr explicit UInt32_8(const uint32_8 & n) : _n(n) {}
	// finline explicit UInt32_8(const uint32_t d) : _n(uint32_8{d, d, d, d, d, d, d, d}) {}
	// finline explicit UInt32_8(const uint32_t d[8]) : _n(uint32_8{d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]}) {}


	// Needed to convert vector
	// finline const Vint::vint & get() const { return _n; }
	// finline void set(const uint32_8 & n) { _n = n; }

	// uint32_t operator[](const size_t i) const { return _n[i]; }
	
	// finline uint32_8 operator==(const UInt32_8 & rhs) const { return (_n == rhs._n); }
	// finline bool operator!=(const UInt32_8 & rhs) const { const uint32_8 c = (_n != rhs._n); bool b = true; for (size_t i = 0; i < 8; ++i) b &= (c[i] != 0); return b; }
	// finline bool is_equal(const UInt32_8 & rhs) const { const Vint::vint c = (_n == rhs._n); bool b = true; for (size_t i = 0; i < 8; ++i) b &= (c[i] != 0); return b; }

	// finline UInt32_8 operator^(const UInt32_8 & rhs) const { return UInt32_8(_n ^ rhs._n); }

	// finline uint32_t max() const
	// {
 	// 	uint32_t m = 0; for (size_t i = 0; i < 8; ++i) m = std::max(m, _n[i]);
	// 	return m;
	// }
};*/

// typedef int32_t 	int32_8  __attribute__ ((vector_size(8 * sizeof(int32_t))));



/*class Int32_8 : public Vint<int32_t, 8>
{
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
	// int32_8 _n;

public:
	finline Int32_8() : Vint() {}
	finline Int32_8(const Vint::vtype & n) : Vint(n) {}
	finline Int32_8(const Vint & rhs) : Vint(rhs) {}
	finline Int32_8(const Int32_8 & rhs) : Vint(rhs._n) {}
	finline Int32_8(const int32_t d) : Vint(d) {}	// : Vint(Vint::vint{d, d, d, d, d, d, d, d}) {}

	finline Int32_8 & operator=(const Int32_8 & rhs) { _n = rhs._n; return *this; }

	// Needed to convert vector
	// finline const Vint::vint & get() const { return _n; }
	finline void set(const Vint::vtype & n) { _n = n; }

	// int32_t operator[](const size_t i) const { return _n[i]; }	// TODO
	// void set_i(const size_t i, const int32_t d) { _n[i] = d; }	// TODO

	// finline bool is_zero() const
	// {
	// 	bool b = true; for (size_t i = 0; i < 8; ++i) b &= (_n[i] == 0); return b;
	// }

	// finline bool is_true() const
	// {
	// 	bool b = true; for (size_t i = 0; i < 8; ++i) b &= (_n[i] == -1); return b;
	// }

	// finline int32_8 operator==(const Int32_8 & rhs) const { return (_n == rhs._n); }
	// finline int32_8 operator!=(const Int32_8 & rhs) const { return (_n != rhs._n); }
	// finline int32_8 operator>(const Int32_8 & rhs) const { return (_n > rhs._n); }
	// finline int32_8 operator<(const Int32_8 & rhs) const { return (_n < rhs._n); }
	// finline int32_8 operator>=(const Int32_8 & rhs) const { return (_n >= rhs._n); }
	// finline int32_8 operator<=(const Int32_8 & rhs) const { return (_n <= rhs._n); }

	// finline Int32_8 & operator&=(const Int32_8 & rhs) { _n &= rhs._n; return *this; }
	// finline Int32_8 operator&(const Int32_8 & rhs) const { return Int32_8(_n & rhs._n); }

	// finline Int32_8 operator-() const { return Int32_8(-_n); }

	// finline Int32_8 & operator+=(const Int32_8 & rhs) { _n += rhs._n; return *this; }
	// finline Int32_8 & operator-=(const Int32_8 & rhs) { _n -= rhs._n; return *this; }

	// finline Int32_8 operator+(const Int32_8 & rhs) const { return Int32_8(_n + rhs._n); }
	// finline Int32_8 operator-(const Int32_8 & rhs) const { return Int32_8(_n - rhs._n); }

	// finline Int32_8 operator/(const int32_t d) const { return Int32_8(_n / d); }
};*/

// typedef uint64_t	uint64_8 __attribute__ ((vector_size(8 * sizeof(uint64_t))));

/*class UInt64_8 : public Vint<uint64_t, 8>
{
public:
	// typedef uint64_t	uint64_8 __attribute__ ((vector_size(8 * sizeof(uint64_t))));

private:
	// uint64_8 _n;

public:
	finline UInt64_8() : Vint() {}
	finline UInt64_8(const Vint::vtype & n) : Vint(n) {}
	finline UInt64_8(const Vint & rhs) : Vint(rhs) {}
	finline UInt64_8(const UInt64_8 & rhs) : Vint(rhs._n) {}
	finline UInt64_8(const uint64_t d) : Vint(d) {}	// : Vint(Vint::vint{d, d, d, d, d, d, d, d}) {}

	finline UInt64_8 & operator=(const UInt64_8 & rhs) { _n = rhs._n; return *this; }


	// finline explicit UInt64_8() {}
	// finline constexpr explicit UInt64_8(const uint64_8 & n) : _n(n) {}
	// finline explicit UInt64_8(const uint64_t d) : _n(uint64_8{d, d, d, d, d, d, d, d}) {}
	// finline explicit UInt64_8(const Int32_8 & rhs) : _n(__builtin_convertvector(rhs.get(), uint64_8)) {}
	// finline explicit UInt64_8(const UInt32_8 & rhs) : _n(__builtin_convertvector(rhs.get(), uint64_8)) {}

	// finline const uint64_8 & get() const { return _n; }
	// finline UInt32_8 toUInt32_8() const { return UInt32_8(__builtin_convertvector(_n, UInt32_8::vtype)); }

	// uint64_t operator[](const size_t i) const { return _n[i]; }

	// finline bool operator==(const UInt64_8 & rhs) const { const Vint::vint c = (_n == rhs._n); bool b = true; for (size_t i = 0; i < 8; ++i) b &= (c[i] != 0); return b; }
	// finline bool operator!=(const UInt64_8 & rhs) const { const Vint::vint c = (_n != rhs._n); bool b = true; for (size_t i = 0; i < 8; ++i) b &= (c[i] != 0); return b; }

	// finline UInt64_8 & operator&=(const UInt64_8 & rhs) { _n &= rhs._n; return *this; }
	// finline UInt64_8 operator&(const UInt64_8 & rhs) const { return UInt64_8(_n & rhs._n); }

	// finline UInt64_8 & operator+=(const UInt64_8 & rhs) { _n += rhs._n; return *this; }
	// finline UInt64_8 & operator-=(const UInt64_8 & rhs) { _n -= rhs._n; return *this; }
	// finline UInt64_8 & operator*=(const UInt64_8 & rhs) { _n *= rhs._n; return *this; }

	// finline UInt64_8 operator+(const UInt64_8 & rhs) const { return UInt64_8(_n + rhs._n); }
	// finline UInt64_8 operator-(const UInt64_8 & rhs) const { return UInt64_8(_n - rhs._n); }
	// finline UInt64_8 operator*(const UInt64_8 & rhs) const { return UInt64_8(_n * rhs._n); }

	// finline UInt64_8 & operator^=(const UInt64_8 & rhs) { _n ^= rhs._n; return *this; }

	// finline UInt64_8 operator>>(const int s) const { return UInt64_8(_n >> s); }
	// finline UInt64_8 rotl(const UInt64_8 & s) const { return UInt64_8((_n << s._n) | (_n >> (-s._n & 63))); }
};*/

// conversions

finline Int32_8 UInt32_8_to_Int32_8(const UInt32_8 & rhs) { return Int32_8(__builtin_convertvector(rhs.get(), Int32_8::vtype)); }
finline UInt64_8 Int32_8_to_UInt64_8(const Int32_8 & rhs) { return UInt64_8(__builtin_convertvector(rhs.get(), UInt64_8::vtype)); }
finline UInt64_8 UInt32_8_to_UInt64_8(const UInt32_8 & rhs) { return UInt64_8(__builtin_convertvector(rhs.get(), UInt64_8::vtype)); }
finline UInt32_8 UInt64_8_to_UInt32_8(const UInt64_8 & rhs) { return UInt32_8(__builtin_convertvector(rhs.get(), UInt32_8::vtype)); }

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

// finline void set1(Vint32 & x, const int32_t a)
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
