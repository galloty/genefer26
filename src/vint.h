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

	finline Vint operator==(const Vint & rhs) const { return Vint(_n == rhs._n); }
	finline Vint operator!=(const Vint & rhs) const { return Vint(_n != rhs._n); }
	finline Vint operator>(const Vint & rhs) const { return Vint(_n > rhs._n); }
	finline Vint operator<(const Vint & rhs) const { return Vint(_n < rhs._n); }
	finline Vint operator>=(const Vint & rhs) const { return Vint(_n >= rhs._n); }
	finline Vint operator<=(const Vint & rhs) const { return Vint(_n <= rhs._n); }

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

	finline stype min() const { stype m = _n[0]; for (size_t i = 1; i < SIZE; ++i) m = std::min(m, _n[i]); return m; }
	finline stype max() const { stype m = _n[0]; for (size_t i = 1; i < SIZE; ++i) m = std::max(m, _n[i]); return m; }

	finline mtype get_bit_mask(const int i) const
	{
		const vtype r = _n & (vtype{1, 1, 1, 1, 1, 1, 1, 1} << i);
		mtype mask = (r[0] != 0) ? mtype(1) : mtype(0);
		for (size_t j = 1; j < SIZE; ++j) mask |= ((r[j] != 0) ? mtype(1) : mtype(0)) << j;
		return mask;
	}
};

typedef Vint<uint32_t, 8, uint8_t> UInt32_8;
typedef Vint<int32_t, 8, uint8_t> Int32_8;
typedef Vint<uint64_t, 8, uint8_t> UInt64_8;

// specializations

template<>
finline UInt64_8 UInt64_8::rotl(const UInt64_8 & rhs) const
{
#if defined(__AVX512F__)
	return Vint((vtype)_mm512_rolv_epi64((__m512i)_n, (__m512i)rhs._n));
#else
	return Vint((_n << rhs._n) | (_n >> (-rhs._n & (8 * sizeof(uint64_t) - 1))));
#endif
}

template<>
finline uint8_t UInt32_8::get_bit_mask(const int i) const
{
	const vtype vmask = vtype{1, 1, 1, 1, 1, 1, 1, 1} << i;
/*#if defined(__AVX512F__)
	const __mmask16 mask = _mm512_test_epi32_mask(_mm512_zextsi256_si512((__m256i)_n), _mm512_zextsi256_si512((__m256i )vmask));
	return uint8_t(mask); */	// because of kmovw	%k0, mem, it may not be faster
	const vtype r = _n & vmask;
#if defined(__AVX2__)
	const int mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32((__m256i)r, _mm256_setzero_si256())));
	return ~uint8_t(mask);
#else
	uint32_t mask = (r[0] != 0) ? 1 : 0;
	for (size_t i = 1; i < 8; ++i) mask |= ((r[i] != 0) ? 1u : 0u) << i;
	return mask;
#endif
}

// conversions

finline Int32_8 UInt32_8_to_Int32_8(const UInt32_8 & rhs) { return Int32_8(__builtin_convertvector(rhs.get(), Int32_8::vtype)); }
finline UInt64_8 Int32_8_to_UInt64_8(const Int32_8 & rhs) { return UInt64_8(__builtin_convertvector(rhs.get(), UInt64_8::vtype)); }
finline UInt64_8 UInt32_8_to_UInt64_8(const UInt32_8 & rhs) { return UInt64_8(__builtin_convertvector(rhs.get(), UInt64_8::vtype)); }
finline UInt32_8 UInt64_8_to_UInt32_8(const UInt64_8 & rhs) { return UInt32_8(__builtin_convertvector(rhs.get(), UInt32_8::vtype)); }
