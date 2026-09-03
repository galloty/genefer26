/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "b_vec.h"

class mpz_vec
{
private:
	const size_t _size;
	mpz_t _z[32];

public:
	explicit mpz_vec(const size_t size) : _size(size) { for (size_t j = 0; j < size; ++j) mpz_init(_z[j]); }
	virtual ~mpz_vec() { for (size_t j = 0, size = _size; j < size; ++j) mpz_clear(_z[j]); }

	size_t get_size() const { return _size; }

	const mpz_t & operator[](const size_t i) const { return _z[i]; }
	mpz_t & operator[](const size_t i) { return _z[i]; }

	void set_ui(const b_vec & n)
	{
		const size_t size_8 = _size / 8;
		mpz_t * const z = _z;

		for (size_t j = 0; j < size_8; ++j)
		{
			for (size_t i = 0; i < 8; ++i) mpz_set_ui(z[8 * j + i], n[j][i]);
		}
	}

	void set_exponent(const b_vec & b, const int n)
	{
		const size_t size_8 = _size / 8;
		mpz_t * const z = _z;

		for (size_t j = 0; j < size_8; ++j)
		{
			for (size_t i = 0; i < 8; ++i) mpz_ui_pow_ui(z[8 * j + i], b[j][i], 1u << n);
		}
	}

	void set_GL_residue(const mpz_vec & exponent, const int B_GL)
	{
		const size_t size = _size;
		mpz_t * const z = _z;
		const mpz_t * const ez = exponent._z;

		for (size_t j = 0; j < size; ++j) mpz_set_ui(z[j], 0);

		mpz_t e, t; mpz_inits(e, t, nullptr);
		for (size_t j = 0; j < size; ++j)
		{
			mpz_set(e, ez[j]);

			while (mpz_sgn(e) != 0)
			{
				mpz_mod_2exp(t, e, static_cast<unsigned long int>(B_GL));
				mpz_add(z[j], z[j], t);
				mpz_div_2exp(e, e, static_cast<unsigned long int>(B_GL));
			}
		}
		mpz_clears(e, t, nullptr);
	}

	void set_PL_residue(const mpz_vec & exponent, const int B_PL, const mpz_vec * const w, const size_t L)
	{
		const size_t size = _size;
		mpz_t * const z = _z;
		const mpz_t * const ez = exponent._z;

		for (size_t j = 0; j < size; ++j) mpz_set_ui(_z[j], 0);

		mpz_t e, t; mpz_inits(e, t, nullptr);
		for (size_t j = 0; j < size; ++j)
		{
			mpz_set(e, ez[j]);

			for (size_t i = 0; i < L; i++)
			{
				mpz_mod_2exp(t, e, static_cast<unsigned long int>(B_PL));
				mpz_addmul(z[j], t, w[i][j]);
				mpz_div_2exp(e, e, static_cast<unsigned long int>(B_PL));
			}
		}
		mpz_clears(e, t, nullptr);
	}

	void add_ui(const uint32_t n)
	{
		const size_t size_8 = _size / 8;
		mpz_t * const z = _z;

		for (size_t j = 0; j < size_8; ++j)
		{
			for (size_t i = 0; i < 8; ++i) mpz_add_ui(z[8 * j + i], z[8 * j + i], n);
		}
	}

	void mul_ui(const uint32_t n)
	{
		const size_t size_8 = _size / 8;
		mpz_t * const z = _z;

		for (size_t j = 0; j < size_8; ++j)
		{
			for (size_t i = 0; i < 8; ++i) mpz_mul_ui(z[8 * j + i], z[8 * j + i], n);
		}
	}

	void mul_ui(const mpz_vec & rhs, const b_vec & n)
	{
		const size_t size_8 = _size / 8;
		mpz_t * const z = _z;
		const mpz_t * const rz = rhs._z;

		for (size_t j = 0; j < size_8; ++j)
		{
			for (size_t i = 0; i < 8; ++i) mpz_mul_ui(z[8 * j + i], rz[8 * j + i], n[j][i]);
		}
	}

	void mul_2exp(const mpz_vec & rhs, const unsigned long int e)
	{
		const size_t size = _size;
		mpz_t * const z = _z;
		const mpz_t * const rz = rhs._z;

		for (size_t j = 0; j < size; ++j) mpz_mul_2exp(z[j], rz[j], e);
	}

	size_t get_max_size() const
	{
		const size_t size = _size;
		const mpz_t * const z = _z;

		size_t max_index = 0;
		for (size_t j = 0; j < size; ++j) max_index = std::max(max_index, mpz_sizeinbase(z[j], 2));
		return max_index;
	}

	uint32_t get_bit_mask(const int i) const
	{
		const size_t size = _size;
		const mpz_t * const z = _z;

		uint32_t mask = 0;
		for (size_t j = 0; j < size; ++j) mask |= ((mpz_tstbit(z[j], mp_bitcnt_t(i)) != 0) ? 1u : 0u) << j;
		return mask;
	}

	uint32_t cmp_sub(const mpz_t & rhs)
	{
		const size_t size = _size;
		mpz_t * const z = _z;

		uint32_t mask = 0;
		for (size_t j = 0; j < size; ++j)
		{
			if (mpz_cmp(z[j], rhs) > 0)
			{
				mpz_sub(z[j], z[j], rhs);
				mask |= 1u << j;
			}
		}
		return mask;
	}
};
