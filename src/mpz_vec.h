/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "vint.h"

template<size_t SIZE>
class mpz_vec
{
private:
	mpz_t _z[SIZE];

public:
	mpz_vec() { for (size_t j = 0; j < SIZE; ++j) mpz_init(_z[j]); }
	virtual ~mpz_vec() { for (size_t j = 0; j < SIZE; ++j) mpz_clear(_z[j]); }

	const mpz_t & operator[](const size_t i) const { return _z[i]; }
	mpz_t & operator[](const size_t i) { return _z[i]; }

	void set_ui(const UInt32_8 & n)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_set_ui(_z[j], n[j]);
	}

	void set_exponent(const UInt32_8 & b, const uint32_t n)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_ui_pow_ui(_z[j], b[j], 1u << n);
	}

	void set_GL_residue(const mpz_vec & exponent, const int B_GL)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_set_ui(_z[j], 0);

		mpz_t e, t; mpz_inits(e, t, nullptr);
		for (size_t j = 0; j < SIZE; ++j)
		{
			mpz_set(e, exponent._z[j]);

			while (mpz_sgn(e) != 0)
			{
				mpz_mod_2exp(t, e, static_cast<unsigned long int>(B_GL));
				mpz_add(_z[j], _z[j], t);
				mpz_div_2exp(e, e, static_cast<unsigned long int>(B_GL));
			}
		}
		mpz_clears(e, t, nullptr);
	}

	void set_PL_residue(const mpz_vec & exponent, const int B_PL, const mpz_vec * const w, const size_t L)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_set_ui(_z[j], 0);

		mpz_t e, t; mpz_inits(e, t, nullptr);
		for (size_t j = 0; j < SIZE; ++j)
		{
			mpz_set(e, exponent._z[j]);

			for (size_t i = 0; i < L; i++)
			{
				mpz_mod_2exp(t, e, static_cast<unsigned long int>(B_PL));
				mpz_addmul(_z[j], t, w[i][j]);
				mpz_div_2exp(e, e, static_cast<unsigned long int>(B_PL));
			}
		}
		mpz_clears(e, t, nullptr);
	}

	void add_ui(const mpz_vec & rhs, const UInt32_8 & n)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_add_ui(_z[j], rhs._z[j], n[j]);
	}

	void mul_ui(const mpz_vec & rhs, const UInt32_8 & n)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_mul_ui(_z[j], rhs._z[j], n[j]);
	}

	void mul_2exp(const mpz_vec & rhs, const unsigned long int e)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_mul_2exp(_z[j], rhs._z[j], e);
	}

	int get_max_index() const	// TODO get_size => + 1
	{
		int max_index = -1;
		for (size_t j = 0; j < SIZE; ++j) max_index = std::max(max_index, int(mpz_sizeinbase(_z[j], 2)) - 1);
		return max_index;
	}

	uint32_t get_bit_mask(const size_t i) const
	{
		uint32_t mask = 0;
		for (size_t j = 0; j < SIZE; ++j) mask |= ((mpz_tstbit(_z[j], mp_bitcnt_t(i)) != 0) ? 1u : 0u) << j;
		return mask;
	}

	uint32_t cmp_sub(const mpz_t & rhs)
	{
		uint32_t mask = 0;
		for (size_t j = 0; j < SIZE; ++j)
		{
			if (mpz_cmp(_z[j], rhs) > 0)
			{
				mpz_sub(_z[j], _z[j], rhs);
				mask |= 1u << j;
			}
		}
		return mask;
	}
};
