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

	void set_ui(const vuint32 & n)
	{
		for (size_t j = 0; j < VSIZE; ++j) mpz_set_ui(_z[j], n[j]);
	}

	void set_exponent(const vuint32 & b, const uint32_t n)
	{
		for (size_t j = 0; j < VSIZE; ++j) mpz_ui_pow_ui(_z[j], b[j], 1u << n);
	}

	void set_GL_residue(const mpz_vec & rhs, const int B_GL)
	{
		for (size_t j = 0; j < SIZE; ++j) mpz_set_ui(_z[j], 0);

		mpz_t e, t; mpz_inits(e, t, nullptr);
		for (size_t j = 0; j < VSIZE; ++j)
		{
			mpz_set(e, rhs._z[j]);

			while (mpz_sgn(e) != 0)
			{
				mpz_mod_2exp(t, e, static_cast<unsigned long int>(B_GL));
				mpz_add(_z[j], _z[j], t);
				mpz_div_2exp(e, e, static_cast<unsigned long int>(B_GL));
			}
		}
		mpz_clears(e, t, nullptr);
	}

	void mul_ui(const mpz_vec & rhs, const vuint32 & n)
	{
		for (size_t j = 0; j < VSIZE; ++j) mpz_mul_ui(_z[j], rhs._z[j], n[j]);
	}

	int get_max_index() const	// TODO get_size => + 1
	{
		int max_index = -1;
		for (size_t j = 0; j < VSIZE; ++j) max_index = std::max(max_index, int(mpz_sizeinbase(_z[j], 2)) - 1);
		return max_index;
	}

	uint32_t get_bit_mask(const size_t i) const
	{
		uint32_t m = 0;
		for (size_t j = 0; j < SIZE; ++j) m |= ((mpz_tstbit(_z[j], mp_bitcnt_t(i)) != 0) ? uint32_t(1) : uint32_t(0)) << j;
		return m;
	}
};
