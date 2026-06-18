/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include <gmp.h>

#include "transform.h"

class transformGMP : public transform
{
private:
	const size_t _num_regs;
	mpz_t * const _x;
	mpz_t _g, _y;

public:
	transformGMP(const uint32_t b, const uint32_t n, const size_t num_regs)
				: transform(size_t(1) << n, b, EKind::GPU), _num_regs(num_regs), _x(new mpz_t[num_regs])
	{
		for (size_t i = 0; i < num_regs; ++i) mpz_init(_x[i]);
		mpz_inits(_g, _y, nullptr);
		mpz_ui_pow_ui(_g, b, uint32_t(1) << n);
		mpz_add_ui(_g, _g, 1);
	}

	virtual ~transformGMP()
	{
		mpz_clears(_g, _y, nullptr);
		for (size_t i = 0, num_regs = _num_regs; i < num_regs; ++i) mpz_clear(_x[i]);
		delete[] _x;
	}

protected:
	void getZi(int32_t * const zi) const override
	{
		const size_t size = get_size();
		const uint32_t b = get_b();
		const mpz_t & x = _x[0];

		mpz_t q; mpz_init_set(q, x);
		for (size_t i = 0; i < size; ++i)
		{
			const uint32_t r = mpz_fdiv_q_ui(q, q, b);
			zi[i] = int32_t(r);
		}
		mpz_clear(q);
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size = get_size();
		const uint32_t b = get_b();
		mpz_t & x = _x[0];

		mpz_set_ui(x, 0);
		for (size_t i = 0; i < size; ++i)
		{
			mpz_mul_ui(x, x, b);
			const int32_t r = zi[size - 1 - i];
			if (r >= 0) mpz_add_ui(x, x, uint32_t(r)); else mpz_sub_ui(x, x, uint32_t(-r));
		}
	}

public:
	void set(const uint32_t a) override
	{
		mpz_set_ui(_x[0], a);
	}

	void square_dup(const bool dup) override
	{
		mpz_t & x = _x[0];

		mpz_mul(x, x, x);
		if (dup) mpz_add(x, x, x);
		mpz_mod(x, x, _g);
	}

	void init_multiplicand(const size_t src) override
	{
		mpz_set(_y, _x[src]);
	}

	void mul() override
	{
		mpz_t & x = _x[0];

		mpz_mul(x, _y, x);
		mpz_mod(x, x, _g);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		mpz_set(_x[dst], _x[src]);
	}

	size_t get_mem_size() const override { return 0; }
	size_t get_cache_size() const override { return 0; }

	bool read_checkpoint(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		for (size_t i = 0; i < num_regs; ++i) if (!cFile.read(_x[i])) return false;

		return true;
	}

	void save_checkpoint(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t num_regs = (nregs != 0) ? nregs : _num_regs;
		for (size_t i = 0; i < num_regs; ++i) if (!cFile.write(_x[i])) return;
	}

	double getError() const override { return 0.0; }
};
