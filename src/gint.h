/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cstdlib>
#include <algorithm>

#include "vint.h"
#include "alignment.h"
#include "file.h"
#include "pio.h"

class gint
{
private:
	const size_t _size;
	const vuint32 _base;
	vint32 * const _d;

	enum class EState { Unknown, Balanced, Unbalanced };
	EState _state;

	class Vint64
	{
	private:
		vint64 _l;

		finline constexpr explicit Vint64(const vint64 & l) : _l(l) {}

		finline int32_t _mod_pos(const size_t i, const int32_t m)
		{
			int32_t r = int32_t(_l[i] % m); _l[i] /= m;
			if (r < 0) { r += m; _l[i] -= 1; }
			return r;
		}

		finline int32_t _mod_balanced(const size_t i, const int32_t m)
		{
			int32_t r = int32_t(_l[i] % m); _l[i] /= m;
			if (r > m / 2) { r -= m; _l[i] += 1; }
			else if (r <= -(m / 2)) { r += m; _l[i] -= 1; }
			return r;
		}

	public:
		finline explicit Vint64() {}

		finline int64_t operator[](const size_t i) const { return _l[i]; }
		finline void set(const size_t i, const int64_t x) { _l[i] = x; }

		finline static Vint64 zero() { vint64 l; set_zero(l); return Vint64(l); }
		finline bool is_zero() const { return cmp_zero(_l); }

		finline Vint64 operator-() const { return Vint64(-_l); }

		finline Vint64 & operator+=(const vint32 & rhs) { _l += __builtin_convertvector(rhs, vint64); return *this; }

		finline void mod_pos(vint32 & y, const vint32 & m)
		{
			for (size_t i = 0; i < VSIZE; ++i) y[i] = _mod_pos(i, m[i]);
		}

		finline void mod_balanced(vint32 & y, const vint32 & m)
		{
			for (size_t i = 0; i < VSIZE; ++i) y[i] = _mod_balanced(i, m[i]);
		}

		finline void add_mod_pos(vint32 & y, const vint32 & m)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				if (_l[i] != 0) { _l[i] += y[i]; y[i] = _mod_pos(i, m[i]); }
			}
		}

		finline void add_mod_balanced(vint32 & y, const vint32 & m)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				if (_l[i] != 0) { _l[i] += y[i]; y[i] = _mod_balanced(i, m[i]); }
			}
		}
	};

	class Vuint64
	{
	private:
		vuint64 _l;

	public:
		finline explicit Vuint64() {}
		finline constexpr explicit Vuint64(const vuint64 & l) : _l(l) {}
		finline explicit Vuint64(const vint32 & rhs) : _l(__builtin_convertvector(rhs, vuint64)) {}
		finline explicit Vuint64(const vuint32 & rhs) : _l(__builtin_convertvector(rhs, vuint64)) {}

		finline const vuint64 & get() const { return _l; }

		finline Vuint64 & operator+=(const Vuint64 & rhs) { _l += rhs._l; return *this; }
		finline Vuint64 & operator*=(const Vuint64 & rhs) { _l *= rhs._l; return *this; }

		finline Vuint64 operator*(const vint32 & rhs) const { return Vuint64(_l * __builtin_convertvector(rhs, vuint64)); }

		finline void get_hash32(vuint32 & hash32) const
		{
			const vuint32 r = __builtin_convertvector(_l, vuint32) ^ __builtin_convertvector(_l >> 32, vuint32);
			hash32 = (r >= 2) ? r : 2;
		}
	};

private:
	static constexpr uint64_t rotl64(const uint64_t x, const uint8_t n) { return (x << n) | (x >> (-n & 63)); }

public:
	gint(const size_t size, const vuint32 & base) : _size(size), _base(base),
				_d(static_cast<vint32 *>(align_new(size * sizeof(vint32), sizeof(vint32)))),
				_state(EState::Unknown) {}
	virtual ~gint() { align_delete(_d); }

	size_t get_size() const { return _size; }
	const vuint32 & get_base() const { return _base; }
	vint32 * data() const { return _d; }

	void reset() { _state = EState::Unknown; }

	void unbalance()
	{
		if (_state == EState::Unbalanced) return;
		_state = EState::Unbalanced;

		std::cout << "unbalance" << std::endl;

		const size_t size = _size;
		const vint32 base = (vint32)_base;
		vint32 * const d = _d;
		Vint64 f = Vint64::zero();

		for (size_t i = 0; i < size; ++i)
		{
			f += d[i];
			f.mod_pos(d[i], base);
		}

		while (!f.is_zero())
		{
			f = -f;	// f * x^size = -f
			for (size_t i = 0; i < size; ++i)
			{
				f.add_mod_pos(d[i], base);
				if (f.is_zero()) return;
			}

			bool is_zero_or_minus_one = true;
			for (size_t j = 0; j < VSIZE; ++j)
			{
				if (f[j] == 0) continue;
				else if (f[j] == 1)
				{
					bool is_minus_one = true; for (size_t i = 0; i < size; ++i) if (d[i][j] != 0) { is_minus_one = false; break; }
					if (is_minus_one) { d[0][j] = -1; f.set(j, 0); }	// -1 cannot be unbalanced
					else { is_zero_or_minus_one = false; break; }
				}
				else { is_zero_or_minus_one = false; break; }
			}

			if (is_zero_or_minus_one) return;
		}
	}

	void balance()
	{
		if (_state == EState::Balanced) return;
		_state = EState::Balanced;

		std::cout << "balance" << std::endl;

		const size_t size = _size;
		const vint32 base = (vint32)_base;
		vint32 * const d = _d;
		Vint64 f = Vint64::zero();

		for (size_t i = 0; i < size; ++i)
		{
			f += d[i];
			f.mod_balanced(d[i], base);
		}

		while (!f.is_zero())
		{
			f = -f;	// f * x^size = -f
			for (size_t i = 0; i < size; ++i)
			{
				f.add_mod_balanced(d[i], base);
				if (f.is_zero()) return;
			}
		}
	}

	void read(file & cFile)
	{
		uint32_t size; cFile.read(reinterpret_cast<char *>(&size), sizeof(size));
		vuint32 base; cFile.read(reinterpret_cast<char *>(&base), sizeof(base));
		if ((size_t(size) != _size) || !cmp(base, _base)) cFile.error("bad file");
		cFile.read(reinterpret_cast<char *>(_d), _size * sizeof(vint32));
		_state = EState::Unbalanced;
	}

	void write(file & cFile)
	{
		unbalance();
		const uint32_t size = static_cast<uint32_t>(_size);
		cFile.write(reinterpret_cast<const char *>(&size), sizeof(size));
		cFile.write(reinterpret_cast<const char *>(&_base), sizeof(_base));
		cFile.write(reinterpret_cast<const char *>(_d), _size * sizeof(vint32));
	}

	void is_one(bool b[VSIZE], vuint64 & res64)
	{
		unbalance();

		const size_t size = _size;
		const vuint32 base = _base;
		const vint32 * const d = _d;

		std::cout << "toto" << std::endl;

		Vuint64 r64 = Vuint64(d[0]), bi = Vuint64(base);
		vint32 one; one = (d[0] == 1);
		for (size_t i = 1; i < size; ++i)
		{
			r64 += bi * d[i];
			bi *= Vuint64(base);
			one &= (d[i] == 0);
		}
		res64 = r64.get();

		for (size_t j = 0; j < VSIZE; ++j) b[j] = (one[j] == -1);
	}

	void gethash64(vuint64 & hash64)
	{
		unbalance();

		const size_t size = _size;
		const vint32 * const d = _d;

		for (size_t j = 0; j < VSIZE; ++j)
		{
			uint64_t hash = 0;
			bool is_zero = true;
			for (size_t i = 0; i < size; ++i)
			{
				const uint32_t a_i = static_cast<uint32_t>(d[i][j]);
				hash += a_i;
				hash ^= rotl64(a_i + 0xc39d8a0552b073e8ull, (17 * static_cast<uint64_t>(a_i) + 5) % 64);
				is_zero &= (a_i == 0);
			}
			if (is_zero) pio::error("value is zero", true);
			hash64[j] = hash;
		}
	}

	void gethash32(vuint32 & hash32)
	{
		vuint64 hash64; gethash64(hash64);
		Vuint64 t(hash64); t.get_hash32(hash32);
	}
};
