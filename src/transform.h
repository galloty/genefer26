/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <string>
#include <sstream>

#include "vint.h"
#include "file.h"
#include "alignment.h"

class transform
{
protected:
	enum class EKind { GPU, CPU };

private:
	const UInt32_8 _b;
	const uint32_t _n;
	const EKind _kind;
	Int32_8 * const _d;
	std::string _type;
	mutable bool _unbalanced;

protected:
	virtual void getZi(Int32_8 * const zi) const = 0;
	virtual void setZi(const Int32_8 * const zi) = 0;

public:
	virtual void set(const uint32_t a) = 0;					// r_0 = a
	virtual void square_dup(const uint32_t dup) = 0;		// r_0 = r_0^2 or 2*r_0^2
	virtual void init_multiplicand(const size_t src) = 0;	// r_m = transform(r_src)
	virtual void mul() = 0;									// r_0 *= r_m
	virtual void mul_mask(const uint8_t mask) = 0;

	virtual void copy(const size_t dst, const size_t src) const = 0;	// r_dst = r_src
	virtual void copy_mask(const size_t dst, const size_t src, const uint8_t mask) const = 0;

	virtual bool read_checkpoint(file & cfile, const size_t num_regs) = 0;
	virtual void save_checkpoint(file & cfile, const size_t num_regs) const = 0;

	virtual size_t get_cache_size() const = 0;
	virtual double get_error() const { return 0; }

	virtual void cosmic_ray() = 0;

private:
	static transform * create_ocl(const UInt32_8 & b, const uint32_t n, const size_t num_regs, const size_t device,
								  const bool is_boinc, const bool get_boinc_ids);
	static transform * create_avx10(const UInt32_8 & b, const uint32_t n, const size_t num_regs);
	static transform * create_512(const UInt32_8 & b, const uint32_t n, const size_t num_regs);
	static transform * create_fma(const UInt32_8 & b, const uint32_t n, const size_t num_regs);
	static transform * create_avx(const UInt32_8 & b, const uint32_t n, const size_t num_regs);
	static transform * create_sse4(const UInt32_8 & b, const uint32_t n, const size_t num_regs);

public:
	transform(const UInt32_8 & b, const uint32_t n, const EKind kind) : _b(b), _n(n), _kind(kind),
		_d(static_cast<Int32_8 *>(align_new(sizeof(Int32_8) << n, sizeof(Int32_8)))) { _unbalanced = false; }
	virtual ~transform() { align_delete(_d); }

private:
	static constexpr uint64_t rotl64(const uint64_t x, const uint8_t n) { return (x << n) | (x >> (-n & 63)); }

	void unbalance() const
	{
		if (_unbalanced) return;

		const size_t size = size_t(1) << _n;
		const Int32_8 base = Int32_8(_b);
		Int32_8 * const d = _d;
		Int32_8 f = Int32_8(0);

		// We have -base <= d[i] <= base, -1 <= f <= 1
		for (size_t i = 0; i < size; ++i)
		{
			Int32_8 r = d[i] + f;
			f = Int32_8(0);
			for (size_t j = 0; j < VSIZE; ++j)
			{
				if (r[j] >= base[j]) { r.set_i(j, r[j] - base[j]); f.set_i(j, f[j] + 1); }
				else if (r[j] < 0) { r.set_i(j, r[j] + base[j]); f.set_i(j, f[j] - 1); }
			}

			d[i] = r;
		}

		for (size_t j = 0; j < VSIZE; ++j)
		{
			while (f[j] != 0)
			{
				f.set_i(j, -f[j]);	// f * x^size = -f

				for (size_t i = 0; i < size; ++i)
				{
					int32_t r = d[i][j] + f[j];
					f.set_i(j, 0);
					if (r >= base[j]) { r -= base[j]; f.set_i(j, f[j] + 1); }
					else if (r < 0) { r += base[j]; f.set_i(j, f[j] - 1); }
					d[i].set_i(j, r);
					if (f[j] == 0) break;
				}

				if (f[j] == 1)
				{
					bool is_minus_one = true;
					for (size_t i = 0; i < size; ++i) if (d[i][j] != 0) { is_minus_one = false; break; }
					if (is_minus_one) { d[0].set_i(j, -1); f.set_i(j, 0); }	// -1 cannot be unbalanced
				}
			}
		}

		_unbalanced = true;
	}

protected:
	const UInt32_8 & get_b() const { return _b; }
	uint32_t get_n() const { return _n; }
	EKind get_kind() const { return _kind; }
	void set_type(const std::string & type) { _type = type; }

	static size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

public:
	static transform * create_gpu(const UInt32_8 & b, const uint32_t n, const size_t num_regs, const size_t device,
								  const bool isBoinc, const bool get_boinc_ids)
	{
		transform * const ptransform = transform::create_ocl(b, n, num_regs, device, isBoinc, get_boinc_ids);

		if (ptransform == nullptr) throw std::runtime_error("OpenCL device not found");
		return ptransform;
	}

	static transform * create_cpu(const UInt32_8 & b, const uint32_t n, const size_t num_regs)
	{
		transform * ptransform = nullptr;

		__builtin_cpu_init();
		if (__builtin_cpu_supports("avx10.2"))
		{
			ptransform = transform::create_avx10(b, n, num_regs);
		}
		else if (__builtin_cpu_supports("avx512f"))
		{
			ptransform = transform::create_512(b, n, num_regs);
		}
		else if (__builtin_cpu_supports("fma"))
		{
			ptransform = transform::create_fma(b, n, num_regs);
		}
		else if (__builtin_cpu_supports("avx"))
		{
			ptransform = transform::create_avx(b, n, num_regs);
		}
		else if (__builtin_cpu_supports("sse4.1"))
		{
			ptransform = transform::create_sse4(b, n, num_regs);
		}

		if (ptransform == nullptr) throw std::runtime_error("processor must support SSE4.1");
		return ptransform;
	}

	static size_t display_devices();

	const std::string get_type() const { return _type; }

	void mul(const size_t src)
	{
		init_multiplicand(src);
		mul();
	}

	void to_int() const
	{
		getZi(_d);
		_unbalanced = false;
	}

	void from_int()
	{
		if (_unbalanced) pio::error("from_int unbalanced data", true);
		setZi(_d);
	}

	void read(file & cFile)
	{
		uint32_t size; cFile.read(reinterpret_cast<char *>(&size), sizeof(size));
		UInt32_8::uint32_8 nbase; cFile.read(reinterpret_cast<char *>(&nbase), sizeof(nbase));
		if ((size != (1u << _n)) || (UInt32_8(nbase) != _b)) cFile.error("bad file");
		cFile.read(reinterpret_cast<char *>(_d), sizeof(UInt32_8) << _n);

		_unbalanced = false;
	}

	void write(file & cFile) const
	{
		if (_unbalanced) pio::error("write unbalanced data", true);

		const uint32_t size = 1u << _n;
		cFile.write(reinterpret_cast<const char *>(&size), sizeof(size));
		const UInt32_8::uint32_8 nbase = _b.get();
		cFile.write(reinterpret_cast<const char *>(&nbase), sizeof(nbase));
		cFile.write(reinterpret_cast<const char *>(_d), sizeof(UInt32_8) << _n);
	}

	void is_one(bool b[VSIZE], UInt64_8 & res64) const
	{
		unbalance();

		const size_t size = size_t(1) << _n;
		const Int32_8 base = Int32_8(_b);
		const Int32_8 * const d = _d;

		UInt64_8 r64 = UInt64_8(d[0]), bi = UInt64_8(base);
		Int32_8::int32_8 one; one = (d[0] == Int32_8(1));
		for (size_t i = 1; i < size; ++i)
		{
			r64 += bi * UInt64_8(d[i]);
			bi *= UInt64_8(base);
			one &= (d[i] == Int32_8(0));
		}
		res64 = r64;

		for (size_t j = 0; j < VSIZE; ++j) b[j] = (one[j] == -1);
	}

	UInt64_8 gethash64() const
	{
		unbalance();

		const size_t size = size_t(1) << _n;
		const Int32_8 * const d = _d;
		UInt64_8::uint64_8 hash64;
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

		return UInt64_8(hash64);
	}

	UInt32_8 gethash32() const
	{
		const UInt64_8 hash64 = gethash64();
		const UInt32_8::uint32_8 r = __builtin_convertvector(hash64.get(), UInt32_8::uint32_8) ^ __builtin_convertvector(hash64.get() >> 32, UInt32_8::uint32_8);
	 	return UInt32_8((r >= 2) ? r : 2);
	}
};
