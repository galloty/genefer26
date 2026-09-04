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
#include "b_vec.h"
#include "file.h"
#include "alignment.h"

#ifndef finline
#define finline	__attribute__((always_inline)) inline
#endif

class itransform
{
private:
	std::string _gpu_type;

protected:
	enum class EKind { GPU, CPU };

	void set_gpu_type(const std::string & type) { _gpu_type = type; }

public:
	const std::string get_gpu_type() const { return _gpu_type; }

	static size_t display_devices();
};

template<size_t VSIZE>
class transform : public itransform
{
private:
	const b_vec _b;
	const int _ln;
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
	virtual void mul_mask(const uint32_t mask) = 0;

	virtual void copy(const size_t dst, const size_t src) const = 0;	// r_dst = r_src
	virtual void copy_mask(const size_t dst, const size_t src, const uint32_t mask) const = 0;

	virtual void power(const size_t src, const uint32_t e) = 0;
	virtual void power_vec(const size_t src, const b_vec & e) = 0;

	virtual bool read_checkpoint(file & cfile) = 0;
	virtual void save_checkpoint(file & cfile) const = 0;

	virtual size_t get_data_size() const = 0;
	virtual size_t get_cache_size() const = 0;
	virtual double get_error() const { return 0; }

	// the binary code must be generated for each instruction set
	virtual void is_one(bool b[32], UInt64_8 res64[4]) const = 0;
	virtual void gethash64(UInt64_8 h[4]) const = 0;
	virtual b_vec gethash32() const = 0;

#ifdef QVALID
	virtual void cosmic_ray() = 0;
#endif

private:
	static transform * create_ocl_avx512(const b_vec & b, const int n, const size_t num_regs, const size_t device,
		const bool is_boinc, const bool get_boinc_ids, int _boinc_argc, char ** _boinc_argv);
	static transform * create_ocl_avx2(const b_vec & b, const int n, const size_t num_regs, const size_t device,
		const bool is_boinc, const bool get_boinc_ids, int _boinc_argc, char ** _boinc_argv);
	static transform * create_ocl_sse2(const b_vec & b, const int n, const size_t num_regs, const size_t device,
		const bool is_boinc, const bool get_boinc_ids, int _boinc_argc, char ** _boinc_argv);

	static transform * create_avx10(const b_vec & b, const int n, const size_t num_regs);
	static transform * create_avx512(const b_vec & b, const int n, const size_t num_regs);
	static transform * create_fma(const b_vec & b, const int n, const size_t num_regs);
	static transform * create_avx(const b_vec & b, const int n, const size_t num_regs);
	static transform * create_sse4(const b_vec & b, const int n, const size_t num_regs);

public:
	transform(const b_vec & b, const int ln, const EKind kind) : _b(b), _ln(ln), _kind(kind),
		_d(static_cast<Int32_8 *>(align_new(sizeof(Int32_8) << ln, sizeof(Int32_8)))) { _unbalanced = false; }
	virtual ~transform() { align_delete(_d); }

private:
	finline void unbalance() const
	{
		if (_unbalanced) return;
		_unbalanced = true;

		const size_t n = size_t(1) << _ln;
		const Int32_8 base = UInt32_8_to_Int32_8(_b[0]);	// TODO
		Int32_8 * const d = _d;
		Int32_8 f = Int32_8(0);

		// We have -base <= d[i] <= base, -1 <= f <= 1
		for (size_t i = 0; i < n; ++i)
		{
			Int32_8 r = d[i] + f;
			f = Int32_8(0);

			const Int32_8 l = (r < Int32_8(0)), ge = (r >= base);
			r += (base & l); f -= (Int32_8(1) & l);
			const Int32_8 l2 = (r < Int32_8(0));	// This should not occur but quick and safer
			r -= (base & ge); f += (Int32_8(1) & ge);
			r += (base & l2); f -= (Int32_8(1) & l2);

			d[i] = r;
		}

		while (!f.is_zero())
		{
			f = -f;	// f * x^size = -f

			for (size_t i = 0; i < n; ++i)
			{
				Int32_8 r = d[i] + f;
				f = Int32_8(0);

				const Int32_8 l = (r < Int32_8(0)), ge = (r >= base);
				r += (base & l); f -= (Int32_8(1) & l);
				r -= (base & ge); f += (Int32_8(1) & ge);

				d[i] = r;

				if (f.is_zero()) return;
			}

			// -1 cannot be unbalanced
			Int32_8 is_minus_one = (f == Int32_8(1));
			for (size_t i = 0; i < n; ++i) is_minus_one &= (d[i] == Int32_8(0));
			d[0] -= (Int32_8(1) & is_minus_one);
			f -= (Int32_8(1) & is_minus_one);
		}
	}

protected:
	const b_vec & get_b() const { return _b; }
	int get_ln() const { return _ln; }
	EKind get_kind() const { return _kind; }
	void set_type(const std::string & type) { _type = type; }

	static constexpr int ilog2_32(const uint32_t n) { return (n == 0) ? -1 : (31 - __builtin_clz(n)); }

	static size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	finline void _power(const size_t src, const uint32_t e)
	{
		init_multiplicand(src);
		set(1);
		for (int i = ilog2_32(e); i >= 0; --i)
		{
			square_dup(0);
			if ((e & (1u << i)) != 0) mul();
		}
	}

	finline void _power_vec(const size_t src, const b_vec & e)
	{
		init_multiplicand(src);
		set(1);
		for (int i = ilog2_32(e.max()); i >= 0; --i)
		{
			square_dup(0);
			mul_mask(e.get_bit_mask(i));
		}
	}

	finline void _is_one(bool b[32], UInt64_8 res64[4]) const
	{
		unbalance();

		const size_t n = size_t(1) << _ln;
		const size_t size = _b.get_size();

		for (size_t j = 0; j < size; ++j)	// TODO
		{
			const UInt64_8 base = UInt32_8_to_UInt64_8(_b[j]);
			const Int32_8 * const d = _d;

			UInt64_8 r64 = Int32_8_to_UInt64_8(d[0]), bi = base;
			Int32_8 one = (d[0] == Int32_8(1));
			for (size_t i = 1; i < n; ++i)
			{
				r64 += bi * Int32_8_to_UInt64_8(d[i]);
				bi *= base;
				one &= (d[i] == Int32_8(0));
			}
			res64[j] = r64;

			for (size_t i = 0; i < 8; ++i) b[8 * j + i] = (one[i] == -1);
		}
	}

	finline void _gethash64(UInt64_8 h[4]) const
	{
		unbalance();

		const size_t n = size_t(1) << _ln;
		const size_t size = _b.get_size();

		for (size_t j = 0; j < size; ++j)	// TODO
		{
			const Int32_8 * const d = _d;
			UInt64_8 hash64 = UInt64_8(uint64_t(0));

			Int32_8 zero = Int32_8(-1);
			for (size_t i = 0; i < n; ++i)
			{
				const Int32_8 d_i = d[i];
				const UInt64_8 a_i = Int32_8_to_UInt64_8(d_i);
				hash64 += a_i;
				hash64 ^= (a_i + UInt64_8(0xc39d8a0552b073e8ull)).rotl((UInt64_8(17) * a_i + UInt64_8(5)) & UInt64_8(63));
				zero &= Int32_8(d_i == Int32_8(0));
			}
			if (zero.is_true()) pio::error("value is zero", true);

			h[j] = hash64;
		}
	}

	finline b_vec _gethash32() const
	{
		UInt64_8 hash64[4]; _gethash64(hash64);
		const size_t size = _b.get_size();
		b_vec r(size);
		for (size_t j = 0; j < size; ++j)
		{
			const UInt32_8 t = UInt64_8_to_UInt32_8(hash64[j]) ^ UInt64_8_to_UInt32_8(hash64[j] >> 32);
			r[j] = t.max(UInt32_8(2));
		}
		return r;
	}

public:
	static transform * create_gpu(const b_vec & b, const int n, const size_t num_regs, const size_t device,
								  const bool isBoinc, const bool get_boinc_ids, int _boinc_argc, char ** _boinc_argv)

	{
		transform * ptransform = nullptr;

		__builtin_cpu_init();
		if (__builtin_cpu_supports("avx512f") != 0)
		{
			ptransform = transform::create_ocl_avx512(b, n, num_regs, device, isBoinc, get_boinc_ids, _boinc_argc, _boinc_argv);
		}
		else if (__builtin_cpu_supports("avx2") != 0)
		{
			ptransform = transform::create_ocl_avx2(b, n, num_regs, device, isBoinc, get_boinc_ids, _boinc_argc, _boinc_argv);
		}
		else ptransform = transform::create_ocl_sse2(b, n, num_regs, device, isBoinc, get_boinc_ids, _boinc_argc, _boinc_argv);	// SSE2 is mandatory for x64 

		return ptransform;
	}

	static transform * create_cpu(const b_vec & b, const int n, const size_t num_regs)
	{
		transform * ptransform = nullptr;

		__builtin_cpu_init();

#if (defined(__GNUC__) && (__GNUC__ >= 15)) || (defined(__clang__) && (__clang_major__ >= 22))
		if (__builtin_cpu_supports("avx10.2") != 0)
		{
			ptransform = transform::create_avx10(b, n, num_regs);
		}
		else
#endif
		if (__builtin_cpu_supports("avx512f") != 0)
		{
			ptransform = transform::create_avx512(b, n, num_regs);
		}
		else if ((__builtin_cpu_supports("fma") != 0) && (__builtin_cpu_supports("avx2") != 0))
		{
			ptransform = transform::create_fma(b, n, num_regs);
		}
		else if (__builtin_cpu_supports("avx") != 0)
		{
			ptransform = transform::create_avx(b, n, num_regs);
		}
		else if (__builtin_cpu_supports("sse4.1") != 0)
		{
			ptransform = transform::create_sse4(b, n, num_regs);
		}

		if (ptransform == nullptr) throw std::runtime_error("processor must support SSE4.1");
		return ptransform;
	}

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
		UInt32_8::vtype nbase; cFile.read(reinterpret_cast<char *>(&nbase), sizeof(nbase));
		if ((size != (1u << _ln)) || !UInt32_8(nbase).is_equal(_b[0])) cFile.error("bad file");	// TODO
		cFile.read(reinterpret_cast<char *>(_d), sizeof(UInt32_8) << _ln);

		_unbalanced = false;
	}

	void write(file & cFile) const
	{
		if (_unbalanced) pio::error("write unbalanced data", true);

		const uint32_t n = 1u << _ln;
		cFile.write(reinterpret_cast<const char *>(&n), sizeof(n));
		const UInt32_8::vtype nbase = _b[0].get();	// TODO
		cFile.write(reinterpret_cast<const char *>(&nbase), sizeof(nbase));
		cFile.write(reinterpret_cast<const char *>(_d), sizeof(UInt32_8) << _ln);
	}
};
