/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <string>
#include <sstream>

#include "gint.h"
#include "file.h"

class transform
{
protected:
	enum class EKind { GPU, CPU };

private:
	const uint32_t _b;
	const uint32_t _n;
	const EKind _kind;
	std::string _type;

protected:
	virtual void getZi(int32_t * const zi) const = 0;
	virtual void setZi(const int32_t * const zi) = 0;

public:
	virtual void set(const uint32_t a) = 0;					// r_0 = a
	virtual void square_dup(const bool dup) = 0;			// r_0 = r_0^2 or 2*r_0^2
	virtual void init_multiplicand(const size_t src) = 0;	// r_m = transform(r_src)
	virtual void mul() = 0;									// r_0 *= r_m

	virtual void copy(const size_t dst, const size_t src) const = 0;	// r_dst = r_src

	virtual bool read_checkpoint(file & cfile, const size_t num_regs) = 0;
	virtual void save_checkpoint(file & cfile, const size_t num_regs) const = 0;

	virtual size_t get_cache_size() const = 0;
	virtual double get_error() const { return 0; }

private:
	static transform * create_ocl(const uint32_t b, const uint32_t n, const size_t num_regs, const size_t device,
								  const bool is_boinc, const bool get_boinc_ids);
	static transform * create_512(const uint32_t b, const uint32_t n, const size_t num_regs);
	static transform * create_fma(const uint32_t b, const uint32_t n, const size_t num_regs);
	static transform * create_avx(const uint32_t b, const uint32_t n, const size_t num_regs);
	static transform * create_sse4(const uint32_t b, const uint32_t n, const size_t num_regs);

public:
	transform(const uint32_t b, const uint32_t n, const EKind kind) : _b(b), _n(n), _kind(kind) {}
	virtual ~transform() {}

protected:
	uint32_t get_b() const { return _b; }
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
	static transform * create_gpu(const uint32_t b, const uint32_t n, const size_t num_regs, const size_t device,
								  const bool isBoinc, const bool get_boinc_ids)
	{
		transform * const ptransform = transform::create_ocl(b, n, num_regs, device, isBoinc, get_boinc_ids);

		if (ptransform == nullptr) throw std::runtime_error("OpenCL device not found");
		return ptransform;
	}

	static transform * create_cpu(const uint32_t b, const uint32_t n, const size_t num_regs)
	{
		transform * ptransform = nullptr;

		__builtin_cpu_init();
		if (__builtin_cpu_supports("avx512f"))
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

	void getInt(gint & g) const
	{
		if ((g.get_size() != (size_t(1) << _n)) || (g.get_base() != _b)) throw std::runtime_error("getInt");
		getZi(g.data());
		g.reset();
	}

	void setInt(gint & g)
	{
		if ((g.get_size() != (size_t(1) << _n)) || (g.get_base() != _b)) throw std::runtime_error("setInt");
		g.balance();
		setZi(g.data());
	}
};
