/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "ocl.h"
#include "vint.h"

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

class ZP
{
protected:
	uint32 _n;

public:
	ZP() {}
	explicit ZP(const uint32 n) : _n(n) {}

	uint32 get() const { return _n; }
};

template<uint32 P, uint32 Q, uint32 R, uint32 H>
class ZPT : public ZP
{
private:
	static uint32 _add(const uint32 a, const uint32 b) { return a + b - ((a >= P - b) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { return a - b + ((a < b) ? P : 0); }

	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

public:
	ZPT() {}
	explicit ZPT(const uint32 n) : ZP(n) {}

	int32 get_int() const { return (_n >= P / 2) ? int32(_n - P) : int32(_n); }
	ZPT & set_int(const int32 i) { _n = (i < 0) ? (uint32(i) + P) : uint32(i); return *this; }

	ZPT mul(const ZPT & rhs) const { return ZPT(_mul(_n, rhs._n)); }

	ZPT pow(const size_t e) const
	{
		if (e == 0) return ZPT(R);	// MF of one is R
		ZPT r = ZPT(R), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.mul(y); }
		r = r.mul(y);
		return r;
	}

	static const ZPT primroot_n(const uint32 n) { return ZPT(H).pow((P - 1) / n); }
	static ZPT norm(const uint32 n) { return ZPT(P - (P - 1) / n); }
};

#define CREATE_TRANSFORM_KERNEL(name) _##name = create_transform_kernel(#name);

template<size_t VSIZE, bool IS32>
class engine : public device
{
private:
	const size_t _n;
	const int _ln;
	const bool _is_boinc;
	const size_t _num_regs;

	cl_mem _z = nullptr, _zp = nullptr, _w = nullptr, _c = nullptr;

	cl_kernel _forward8 = nullptr, _backward8 = nullptr, _forward8_0 = nullptr;
	cl_kernel _square2x4 = nullptr, _square4x2 = nullptr, _square8 = nullptr;

public:
	engine(const platform & platform, const size_t device_id, const int ln, const bool is_boinc, const size_t num_regs)
		: device(platform, device_id), _n(size_t(1) << ln), _ln(ln), _is_boinc(is_boinc), _num_regs(num_regs) {}
	virtual ~engine() {}


///////////////////////////////

public:
	void alloc_memory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t n = _n;
		if (n != 0)
		{
			_z = _createBuffer(CL_MEM_READ_WRITE, 3 * VSIZE * _num_regs * n * sizeof(ZP));
			_zp = _createBuffer(CL_MEM_READ_WRITE, 3 * VSIZE * n * sizeof(ZP));
			_w = _createBuffer(CL_MEM_READ_ONLY, 3 * n / 2 * sizeof(ZP));
			_c = _createBuffer(CL_MEM_READ_WRITE, n / 4 * sizeof(int64));
		}
	}
	void release_memory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Free gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		if (_n != 0)
		{
			_releaseBuffer(_z); _releaseBuffer(_zp);
			_releaseBuffer(_w); _releaseBuffer(_c);
		}
	}

///////////////////////////////

private:
	cl_kernel create_transform_kernel(const char * const kernel_name, const bool is_multiplier = true)
	{
		cl_kernel kernel = _createKernel(kernel_name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), is_multiplier ? &_z : &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_w);
		return kernel;
	}

public:
	void create_kernels(const UInt32_8 & b)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl kernels." << std::endl;
		pio::display(ss.str());
#endif

		CREATE_TRANSFORM_KERNEL(forward8);
		CREATE_TRANSFORM_KERNEL(backward8);
		CREATE_TRANSFORM_KERNEL(forward8_0);

		CREATE_TRANSFORM_KERNEL(square2x4);
		CREATE_TRANSFORM_KERNEL(square4x2);
		CREATE_TRANSFORM_KERNEL(square8);
	}

	void release_kernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif

		_releaseKernel(_forward8); _releaseKernel(_backward8); _releaseKernel(_forward8_0);
		_releaseKernel(_square2x4); _releaseKernel(_square4x2); _releaseKernel(_square8);
	}

///////////////////////////////

public:
	void read_memory_z(ZP * const z_ptr, const size_t count = 1) { _readBuffer(_z, z_ptr, 3 * VSIZE * count * _n * sizeof(ZP)); }
	void write_memory_z(const ZP * const z_ptr, const size_t count = 1) { _writeBuffer(_z, z_ptr, 3 * VSIZE * count * _n * sizeof(ZP)); }
	void write_memory_w(const ZP * const w_ptr, const size_t offset) { _writeBuffer(_w, w_ptr, _n / 2 * sizeof(ZP), offset * _n / 2 * sizeof(ZP)); }

///////////////////////////////

public:
	void forward8(const int lm, const size_t s)
	{
		const int32 ilm = int32(lm); const uint32 is = uint32(s);
		_setKernelArg(_forward8, 2, sizeof(int32), &ilm);
		_setKernelArg(_forward8, 3, sizeof(uint32), &is);
		_executeKernel(_forward8, 3 * VSIZE * _n / 8);
	}

	void backward8(const int lm, const size_t s)
	{
		const int32 ilm = int32(lm); const uint32 is = uint32(s);
		_setKernelArg(_backward8, 2, sizeof(int32), &ilm);
		_setKernelArg(_backward8, 3, sizeof(uint32), &is);
		_executeKernel(_backward8, 3 * VSIZE * _n / 8);
	}

	void forward8_0() { _executeKernel(_forward8_0, 3 * VSIZE * _n / 8); }

	void square2x4() { _executeKernel(_square2x4, 3 * VSIZE * _n / 8); }
	void square4x2() { _executeKernel(_square4x2, 3 * VSIZE * _n / 8); }
	void square8() { _executeKernel(_square8, 3 * VSIZE * _n / 8); }
};
