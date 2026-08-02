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

	cl_kernel _backward8 = nullptr;

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

		CREATE_TRANSFORM_KERNEL(backward8);
	}

	void release_kernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif

		_releaseKernel(_backward8);
	}

///////////////////////////////

public:
	void read_memory_z(ZP * const z_ptr, const size_t count = 1) { _readBuffer(_z, z_ptr, 3 * VSIZE * count * _n * sizeof(ZP)); }
	void write_memory_z(const ZP * const z_ptr, const size_t count = 1) { _writeBuffer(_z, z_ptr, 3 * VSIZE * count * _n * sizeof(ZP)); }
	void write_memory_w(const ZP * const w_ptr, const size_t offset) { _writeBuffer(_w, w_ptr, _n / 2 * sizeof(ZP), offset * _n / 2 * sizeof(ZP)); }

///////////////////////////////

public:
	void backward8(const size_t m, const size_t s)
	{
		const uint32 im = uint32(m), is = uint32(s);
		_setKernelArg(_backward8, 2, sizeof(uint32), &im);
		_setKernelArg(_backward8, 3, sizeof(uint32), &is);
		_executeKernel(_backward8, 3 * VSIZE * _n / 8);
	}
};
