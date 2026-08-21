/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "ocl.h"

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;
typedef cl_uint2	uint32_2;

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
#define CREATE_TRANSFORM_KERNELP(name) _##name = create_transform_kernel(#name, false);
#define CREATE_MUL_KERNEL(name) _##name = create_mul_kernel(#name);
#define CREATE_CARRY_KERNEL(name) _##name = create_carry_kernel(#name);
#define CREATE_SETCOPY_KERNEL(name) _##name = create_set_copy_kernel(#name);
#define CREATE_COPYP_KERNEL(name) _##name = create_copyp_kernel(#name);

template<size_t VSIZE, size_t CARRY_LENGTH, size_t OCL_VSIZE, size_t OCL_CARRY_VSIZE, bool IS32>
class engine : public device
{
private:
	const size_t _n;
	const int _ln;
	const bool _is_boinc;
	const size_t _num_regs;
	const int _carry_shift;

	cl_mem _z = nullptr, _zp = nullptr, _w = nullptr, _c = nullptr, _bb_inv = nullptr, _bs = nullptr;

	cl_kernel _forward8 = nullptr, _backward8 = nullptr, _forward8_0 = nullptr;
	cl_kernel _square2x4 = nullptr, _square4x2 = nullptr, _square8 = nullptr;
	cl_kernel _fwd4x2 = nullptr, _fwd8 = nullptr;
	cl_kernel _mul2x4 = nullptr, _mul4x2 = nullptr, _mul8 = nullptr;
	cl_kernel _mul2x4_mask = nullptr, _mul4x2_mask = nullptr, _mul8_mask = nullptr;
	cl_kernel _carry1 = nullptr, _carry2 = nullptr;
	cl_kernel _set = nullptr, _copy = nullptr, _copyp = nullptr, _copy_mask = nullptr, _cosmic_ray = nullptr;

	static constexpr int ilog2_32(const uint32_t n) { return (n == 0) ? -1 : (31 - __builtin_clz(n)); }

public:
	engine(const platform & platform, const size_t device_id, const int ln, const bool is_boinc, const size_t num_regs)
		: device(platform, device_id), _n(size_t(1) << ln), _ln(ln), _is_boinc(is_boinc), _num_regs(num_regs),
		_carry_shift(ilog2_32(uint32_t(std::min(_n / CARRY_LENGTH, std::min(size_t(256), getMaxWorkGroupSize())) / OCL_VSIZE))) {}
	virtual ~engine() {}

	int get_carry_shift() const { return _carry_shift; }
	size_t get_carry_workgroup_size() const { return (OCL_VSIZE << _carry_shift) / OCL_CARRY_VSIZE; }

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
			_c = _createBuffer(CL_MEM_READ_WRITE, VSIZE * n / 8 * sizeof(int64));
			_bb_inv = _createBuffer(CL_MEM_READ_ONLY, VSIZE * sizeof(uint32_2));
			_bs = _createBuffer(CL_MEM_READ_ONLY, VSIZE * sizeof(int32));
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
			_releaseBuffer(_z); _releaseBuffer(_zp); _releaseBuffer(_w);
			_releaseBuffer(_c); _releaseBuffer(_bb_inv); _releaseBuffer(_bs);
		}
	}

///////////////////////////////

private:
	void set_transform_arg0(cl_kernel & kernel, const bool is_multiplier = true)
	{
		_setKernelArg(kernel, 0, sizeof(cl_mem), is_multiplier ? &_z : &_zp);
	}

	cl_kernel create_transform_kernel(const char * const kernel_name, const bool is_multiplier = true)
	{
		cl_kernel kernel = _createKernel(kernel_name);
		set_transform_arg0(kernel, is_multiplier);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_w);
		return kernel;
	}

	cl_kernel create_mul_kernel(const char * const kernel_name)
	{
		cl_kernel kernel = _createKernel(kernel_name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_zp);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_w);
		return kernel;
	}

	cl_kernel create_carry_kernel(const char * const kernel_name)
	{
		cl_kernel kernel = _createKernel(kernel_name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_bs);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_c);
		return kernel;
	}

	cl_kernel create_set_copy_kernel(const char * const kernel_name)
	{
		cl_kernel kernel = _createKernel(kernel_name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		return kernel;
	}

	cl_kernel create_copyp_kernel(const char * const kernel_name)
	{
		cl_kernel kernel = _createKernel(kernel_name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_z);
		return kernel;
	}

public:
	void create_kernels()
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

		CREATE_TRANSFORM_KERNELP(fwd4x2);
		CREATE_TRANSFORM_KERNELP(fwd8);

		CREATE_MUL_KERNEL(mul2x4);
		CREATE_MUL_KERNEL(mul4x2);
		CREATE_MUL_KERNEL(mul8);

		CREATE_MUL_KERNEL(mul2x4_mask);
		CREATE_MUL_KERNEL(mul4x2_mask);
		CREATE_MUL_KERNEL(mul8_mask);

		CREATE_CARRY_KERNEL(carry1);
		CREATE_CARRY_KERNEL(carry2);

		CREATE_SETCOPY_KERNEL(set);
		CREATE_SETCOPY_KERNEL(copy);
		CREATE_COPYP_KERNEL(copyp);
		CREATE_SETCOPY_KERNEL(copy_mask);
		CREATE_SETCOPY_KERNEL(cosmic_ray);
	}

	void release_kernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif

		_releaseKernel(_forward8); _releaseKernel(_backward8); _releaseKernel(_forward8_0);
		_releaseKernel(_square2x4); _releaseKernel(_square4x2); _releaseKernel(_square8);
		_releaseKernel(_fwd4x2); _releaseKernel(_fwd8);
		_releaseKernel(_mul2x4); _releaseKernel(_mul4x2); _releaseKernel(_mul8);
		_releaseKernel(_mul2x4_mask); _releaseKernel(_mul4x2_mask); _releaseKernel(_mul8_mask);
		_releaseKernel(_carry1); _releaseKernel(_carry2);
		_releaseKernel(_set); _releaseKernel(_copy); _releaseKernel(_copyp); _releaseKernel(_copy_mask); _releaseKernel(_cosmic_ray);
	}

///////////////////////////////

public:
	void read_memory_z(ZP * const z_ptr, const size_t count = 1) { _readBuffer(_z, z_ptr, 3 * VSIZE * count * _n * sizeof(ZP)); }
	void write_memory_z(const ZP * const z_ptr, const size_t count = 1) { _writeBuffer(_z, z_ptr, 3 * VSIZE * count * _n * sizeof(ZP)); }
	void write_memory_w(const ZP * const w_ptr) { _writeBuffer(_w, w_ptr, 3 * _n / 2 * sizeof(ZP)); }
	void write_memory_b(const uint32_t * const b, const uint32_t * const b_inv, const int * const b_s)
	{
		uint32_2 bb_inv[VSIZE]; for (size_t i = 0; i < VSIZE; ++i) { bb_inv[i].s[0] = b[i]; bb_inv[i].s[1] = b_inv[i]; }
		int32 bs[8]; for (size_t i = 0; i < VSIZE; ++i) bs[i] = static_cast<int32>(b_s[i]);
		_writeBuffer(_bb_inv, bb_inv, VSIZE * sizeof(uint32_2));
		_writeBuffer(_bs, bs, VSIZE * sizeof(int32));
	}

///////////////////////////////

public:
	void forward8(const int lm, const size_t s)
	{
		const int32 ilm = int32(lm); const uint32 is = uint32(s);
		_setKernelArg(_forward8, 2, sizeof(int32), &ilm);
		_setKernelArg(_forward8, 3, sizeof(uint32), &is);
		_executeKernel(_forward8, 3 * VSIZE / OCL_VSIZE * _n / 8);
	}

	void backward8(const int lm, const size_t s)
	{
		const int32 ilm = int32(lm); const uint32 is = uint32(s);
		_setKernelArg(_backward8, 2, sizeof(int32), &ilm);
		_setKernelArg(_backward8, 3, sizeof(uint32), &is);
		_executeKernel(_backward8, 3 * VSIZE / OCL_VSIZE * _n / 8);
	}

	void forward8_0() { _executeKernel(_forward8_0, 3 * VSIZE / OCL_VSIZE * _n / 8); }

	void square2x4() { _executeKernel(_square2x4, 3 * VSIZE / OCL_VSIZE * _n / 8); }
	void square4x2() { _executeKernel(_square4x2, 3 * VSIZE / OCL_VSIZE * _n / 8); }
	void square8() { _executeKernel(_square8, 3 * VSIZE / OCL_VSIZE * _n / 8); }

	void forward8p(const int lm, const size_t s)
	{
		set_transform_arg0(_forward8, false);
		forward8(lm, s);
		set_transform_arg0(_forward8);
	}

	void forward8_0p()
	{
		set_transform_arg0(_forward8_0, false);
		forward8_0();
		set_transform_arg0(_forward8_0);
	}

	void fwd4x2() { _executeKernel(_fwd4x2, 3 * VSIZE / OCL_VSIZE * _n / 8); }
	void fwd8() { _executeKernel(_fwd8, 3 * VSIZE / OCL_VSIZE * _n / 8); }

	void mul2x4() { _executeKernel(_mul2x4, 3 * VSIZE / OCL_VSIZE * _n / 8); }
	void mul4x2() { _executeKernel(_mul4x2, 3 * VSIZE / OCL_VSIZE * _n / 8); }
	void mul8() { _executeKernel(_mul8, 3 * VSIZE / OCL_VSIZE * _n / 8); }

	void mul2x4_mask(const uint32_t mask)
	{
		const uint32 imask = uint32(mask);
		_setKernelArg(_mul2x4_mask, 3, sizeof(uint32), &imask);
		_executeKernel(_mul2x4_mask, 3 * VSIZE * _n / 8);
	}
	void mul4x2_mask(const uint32_t mask)
	{
		const uint32 imask = uint32(mask);
		_setKernelArg(_mul4x2_mask, 3, sizeof(uint32), &imask);
		_executeKernel(_mul4x2_mask, 3 * VSIZE * _n / 8);
	}
	void mul8_mask(const uint32_t mask)
	{
		const uint32 imask = uint32(mask);
		_setKernelArg(_mul8_mask, 3, sizeof(uint32), &imask);
		_executeKernel(_mul8_mask, 3 * VSIZE * _n / 8);
	}

	void carry(const uint32_t dup)
	{
		const uint32 idup = uint32(dup);
		_setKernelArg(_carry1, 4, sizeof(uint32), &idup);
		_executeKernel(_carry1, VSIZE / OCL_CARRY_VSIZE * _n / CARRY_LENGTH, get_carry_workgroup_size());
		_executeKernel(_carry2, (VSIZE * _n / CARRY_LENGTH) >> _carry_shift);
	}

	void set(const uint32_t a)
	{
		const uint32 ia = uint32(a);
		_setKernelArg(_set, 1, sizeof(uint32), &ia);
		_executeKernel(_set, 3 * VSIZE / OCL_VSIZE * _n);
	}

	void copy(const size_t dst, const size_t src)
	{
		const uint32 idst = uint32(dst), isrc = uint32(src);
		_setKernelArg(_copy, 1, sizeof(uint32), &idst);
		_setKernelArg(_copy, 2, sizeof(uint32), &isrc);
		_executeKernel(_copy, 3 * VSIZE / OCL_VSIZE * _n);
	}

	void copyp(const size_t src)
	{
		const uint32 isrc = uint32(src);
		_setKernelArg(_copyp, 2, sizeof(uint32), &isrc);
		_executeKernel(_copyp, 3 * VSIZE / OCL_VSIZE * _n);
	}

	void copy_mask(const size_t dst, const size_t src, const uint32_t mask)
	{
		const uint32 idst = uint32(dst), isrc = uint32(src), imask = uint32(mask);
		_setKernelArg(_copy_mask, 1, sizeof(uint32), &idst);
		_setKernelArg(_copy_mask, 2, sizeof(uint32), &isrc);
		_setKernelArg(_copy_mask, 3, sizeof(uint32), &imask);
		_executeKernel(_copy_mask, 3 * VSIZE * _n);
	}

	void cosmic_ray() { _executeKernel(_cosmic_ray, 3 * VSIZE * _n); }
};
