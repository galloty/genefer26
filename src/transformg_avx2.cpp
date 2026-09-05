/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define arch_g_namespace	arch_g_avx2_namespace

#include "transformGPU.h"

// Defined in transformg_sse2
extern void get_opencl_ids(int argc, char * argv[], cl_device_id & boinc_device_id, cl_platform_id & boinc_platform_id);

#define _create_ocl_avx2(SIZE) \
template<> \
transform<SIZE> * transform<SIZE>::create_ocl_avx2(const b_vec<SIZE / 8> & b, const int n, const size_t num_regs, const size_t device, \
											const bool is_boinc, const bool get_boinc_ids, int _boinc_argc, char ** _boinc_argv) \
{ \
	cl_platform_id boinc_platform_id = 0; \
	cl_device_id boinc_device_id = 0; \
	if (get_boinc_ids) get_opencl_ids(_boinc_argc, _boinc_argv, boinc_device_id, boinc_platform_id); \
 \
	transform<SIZE> * ptransform = nullptr; \
	const uint32_t b_max = b.max(); \
 \
	if (b_max <= 1000000000) \
	{ \
		ptransform = arch_g_avx2_namespace::create_transformGPU<SIZE, false>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id); \
	} \
	else \
	{ \
		ptransform = arch_g_avx2_namespace::create_transformGPU<SIZE, true>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id); \
	} \
	ptransform->set_type("AVX2"); \
	return ptransform; \
}

template<size_t VSIZE>
transform<VSIZE> * transform<VSIZE>::create_ocl_avx2(const b_vec<VSIZE / 8> & b, const int n, const size_t num_regs, const size_t device,
											const bool is_boinc, const bool get_boinc_ids, int _boinc_argc, char ** _boinc_argv)
{
	return nullptr;
}

#ifndef NO8
_create_ocl_avx2(8)
#endif
#ifndef NO16
_create_ocl_avx2(16)
#endif
#ifndef NO32
_create_ocl_avx2(32)
#endif
