/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define arch_g_namespace	arch_g_avx512_namespace

#include "transformGPU.h"

#if defined(BOINC)
// Defined in transformg_sse2
extern void get_opencl_ids(int argc, char * argv[], cl_device_id & boinc_device_id, cl_platform_id & boinc_platform_id);
#endif

transform * transform::create_ocl_avx512(const UInt32_8 & b, const int n, const size_t num_regs, const size_t device,
								  const bool is_boinc, const bool get_boinc_ids, int _boinc_argc, char ** _boinc_argv)
{
	cl_platform_id boinc_platform_id = 0;
	cl_device_id boinc_device_id = 0;
#if defined(BOINC)
	if (get_boinc_ids) get_opencl_ids(_boinc_argc, _boinc_argv, boinc_device_id, boinc_platform_id);
#else
	(void)get_boinc_ids;
	(void)_boinc_argc;
	(void)_boinc_argv;
#endif

	transform * ptransform = nullptr;
	const uint32_t b_max = b.max();

	if (b_max <= 1000000000)
	{
		ptransform = new arch_g_avx512_namespace::transformGPU<8, 1, false>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id);
	}
	else
	{
		ptransform = new arch_g_avx512_namespace::transformGPU<8, 1, true>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id);
	}
	ptransform->set_type("AVX-512");
	return ptransform;
}
