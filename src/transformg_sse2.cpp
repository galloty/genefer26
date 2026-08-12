/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define arch_g_namespace	arch_g_sse2_namespace

#include "transformGPU.h"

// OpenCL headers are required
size_t transform::display_devices() { platform pfm; return pfm.displayDevices(); }

#if defined(BOINC)

#include "boinc_opencl.h"

void get_opencl_ids(int argc, char * argv[], cl_device_id & boinc_device_id, cl_platform_id & boinc_platform_id)
{
	const int err = boinc_get_opencl_ids(argc, argv, 0, &boinc_device_id, &boinc_platform_id);
	if ((err != 0) || (boinc_device_id == 0) || (boinc_platform_id == 0))
	{
		std::ostringstream ss; ss << "boinc_get_opencl_ids() failed, err = " << err;
		pio::error(ss.str());
		// continue using default OpenCL device
		boinc_device_id = 0; boinc_platform_id = 0;
	}
}
#endif

transform * transform::create_ocl_sse2(const UInt32_8 & b, const int n, const size_t num_regs, const size_t device,
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
		ptransform = new arch_g_sse2_namespace::transformGPU<8, 1, false>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id);
	}
	else
	{
		ptransform = new arch_g_sse2_namespace::transformGPU<8, 1, true>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id);
	}
	ptransform->set_type("SSE2");
	return ptransform;
}
