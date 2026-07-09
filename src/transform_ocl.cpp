/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformGPU.h"

size_t transform::display_devices()
{
	platform pfm;
	return pfm.displayDevices();
}

transform * transform::create_ocl(const vint32 & b, const uint32_t n, const size_t num_regs, const size_t device,
								  const bool is_boinc, const bool get_boinc_ids)
{
	cl_platform_id boinc_platform_id = 0;
	cl_device_id boinc_device_id = 0;
#if defined(BOINC)
	if (get_boinc_ids)
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
#else
	(void)get_boinc_ids;
#endif

	transform * ptransform = nullptr;
	uint32_t b_max = 0; for (size_t i = 0; i < VSIZE; ++i) b_max = std::max(b_max, b[i]);

	if (b_max <= 1000000000)
	{
		ptransform = new transformGPU<false>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id);
	}
	else
	{
		ptransform = new transformGPU<true>(b, n, num_regs, device, is_boinc, boinc_platform_id, boinc_device_id);
	}
	return ptransform;
}
