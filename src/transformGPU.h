/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "ocl.h"

#include "transformGMP.h"

template<bool is32>
class transformGPU : public transformGMP
{
private:
	device * _pdevice;

public:
	transformGPU(const uint32_t b, const uint32_t n, const size_t num_regs, const size_t d,	// device
				 const bool is_boinc, const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id)
				: transformGMP(b, n, num_regs)
	{
		const bool is_boinc_platform = is_boinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pdevice = new device(eng_platform, is_boinc_platform ? 0 : d);
		set_type(_pdevice->getType());
	}

	virtual ~transformGPU()
	{
		delete _pdevice;
	}
};
