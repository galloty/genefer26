/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define transformCPU_namespace	transformCPU_AVX10
#include "transformCPU.h"

transform * transform::create_avx10(const uint32_t b, const uint32_t n, const size_t num_regs)
{
	transform * ptransform = transformCPU_AVX10::create_transformCPU(b, n, num_regs);
	ptransform->set_type("AVX10.2");
	return ptransform;
}

