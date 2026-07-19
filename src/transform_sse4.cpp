/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define arch_namespace	arch_sse4_namespace

#include "transformCPU.h"

transform * transform::create_sse4(const vuint32 & b, const uint32_t n, const size_t num_regs)
{
	transform * ptransform = arch_sse4_namespace::create_transformCPU(b, n, num_regs);
	ptransform->set_type("SSE4.1");
	return ptransform;
}
