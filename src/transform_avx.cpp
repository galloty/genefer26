/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define arch_namespace	arch_avx_namespace

#include "transformCPU.h"

#define _create_avx(SIZE) \
template<> \
transform<SIZE> * transform<SIZE>::create_avx(const b_vec & b, const int n, const size_t num_regs) \
{ \
	transform<SIZE> * ptransform = arch_avx_namespace::create_transformCPU<SIZE>(b, n, num_regs); \
	ptransform->set_type("AVX"); \
	return ptransform; \
}

template<size_t VSIZE>
transform<VSIZE> * transform<VSIZE>::create_avx(const b_vec & b, const int n, const size_t num_regs)
{
	return nullptr;
}

_create_avx(8)
_create_avx(16)
_create_avx(32)
