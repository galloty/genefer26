/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define arch_namespace	arch_512_namespace

#include "transformCPU.h"

#define _create_avx512(SIZE) \
template<> \
transform<SIZE> * transform<SIZE>::create_avx512(const b_vec<SIZE / 8> & b, const int n, const size_t num_regs) \
{ \
	transform<SIZE> * ptransform = arch_512_namespace::create_transformCPU<SIZE>(b, n, num_regs); \
	ptransform->set_type("AVX-512"); \
	return ptransform; \
}

template<size_t VSIZE>
transform<VSIZE> * transform<VSIZE>::create_avx512(const b_vec<VSIZE / 8> & b, const int n, const size_t num_regs)
{
	return nullptr;
}

#ifndef NO8
_create_avx512(8)
#endif
#ifndef NO16
_create_avx512(16)
#endif
#ifndef NO32
_create_avx512(32)
#endif
