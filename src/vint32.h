/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#define VSIZE	8
typedef uint32_t	vint32 __attribute__ ((vector_size(VSIZE * sizeof(uint32_t))));

inline bool is_equal(const vint32 & x, const vint32 & y)
{
	// TODO AVX-512, AVX2, ...
	bool b = true;
	for (size_t j = 0; j < VSIZE; ++j) b &= (x[j] == y[j]);
	return b;
}