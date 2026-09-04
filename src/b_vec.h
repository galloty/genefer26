/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "vint.h"
#include "pio.h"

class b_vec
{
private:
	const size_t _size;
	UInt32_8 _b[4];

public:
	explicit b_vec(const size_t size) : _size(size) {}
	virtual ~b_vec() {}

	size_t get_size() const { return _size; }

	const UInt32_8 & operator[](const size_t i) const { return _b[i]; }
	UInt32_8 & operator[](const size_t i) { return _b[i]; }

	void init(const uint32_t b_max, const uint32_t step_min, const uint32_t step_max)
	{
		uint32_t b = b_max;
		for (size_t j = 0, size = _size; j < size; ++j)
		{
			uint32_t nb[8];
			for (size_t i = 0; i < 8; ++i)
			{
				nb[7 - i] = b;
				if (step_min == step_max) b -= step_min;
				else b -= uint32_t(std::rand()) % (step_max - step_min) + step_min;
				b &= ~1u;	// even
			}
			_b[j] = UInt32_8(nb);
		}
	}

	void init(const std::string & b_filename)
	{
		std::ifstream file(b_filename);
		if (!file.is_open()) pio::error("cannot open input file", true);

		for (size_t j = 0, size = _size; j < size; ++j)
		{
			size_t i = 0;
			std::string line;
			uint32_t nb[8];
			while (std::getline(file, line))
			{
				const uint32_t b_i = uint32_t(std::stoi(line));
				if (b_i % 2 != 0) pio::error("b must be even", true);
				if (b_i < 10000) pio::error("b < 10000 is not supported", true);
				if (b_i > 2000000000) pio::error("b > 2000000000 is not supported", true);
				if ((b_i == 0) || ((b_i & (~b_i + 1)) == b_i)) pio::error("b must not be a power of two", true);
				nb[i] = b_i;
				++i; if (i == 8) break;
			}
			while (i < 8) { nb[i] = nb[i - 1]; ++i; }
			_b[j] = UInt32_8(nb);
		}

		file.close();
	}

	uint32_t min() const
	{
		const UInt32_8 * const b = _b;
		uint32_t b_min = b[0].min();
		for (size_t j = 1, size = _size; j < size; ++j) b_min = std::min(b_min, b[j].min());
		return b_min;
	}

	uint32_t max() const
	{
		const UInt32_8 * const b = _b;
		uint32_t b_max = b[0].max();
		for (size_t j = 1, size = _size; j < size; ++j) b_max = std::max(b_max, b[j].max());
		return b_max;
	}

	uint32_t get_bit_mask(const int i) const
	{
		const UInt32_8 * const b = _b;
		uint32_t mask = 0;
		for (size_t j = 0, size = _size; j < size; ++j) mask |= uint32_t(b[j].get_bit_mask(i)) << (8 * j);
		return mask;
	}
};
