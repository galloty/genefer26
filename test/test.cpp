/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, char * argv[])
{
	const int n = 13;
	const size_t size = 32;

	std::ifstream file_in("GFN13.txt");
	if (!file_in.is_open()) { std::cerr << "Cannot open input file." << std::endl; return EXIT_FAILURE; }

	bool eof = false;
	while (!eof)
	{
		size_t i = 0;
		std::string line;
		uint32_t b[size];
		bool eof = true;
		while (std::getline(file_in, line))
		{
			b[i] = uint32_t(std::stoi(line));
			++i; if (i == size) { eof = false; break; }
		}
		while (i < size) { b[i] = b[i - 1]; ++i; }

		std::ofstream file_out("b.txt");
		for (size_t i = 0; i < size; ++i) file_out << b[i] << std::endl;
		file_out.close();

		std::system("geneferv -n 13 -b b.txt -q");
	}

	file_in.close();

	return EXIT_SUCCESS;
}
