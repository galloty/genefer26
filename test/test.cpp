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
	if (argc != 4) { std::cerr << "Usage: test <n> <size> file" << std::endl; return EXIT_FAILURE; }

	const int n = atoi(argv[1]);
	const size_t size = size_t(atoi(argv[2]));

	std::ifstream file_in(argv[3]);
	if (!file_in.is_open()) { std::cerr << "Cannot open input file." << std::endl; return EXIT_FAILURE; }

	bool eof = false;
	while (!eof)
	{
		size_t i = 0;
		std::string line;
		uint32_t b[size];
		eof = true;
		while (std::getline(file_in, line))
		{
			b[i] = uint32_t(std::stoi(line));
			++i; if (i == size) { eof = false; break; }
		}
		while (i < size) { b[i] = b[i - 1]; ++i; }

		std::ofstream file_out("b.txt");
		for (size_t i = 0; i < size; ++i) file_out << b[i] << std::endl;
		file_out.close();

		std::ostringstream ss; ss << "geneferv -n " << n << " -b b.txt -l " << size;
		std::ostringstream ss1; ss1 << ss.str() << " -p";
		std::ostringstream ss2; ss2 << ss.str() << " -s -cpu";
		std::ostringstream ss3; ss3 << ss.str() << " -c";
		std::system(ss1.str().c_str());
		std::system(ss2.str().c_str());
		std::system(ss3.str().c_str());

		std::ostringstream ss_gname; ss_gname << "g" << n << "_" << b[0] << "_" << b[size - 1];
		std::ostringstream ss_proof; ss_proof << ss_gname.str() << ".proof";
		std::ostringstream ss_cert; ss_cert << ss_gname.str() << ".cert";
		std::remove(ss_proof.str().c_str());
		std::remove(ss_cert.str().c_str());

		std::string pkey[size], ckey[size];

		std::ifstream file_res("results.txt");
		for (size_t i = 0; i < size; ++i)
		{
			std::string line;
			std::getline(file_res, line);
			if (line.find("is a probable prime") == std::string::npos) { std::cerr << b[i] << "is not prime." << std::endl; return EXIT_FAILURE; }
			pkey[i] = line.substr(line.find("pkey =") + 7, 16);
		}

		for (size_t i = 0; i < size; ++i)
		{
			std::string line;
			std::getline(file_res, line);
			if (line.find("is a probable prime") == std::string::npos) { std::cerr << b[i] << "is not prime." << std::endl; return EXIT_FAILURE; }
			if (line.substr(line.find("pkey =") + 7, 16) != pkey[i]) { std::cerr << b[i] << ", pkey failed." << std::endl; return EXIT_FAILURE; }
			ckey[i] = line.substr(line.find("ckey =") + 7, 16);
		}

		for (size_t i = 0; i < size; ++i)
		{
			std::string line;
			std::getline(file_res, line);
			if (line.substr(line.find("ckey =") + 7, 16) != ckey[i]) { std::cerr << b[i] << ", ckey failed." << std::endl; return EXIT_FAILURE; }
		}

		file_res.close();
		std::remove("results.txt");
	}

	file_in.close();

	return EXIT_SUCCESS;
}
