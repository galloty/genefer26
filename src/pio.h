/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <thread>
#include <chrono>

#include "boinc.h"

class pio
{
private:
	struct deleter { void operator()(const pio * const p) { delete p; } };

public:
	pio() {}
	virtual ~pio() {}

	static pio & get_instance()
	{
		static std::unique_ptr<pio, deleter> p_instance(new pio());
		return *p_instance;
	}

public:
	void set_boinc(const bool is_boinc) { _is_boinc = is_boinc; }

private:
	bool _is_boinc = false;

private:
	// print: console: stdout, boinc: stderr
	void _print(const std::string & str) const
	{
		if (_is_boinc) { std::cerr << str << std::flush; }
		else { std::cout << str; }
	}

private:
	// display: console: stdout, boinc: -
	void _display(const std::string & str) const
	{
		if (!_is_boinc) { std::cout << str << std::flush; }
	}

private:
	// error: normal: stderr, boinc: stderr
	void _error(const std::string & str, const bool fatal) const
	{
		if (fatal) std::cerr << std::endl;
		std::cerr << "Error: " << str << "." << std::endl << std::flush;
		if (fatal)
		{
			if (_is_boinc)
			{
				// delay five minutes before reporting to the host in order to slow down the error rate.
				std::this_thread::sleep_for(std::chrono::minutes(5));
				boinc_finish(EXIT_FAILURE);
			}
			else
			{
				exit(EXIT_FAILURE);
			}
		}
	}

private:
	// result: normal: 'results.txt' file
	bool _result(const std::string & str, const std::string & filename) const
	{
		const char * const file_name = filename.empty() ? "results.txt" : filename.c_str();
		if (_is_boinc)
		{
			FILE * const out_file = _open(file_name, "a");
			if (out_file == nullptr) throw std::runtime_error("Cannot write 'results.txt' file");
			std::fprintf(out_file, "%s", str.c_str());
			std::fclose(out_file);
			return true;
		}
		std::ofstream resFile(file_name, std::ios::app);
		if (!resFile.is_open()) return false;
		resFile << str;
		resFile.close();
		return true;
	}

private:
	FILE * _open(const char * const filename, const char * const mode) const
	{
		if (_is_boinc)
		{
			char path[512];
			boinc_resolve_filename(filename, path, sizeof(path));
			return boinc_fopen(path, mode);
		}
		return std::fopen(filename, mode);
	}

public:
	static void print(const std::string & str) { get_instance()._print(str); }
	static void display(const std::string & str) { get_instance()._display(str); }
	static void error(const std::string & str, const bool fatal = false) { get_instance()._error(str, fatal); }
	static bool result(const std::string & str, const std::string & filename = "") { return get_instance()._result(str, filename); }

	static FILE * open(const char * const filename, const char * const mode) { return get_instance()._open(filename, mode); }
};
