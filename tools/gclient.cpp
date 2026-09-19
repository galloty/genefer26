/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

// #include <cstdint>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <thread>
// #include <fstream>
// #include <sstream>


#if defined(_WIN64)
#include <Windows.h>
#else
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

#include "boinc_api.h"
#include "version.h"

struct io
{
	static void print(const std::string & str)
	{
		std::cerr << str << std::flush;
	}

	static void error(const std::string & str, const bool fatal = false)
	{
		if (fatal) std::cerr << std::endl;
		std::cerr << "Error: " << str << "." << std::endl << std::flush;
		if (fatal)
		{
			// delay five minutes before reporting to the host in order to slow down the error rate.
			std::this_thread::sleep_for(std::chrono::minutes(5));
			boinc_finish(EXIT_FAILURE);
		}
	}
};

class application
{
private:
	struct deleter { void operator()(const application * const p) { delete p; } };

private:
	static void quit(int)
	{
		boinc_finish(EXIT_SUCCESS);
	}

private:
#if defined(_WIN64)
	static BOOL WINAPI HandlerRoutine(DWORD)
	{
		quit(1);
		return TRUE;
	}
#endif

public:
	application()
	{
#if defined(_WIN64)	
		SetConsoleCtrlHandler(HandlerRoutine, TRUE);
#else
		signal(SIGTERM, quit);
		signal(SIGINT, quit);
#endif
	}

	virtual ~application() {}

	static application & get_instance()
	{
		static std::unique_ptr<application, deleter> p_instance(new application());
		return *p_instance;
	}

private:
	enum class EMode { None, Proof, Check };

private:
	static std::string header(const std::vector<std::string> & args, const bool nl = false)
	{
		const char * const sysver =
#if defined(_WIN64)
#if defined(__aarch64__)
			"win arm64";
#else
			"win x64";
#endif
#elif defined(__linux__)
#if defined(__aarch64__)
			"linux arm64";
#else
			"linux x64";
#endif
#elif defined(__APPLE__)
#if defined(__aarch64__)
			"macOS arm64";
#else
			"macOS x64";
#endif
#else
			"unknown";
#endif

		std::ostringstream ssc;
#if defined(__clang__)
		ssc << ", clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
		ssc << ", gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif

		ssc << ", boinc-" << BOINC_VERSION_STRING;

		std::ostringstream ss;
		ss << "genefer" << " version 26.09.0 (" << sysver << ssc.str() << ")" << std::endl;
		ss << "Copyright (c) 2022, Yves Gallot" << std::endl;
		ss << "genefer" << " is free source code, under the MIT license." << std::endl;
		if (nl)
		{
			ss << std::endl << "Command line: '";
			bool first = true;
			for (const std::string & arg : args)
			{
				if (first) first = false; else ss << " ";
				ss << arg;
			}
			ss << "'" << std::endl << std::endl;
		}
		return ss.str();
	}

public:
	void run(int argc, char * argv[])
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);

		BOINC_OPTIONS boinc_options;
		boinc_options_defaults(boinc_options);
		boinc_options.direct_process_action = 0;
		const int retval = boinc_init_options(&boinc_options);
		if (retval != 0)
		{
			std::ostringstream ss; ss << "boinc_init returned " << retval;
			throw std::runtime_error(ss.str());
		}

		// if -v or -V then print header and exit
		for (const std::string & arg : args)
		{
			if ((arg[0] == '-') && ((arg[1] == 'v') || (arg[1] == 'V')))
			{
				io::print(header(args));
				boinc_finish(EXIT_SUCCESS);
				return;
			}
		}

		io::print(header(args, true));

		uint32_t b = 0, n = 0;
		EMode mode = EMode::None;
		std::string main_filename;

		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if ((arg.substr(0, 2) == "-b") && (arg.substr(0, 3) != "-bo"))
			{
				const std::string bstr = ((arg == "-b") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				b = static_cast<uint32_t>(std::atoi(bstr.c_str()));
			}
			if (arg.substr(0, 2) == "-n")
			{
				const std::string nstr = ((arg == "-n") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				n = static_cast<uint32_t>(std::atoi(nstr.c_str()));
			}
			if (arg.substr(0, 2) == "-p")
			{
				mode = EMode::Proof;
			}
			if (arg.substr(0, 2) == "-c")
			{
				mode = EMode::Check;
			}
			if (arg.substr(0, 2) == "-f")
			{
				main_filename = ((arg == "-f") && (i + 1 < size)) ? args[++i] : arg.substr(2);
			}
		}

		std::ostringstream ssg;
		if (mode == EMode::Proof)
		{
			ssg << "Running on device 'NVIDIA GeForce RTX 4060', vendor 'NVIDIA Corporation', version 'OpenCL 3.0 CUDA', driver '616.92', data size: 1.12 MB." << std::endl;
		}
		else if (mode == EMode::Check)
		{
			ssg << "Using fma implementation, 1 thread(s), data size: 1.2 MB." << std::endl;
		}
		io::print(ssg.str());

		boinc_fraction_done(0.1);

		std::ostringstream sse;
		if (mode == EMode::Proof)
		{
			sse << "C:\\genefer\\genefer22g.exe -p -n " << n << " -b " << b << " -f gproof";
		}
		else if (mode == EMode::Check)
		{
			sse << "C:\\genefer\\genefer22.exe -c -n " << n << " -b " << b << " -f gproof";
		}
		std::system(sse.str().c_str());

		std::ostringstream ssr;
		if (mode == EMode::Proof)
		{
			ssr << b << "^{2^" << n << "} + 1: proof file is generated, time = 00:01:00." << std::endl;
		}
		else if (mode == EMode::Check)
		{
			char path_res[512];
			boinc_resolve_filename("results.txt", path_res, sizeof(path_res));
			std::ifstream file_res(path_res);
			std::string line;
			if (file_res.is_open())
			{
				std::getline(file_res, line);
				file_res.close();
			}
			const std::string::size_type pos = line.find("ckey =");
			const std::string ckey = (pos != std::string::npos) ? line.substr(pos + 7, 16) : line;
			ssr << b << "^{2^" << n << "} + 1 is checked, ckey = " << ckey << ", time = 00:00:01." << std::endl;
		}
		io::print(ssr.str());

	// const int depth = 7, m = 1 << 17;
	// int B_PL_prev = 0, b_prev = 0;
	// for (int b = 500000000; b <= 510000000; b += 2)
	// {
	// 	const size_t esize = size_t(m * log2(b)) + 1;
	// 	const int B_PL = static_cast<int>((esize - 1) >> depth) + 1;
	// 	if (B_PL != B_PL_prev)
	// 	{
	// 		if (B_PL_prev != 0) std::cout << b << ": " << B_PL << ", " << b - b_prev << std::endl;
	// 		B_PL_prev = B_PL; b_prev = b;
	// 	}
	// }

		boinc_finish(EXIT_SUCCESS);
	}
};

int main(int argc, char * argv[])
{
	std::setvbuf(stderr, nullptr, _IONBF, 0);	// no buffer

	try
	{
		application & app = application::get_instance();
		app.run(argc, argv);
	}
	catch (const std::runtime_error & e)
	{
		io::error(e.what(), true);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
