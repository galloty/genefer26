/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(_WIN64)
#include <Windows.h>
#else
#include <signal.h>
// #include <sys/time.h>
// #include <sys/resource.h>
#endif

#if defined(BOINC)
#include "version.h"
#endif

#include "genefer.h"

class application
{
private:
	struct deleter { void operator()(const application * const p) { delete p; } };

private:
	static void quit(int)
	{
		genefer::get_instance().quit();
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

#if defined(BOINC)
		ssc << ", boinc-" << BOINC_VERSION_STRING;
#endif

		std::ostringstream ss;
		ss << "geneferv version 26.06.0 (" << sysver << ssc.str() << ")" << std::endl;
		ss << "Copyright (c) 2026, Yves Gallot" << std::endl;
		ss << "genefer is free source code, under the MIT license." << std::endl;
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

private:
	static std::string usage()
	{
		std::ostringstream ss;
		ss << "Usage: geneferv [options]  options may be specified in any order" << std::endl;
		ss << "  -n <n>                   exponent of the GFN (5 <= n <= 17)" << std::endl;
		ss << "  -b <b>                   base of the GFN (6 <= b <= 2000000000)" << std::endl;
		ss << "  -q                       quick test" << std::endl;
		ss << "  -p                       full test: a proof is generated" << std::endl;
		ss << "  -s                       server job: convert the proof into a certificate and a 64-bit key" << std::endl;
		ss << "  -c                       check certificate: a 64-bit key is generated (must be identical to server key)" << std::endl;
		ss << "  -h                       validate and bench your hardware" << std::endl;
		ss << "  -d <n> or --device <n>   set the OpenCL device number (default 0)" << std::endl;
		ss << "  -cpu                     execute the implementation for CPU" << std::endl;
		ss << "  -f <filename>            main filename (without extension) of input and output files" << std::endl;
		ss << "  -v or -V                 print the startup banner and exit" << std::endl;
#if defined(BOINC)
		ss << "  -boinc                   operate as a BOINC client app" << std::endl;
#endif
		ss << std::endl;
		return ss.str();
	}

public:
	void run(int argc, char * argv[])
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);

		bool b_boinc = false;
#if defined(BOINC)
		for (const std::string & arg : args) if (arg == "-boinc") b_boinc = true;
#endif
		pio::get_instance().set_boinc(b_boinc);

		if (b_boinc)
		{
			BOINC_OPTIONS boinc_options;
			boinc_options_defaults(boinc_options);
			boinc_options.direct_process_action = 0;
			boinc_options.normal_thread_priority = 1;
			const int retval = boinc_init_options(&boinc_options);
			if (retval != 0)
			{
				std::ostringstream ss; ss << "boinc_init returned " << retval;
				throw std::runtime_error(ss.str());
			}
		}

		// if -v or -V then print header and exit
		for (const std::string & arg : args)
		{
			if ((arg[0] == '-') && ((arg[1] == 'v') || (arg[1] == 'V')))
			{
				pio::print(header(args));
				if (b_boinc) boinc_finish(EXIT_SUCCESS);
				return;
			}
		}

		pio::print(header(args, true));

		uint32_t b = 0, n = 0;
		genefer::EMode mode = genefer::EMode::None;
		size_t device = 0;
		bool isCPU = false;
#if defined(BOINC)
		bool ext_device = false;
#endif
		std::string main_filename = "";

		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg.substr(0, 2) == "-n")
			{
				const std::string nstr = ((arg == "-n") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				n = static_cast<uint32_t>(std::atoi(nstr.c_str()));
				if (n < 5) throw std::runtime_error("n < 5 is not supported");
				if (n > 17) throw std::runtime_error("n > 17 is not supported");
			}
			if ((arg.substr(0, 2) == "-b") && (arg.substr(0, 3) != "-bo"))
			{
				const std::string bstr = ((arg == "-b") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				b = static_cast<uint32_t>(std::atoi(bstr.c_str()));
				if (b % 2 != 0) throw std::runtime_error("b must be even");
				if (b < 6) throw std::runtime_error("b < 6 is not supported");
				if (b > 2000000000) throw std::runtime_error("b > 2000000000 is not supported");
				if ((b == 0) || ((b & (~b + 1)) == b)) throw std::runtime_error("b must not be a power of two");
			}
			if (arg.substr(0, 2) == "-q")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-q used with an incompatible option (-p, -s, -c, -e, -h)");
				mode = genefer::EMode::Quick;
			}
			if (arg.substr(0, 2) == "-p")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-p used with an incompatible option (-q, -s, -c, -e, -h)");
				mode = genefer::EMode::Proof;
			}
			if (arg.substr(0, 2) == "-s")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-s used with an incompatible option (-q, -p, -c, -e, -h)");
				mode = genefer::EMode::Server;
			}
			if ((arg.substr(0, 2) == "-c")  && (arg.substr(0, 3) != "-cp"))
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-c used with an incompatible option (-q, -p, -s, -e, -h)");
				mode = genefer::EMode::Check;
			}
			if (arg.substr(0, 2) == "-h")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-h used with an incompatible option (-q, -p, -s, -e, -c)");
				mode = genefer::EMode::Bench;
			}
			if (arg.substr(0, 2) == "-d")
			{
				const std::string dstr = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = size_t(std::atoi(dstr.c_str()));
#if defined(BOINC)
				ext_device = true;
#endif
			}
			if (arg.substr(0, 8) == "--device")
			{
				const std::string dstr = ((arg == "--device") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = size_t(std::atoi(dstr.c_str()));
#if defined(BOINC)
				ext_device = true;
#endif
			}
			if (arg.substr(0, 2) == "-f")
			{
				main_filename = ((arg == "-f") && (i + 1 < size)) ? args[++i] : arg.substr(2);
			}
			if (arg.substr(0, 4) == "-cpu")
			{
				isCPU = true;
			}
		}

		genefer & g = genefer::get_instance();
		g.set_boinc(b_boinc);
#if defined(BOINC)
		g.set_boinc_param(b_boinc && !boinc_is_standalone() && !ext_device);
#endif
		g.set_filename(main_filename);

		if (mode == genefer::EMode::Bench)
		{
			const bool success = 
#if defined(_WIN64)
			(SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) != 0);
#else
			(setpriority(PRIO_PROCESS, 0, -20) == 0);
#endif
			std::ostringstream ss; ss << "Set high priority";
			if (!success) ss << ": failed (permission denied)";
			ss << "." << std::endl;
			pio::print(ss.str());

			for (size_t n = 5; n <= 17; ++n)
			{
				if (g.check(0, n, mode, device, isCPU) != genefer::EReturn::Success) return;
			}
			return;
		}

		if ((mode == genefer::EMode::None) || (b == 0) || (n == 0))
		{
			// internal test
			// if (g.check(278, 8, genefer::EMode::Quick, device, true, 5) != genefer::EReturn::Success) return;
			// return;

			pio::print(usage());
			if (genefer::display_devices() == 0) throw std::runtime_error("No OpenCL device");
			return;
		}

		const genefer::EReturn ret = g.check(b, n, mode, device, isCPU);
		if (b_boinc)
		{
			if (ret == genefer::EReturn::Success) boinc_finish(BOINC_SUCCESS);
			if (ret == genefer::EReturn::Failed) boinc_finish(EXIT_CHILD_FAILED);
		}

		if (ret == genefer::EReturn::Aborted)
		{
			std::ostringstream ss; ss << std::endl;
			pio::print(ss.str());
		}
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
		pio::error(e.what(), true);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
