/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

struct timer
{
	typedef std::chrono::high_resolution_clock::time_point time;

	static time current_time()
	{
		return std::chrono::high_resolution_clock::now();
	}

	static double diff_time(const time & end, const time & start)
	{
		return std::chrono::duration<double>(end - start).count();
	}

	static std::string format_time(const double time)
	{
		uint64_t seconds = static_cast<uint64_t>(time), minutes = seconds / 60, hours = minutes / 60;
		seconds -= minutes * 60; minutes -= hours * 60;

		std::stringstream ss;
		ss << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
		return ss.str();
	}
};

class watch
{
private:
	const double _elapsed_time;
	const timer::time _start_time;
	timer::time _current_time;
	timer::time _display_start_time;
	timer::time _record_start_time;

public:
	watch(const double restoredTime = 0) : _elapsed_time(restoredTime), _start_time(timer::current_time())
	{
		_current_time = _display_start_time = _record_start_time = _start_time;
	}

	virtual ~watch() {}

	void read() { _current_time = timer::current_time(); }

	double get_elapsed_time() { read(); return _elapsed_time + timer::diff_time(_current_time, _start_time); }

	double get_display_time() const { return timer::diff_time(_current_time, _display_start_time); }
	double get_record_time() const { return timer::diff_time(_current_time, _record_start_time); }

	void reset_display_time() { _display_start_time = _current_time; }
	void reset_record_time() { _record_start_time = _current_time; }
};
