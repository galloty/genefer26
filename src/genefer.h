/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>
#include <ctime>
#include <sys/stat.h>

#include "vint.h"
#include "mpz_vec.h"
#include "pio.h"
#include "transform.h"
#include "file.h"
#include "timer.h"

inline int ilog2_32(const uint32_t n) { return (n == 0) ? -1 : (31 - __builtin_clz(n)); }

class genefer
{
public:
	enum class EReturn { Success, Failed, Aborted };
	enum class EMode { None, Quick, Proof, Server, Check, Bench };

private:
	struct deleter { void operator()(const genefer * const p) { delete p; } };

public:
	genefer() {}
	virtual ~genefer() {}

	static genefer & get_instance()
	{
		static std::unique_ptr<genefer, deleter> pInstance(new genefer());
		return *pInstance;
	}

private:
	std::atomic_bool _quit = false;
	bool _is_boinc = false;
	bool _get_boinc_ids = false;
	transform * _transform = nullptr;
	gint * _gi = nullptr;
	std::string _main_filename;
	uint32_t _n = 0;
	int _print_range = 0, _print_i = 0;
	bool _print_sr = true;

	using mpzv = mpz_vec<VSIZE>;

protected:
	bool quitting() const
	{
		const bool b = _quit.load(std::memory_order_relaxed);
		if (b) std::atomic_thread_fence(std::memory_order_acquire);
		return b;
	}

public:
	void quit()
	{
		std::atomic_thread_fence(std::memory_order_release);
		_quit.store(true, std::memory_order_relaxed);
	}

	void set_boinc(const bool is_boinc) { _is_boinc = is_boinc; }
	void set_boinc_param(const bool get_boinc_ids) { _get_boinc_ids = get_boinc_ids; }

	void set_filename(const std::string & main_filename) { _main_filename = main_filename; }

	static size_t display_devices() { return transform::display_devices(); }

private:
	void create_transform_GPU(const vuint32 & b, const uint32_t n, const size_t num_regs, const size_t device,
							const bool verbose = true, const bool full = true)
	{
		delete_transform();

		std::string ttype;
		_transform = transform::create_gpu(b, n, num_regs, device, _is_boinc, _get_boinc_ids);
		if (verbose)
		{
			std::ostringstream ss;
			if (full) ss << "Running on " << _transform->get_type() << ", data size: " << std::setprecision(3) << _transform->get_cache_size() / (1024 * 1024.0) << " MB";
			ss << "." << std::endl;
			pio::print(ss.str());
		}
	}

	void create_transform_CPU(const vuint32 & b, const uint32_t n, const size_t num_regs, const bool verbose = true, const bool full = true)
	{
		delete_transform();

		std::string ttype;
		_transform = transform::create_cpu(b, n, num_regs);
		if (verbose)
		{
			std::ostringstream ss; ss << "Using " << _transform->get_type() << " implementation, 4 threads";
			if (full) ss << ", data size: " << std::setprecision(3) << _transform->get_cache_size() / (1024 * 1024.0) << " MB";
			ss << "." << std::endl;
			pio::print(ss.str());
		}
	}

	void delete_transform()
	{
		if (_transform != nullptr)
		{
			delete _transform;
			_transform = nullptr;
		}
	}

	std::string checkpoint_filename() const { return _main_filename + ".chk"; }
	std::string proof_filename() const { return _main_filename + ".proof"; }
	std::string cert_filename() const { return _main_filename + ".cert"; }
	std::string ckpt_filename(const size_t i) const
	{
		std::ostringstream ss; ss << _main_filename << "_" << i << ".ckpt";
		return ss.str();
	}

	static std::string uint64toString(const uint64_t u)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u;
		return ss.str();
	}

	static std::string gfn(const uint32_t b, const uint32_t n)
	{
		std::ostringstream ss;
		ss << b << "^{2^" << n << "} + 1";
		return ss.str();
	}

	static std::string gfn_status(const bool isPrp, const uint64_t pkey, const uint64_t ckey, const uint64_t res64)
	{
		std::ostringstream ss; ss << " is ";
		if (isPrp) ss << "a probable prime"; else ss << "composite, res64 = " << uint64toString(res64);
		if (pkey != 0) ss << ", pkey = " << uint64toString(pkey);
		if (ckey != 0) ss << ", ckey = " << uint64toString(ckey);
		ss << ".";
		return ss.str();
	}

	void init_print_progress(const int i0, const int i_start)
	{
		_print_range = i0; _print_i = i_start;
		if (_is_boinc) boinc_fraction_done((i0 > i_start) ? static_cast<double>(i0 - i_start) / i0 : 0.0);
	}

	int print_progress(const double elapsed_time, const double display_time, const int i)
	{
		if (_print_i == i) return 1;
		const double percent = static_cast<double>(_print_range - i) / _print_range;
		const double mul_time = display_time / (_print_i - i); _print_i = i;
		const int dcount = std::max(static_cast<int>(1.0 / mul_time), 2);
		if (_is_boinc) boinc_fraction_done(percent);
		else
		{
			const double remaining_time = mul_time * i, expected_time = elapsed_time / percent;
			std::ostringstream ss; ss << std::setprecision(3) << percent * 100.0 << "% done, " << timer::format_time(remaining_time)
									<< "/" << timer::format_time(expected_time) << " remaining, " << mul_time * 1e3 << " ms/bit.        \r";
			pio::display(ss.str());
		}
		return dcount;
	}

	static void clearline() { pio::display("                                                            \r"); }

	static void parse_b(const std::string & b_filename, vuint32 & b)
	{
		std::ifstream file(b_filename);
		if (!file.is_open()) pio::error("cannot open input file", true);

		size_t i = 0;
		std::string line;
		while (std::getline(file, line))
		{
			const uint32_t b_i = uint32_t(std::stoi(line));
			if (b_i % 2 != 0) pio::error("b must be even", true);
			if (b_i < 6) pio::error("b < 6 is not supported", true);
			if (b_i > 2000000000) pio::error("b > 2000000000 is not supported", true);
			if ((b_i == 0) || ((b_i & (~b_i + 1)) == b_i)) pio::error("b must not be a power of two", true);
			b[i] = b_i;
			++i; if (i == VSIZE) break;
		}
		while (i < VSIZE) { b[i] = b[i - 1]; ++i; }

		file.close();
	}

	int _read_checkpoint(const std::string & filename, const int where, int & i, double & elapsed_time)
	{
		file checkpoint_file(filename);
		if (!checkpoint_file.exists()) return -1;

		int version = 0;
		if (!checkpoint_file.read(reinterpret_cast<char *>(&version), sizeof(version))) return -2;
		if (version != 1) return -2;
		int rwhere = 0;
		if (!checkpoint_file.read(reinterpret_cast<char *>(&rwhere), sizeof(rwhere))) return -2;
		if (rwhere != where) return -2;
		if (!checkpoint_file.read(reinterpret_cast<char *>(&i), sizeof(i))) return -2;
		if (!checkpoint_file.read(reinterpret_cast<char *>(&elapsed_time), sizeof(elapsed_time))) return -2;
		const size_t num_reg = (where == 0) ? 2 : 3;
		if (!_transform->read_checkpoint(checkpoint_file, num_reg)) return -2;
		if (!checkpoint_file.check_crc32()) return -2;
		return 0;
	}

	bool read_checkpoint(const int where, int & i, double & elapsed_time)
	{
		std::string checkpoint_file = checkpoint_filename();
		int error = _read_checkpoint(checkpoint_file, where, i, elapsed_time);
		if (error < -1)
		{
			std::ostringstream ss; ss << checkpoint_file << ": invalid checkpoint";
			pio::error(ss.str());
		}
		checkpoint_file += ".old";
		if (error < 0)
		{
			error = _read_checkpoint(checkpoint_file, where, i, elapsed_time);
			if (error < -1)
			{
				std::ostringstream ss; ss << checkpoint_file << ": invalid checkpoint";
				pio::error(ss.str());
			}
		}
		return (error == 0);
	}

	void save_checkpoint(const int where, const int i, const double elapsed_time) const
	{
		const std::string checkpoint_file = checkpoint_filename(), old_checkpoint_file = checkpoint_file + ".old", new_checkpoint_file = checkpoint_file + ".new";

		{
			file checkpoint_file(new_checkpoint_file, "wb", false);
			int version = 1;
			if (!checkpoint_file.write(reinterpret_cast<const char *>(&version), sizeof(version))) return;
			if (!checkpoint_file.write(reinterpret_cast<const char *>(&where), sizeof(where))) return;
			if (!checkpoint_file.write(reinterpret_cast<const char *>(&i), sizeof(i))) return;
			if (!checkpoint_file.write(reinterpret_cast<const char *>(&elapsed_time), sizeof(elapsed_time))) return;
			const size_t num_reg = (where == 0) ? 2 : 3;
			_transform->save_checkpoint(checkpoint_file, num_reg);
			checkpoint_file.write_crc32();
		}

		std::remove(old_checkpoint_file.c_str());

		struct stat s;
		if ((stat(checkpoint_file.c_str(), &s) == 0) && (std::rename(checkpoint_file.c_str(), old_checkpoint_file.c_str()) != 0))	// file exists and cannot rename it
		{
			pio::error("cannot save checkpoint");
			return;
		}

		if (std::rename(new_checkpoint_file.c_str(), checkpoint_file.c_str()) != 0)
		{
			pio::error("cannot save checkpoint");
			return;
		}
	}

	void clear_checkpoint() const
	{
		const std::string checkpoint_file = checkpoint_filename();
		std::remove(checkpoint_file.c_str());
		std::remove(std::string(checkpoint_file + ".old").c_str());
	}

	static bool boinc_quit_request(const BOINC_STATUS & status)
	{
		if ((status.quit_request | status.abort_request | status.no_heartbeat) == 0) return false;

		std::ostringstream ss; ss << "Terminating because Boinc ";
		if (status.quit_request != 0) ss << "requested that we should quit.";
		else if (status.abort_request != 0) ss << "requested that we should abort.";
		else if (status.no_heartbeat != 0) ss << "heartbeat was lost.";
		ss << std::endl;
		pio::print(ss.str());
		return true;
	}

	void print_state(const bool suspended)
	{
		if (_print_sr)
		{
			std::ostringstream ss_s; ss_s << "Boinc client is " << (suspended ? "suspended." : "resumed.") << std::endl;
			pio::print(ss_s.str());
		}
		if (!suspended) _print_sr = false;
	}

	void boinc_monitor()
	{
		BOINC_STATUS status; boinc_get_status(&status);
		if (boinc_quit_request(status)) { quit(); return; }

		if (status.suspended != 0)
		{
			print_state(true);
			while (status.suspended != 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				boinc_get_status(&status);
				if (boinc_quit_request(status)) { quit(); return; }
			}
			print_state(false);
		}
	}

	void boinc_monitor(const int where, const int i, watch & chrono)
	{
		BOINC_STATUS status; boinc_get_status(&status);
		if (boinc_quit_request(status)) { quit(); return; }

		if (status.suspended != 0)
		{
			print_state(true);
			save_checkpoint(where, i, chrono.get_elapsed_time());
			while (status.suspended != 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				boinc_get_status(&status);
				if (boinc_quit_request(status)) { quit(); return; }
			}
			print_state(false);
		}

		if (boinc_time_to_checkpoint() != 0)
		{
			save_checkpoint(where, i, chrono.get_elapsed_time());
			boinc_checkpoint_completed();
		}
	}

	uint8_t get_bit_mask(const vuint32 & e, const size_t i) const
	{
		uint8_t m = 0;
		for (size_t j = 0; j < VSIZE; ++j) m |= (((e[j] & (1u << i)) != 0) ? uint8_t(1) : uint8_t(0)) << j;	// TODO vectorize
		return m;
	}

	void power(const size_t reg, const vuint32 & e) const
	{
		transform * const ptransform = _transform;
		ptransform->init_multiplicand(reg);
		ptransform->set(1);
		for (int i = ilog2_32(reduce_max(e)); i >= 0; --i)
		{
			ptransform->square_dup(0);
			const uint8_t mask = get_bit_mask(e, size_t(i));
			ptransform->mul_mask(mask);
		}
	}

	void powerz(const size_t reg, const mpzv & e) const
	{
		transform * const ptransform = _transform;
		ptransform->init_multiplicand(reg);
		ptransform->set(1);
		for (int i = e.get_max_index(); i >= 0; --i)
		{
			ptransform->square_dup(0);
			const uint32_t mask = e.get_bit_mask(size_t(i));
			ptransform->mul_mask(uint8_t(mask));
		}
	}

	static int B_GerbiczLi(const size_t esize)
	{
		const size_t L = (size_t(2) << (ilog2_32(static_cast<uint32_t>(esize)) / 2));
		return static_cast<int>((esize - 1) / L) + 1;
	}

	static int B_PietrzakLi(const size_t esize, const int depth)
	{
		return static_cast<int>((esize - 1) >> depth) + 1;
	}

	// out: reg_0 is 2^exponent and reg_1 is d(t)
	EReturn prp(const mpzv & exponent, const int B_GL, const int B_PL, double & test_time)
	{
		transform * const ptransform = _transform;
		gint & gi = *_gi;

		int ri = 0; double restored_time = 0;
		const bool found = read_checkpoint(0, ri, restored_time);
		if (!found)
		{
			ri = 0; restored_time = 0;
			ptransform->set(1);
			ptransform->copy(1, 0);	// d(t)
		}
		else
		{
			std::ostringstream ss; ss << "Resuming from a checkpoint." << std::endl;
			pio::print(ss.str());
			if (ri == -1)
			{
				test_time = 0;
				return EReturn::Success;
			}
		}

		watch chrono(found ? restored_time : 0);
		int i0 = exponent.get_max_index();
		const int i_start = found ? ri : i0;
		init_print_progress(i0, i_start);
		int dcount = 100;

		for (int i = i_start; i >= 0; --i)
		{
			if (_is_boinc) boinc_monitor(0, i, chrono);

			if (quitting())
			{
				save_checkpoint(0, i, chrono.get_elapsed_time());
				return EReturn::Aborted;
			}

			if (i % dcount == 0)
			{
				const double elapsed_time = chrono.get_elapsed_time(), displayTime = chrono.get_display_time();
				if (displayTime >= 10) { dcount = print_progress(elapsed_time, displayTime, i); chrono.reset_display_time(); }
				if (!_is_boinc && (chrono.get_record_time() > 600)) { save_checkpoint(0, i, chrono.get_elapsed_time()); chrono.reset_record_time(); }
			}

			ptransform->square_dup(exponent.get_bit_mask(size_t(i)));
			// if (i == i_start) ptransform->cosmic_ray();	// => invalid
			// if (i == 0) ptransform->cosmic_ray();	// => invalid

			if ((i % B_GL == 0) && (i / B_GL != 0))
			{
				ptransform->copy(2, 0);
				ptransform->mul(1);	// d(t)
				ptransform->copy(1, 0);
				ptransform->copy(0, 2);
			}
			if ((B_PL != 0) && (i % B_PL == 0))
			{
				ptransform->getInt(gi);
				file ckptFile(ckpt_filename(size_t(i / B_PL)), "wb", true);
				gi.write(ckptFile);
				ckptFile.write_crc32();
			}
		}

		test_time = chrono.get_elapsed_time();
		save_checkpoint(0, -1, test_time);
		return EReturn::Success;
	}

	// Gerbicz-Li error checking
	// in: reg_0 is 2^exponent and reg_1 is d(t)
	// out: return valid/invalid
	EReturn GL(const mpzv & exponent, const int B_GL, double & valid_time)
	{
		transform * const ptransform = _transform;
		gint & gi = *_gi;

		clearline(); pio::display("Validating...\r");

		watch chrono;

		// d(t + 1) = d(t) * result
		ptransform->mul(1);
		ptransform->copy(2, 0);

		// d(t)^{2^B}
		ptransform->copy(0, 1);
		for (int i = B_GL - 1; i >= 0; --i)
		{
			if (_is_boinc) boinc_monitor();
			if (quitting()) return EReturn::Aborted;
			ptransform->square_dup(0);
		}
		ptransform->copy(1, 0);

		mpzv res; res.set_GL_residue(exponent, B_GL);

		// 2^res
		ptransform->set(1);
		for (int i = res.get_max_index(); i >= 0; --i)
		{
			if (_is_boinc) boinc_monitor();
			if (quitting()) return EReturn::Aborted;
			ptransform->square_dup(res.get_bit_mask(size_t(i)));
		}

		// d(t)^{2^B} * 2^res
		ptransform->mul(1);

		// d(t)^{2^B} * 2^res ?= d(t + 1)
		ptransform->getInt(gi);
		vuint64 h1; gi.gethash64(h1);
		ptransform->copy(0, 2);
		ptransform->getInt(gi);
		vuint64 h2; gi.gethash64(h2);

		const bool success = cmp(h1, h2);

		valid_time = chrono.get_elapsed_time();
		return success ? EReturn::Success : EReturn::Failed;
	}

	// (Pietrzak-Li proof generation
	// in: ckpt[i]
	// out: proof file, proof key
	EReturn PL(const int depth, double & proof_time, vuint64 & pkey)
	{
		transform * const ptransform = _transform;
		gint & gi = *_gi;

		clearline(); pio::display("Generating proof...\r");

		watch chrono;

		file proof_file(proof_filename(), "wb", true);
		int version = 1;
		proof_file.write(reinterpret_cast<const char *>(&version), sizeof(version));
		proof_file.write(reinterpret_cast<const char *>(&depth), sizeof(depth));

		// mu[0] = ckpt[0]
		file ckpt_file(ckpt_filename(0), "rb", true);
		gi.read(ckpt_file);
		ckpt_file.check_crc32();

		gi.write(proof_file);
		// v1 = mu[0]^w[0]
		vuint32 q; gi.gethash32(q);
		ptransform->setInt(gi);
		power(0, q);
		ptransform->copy(2, 0);

		const size_t L = size_t(1) << depth;
		mpzv * const w = new mpzv[L / 2];
		w[0].set_ui(q);

// size_t s = 0;	// complexity

		for (int k = 1; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// mu[k] = ckpt[i]^w[0]
			file ckpt_file(ckpt_filename(i), "rb", true);
			gi.read(ckpt_file);
			ckpt_file.check_crc32();
			ptransform->setInt(gi);

			powerz(0, w[0]);
// s += mpz_sizeinbase(w[0], 2);
			ptransform->copy(1, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// mu[k] *= ckpt[i + 2 * j]^w[j]
				file ckpt_file(ckpt_filename(i + 2 * j), "rb", true);
				gi.read(ckpt_file);
				ckpt_file.check_crc32();
				ptransform->setInt(gi);

				powerz(0, w[j]);
// s += mpz_sizeinbase(w[j], 2);
				ptransform->mul(1);
				ptransform->copy(1, 0);

				if (_is_boinc) boinc_monitor();
				if (quitting())
				{
					delete[] w;
					return EReturn::Aborted;
				}
			}
// std::cout << k << ": " << s << ", " << 32 * k * (1 << (k - 1))<< std::endl;
			ptransform->getInt(gi);
			gi.write(proof_file);
			vuint32 q; gi.gethash32(q);
			// v1 = v1 * mu[k]^w[k]
			power(0, q);
			ptransform->mul(2);
			ptransform->copy(2, 0);

			if (i > 1)
			{
				for (size_t j = 0; j < L / 2; j += i) w[i / 2 + j].mul_ui(w[j], q);
			}
		}

		proof_file.write_crc32();

		delete[] w;

		// pkey = hash64(v1);
		ptransform->copy(0, 2);
		ptransform->getInt(gi);
		gi.gethash64(pkey);

		proof_time = chrono.get_elapsed_time();
		return EReturn::Success;
	}

	EReturn quick(const mpzv & exponent, double & test_time, double & valid_time, bool is_prp[VSIZE], vuint64 & res64)
	{
		const int B_GL = B_GerbiczLi(size_t(exponent.get_max_index() + 1));

		const EReturn rPrp = prp(exponent, B_GL, 0, test_time);
		if (rPrp != EReturn::Success) return rPrp;
		{
			gint & gi = *_gi;
			_transform->getInt(gi);
			gi.is_one(is_prp, res64);
		}
		return GL(exponent, B_GL, valid_time);
	}

	EReturn proof(const mpzv & exponent, const int depth, double & test_time, double & valid_time, double & proof_time,
				  bool is_prp[VSIZE], vuint64 & pkey, vuint64 & res64)
	{
		size_t esize = size_t(exponent.get_max_index() + 1);
		const int B_GL = B_GerbiczLi(esize), B_PL = B_PietrzakLi(esize, depth);

		const EReturn rPrp = prp(exponent, B_GL, B_PL, test_time);
		if (rPrp != EReturn::Success) return rPrp;
		{
			gint & gi = *_gi;
			_transform->getInt(gi);
			gi.is_one(is_prp, res64);
		}
		const EReturn rGL = GL(exponent, B_GL, valid_time);
		if (rGL != EReturn::Success) return rGL;
		return PL(depth, proof_time, pkey);
	}

	static uint32_t rand32(const uint32_t rmin, const uint32_t rmax) { return (static_cast<uint32_t>(std::rand()) % (rmax - rmin)) + rmin; }

	EReturn server(const mpzv & exponent, double & time, bool is_prp[VSIZE], uint64_t & pkey, uint64_t & ckey, vuint64 & res64)
	{
/*		transform * const ptransform = _transform;
		gint & gi = *_gi;

		watch chrono;

		file proof_file(proof_filename(), "rb", true);
		int version = 0; proof_file.read(reinterpret_cast<char *>(&version), sizeof(version));
		int depth = 0; proof_file.read(reinterpret_cast<char *>(&depth), sizeof(depth));

		const size_t L = size_t(1) << depth;
		size_t esize = 0; for (size_t j = 0; j < VSIZE; ++j) esize = std::max(esize, mpz_sizeinbase(exponent[j], 2));
		const int B = B_PietrzakLi(esize, depth);

		mpz_t * const w = new mpz_t[L]; for (size_t i = 0; i < L; ++i) mpz_init(w[i]);

		// v1 = mu[0]^w[0], v1: reg = 1
		gi.read(proof_file);
		gi.is_one(is_prp, res64);
		const uint32_t q = gi.gethash32();
		mpz_set_ui(w[0], q);
		ptransform->setInt(gi);
		power(0, q);
		ptransform->copy(1, 0);

		// v2 = 1, v2: reg = 2
		ptransform->set(1);
		ptransform->copy(2, 0);

		for (int k = 1; k <= depth; ++k)
		{
			// mu[k]: reg = 3
			gi.read(proof_file);
			const uint32_t q = gi.gethash32();
			ptransform->setInt(gi);
			ptransform->copy(3, 0);

			// v1 = v1 * mu[k]^w[k]
			power(0, q);
			ptransform->mul(1);
			ptransform->copy(1, 0);

			// v2 = v2^w[k] * mu[k]
			power(2, q);
			ptransform->mul(3);
			ptransform->copy(2, 0);

			const size_t i = size_t(1) << (depth - k);
			for (size_t j = 0; j < L; j += 2 * i) mpz_mul_ui(w[i + j], w[j], q);

			if (quitting()) return EReturn::Aborted;
		}

		proof_file.check_crc32();

		// pkey = hash64(v1);
		ptransform->copy(0, 1);
		ptransform->getInt(gi);
		pkey = gi.gethash64();

		mpz_t p2, t; mpz_init_set_ui(p2, 0); mpz_init(t);
		mpz_t e; mpz_init_set(e, exponent[0]);	// TODO
		for (size_t i = 0; i < L; i++)
		{
			mpz_mod_2exp(t, e, static_cast<unsigned long int>(B));
			mpz_addmul(p2, t, w[i]);
			mpz_div_2exp(e, e, static_cast<unsigned long int>(B));
		}
		mpz_clear(e);

		for (size_t i = 0; i < L; ++i) mpz_clear(w[i]);
		delete[] w;

		// encode
		std::srand(static_cast<unsigned int>(std::time(nullptr))); std::rand();	// use current time as seed for random generator
		const uint32_t rnd1 = rand32(2, 64), rnd2 = rand32(16, 256), rnd3 = rand32(4, 65536);

		mpz_set_ui(t, rnd1);
		mpz_mul_2exp(t, t, static_cast<unsigned long int>(B));
		if (mpz_cmp(p2, t) > 0)
		{
			mpz_sub(p2, p2, t);
			ptransform->set(2);
			power(0, rnd1);
			ptransform->mul(2);
			ptransform->copy(2, 0);
		}

		power(1, rnd2);
		ptransform->copy(1, 0);
		ptransform->set(2);
		power(0, rnd3);
		ptransform->mul(1);
		ptransform->getInt(gi);
		ckey = gi.gethash64();

		power(2, rnd2);
		ptransform->getInt(gi);
		mpz_mul_ui(p2, p2, rnd2);
		mpz_add_ui(p2, p2, rnd3);

		{
			file cert_file(cert_filename(), "wb", true);
			int version = 1;
			cert_file.write(reinterpret_cast<const char *>(&version), sizeof(version));
			cert_file.write(reinterpret_cast<const char *>(&B), sizeof(B));
			gi.write(cert_file);
			cert_file.write(p2);
			cert_file.write_crc32();
		}

		mpz_clear(p2);  mpz_clear(t);

		time = chrono.get_elapsed_time();*/
		return EReturn::Success;
	}

	EReturn check(double & time, uint64_t & ckey)
	{
/*		transform * const ptransform = _transform;
		gint & gi = *_gi;

		int ri = 0; double restored_time = 0;
		const bool found = read_checkpoint(1, ri, restored_time);
		if (!found)
		{
			ri = 0; restored_time = 0;
		}
		else
		{
			std::ostringstream ss; ss << "Resuming from a checkpoint." << std::endl;
			pio::print(ss.str());
		}

		int B = 0;
		mpz_t p2; mpz_init(p2);
		{
			file cert_file(cert_filename(), "rb", true);
			int version = 0; cert_file.read(reinterpret_cast<char *>(&version), sizeof(version));
			cert_file.read(reinterpret_cast<char *>(&B), sizeof(B));
			gi.read(cert_file);
			cert_file.read(p2);
			cert_file.check_crc32();
		}
		const int p2size = static_cast<int>(mpz_sizeinbase(p2, 2));

		// Gerbicz test for v2^{2^B} and Gerbicz-Li test for 2^p2
		const int L = B_GerbiczLi(static_cast<size_t>(B)), GL = B_GerbiczLi(static_cast<size_t>(p2size));

		watch chrono(found ? restored_time : 0);
		const int i0 = p2size + B - 1;
		init_print_progress(i0, found ? ri : i0);
		int dcount = 100;

		// v2 = v2^{2^B}
		if (!found || (ri >= p2size))
		{
			if (!found)
			{
				ptransform->setInt(gi);
				ptransform->copy(1, 0);	// d(t) = u(0)
			}
			for (int i = found ? ri : i0; i >= p2size; --i)
			{
				if (_is_boinc) boinc_monitor(1, i, chrono);

				if (quitting())	// || (i == p2size + B/2))	// test context
				{
					save_checkpoint(1, i, chrono.get_elapsed_time());
					mpz_clear(p2);
					return EReturn::Aborted;
				}

				if (i % dcount == 0)
				{
					const double elapsed_time = chrono.get_elapsed_time(), displayTime = chrono.get_display_time();
					if (displayTime >= 10) { dcount = print_progress(elapsed_time, displayTime, i); chrono.reset_display_time(); }
					if (!_is_boinc && (chrono.get_record_time() > 600)) { save_checkpoint(1, i, chrono.get_elapsed_time()); chrono.reset_record_time(); }
				}

				const int j = i0 - i;
				if ((j % L == 0) && (j != 0))
				{
					ptransform->copy(2, 0);
					ptransform->mul(1);	// d(t) = d(t - 1) * u(t * L)
					ptransform->copy(1, 0);
					ptransform->copy(0, 2);
				}

				ptransform->square_dup(0);
				// if (i == i0) ptransform->add1();	// => invalid
				// if (i == p2size) ptransform->add1();	// => invalid
			}

			ptransform->copy(2, 0);	// v2

			// u((t + 1) * L)
			if (B % L != 0)
			{
				for (int i = L - (B % L); i > 0; --i)
				{
					if (_is_boinc) boinc_monitor();
					if (quitting()) { mpz_clear(p2); return EReturn::Aborted; }
					ptransform->square_dup(0);
				}
			}
			// d(t + 1) = d(t) * u((t + 1) * L)
			ptransform->mul(1);
			ptransform->copy(3, 0);

			// d(t)^{2^L}
			ptransform->copy(0, 1);
			for (int i = L; i > 0; --i)
			{
				if (_is_boinc) boinc_monitor();
				if (quitting()) { mpz_clear(p2); return EReturn::Aborted; }
				ptransform->square_dup(0);
			}
			ptransform->copy(1, 0);

			// u(0) * d(t)^{2^L}
			ptransform->setInt(gi);
			ptransform->mul(1);

			// u(0) * d(t)^{2^L} ?= d(t + 1)
			ptransform->getInt(gi);
			const uint64_t h1 = gi.gethash64();
			ptransform->copy(0, 3);
			ptransform->getInt(gi);
			const uint64_t h2 = gi.gethash64();

			if (h1 != h2) { mpz_clear(p2); return EReturn::Failed; }
		}

		// 2^p2
		if (!found || (ri >= p2size))
		{
			ptransform->set(1);
			ptransform->copy(1, 0);	// d(t)
		}
		for (int i = found ? std::min(ri, p2size - 1) : p2size - 1; i >= 0; --i)
		{
			if (_is_boinc) { boinc_monitor(1, i, chrono); }

			if (quitting())	// || (i == p2size/2))	// test context
			{
				save_checkpoint(1, i, chrono.get_elapsed_time());
				mpz_clear(p2);
				return EReturn::Aborted;
			}

			if (i % dcount == 0)
			{
				const double elapsed_time = chrono.get_elapsed_time(), displayTime = chrono.get_display_time();
				if (displayTime >= 10) { dcount = print_progress(elapsed_time, displayTime, i); chrono.reset_display_time(); }
				if (!_is_boinc && (chrono.get_record_time() > 600)) { save_checkpoint(1, i, chrono.get_elapsed_time()); chrono.reset_record_time(); }
			}

			ptransform->square_dup(mpz_tstbit(p2, mp_bitcnt_t(i)) != 0);
			// if (i == p2size - 1) ptransform->add1();	// => invalid
			// if (i == 0) ptransform->add1();	// => invalid

			if ((i % GL == 0) && (i / GL != 0))
			{
				ptransform->copy(3, 0);
				ptransform->mul(1);	// d(t)
				ptransform->copy(1, 0);
				ptransform->copy(0, 3);
			}
		}

		ptransform->copy(3, 0);

		// v1' = v2 * 2^p2
		ptransform->mul(2);

		// ckey = hash64(v1')
		ptransform->getInt(gi);
		ckey = gi.gethash64();

		// d(t + 1) = d(t) * result
		ptransform->copy(0, 3);
		ptransform->mul(1);
		ptransform->copy(2, 0);

		// d(t)^{2^GL}
		ptransform->copy(0, 1);
		for (int i = GL - 1; i >= 0; --i)
		{
			if (_is_boinc) boinc_monitor();
			if (quitting()) { mpz_clear(p2); return EReturn::Aborted; }
			ptransform->square_dup(0);
		}
		ptransform->copy(1, 0);

		mpz_t res, t; mpz_init_set_ui(res, 0); mpz_init(t);
		while (mpz_sgn(p2) != 0)
		{
			mpz_mod_2exp(t, p2, static_cast<unsigned long int>(GL));
			mpz_add(res, res, t);
			mpz_div_2exp(p2, p2, static_cast<unsigned long int>(GL));
		}
		mpz_clear(p2); mpz_clear(t);

		// 2^res
		ptransform->set(1);
		for (int i = static_cast<int>(mpz_sizeinbase(res, 2)) - 1; i >= 0; --i)
		{
			if (_is_boinc) boinc_monitor();
			if (quitting()) return EReturn::Aborted;
			ptransform->square_dup(mpz_tstbit(res, mp_bitcnt_t(i)) != 0);
		}

		mpz_clear(res);

		// d(t)^{2^GL} * 2^res
		ptransform->mul(1);

		// d(t)^{2^GL} * 2^res ?= d(t + 1)
		ptransform->getInt(gi);
		const uint64_t h1 = gi.gethash64();
		ptransform->copy(0, 2);
		ptransform->getInt(gi);
		const uint64_t h2 = gi.gethash64();

		if (h1 != h2) return EReturn::Failed;

		time = chrono.get_elapsed_time();*/
		return EReturn::Success;
	}

	EReturn bench(const uint32_t m, const size_t device, const bool isCPU)
	{
		static constexpr uint32_t bm[12] = { 1000000000, 1000000000, 1000000000, 460000000, 550000000, 400000000,
											 80000000, 20000000, 6000000, 3000000, 550000, 150000 };
		const size_t num_regs = 3;

		const uint32_t b = bm[m - 12], n = m;
		vuint32 vb; for (size_t j = 0; j < VSIZE; ++j) vb[j] = b;

		if (isCPU) create_transform_CPU(vb, n, num_regs, m == 16, false);
		else create_transform_GPU(vb, n, num_regs, device, m == 16, false);

		transform * const pTransform = _transform;

		// pTransform->info();

		_gi = new gint(size_t(1) << n, vb);
		vuint32 svb; for (size_t j = 0; j < VSIZE; ++j) svb[j] = 6 + 2 * uint32_t(j);
		mpzv exponent; exponent.set_exponent(svb, 6);

		double testTime = 0, validTime = 0; bool is_prp[VSIZE]; vuint64 res64;
		const EReturn qret = quick(exponent, testTime, validTime, is_prp, res64);
		clear_checkpoint();
		delete _gi; _gi = nullptr;

		pio::print(gfn(b, n));

		std::ostringstream ss;
		if (qret == EReturn::Failed) ss << ": test failed!" << std::endl;
		else if (qret == EReturn::Aborted) ss << ": aborted." << std::endl;
		else
		{
			static std::atomic_bool _break;

			std::atomic_thread_fence(std::memory_order_release);
			_break.store(false, std::memory_order_relaxed);

			std::thread delay([=]()
			{
				std::atomic_thread_fence(std::memory_order_release);
				std::this_thread::sleep_for(std::chrono::seconds(10));
				_break.store(true, std::memory_order_relaxed);
			}); delay.detach();

			watch chrono(0);
			size_t i = 1;
			while (!_break.load(std::memory_order_relaxed))
			{
				pTransform->square_dup(((i % 2) != 0) ? uint32_t(-1) : 0);
				++i;
				if (quitting())
				{
					ss << ": aborted." << std::endl;
					pio::print(ss.str());
					delete_transform();
					return EReturn::Aborted;
				}
			}
			std::atomic_thread_fence(std::memory_order_acquire);

			pTransform->copy(1, 0);	// synchro

			const size_t memsize = _transform->get_cache_size();

			const double error = _transform->get_error();
			const double mul_time = chrono.get_elapsed_time() / i, estimated_time = mul_time * std::log2(b) * (size_t(1) << n);
			ss << ": " << timer::format_time(estimated_time) << std::setprecision(3) << ", " << mul_time * 1e3 << " ms/bit, ";
			if (error != 0) ss << "error = " << std::setprecision(4) << error << ", ";
			ss << "data size: " << memsize / (1024 * 1024.0) << " MB." << std::endl;
		}
		pio::print(ss.str());

		delete_transform();
		return qret;
	}

public:
	EReturn check(const std::string & b_filename, const uint32_t n, const EMode mode, const size_t device, const bool isCPU, const int depth = 7)
	{
		_n = n;

		vuint32 b; parse_b(b_filename, b);
		uint32_t b_min = uint32_t(-1), b_max = 0;
		for (size_t i = 0; i < VSIZE; ++i) { b_min = std::min(b_min, b[i]); b_max = std::max(b_max, b[i]); }
		const uint32_t b_TODO = b[0];

		const bool empty_main_filename = _main_filename.empty();
		if (empty_main_filename)
		{
			std::ostringstream ss; ss << "g" << n << "_" << b_min << "_" << b_max;
			_main_filename = ss.str();
		}

		if (mode == EMode::Bench) return bench(n, device, isCPU);

		size_t num_regs;
		if (mode == EMode::Quick) num_regs = 3;
		else if (mode == EMode::Proof) num_regs = 3;
		else if (mode == EMode::Server) num_regs = 4;
		else if (mode == EMode::Check) num_regs = 4;
		else return EReturn::Failed;

		bool checkError = true;
		if (isCPU) create_transform_CPU(b, n, num_regs, checkError);
		else create_transform_GPU(b, n, num_regs, device);

		_gi = new gint(size_t(1) << n, b);

		EReturn success = EReturn::Failed;

		if (mode == EMode::Check)
		{
			double time = 0; uint64_t ckey = 0;
			success = check(time, ckey);
			const double error = _transform->get_error();
			clearline();
			std::ostringstream ss; ss << gfn(b_TODO, n);
			if (success == EReturn::Success)
			{
				ss << " is checked, ckey = " << uint64toString(ckey);
				if (error != 0) ss << ", error = " << std::setprecision(4) << error;
				ss << ", time = " << timer::format_time(time) << ".";
			}
			else if (success == EReturn::Failed) ss << ": check failed!";
			else ss << ": terminated.";
			ss << std::endl; pio::print(ss.str());
			if (success == EReturn::Success)
			{
				pio::result(ss.str());
				if (!_is_boinc) clear_checkpoint();
			}
		}
		else
		{
			mpzv exponent; exponent.set_exponent(b, n);

			if (mode == EMode::Quick)
			{
				double testTime = 0, validTime = 0; bool is_prp[VSIZE]; vuint64 res64;
				success = quick(exponent, testTime, validTime, is_prp, res64);
				const double error = _transform->get_error();
				clearline();
				std::ostringstream ss;
				if (success == EReturn::Failed) ss << "Validation failed!";
				else if (success == EReturn::Aborted) ss << "Test was aborted.";
				if (success == EReturn::Success)
				{
					ss << "Test succeeded";
					if (error != 0) ss << ", error = " << std::setprecision(4) << error;
					ss << ", time = " << timer::format_time(testTime + validTime) << "." << std::endl;;
					for (size_t j = 0; j < VSIZE; ++j) ss << gfn(b[j], n) << gfn_status(is_prp[j], 0, 0, res64[j]) << std::endl;
				}
				pio::print(ss.str());
				if ((success == EReturn::Success) || (!_is_boinc && (success == EReturn::Failed))) pio::result(ss.str());
				if (!_is_boinc && (success != EReturn::Aborted)) clear_checkpoint();
			}
			else if (mode == EMode::Proof)
			{
				double testTime = 0, validTime = 0, proofTime = 0; bool is_prp[VSIZE]; vuint64 pkey, res64;
				success = proof(exponent, depth, testTime, validTime, proofTime, is_prp, pkey, res64);
				const double error = _transform->get_error();
				const double time = testTime + validTime + proofTime;
				clearline();
				std::ostringstream ss;
				if (success == EReturn::Failed) ss << "Validation failed!";
				else if (success == EReturn::Aborted) ss << "Test was aborted.";
				if (success == EReturn::Success)
				{
					ss << "Proof file is generated";
					if (error != 0) ss << ", error = " << std::setprecision(4) << error;
					ss << ", time = " << timer::format_time(time) << ".";
				}
				ss << std::endl; pio::print(ss.str());
				if (success == EReturn::Success)
				{
					std::ostringstream ssr;
					for (size_t j = 0; j < VSIZE; ++j) ssr << gfn(b[j], n) << gfn_status(is_prp[j], pkey[j], 0, res64[j]) << std::endl;
					pio::result(ssr.str());
					if (!_is_boinc)
					{
						for (size_t i = 0, L = size_t(1) << depth; i < L; ++i) std::remove(ckpt_filename(i).c_str());
						clear_checkpoint();
					}
				}
			}
			else if (mode == EMode::Server)
			{
				double time = 0; bool is_prp[VSIZE]; uint64_t pkey = 0, ckey = 0; vuint64 res64;
				success = server(exponent, time, is_prp, pkey, ckey, res64);
				const double error = _transform->get_error();
				for (size_t j = 0; j < VSIZE; ++j)
				{
					std::ostringstream ss; ss << gfn(b[j], n);
					if (success == EReturn::Success) ss << gfn_status(is_prp[j], pkey, ckey, res64[j]/*, error, time*/);
					else if (success == EReturn::Failed) ss << ": generation failed!";
					else ss << ": terminated.";
					ss << std::endl; pio::print(ss.str());
					if (success == EReturn::Success) pio::result(ss.str());
				}
			}
		}

		delete _gi; _gi = nullptr;
		delete_transform();
		if (empty_main_filename) _main_filename.clear();

		return success;
	}
};
