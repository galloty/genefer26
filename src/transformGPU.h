/*
Copyright 2026, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "transform.h"
#include "engine.h"
#include "ocl/kernel.h"

namespace arch_g_namespace
{

#define	P1S			(127 * (uint32(1) << 24) + 1)
#define	Q1S			2164260865u		// p * q = 1 (mod 2^32)
#define	R1S			33554430u		// 2^32 mod p
#define	RSQ1S		402124772u		// (2^32)^2 mod p
#define	H1S			100663290u		// Montgomery form of the primitive root 3
#define	IM1S		2063729671u		// MF of I = 3^{(p - 1)/4}
#define	MFIM1S		1930170389u		// MF of MF of I to convert input into MF
#define	SQRTI1S		1626730317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1S	856006302u		// MF of i * sqrt(i)

#define	P2S			(63 * (uint32(1) << 25) + 1)
#define	Q2S			2181038081u
#define	R2S			67108862u
#define	RSQ2S		2111798781u
#define	H2S			335544310u		// MF of the primitive root 5
#define	IM2S		530075385u
#define	MFIM2S		1036950657u
#define	SQRTI2S		338852760u
#define	ISQRTI2S	1090446030u

#define	P3S			(15 * (uint32(1) << 27) + 1)
#define	Q3S			2281701377u
#define	R3S			268435454u
#define	RSQ3S		1172168163u
#define	H3S			268435390u		// MF of the primitive root 31
#define	IM3S		473486609u
#define	MFIM3S		734725699u
#define	SQRTI3S		1032137103u
#define	ISQRTI3S	1964242958u

#define	INVP2_P1S	2130706177u		// MF of 1 / P2 (mod P1)
#define	INVP3_P1S	608773230u		// MF of 1 / P3 (mod P1)
#define	INVP3_P2S	1409286102u		// MF of 1 / P3 (mod P2)

#define	P1P2P3LS	1962934273u					// (P1 * P2 * P3) mod 2^32
#define	P1P2P3HS	2111326211158966273ul		// (P1 * P2 * P3) >> 32

#define	P1P2P3_2LS	3128950784u					// (P1 * P2 * P3 / 2) mod 2^32
#define	P1P2P3_2HS	1055663105579483136ul		// (P1 * P2 * P3 / 2) >> 32

#define P1U			(125 * (uint32(1) << 25) + 1)
#define	Q1U			100663297u		// p * q = 1 (mod 2^32)
#define	R1U			100663295u		// 2^32 mod p
#define	RSQ1U		232465106u		// (2^32)^2 mod p
#define	H1U			301989885u		// Montgomery form of the primitive root 3
#define	IM1U		1486287593u		// MF of I = 3^{(p - 1)/4}
#define	MFIM1U		3645424034u		// MF of MF of I to convert input into MF
#define	SQRTI1U		3580437317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1U	2017881188u		// MF of i * sqrt(i)

#define P2U			(243 * (uint32(1) << 24) + 1)
#define	Q2U			218103809u
#define	R2U			218103807u
#define	RSQ2U		3444438393u
#define	H2U			1526726649u		// MF of the primitive root 7
#define	IM2U		99906823u
#define	MFIM2U		1773796560u
#define	SQRTI2U		2024944857u
#define	ISQRTI2U	2119710515u

#define P3U			(235 * (uint32(1) << 24) + 1)
#define	Q3U			352321537u
#define	R3U			352321535u
#define	RSQ3U		3810498414u
#define	H3U			1056964605u		// MF of the primitive root 3
#define	IM3U		2213106415u
#define	MFIM3U		2454519270u
#define	SQRTI3U		3448990025u
#define	ISQRTI3U	3659377330u

#define	INVP2_P1U	1797558821u		// MF of 1 / P2 (mod P1)
#define	INVP3_P1U	3075822917u		// MF of 1 / P3 (mod P1)
#define	INVP3_P2U	4076863457u		// MF of 1 / P3 (mod P2)

#define	P1P2P3LU	3623878657u					// (P1 * P2 * P3) mod 2^32
#define	P1P2P3HU	15696902887611105282ul		// (P1 * P2 * P3) >> 32

#define	P1P2P3_2LU	1811939328u					// (P1 * P2 * P3 / 2) mod 2^32
#define	P1P2P3_2HU	7848451443805552641ul		// (P1 * P2 * P3 / 2) >> 32

template<size_t VSIZE, bool IS32>
class transformGPU : public transform
{
	template<uint32 P, uint32 Q, uint32 R, uint32 H>
	class ZPT : public ZP
	{
	private:
		// static uint32 _add(const uint32 a, const uint32 b) { return a + b - ((a >= P - b) ? P : 0); }
		static uint32 _sub(const uint32 a, const uint32 b) { return a - b + ((a < b) ? P : 0); }

		static uint32 _mul(const uint32 lhs, const uint32 rhs)
		{
			const uint64 t = lhs * uint64(rhs);
			const uint32 lo = uint32(t), hi = uint32(t >> 32);
			const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
			return _sub(hi, mp);
		}

		ZPT pow(const size_t e) const
		{
			if (e == 0) return ZPT(R);	// MF of one is R
			ZPT r = ZPT(R), y = *this;
			for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r *= y; y *= y; }
			r *= y;
			return r;
		}

	public:
		ZPT() {}
		explicit ZPT(const uint32 n) : ZP(n) {}

		int32 get_int() const { return (_n >= P / 2) ? int32(_n - P) : int32(_n); }
		ZPT & set_int(const int32 i) { _n = (i < 0) ? (uint32(i) + P) : uint32(i); return *this; }

		ZPT & operator*=(const ZPT & rhs) { _n = _mul(_n, rhs._n); return *this; }

		static const ZPT primroot_ln(const int ln) { return ZPT(H).pow((P - 1) >> ln); }
		static ZPT norm_ln(const int ln) { return ZPT(P - ((P - 1) >> ln)); }
	};

	using ZP1 = ZPT<IS32 ? P1U : P1S, IS32 ? Q1U : Q1S, IS32 ? R1U : R1S, IS32 ? H1U : H1S>;
	using ZP2 = ZPT<IS32 ? P2U : P2S, IS32 ? Q2U : Q2S, IS32 ? R2U : R2S, IS32 ? H2U : H2S>;
	using ZP3 = ZPT<IS32 ? P3U : P3S, IS32 ? Q3U : Q3S, IS32 ? R3U : R3S, IS32 ? H3U : H3S>;

private:
	// Necessary conditions are OCL_VSIZE >= OCL_CARRY_VSIZE and CARRY_LENGTH >= OCL_VSIZE.
	static const size_t OCL_VSIZE = 4, OCL_CARRY_VSIZE = 2, CARRY_LENGTH = 8;
	using xengine = engine<VSIZE, CARRY_LENGTH, OCL_VSIZE, OCL_CARRY_VSIZE, IS32>;

	const size_t _num_regs;
	const int _lsize;
	const size_t _size;
	ZP * const _z;
	xengine * _engine = nullptr;

public:
	transformGPU(const UInt32_8 & b, const int n, const size_t num_regs, const size_t device_id,
				 const bool is_boinc, const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id)
				: transform(b, n, EKind::GPU), _num_regs(num_regs), _lsize(n), _size(size_t(1) << n),
				_z(new ZP[3 * VSIZE * num_regs * _size])
	{
		const size_t size = _size;

		const bool is_boinc_platform = is_boinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_engine = new xengine(eng_platform, is_boinc_platform ? 0 : device_id, static_cast<int>(n), is_boinc, num_regs);
		set_gpu_type(_engine->getType());

		std::ostringstream src;

		src << "#define N_SZ\t" << size << "u" << std::endl;
		src << "#define LN_SZ\t" << n << std::endl;
		src << "#define VSIZE\t" << VSIZE << std::endl;
		if (IS32) src << "#define IS32\t" << 1 << std::endl;

		src << "#define P1\t" << (IS32 ? P1U : P1S) << "u" << std::endl;
		src << "#define Q1\t" << (IS32 ? Q1U : Q1S) << "u" << std::endl;
		src << "#define RSQ1\t" << (IS32 ? RSQ1U : RSQ1S) << "u" << std::endl;
		src << "#define IM1\t" << (IS32 ? IM1U : IM1S) << "u" << std::endl;
		src << "#define MFIM1\t" << (IS32 ? MFIM1U : MFIM1S) << "u" << std::endl;
		src << "#define SQRTI1\t" << (IS32 ? SQRTI1U : SQRTI1S) << "u" << std::endl;
		src << "#define ISQRTI1\t" << (IS32 ? ISQRTI1U : ISQRTI1S) << "u" << std::endl;

		src << "#define P2\t" << (IS32 ? P2U : P2S) << "u" << std::endl;
		src << "#define Q2\t" << (IS32 ? Q2U : Q2S) << "u" << std::endl;
		src << "#define RSQ2\t" << (IS32 ? RSQ2U : RSQ2S) << "u" << std::endl;
		src << "#define IM2\t" << (IS32 ? IM2U : IM2S) << "u" << std::endl;
		src << "#define MFIM2\t" << (IS32 ? MFIM2U : MFIM2S) << "u" << std::endl;
		src << "#define SQRTI2\t" << (IS32 ? SQRTI2U : SQRTI2S) << "u" << std::endl;
		src << "#define ISQRTI2\t" << (IS32 ? ISQRTI2U : ISQRTI2S) << "u" << std::endl;

		src << "#define P3\t" << (IS32 ? P3U : P3S) << "u" << std::endl;
		src << "#define Q3\t" << (IS32 ? Q3U : Q3S) << "u" << std::endl;
		src << "#define RSQ3\t" << (IS32 ? RSQ3U : RSQ3S) << "u" << std::endl;
		src << "#define IM3\t" << (IS32 ? IM3U : IM3S) << "u" << std::endl;
		src << "#define MFIM3\t" << (IS32 ? MFIM3U : MFIM3S) << "u" << std::endl;
		src << "#define SQRTI3\t" << (IS32 ? SQRTI3U : SQRTI3S) << "u" << std::endl;
		src << "#define ISQRTI3\t" << (IS32 ? ISQRTI3U : ISQRTI3S) << "u" << std::endl;

		src << "#define INVP2_P1\t" << (IS32 ? INVP2_P1U : INVP2_P1S) << "u" << std::endl;
		src << "#define INVP3_P1\t" << (IS32 ? INVP3_P1U : INVP3_P1S) << "u" << std::endl;
		src << "#define INVP3_P2\t" << (IS32 ? INVP3_P2U : INVP3_P2S) << "u" << std::endl;

		src << "#define P1P2P3L\t" << (IS32 ? P1P2P3LU : P1P2P3LS) << "u" << std::endl;
		src << "#define P1P2P3H\t" << (IS32 ? P1P2P3HU : P1P2P3HS) << "ul" << std::endl;
		src << "#define P1P2P3_2L\t" << (IS32 ? P1P2P3_2LU : P1P2P3_2LS) << "u" << std::endl;
		src << "#define P1P2P3_2H\t" << (IS32 ? P1P2P3_2HU : P1P2P3_2HS) << "ul" << std::endl;

		// Not converted into Montgomery form such that output is converted out of MF
		src << "#define NORM1\t" << ZP1::norm_ln(n - 1).get() << "u" << std::endl;
		src << "#define NORM2\t" << ZP2::norm_ln(n - 1).get() << "u" << std::endl;
		src << "#define NORM3\t" << ZP3::norm_ln(n - 1).get() << "u" << std::endl;

		src << "#define W_SZ\t" << size / 2 << "u" << std::endl;
		src << "#define OCL_VSIZE\t" << OCL_VSIZE << std::endl;
		src << "#define OCL_CARRY_VSIZE\t" << OCL_CARRY_VSIZE << std::endl;
		src << "#define CARRY_LENGTH\t" << CARRY_LENGTH << std::endl;
		src << "#define CARRY_WG_SZ\t" << _engine->get_carry_workgroup_size() << "u" << std::endl;

		std::cout << "N_SZ = " << size << ", VSIZE = " << VSIZE << ", OCL_VSIZE = " << OCL_VSIZE << ", OCL_CARRY_VSIZE = " << OCL_CARRY_VSIZE
			<< ", CARRY_LENGTH = " << CARRY_LENGTH << ", CARRY_WG_SZ = " << _engine->get_carry_workgroup_size() << std::endl;
		std::cout << "transform: " << 3 * VSIZE / OCL_VSIZE * size / 8
			<< ", carry1: " << VSIZE / OCL_CARRY_VSIZE * size / CARRY_LENGTH << " / " << _engine->get_carry_workgroup_size()
			<< ", carry2: " << ((VSIZE * size / CARRY_LENGTH) >> _engine->get_carry_shift()) << std::endl;

		if (is_boinc || !_engine->readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine->loadProgram(src.str());
		_engine->alloc_memory();
		_engine->create_kernels();

		ZP * const w = new ZP[3 * size / 2];
		ZP1 * const w1 = reinterpret_cast<ZP1 *>(&w[0 * size / 2]);
		ZP2 * const w2 = reinterpret_cast<ZP2 *>(&w[1 * size / 2]);
		ZP3 * const w3 = reinterpret_cast<ZP3 *>(&w[2 * size / 2]);

		ZP1 prs1 = ZP1::primroot_ln(n); ZP2 prs2 = ZP2::primroot_ln(n); ZP3 prs3 = ZP3::primroot_ln(n);
		for (int ls = n - 2; ls >= 0; --ls)
		{
			ZP1 r_s1 = prs1; prs1 *= prs1; const ZP1 & r_s1sq = prs1;
			ZP2 r_s2 = prs2; prs2 *= prs2; const ZP2 & r_s2sq = prs2;
			ZP3 r_s3 = prs3; prs3 *= prs3; const ZP3 & r_s3sq = prs3;

			const size_t s = size_t(1) << ls;
			for (size_t j = 0; j < s; ++j)
			{
				const size_t jr = bitrev(j, s);
				w1[s + jr] = r_s1; w2[s + jr] = r_s2; w3[s + jr] = r_s3;
				r_s1 *= r_s1sq; r_s2 *= r_s2sq; r_s3 *= r_s3sq;
			}
		}

		_engine->write_memory_w(w);
		delete[] w;

		uint32_t bu[VSIZE], b_inv[VSIZE]; int b_s[VSIZE];
		for (size_t i = 0; i < VSIZE; ++i)
		{
			const uint32_t bi = b[i];
			const int s = 31 - __builtin_clz(bi) - 1;
			bu[i] = bi;
			b_inv[i] = static_cast<uint32_t>((static_cast<uint64_t>(1) << (s + 32)) / bi);
			b_s[i] = s;
		}

		_engine->write_memory_b(bu, b_inv, b_s);
	}

	virtual ~transformGPU()
	{
		_engine->release_kernels();
		_engine->release_memory();
		_engine->clearProgram();
		delete _engine;

		delete[] _z;
	}

protected:
	static size_t gpu_index(const size_t k, const size_t j, const size_t size)
	{
		return (j % OCL_VSIZE) + OCL_VSIZE * k + (size * OCL_VSIZE) * (j / OCL_VSIZE);
	}

	void getZi(Int32_8 * const zi) const override
	{
		const size_t size = _size;

		_engine->read_memory_z(_z);
		const ZP1 * const z1 = reinterpret_cast<ZP1 *>(&_z[0 * VSIZE * size]);

		for (size_t k = 0; k < size; ++k)
		{
			int32 d[VSIZE]; for (size_t j = 0; j < VSIZE; ++j) d[j] = z1[gpu_index(k, j, size)].get_int();
			zi[k] = Int32_8(d);
		}
	}

	void setZi(const Int32_8 * const zi) override
	{
		const size_t size = _size, vsize = VSIZE * size;
		ZP1 * const z1 = reinterpret_cast<ZP1 *>(&_z[0 * vsize]);
		ZP2 * const z2 = reinterpret_cast<ZP2 *>(&_z[1 * vsize]);
		ZP3 * const z3 = reinterpret_cast<ZP3 *>(&_z[2 * vsize]);

		for (size_t k = 0; k < size; ++k)
		{
			const Int32_8 zk = zi[k];
			for (size_t j = 0; j < VSIZE; ++j)
			{
				const int32 d = zk[j];
				const size_t i = gpu_index(k, j, size);
				z1[i].set_int(d); z2[i].set_int(d); z3[i].set_int(d);
			}
		}

		_engine->write_memory_z(_z);
	}

public:
	void set(const uint32_t a) override
	{
		_engine->set(a);
	}

	void square_dup(const uint32_t dup) override
	{
		const int ln_8 = _lsize - 3;

		_engine->forward8_0();

		int lm = ln_8;
		// for (size_t s = 8; lm > 3; lm -= 3, s *= 8)
		// {
		// 	_engine->forward8(lm - 3, s);
		// 	std::cout << "forward8: " << lm - 3 << ", " << s << std::endl;
		// }
		// std::cout << "square: " << lm << std::endl;
		// exit(0);

		for (size_t s = 8; lm > 6; lm -= 3, s *= 8) _engine->forward8(lm - 3, s);

		if (lm == 6) _engine->square64();
		else if (lm == 5) _engine->square32();
		else if (lm == 4) _engine->square16();

		for (size_t s = size_t(1) << (ln_8 - lm); s >= 1; lm += 3, s /= 8) _engine->backward8(lm, s);

		_engine->carry(dup);
	}

	void init_multiplicand(const size_t src) override
	{
		const int ln_8 = _lsize - 3;

		_engine->copyp(src);

		_engine->forward8_0p();

		int lm = ln_8;
		for (size_t s = 8; lm > 3; lm -= 3, s *= 8) _engine->forward8p(lm - 3, s);

		if (lm == 3) _engine->fwd8();
		else if (lm == 2) _engine->fwd4x2();
	}

	void mul() override
	{
		const int ln_8 = _lsize - 3;

		_engine->forward8_0();

		int lm = ln_8;
		for (size_t s = 8; lm > 3; lm -= 3, s *= 8) _engine->forward8(lm - 3, s);

		if (lm == 3) _engine->mul8();
		else if (lm == 2) _engine->mul4x2();
		else if (lm == 1) _engine->mul2x4();

		for (size_t s = size_t(1) << (ln_8 - lm); s >= 1; lm += 3, s /= 8) _engine->backward8(lm, s);

		_engine->carry(0);
	}

	void mul_mask(const uint32_t mask) override
	{
		const int ln_8 = _lsize - 3;

		_engine->forward8_0();

		int lm = ln_8;
		for (size_t s = 8; lm > 3; lm -= 3, s *= 8) _engine->forward8(lm - 3, s);

		if (lm == 3) _engine->mul8_mask(mask);
		else if (lm == 2) _engine->mul4x2_mask(mask);
		else if (lm == 1) _engine->mul2x4_mask(mask);

		for (size_t s = size_t(1) << (ln_8 - lm); s >= 1; lm += 3, s /= 8) _engine->backward8(lm, s);

		_engine->carry(0);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		_engine->copy(dst, src);
	}

	void copy_mask(const size_t dst, const size_t src, const uint32_t mask) const override
	{
		if (mask == 0) return;
		if (mask == (uint32_t(1) << VSIZE) - 1) { copy(dst, src); return; }

		_engine->copy_mask(dst, src, mask);
	}

	void power(const size_t src, const uint32_t e) override { _power(src, e); }
	void power_vec(const size_t src, const UInt32_8 & e) override { _power_vec(src, e); }

	bool read_checkpoint(file & cFile) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(get_kind())) return false;
		if (!cFile.read(reinterpret_cast<char *>(_z), 3 * VSIZE * _num_regs * _size * sizeof(ZP))) return false;

		_engine->write_memory_z(_z, _num_regs);
		return true;
	}

	void save_checkpoint(file & cFile) const override
	{
		_engine->read_memory_z(_z, _num_regs);

		const int kind = static_cast<int>(get_kind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;
		if (!cFile.write(reinterpret_cast<const char *>(_z), 3 * VSIZE * _num_regs * _size * sizeof(ZP))) return;
	}

	size_t get_data_size() const override { return (3 * (VSIZE * (_num_regs + 1) * _size + _size / 2)) * sizeof(ZP); }
	size_t get_cache_size() const override { return (3 * (VSIZE * _size + _size / 2)) * sizeof(ZP); }
	double get_error() const override { return 0; }

	void is_one(bool b[8], UInt64_8 & res64) const override { _is_one(b, res64); }
	UInt64_8 gethash64() const override { return _gethash64(); }
	UInt32_8 gethash32() const override { return _gethash32(); }

	void cosmic_ray() override { _engine->cosmic_ray(); }
};

}