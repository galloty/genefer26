# genefer&nbsp;26
Generalized Fermat Prime search program

## About

**genefer** performs a fast probable primality test for numbers of the form *b*<sup>2<sup>*n*</sup></sup>&nbsp;+&nbsp;1 ([Generalized Fermat Numbers](https://genefer.great-site.net/)) using a [Fermat test](https://en.wikipedia.org/wiki/Fermat_primality_test). The GPU implementation is based on [OpenCL™](https://www.khronos.org/opencl/) and the GPU implementation is multithreaded.  

**genefer** [version 22](https://github.com/galloty/genefer22) checks a single GFN. It is dedicated to the search for GFN primes in the range 17&nbsp;&le;&nbsp;*n*&nbsp;&le;&nbsp;23.

**genefer** [version 26](https://github.com/galloty/genefer26) checks a list of 8, 16 or 32 GFN. It is dedicated to the search for GFN primes in the range 15&nbsp;&le;&nbsp;*n*&nbsp;&le;&nbsp;18.  
The search for *b*&nbsp;<&nbsp;2,000,000,000 and *n*&nbsp;&le;&nbsp;14 is now complete thanks to the [PRIVATE GFN SERVER](http://boincvm.proxyma.ru:30080/test4vm/index.php) GFN-12/13/14 prime search.  
**genefer** [version 20](https://github.com/galloty/genefer20) was used to test *n*&nbsp;=&nbsp;13 and *n*&nbsp;=&nbsp;14. This application is deprecated. It is replaced with this version.

*geneferv* implements an [Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623) discovered by Darren Li.
The test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking and a proof is generated with ([Pietrzak - Li](https://eprint.iacr.org/2018/627.pdf)) algorithm. Thanks to the Verifiable Delay Function, distributed projects run at twice the speed of double-checked calculations.  

Any number of the form *b*<sup>2<sup>*n*</sup></sup> + 1 such that 10,000 &le; *b* < 2,000,000,000 and 13 &le; *n* &le; 18 can be tested on GPU or CPU.  

The OpenCL implementation is highly optimized for Nvidia *Ada Lovelace* and *Blackwell* architectures (GeForce RTX 40 and 50 series), AMD RDNA 2, 3, 4 architectures (Radeon RX 6000, 7000, 9000 series) and Intel *Alchemist* and *Battlemage* architectures (Arc A7x0 and Arc B5x0).  
The x64 implementation is highly optimized for CPU with AVX-512 or AVX10.2 (AMD Zen 4, 5, 6 [Ryzen 7000, 9000, 10000 series]; Intel X-Series 7th, 9th and 10th Gen; Intel 11th Gen; Intel Xeon Skylake-W and their successors; Intel *Nova Lake*).

## Operating system

 - Linux x64  
 - Windows x64  
 
## Build

Select the [makefile](https://github.com/galloty/genefer26/tree/main/build) of your target. On Windows, [MSYS2](https://www.msys2.org/) distribution and building platform can be installed.  
The compiler (gcc or clang) can be selected, Boinc interface is optional.  
The default settings are gcc and linked to Boinc.  

## Licence

**genefer** is free source code, under the MIT license (see [LICENSE](https://github.com/galloty/genefer26/blob/main/LICENSE)). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
