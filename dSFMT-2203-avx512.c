/**
 * @file dSFMT-2203-AVX512.c
 * @brief double precision SIMD-oriented Fast Mersenne Twister (dSFMT)
 * based on IEEE 754 format. for AVX-512 x4 parallel implementation
 *
 * @author Masaki Ota
 *
 * Copyright (C) 2022 Masaki Ota. All rights reserved.
 * Copyright (C) 2007,2008 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University. All rights reserved.
 * 
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include "dSFMT-params.h"
#include "dSFMT-common.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** dsfmt internal state vector */
dsfmt_t dsfmt_global_data;
/** dsfmt mexp for check */
static const int dsfmt_mexp = DSFMT_MEXP;

/*----------------
  STATIC FUNCTIONS
  ----------------*/
inline static uint32_t ini_func1(uint32_t x);
inline static uint32_t ini_func2(uint32_t x);
#define idxof(n) (n)
static void initial_mask(dsfmt_t *dsfmt);
static void period_certification(dsfmt_t *dsfmt);

/**
 * This function represents the recursion formula.
 * @param r output 128-bit
 * @param a a 128-bit part of the internal state array
 * @param b a 128-bit part of the internal state array
 * @param d a 128-bit part of the internal state array (I/O)
 */
inline static __m512i do_recursion_x4_avx512( __m512i a, __m512i b, __m512i* u) {
    const __m512i shift128_reverse_idx = 
        _mm512_setr_epi32(0, 0, 0, 0, 19,18,17,16, 23,22,21,20, 27,26,25,24);
    const __m512i expand_lung_idx = 
        _mm512_setr_epi32(15,14,13,12, 12,13,14,15, 15,14,13,12, 12,13,14,15);
        
    __m512i v, w, x, y, z;

    z = _mm512_slli_epi64(a, DSFMT_SL1);
    z = _mm512_xor_si512(z, b);

    x = _mm512_permutex2var_epi32(_mm512_setzero_si512(),shift128_reverse_idx , z);
    z = _mm512_xor_si512(z, x);
    x = _mm512_inserti64x4(_mm512_setzero_si512(), _mm512_castsi512_si256(z), 1);
    y = _mm512_permutexvar_epi32(expand_lung_idx, _mm512_load_epi64(u));  /* lung */
    y = _mm512_ternarylogic_epi32(y, x, z, 0x96);
    _mm512_store_epi64(u, y);

    v = _mm512_srli_epi64(y, DSFMT_SR);
    w = _mm512_and_si512(y, _mm512_broadcast_i32x4(sse2_param_mask.i128));
    v = _mm512_ternarylogic_epi32(v, a, w, 0x96);
    return v;
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range [0, 1).
 * @param w 512bit stracture of double precision floating point numbers (I/O)
 */
inline static __m512d convert_c0o1_avx512(__m512i w) {
    return _mm512_add_pd(_mm512_castsi512_pd(w), _mm512_set1_pd(-1.0));
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range (0, 1].
 * @param w 512bit stracture of double precision floating point numbers (I/O)
 */
inline static __m512d convert_o0c1_avx512(__m512i w) {
    return _mm512_sub_pd(_mm512_set1_pd(2.0), _mm512_castsi512_pd(w));
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range (0, 1).
 * @param w 512bit stracture of double precision floating point numbers (I/O)
 */
inline static __m512d convert_o0o1_avx512(__m512i w) {
    w = _mm512_or_si512(w, _mm512_set1_epi64(1));
    return _mm512_add_pd(_mm512_castsi512_pd(w), _mm512_set1_pd(-1.0));
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range (0, 1).
 * @param w 512bit stracture of double precision floating point numbers (I/O)
 */
inline static __m512d no_convert_avx512(__m512i w) {
    return _mm512_castsi512_pd(w);
}


typedef __m512d (*avx512_convert_func_t)(__m512i);

/**
 * This function fills the user-specified array with double precision
 * floating point pseudorandom numbers of the IEEE 754 format.
 * @param dsfmt dsfmt state vector.
 * @param array an 128-bit array to be filled by pseudorandom numbers.
 * @param size number of 128-bit pseudorandom numbers to be generated.
 * @param cvt  converter functicton 
 */
inline static void gen_rand_array_avx512(dsfmt_t* dsfmt, w128_t* array,
    ptrdiff_t size, avx512_convert_func_t cvt) {
    ptrdiff_t i;

    __m512i v0, v1, v2, v3, v4, lung;
    v0 = _mm512_load_epi64(&dsfmt->status_x4[0]);
    v1 = _mm512_load_epi64(&dsfmt->status_x4[1]);
    v2 = _mm512_load_epi64(&dsfmt->status_x4[2]);
    v3 = _mm512_load_epi64(&dsfmt->status_x4[3]);
    v4 = _mm512_load_epi64(&dsfmt->status_x4[4]);
    lung = _mm512_broadcast_i64x2(dsfmt->status[DSFMT_N].si);

    for (i = 0; i < size; i += DSFMT_N) {
        v0 = do_recursion_x4_avx512(v0, _mm512_alignr_epi64(v2, v1, 6), &lung);
        _mm512_storeu_pd(&array[i + 0], cvt(v0));
        v1 = do_recursion_x4_avx512(v1, _mm512_alignr_epi64(v3, v2, 6), &lung);
        _mm512_storeu_pd(&array[i + 4], cvt(v1));
        v2 = do_recursion_x4_avx512(v2, _mm512_alignr_epi64(v4, v3, 6), &lung);
        _mm512_storeu_pd(&array[i + 8], cvt(v2));
        v3 = do_recursion_x4_avx512(v3, _mm512_alignr_epi64(v0, v4, 6), &lung);
        _mm512_storeu_pd(&array[i + 12], cvt(v3));
        v4 = do_recursion_x4_avx512(v4, _mm512_alignr_epi64(v1, v0, 6), &lung);
        _mm512_storeu_pd(&array[i + 16], cvt(v4));
    }

    _mm512_store_epi64(&dsfmt->status_x4[0], v0);
    _mm512_store_epi64(&dsfmt->status_x4[1], v1);
    _mm512_store_epi64(&dsfmt->status_x4[2], v2);
    _mm512_store_epi64(&dsfmt->status_x4[3], v3);
    _mm512_store_epi64(&dsfmt->status_x4[4], v4);

    dsfmt->status[DSFMT_N].si = _mm512_extracti64x2_epi64(lung, 3); // last zmm lane
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func1(uint32_t x) {
    return (x ^ (x >> 27)) * (uint32_t)1664525UL;
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func2(uint32_t x) {
    return (x ^ (x >> 27)) * (uint32_t)1566083941UL;
}

/**
 * This function initializes the internal state array to fit the IEEE
 * 754 format.
 * @param dsfmt dsfmt state vector.
 */
static void initial_mask(dsfmt_t* dsfmt) {
    int i;
    uint64_t* psfmt;

    psfmt = &dsfmt->status[0].u[0];
    for (i = 0; i < DSFMT_N * 2; i++) {
        psfmt[i] = (psfmt[i] & DSFMT_LOW_MASK) | DSFMT_HIGH_CONST;
    }
}

/**
 * This function certificate the period of 2^{SFMT_MEXP}-1.
 * @param dsfmt dsfmt state vector.
 */
static void period_certification(dsfmt_t* dsfmt) {
    uint64_t pcv[2] = { DSFMT_PCV1, DSFMT_PCV2 };
    uint64_t tmp[2];
    uint64_t inner;
    int i;
#if (DSFMT_PCV2 & 1) != 1
    int j;
    uint64_t work;
#endif

    tmp[0] = (dsfmt->status[DSFMT_N].u[0] ^ DSFMT_FIX1);
    tmp[1] = (dsfmt->status[DSFMT_N].u[1] ^ DSFMT_FIX2);

    inner = tmp[0] & pcv[0];
    inner ^= tmp[1] & pcv[1];
    for (i = 32; i > 0; i >>= 1) {
        inner ^= inner >> i;
    }
    inner &= 1;
    /* check OK */
    if (inner == 1) {
        return;
    }
    /* check NG, and modification */
#if (DSFMT_PCV2 & 1) == 1
    dsfmt->status[DSFMT_N].u[1] ^= 1;
#else
    for (i = 1; i >= 0; i--) {
        work = 1;
        for (j = 0; j < 64; j++) {
            if ((work & pcv[i]) != 0) {
                dsfmt->status[DSFMT_N].u[i] ^= work;
                return;
            }
            work = work << 1;
        }
    }
#endif
    return;
}

/*----------------
  PUBLIC FUNCTIONS
  ----------------*/
  /**
   * This function returns the identification string.  The string shows
   * the Mersenne exponent, and all parameters of this generator.
   * @return id string.
   */
const char* dsfmt_get_idstring(void) {
    return DSFMT_IDSTR;
}

/**
 * This function returns the minimum size of array used for \b
 * fill_array functions.
 * @return minimum size of array used for fill_array functions.
 */
int dsfmt_get_min_array_size(void) {
    return DSFMT_N64;
}

void dsfmt_gen_rand_all(dsfmt_t* dsfmt) {
    __m512i v0, v1, v2, v3, v4, lung;
    lung = _mm512_broadcast_i32x4(dsfmt->status[DSFMT_N].si);
    v0 = _mm512_load_si512(&dsfmt->status_x4[0]);
    v1 = _mm512_load_si512(&dsfmt->status_x4[1]);
    v2 = _mm512_load_si512(&dsfmt->status_x4[2]);
    v3 = _mm512_load_si512(&dsfmt->status_x4[3]);
    v4 = _mm512_load_si512(&dsfmt->status_x4[4]);

    v0 = do_recursion_x4_avx512(v0, _mm512_alignr_epi64(v2, v1, 6), &lung);
    _mm512_store_si512(&dsfmt->status_x4[0], v0);
    v1 = do_recursion_x4_avx512(v1, _mm512_alignr_epi64(v3, v2, 6), &lung);
    _mm512_store_si512(&dsfmt->status_x4[1], v1);
    v2 = do_recursion_x4_avx512(v2, _mm512_alignr_epi64(v4, v3, 6), &lung);
    _mm512_store_si512(&dsfmt->status_x4[2], v2);
    v3 = do_recursion_x4_avx512(v3, _mm512_alignr_epi64(v0, v4, 6), &lung);
    _mm512_store_si512(&dsfmt->status_x4[3], v3);
    v4 = do_recursion_x4_avx512(v4, _mm512_alignr_epi64(v1, v0, 6), &lung);    
    _mm512_store_si512(&dsfmt->status_x4[4], v4);

    dsfmt->status[DSFMT_N].si = _mm512_extracti64x2_epi64(lung, 3);
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range [1, 2) to the
 * specified array[] by one call. The number of pseudorandom numbers
 * is specified by the argument \b size, which must be at least (SFMT_MEXP
 * / 128) * 2 and a multiple of two.  The function
 * get_min_array_size() returns this minimum size.  The generation by
 * this function is much faster than the following fill_array_xxx functions.
 *
 * For initialization, init_gen_rand() or init_by_array() must be called
 * before the first call of this function. This function can not be
 * used after calling genrand_xxx functions, without initialization.
 *
 * @param dsfmt dsfmt state vector.
 * @param array an array where pseudorandom numbers are filled
 * by this function.  The pointer to the array must be "aligned"
 * (namely, must be a multiple of 16) in the SIMD version, since it
 * refers to the address of a 128-bit integer.  In the standard C
 * version, the pointer is arbitrary.
 *
 * @param size the number of 64-bit pseudorandom integers to be
 * generated.  size must be a multiple of 2, and greater than or equal
 * to (SFMT_MEXP / 128) * 2.
 *
 * @note \b memalign or \b posix_memalign is available to get aligned
 * memory. Mac OSX doesn't have these functions, but \b malloc of OSX
 * returns the pointer to the aligned memory block.
 */
void dsfmt_fill_array_close1_open2(dsfmt_t *dsfmt, double array[], ptrdiff_t size) {
    assert(size % 2 == 0);
    assert(size >= DSFMT_N64);
    gen_rand_array_avx512(dsfmt, (w128_t *)array, size / 2, no_convert_avx512);
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range (0, 1] to the
 * specified array[] by one call. This function is the same as
 * fill_array_close1_open2() except the distribution range.
 *
 * @param dsfmt dsfmt state vector.
 * @param array an array where pseudorandom numbers are filled
 * by this function.
 * @param size the number of pseudorandom numbers to be generated.
 * see also \sa fill_array_close1_open2()
 */
void dsfmt_fill_array_open_close(dsfmt_t *dsfmt, double array[], ptrdiff_t size) {
    assert(size % DSFMT_N64 == 0);
    gen_rand_array_avx512(dsfmt, (w128_t *)array, size / 2, convert_o0c1_avx512);
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range [0, 1) to the
 * specified array[] by one call. This function is the same as
 * fill_array_close1_open2() except the distribution range.
 *
 * @param array an array where pseudorandom numbers are filled
 * by this function.
 * @param dsfmt dsfmt state vector.
 * @param size the number of pseudorandom numbers to be generated.
 * see also \sa fill_array_close1_open2()
 */
void dsfmt_fill_array_close_open(dsfmt_t *dsfmt, double array[], ptrdiff_t size) {
    assert(size % DSFMT_N64 == 0);
    gen_rand_array_avx512(dsfmt, (w128_t *)array, size / 2, convert_c0o1_avx512);
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range (0, 1) to the
 * specified array[] by one call. This function is the same as
 * fill_array_close1_open2() except the distribution range.
 *
 * @param dsfmt dsfmt state vector.
 * @param array an array where pseudorandom numbers are filled
 * by this function.
 * @param size the number of pseudorandom numbers to be generated.
 * see also \sa fill_array_close1_open2()
 */
void dsfmt_fill_array_open_open(dsfmt_t *dsfmt, double array[], ptrdiff_t size) {
    assert(size % DSFMT_N64 == 0);
    gen_rand_array_avx512(dsfmt, (w128_t *)array, size / 2, convert_o0o1_avx512);
}

#if defined(__INTEL_COMPILER)
#  pragma warning(disable:981)
#endif
/**
 * This function initializes the internal state array with a 32-bit
 * integer seed.
 * @param dsfmt dsfmt state vector.
 * @param seed a 32-bit integer used as the seed.
 * @param mexp caller's mersenne expornent
 */
void dsfmt_chk_init_gen_rand(dsfmt_t *dsfmt, uint32_t seed, int mexp) {
    int i;
    uint32_t *psfmt;

    /* make sure caller program is compiled with the same MEXP */
    if (mexp != dsfmt_mexp) {
	fprintf(stderr, "DSFMT_MEXP doesn't match with dSFMT.c\n");
	exit(1);
    }
    psfmt = &dsfmt->status[0].u32[0];
    psfmt[idxof(0)] = seed;
    for (i = 1; i < (DSFMT_N + 1) * 4; i++) {
        psfmt[idxof(i)] = 1812433253UL
	    * (psfmt[idxof(i - 1)] ^ (psfmt[idxof(i - 1)] >> 30)) + i;
    }
    initial_mask(dsfmt);
    period_certification(dsfmt);
    dsfmt->idx = DSFMT_N64;
}

/**
 * This function initializes the internal state array,
 * with an array of 32-bit integers used as the seeds
 * @param dsfmt dsfmt state vector.
 * @param init_key the array of 32-bit integers, used as a seed.
 * @param key_length the length of init_key.
 * @param mexp caller's mersenne expornent
 */
void dsfmt_chk_init_by_array(dsfmt_t *dsfmt, uint32_t init_key[],
			     int key_length, int mexp) {
    int i, j, count;
    uint32_t r;
    uint32_t *psfmt32;
    int lag;
    int mid;
    int size = (DSFMT_N + 1) * 4;	/* pulmonary */

    /* make sure caller program is compiled with the same MEXP */
    if (mexp != dsfmt_mexp) {
	fprintf(stderr, "DSFMT_MEXP doesn't match with dSFMT.c\n");
	exit(1);
    }
    if (size >= 623) {
	lag = 11;
    } else if (size >= 68) {
	lag = 7;
    } else if (size >= 39) {
	lag = 5;
    } else {
	lag = 3;
    }
    mid = (size - lag) / 2;

    psfmt32 = &dsfmt->status[0].u32[0];
    memset(dsfmt->status, 0x8b, sizeof(dsfmt->status));
    if (key_length + 1 > size) {
	count = key_length + 1;
    } else {
	count = size;
    }
    r = ini_func1(psfmt32[idxof(0)] ^ psfmt32[idxof(mid % size)]
		  ^ psfmt32[idxof((size - 1) % size)]);
    psfmt32[idxof(mid % size)] += r;
    r += key_length;
    psfmt32[idxof((mid + lag) % size)] += r;
    psfmt32[idxof(0)] = r;
    count--;
    for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
	r = ini_func1(psfmt32[idxof(i)]
		      ^ psfmt32[idxof((i + mid) % size)]
		      ^ psfmt32[idxof((i + size - 1) % size)]);
	psfmt32[idxof((i + mid) % size)] += r;
	r += init_key[j] + i;
	psfmt32[idxof((i + mid + lag) % size)] += r;
	psfmt32[idxof(i)] = r;
	i = (i + 1) % size;
    }
    for (; j < count; j++) {
	r = ini_func1(psfmt32[idxof(i)]
		      ^ psfmt32[idxof((i + mid) % size)]
		      ^ psfmt32[idxof((i + size - 1) % size)]);
	psfmt32[idxof((i + mid) % size)] += r;
	r += i;
	psfmt32[idxof((i + mid + lag) % size)] += r;
	psfmt32[idxof(i)] = r;
	i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
	r = ini_func2(psfmt32[idxof(i)]
		      + psfmt32[idxof((i + mid) % size)]
		      + psfmt32[idxof((i + size - 1) % size)]);
	psfmt32[idxof((i + mid) % size)] ^= r;
	r -= i;
	psfmt32[idxof((i + mid + lag) % size)] ^= r;
	psfmt32[idxof(i)] = r;
	i = (i + 1) % size;
    }
    initial_mask(dsfmt);
    period_certification(dsfmt);
    dsfmt->idx = DSFMT_N64;
}
#if defined(__INTEL_COMPILER)
#  pragma warning(default:981)
#endif

#if defined(__cplusplus)
}
#endif
