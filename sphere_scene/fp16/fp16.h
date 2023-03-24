#pragma once
#ifndef HALF
#define HALF

#include <time.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <curand_kernel.h>
#include <cuda_fp16.h>


namespace fp16{
    /* arithmetic functions */
    __device__ __forceinline__ __half operator+(const float a, const __half b) { return __hadd(__float2half(a), b); }
    __device__ __forceinline__ __half operator+(const __half a, const float b) { return __hadd(a, __float2half(b)); }

    __device__ __forceinline__ __half operator-(const float a, const __half b) { return __hsub(__float2half(a), b); }
    __device__ __forceinline__ __half operator-(const __half a, const float b) { return __hsub(a, __float2half(b)); }

    __device__ __forceinline__ __half operator*(const float a, const __half b) { return __hmul(__float2half(a), b); }
    __device__ __forceinline__ __half operator*(const __half a, const float b) { return __hmul(a, __float2half(b)); }

    __device__ __forceinline__ __half operator/(const float a, const __half b) { return __hdiv(__float2half(a), b); }
    __device__ __forceinline__ __half operator/(const __half a, const float b) { return __hdiv(a, __float2half(b)); }

    // __device__ __forceinline__ __half operator-(const __half a) {return __hneg(a);}


    /* comparison functions */
    __device__ __forceinline__ bool operator>(const __half a, const __half b) { return __hgt(a, b); }
    __device__ __forceinline__ bool operator>(const float a, const __half b) { return __hgt(__float2half(a), b); }
    __device__ __forceinline__ bool operator>(const __half a, const float b) { return __hgt(a, __float2half(b)); }

    __device__ __forceinline__ bool operator>=(const __half a, const __half b) { return __hge(a, b); }
    __device__ __forceinline__ bool operator>=(const float a, const __half b) { return __hge(__float2half(a), b); }
    __device__ __forceinline__ bool operator>=(const __half a, const float b) { return __hge(a, __float2half(b)); }

    __device__ __forceinline__ bool operator<(const __half a, const __half b) { return __hlt(a, b); }
    __device__ __forceinline__ bool operator<(const float a, const __half b) { return __hlt(__float2half(a), b); }
    __device__ __forceinline__ bool operator<(const __half a, const float b) { return __hlt(a, __float2half(b)); }

    __device__ __forceinline__ bool operator<=(const __half a, const __half b) { return __hle(a, b); }
    __device__ __forceinline__ bool operator<=(const float a, const __half b) { return __hle(__float2half(a), b); }
    __device__ __forceinline__ bool operator<=(const __half a, const float b) { return __hle(a, __float2half(b)); }

    __device__ __forceinline__ bool operator==(const __half a, const __half b) { return __heq(a, b); }
    __device__ __forceinline__ bool operator==(const float a, const __half b) { return __heq(__float2half(a), b); }
    __device__ __forceinline__ bool operator==(const __half a, const float b) { return __heq(a, __float2half(b)); }

    __device__ __forceinline__ bool operator!=(const __half a, const __half b) { return __hne(a, b); }
    __device__ __forceinline__ bool operator!=(const float a, const __half b) { return __hne(__float2half(a), b); }
    __device__ __forceinline__ bool operator!=(const __half a, const float b) { return __hne(a, __float2half(b)); }

    __device__ __forceinline__ __half max(const __half a, const __half b) {return __hmax(a, b);}
    __device__ __forceinline__ __half min(const __half a, const __half b) {return __hmin(a, b);}

    /* math functions */
    __device__ __forceinline__ __half sqrt(const __half a) {return hsqrt(a);}
    __device__ __forceinline__ __half pow(const __half a, float b) {return __float2half(std::pow(__half2float(a), b));}
    __device__ __forceinline__ __half pow(const __half a, __half b) {return __float2half(std::pow(__half2float(a), __half2float(b)));}
    __device__ __forceinline__ __half sin(const __half a) {return hsin(a);}
    __device__ __forceinline__ __half cos(const __half a) {return hcos(a);}
    __device__ __forceinline__ __half tan(const __half a) {return hsin(a) / hcos(a);}

}

#endif